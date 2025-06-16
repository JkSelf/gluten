/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.shuffle.sort

import org.apache.gluten.exception.GlutenException
import org.apache.gluten.shuffle.SupportsColumnarShuffle

import org.apache.spark.{ShuffleDependency, SparkConf, SparkEnv, TaskContext}
import org.apache.spark.internal.{config, Logging}
import org.apache.spark.internal.config.SHUFFLE_IO_PLUGIN_CLASS
import org.apache.spark.serializer.SerializerManager
import org.apache.spark.shuffle._
import org.apache.spark.shuffle.api.ShuffleExecutorComponents
import org.apache.spark.shuffle.sort.SortShuffleManager.canUseBatchFetch
import org.apache.spark.storage.{BlockId, DisaggregatedShuffleReader}
import org.apache.spark.storage.DisaggregatedBufferedInputStreamAdaptor
import org.apache.spark.util.collection.OpenHashSet

import GlutenDisaggregatedShuffleManager.{bypassDecompressionSerializerManger, loadShuffleExecutorComponents}
import com.ibm.SparkDisaggregatedShuffleBuild

import java.io.InputStream
import java.util.concurrent.ConcurrentHashMap

import scala.collection.JavaConverters._

class GlutenDisaggregatedShuffleManager(
    conf: SparkConf,
    isDriver: Boolean
) extends ShuffleManager
  with SupportsColumnarShuffle
  with Logging {

  conf.set(config.SHUFFLE_CHECKSUM_ENABLED, false)

  val versionString =
    s"${SparkDisaggregatedShuffleBuild.name}-${SparkDisaggregatedShuffleBuild.version} for " +
      s"for ${SparkDisaggregatedShuffleBuild.sparkVersion}" +
      s"_${SparkDisaggregatedShuffleBuild.scalaVersion}"
  logInfo(s"Configured DisaggregatedShuffleManager ($versionString).")

  private lazy val shuffleExecutorComponents = {
    loadShuffleExecutorComponents(conf)
      .asInstanceOf[
        DisaggregatedShuffleDataIO#DisaggregatedShuffleExecutorComponents
      ]
  }

  private lazy val dispatcher = shuffleExecutorComponents.getDispatcher()

  override val shuffleBlockResolver = new IndexShuffleBlockResolver(conf)

  /** A mapping from shuffle ids to the number of mappers producing output for those shuffles. */
  private[this] val taskIdMapsForShuffle = new ConcurrentHashMap[Int, OpenHashSet[Long]]()

  /** Obtains a [[ShuffleHandle]] to pass to tasks. */
  override def registerShuffle[K, V, C](
      shuffleId: Int,
      dependency: ShuffleDependency[K, V, C]): ShuffleHandle = {
    if (dependency.isInstanceOf[ColumnarShuffleDependency[_, _, _]]) {
      logInfo(s"Registering ColumnarShuffle shuffleId: $shuffleId")
      new ColumnarShuffleHandle[K, V](
        shuffleId,
        dependency.asInstanceOf[ColumnarShuffleDependency[K, V, V]])
    } else if (SortShuffleWriter.shouldBypassMergeSort(conf, dependency)) {
      // If there are fewer than spark.shuffle.sort.bypassMergeThreshold partitions and we don't
      // need map-side aggregation, then write numPartitions files directly and just concatenate
      // them at the end. This avoids doing serialization and deserialization twice to merge
      // together the spilled files, which would happen with the normal code path. The downside is
      // having multiple files open at a time and thus more memory allocated to buffers.
      new BypassMergeSortShuffleHandle[K, V](
        shuffleId,
        dependency.asInstanceOf[ShuffleDependency[K, V, V]])
    } else if (SortShuffleManager.canUseSerializedShuffle(dependency)) {
      // Otherwise, try to buffer map outputs in a serialized form, since this is more efficient:
      new SerializedShuffleHandle[K, V](
        shuffleId,
        dependency.asInstanceOf[ShuffleDependency[K, V, V]])
    } else {
      // Otherwise, buffer map outputs in a deserialized form:
      new BaseShuffleHandle(shuffleId, dependency)
    }
  }

  /** Get a writer for a given partition. Called on executors by map tasks. */
  override def getWriter[K, V](
      handle: ShuffleHandle,
      mapId: Long,
      context: TaskContext,
      metrics: ShuffleWriteMetricsReporter): ShuffleWriter[K, V] = {
    val mapTaskIds =
      taskIdMapsForShuffle.computeIfAbsent(handle.shuffleId, _ => new OpenHashSet[Long](16))
    mapTaskIds.synchronized {
      mapTaskIds.add(context.taskAttemptId())
    }
    val env = SparkEnv.get
    val writer: ShuffleWriter[K, V] = handle match {
      case columnarShuffleHandle: ColumnarShuffleHandle[K @unchecked, V @unchecked] =>
        new GlutenDisaggregatedShuffleWriter[K, V](
          shuffleBlockResolver,
          columnarShuffleHandle,
          mapId,
          metrics,
          shuffleExecutorComponents)
      case unsafeShuffleHandle: SerializedShuffleHandle[K @unchecked, V @unchecked] =>
        new UnsafeShuffleWriter(
          env.blockManager,
          context.taskMemoryManager(),
          unsafeShuffleHandle,
          mapId,
          context,
          env.conf,
          metrics,
          shuffleExecutorComponents)
      case bypassMergeSortHandle: BypassMergeSortShuffleHandle[K @unchecked, V @unchecked] =>
        new BypassMergeSortShuffleWriter(
          env.blockManager,
          bypassMergeSortHandle,
          mapId,
          env.conf,
          metrics,
          shuffleExecutorComponents)
      case other: BaseShuffleHandle[K @unchecked, V @unchecked, _] =>
        GlutenShuffleUtils.getSortShuffleWriter(
          other,
          mapId,
          context,
          metrics,
          shuffleExecutorComponents)
    }
    new DisaggregatedShuffleWriter[K, V](writer)
  }

  override def getReader[K, C](
      handle: ShuffleHandle,
      startMapIndex: Int,
      endMapIndex: Int,
      startPartition: Int,
      endPartition: Int,
      context: TaskContext,
      metrics: ShuffleReadMetricsReporter): ShuffleReader[K, C] = {
    val (blocksByAddress, canEnableBatchFetch) = {
      GlutenShuffleUtils.getReaderParam(
        handle,
        startMapIndex,
        endMapIndex,
        startPartition,
        endPartition)
    }
    val shouldBatchFetch =
      canEnableBatchFetch && canUseBatchFetch(startPartition, endPartition, context)
    val serializerManager = if (handle.isInstanceOf[ColumnarShuffleHandle[_, _]]) {
      bypassDecompressionSerializerManger
    } else {
      SparkEnv.get.serializerManager
    }

    if (dispatcher.useSparkShuffleFetch) {
      new BlockStoreShuffleReader(
        handle.asInstanceOf[BaseShuffleHandle[K, _, C]],
        blocksByAddress,
        context,
        metrics,
        serializerManager = ColumnarShuffleManager.bypassDecompressionSerializerManger,
        shouldBatchFetch = shouldBatchFetch
      )
    } else {
      new DisaggregatedShuffleReader(
        dispatcher,
        conf,
        handle.asInstanceOf[BaseShuffleHandle[K, _, C]],
        context,
        metrics,
        startMapIndex,
        endMapIndex,
        startPartition,
        endPartition,
        serializerManager = serializerManager,
        shouldBatchFetch = shouldBatchFetch
      )
    }
  }

  /** Remove a shuffle's metadata from the ShuffleManager. */
  override def unregisterShuffle(shuffleId: Int): Boolean = {
    logInfo(f"Unregister shuffle $shuffleId")

    Option(taskIdMapsForShuffle.remove(shuffleId)).foreach {
      mapTaskIds =>
        mapTaskIds.iterator.foreach {
          mapId => shuffleBlockResolver.removeDataByMap(shuffleId, mapId)
        }
    }

    dispatcher.purgeCachedDataForShuffle(shuffleId)
    if (isDriver && dispatcher.cleanupShuffleFiles) {
      logInfo(s"Removing shuffle files for $shuffleId.")
      dispatcher.removeShuffle(shuffleId)
    }
    true
  }

  /** Shut down this ShuffleManager. */
  override def stop(): Unit = {
    shuffleBlockResolver.stop()

    dispatcher.purgeCachedData()
    if (isDriver && dispatcher.cleanupShuffleFiles) {
      dispatcher.removeRoot()
    }
  }
}

object GlutenDisaggregatedShuffleManager {
  private def loadShuffleExecutorComponents(
      conf: SparkConf
  ): ShuffleExecutorComponents = {
    if (
      !conf.contains(SHUFFLE_IO_PLUGIN_CLASS) ||
      conf.get(SHUFFLE_IO_PLUGIN_CLASS) != "org.apache.spark.shuffle.DisaggregatedShuffleDataIO"
    ) {
      throw new GlutenException(
        s"${SHUFFLE_IO_PLUGIN_CLASS.key} needs to be set to " +
          "\"org.apache.spark.shuffle.DisaggregatedShuffleDataIO\" " +
          "in order for this plugin to work!")
    }
    val executorComponents =
      ShuffleDataIOUtils.loadShuffleDataIO(conf).executor()
    val extraConfigs =
      conf.getAllWithPrefix(ShuffleDataIOUtils.SHUFFLE_SPARK_CONF_PREFIX).toMap
    // Workaround for testing
    val executorId = if (SparkEnv.get == null) "99" else SparkEnv.get.executorId
    executorComponents.initializeExecutor(
      conf.getAppId,
      executorId,
      extraConfigs.asJava
    )
    executorComponents
  }

  def bypassDecompressionSerializerManger: SerializerManager =
    new SerializerManager(
      SparkEnv.get.serializer,
      SparkEnv.get.conf,
      SparkEnv.get.securityManager.getIOEncryptionKey()) {
      // Bypass the shuffle read decompression, decryption is not supported
      override def wrapStream(blockId: BlockId, s: InputStream): InputStream = {
        if (s.isInstanceOf[DisaggregatedBufferedInputStreamAdaptor]) {
          // Suppress double close warning in DisaggregatedBufferedInputStreamAdaptor.
          new InputStream {
            private var closed = false

            override def read(): Int = s.read()

            override def read(b: Array[Byte], off: Int, len: Int): Int = {
              s.read(b, off, len)
            }

            override def skip(n: Long): Long = s.skip(n)

            override def available(): Int = s.available()

            override def close(): Unit = {
              if (!closed) {
                closed = true
                s.close()
              }
            }
          }
        } else {
          s
        }
      }
    }
}
