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
package org.apache.gluten.execution

import org.apache.gluten.backendsapi.BackendsApiManager
import org.apache.gluten.backendsapi.velox.VeloxBackendSettings
import org.apache.gluten.vectorized.HashJoinBuilder

import org.apache.spark.SparkEnv
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution.ColumnarBuildSideRelation
import org.apache.spark.sql.execution.joins.BuildSideRelation
import org.apache.spark.sql.execution.unsafe.UnsafeColumnarBuildSideRelation

import com.github.benmanes.caffeine.cache.{Cache, Caffeine, RemovalCause, RemovalListener}

import java.util.concurrent.TimeUnit

case class BroadcastHashTable(pointer: Long, relation: BuildSideRelation)

/**
 * `VeloxBroadcastBuildSideCache` is used for controlling to build bhj hash table once.
 *
 * The complicated part is due to reuse exchange, where multiple BHJ IDs correspond to a
 * `BuildSideRelation`.
 *
 * This implementation supports two modes:
 *   1. Driver-side build (new): Hash table is built and serialized on driver, then broadcast 2.
 *      Executor-side build (legacy): Each executor builds its own hash table from broadcast data
 */
object VeloxBroadcastBuildSideCache
  extends Logging
  with RemovalListener[String, BroadcastHashTable] {

  private lazy val expiredTime = SparkEnv.get.conf.getLong(
    VeloxBackendSettings.GLUTEN_VELOX_BROADCAST_CACHE_EXPIRED_TIME,
    VeloxBackendSettings.GLUTEN_VELOX_BROADCAST_CACHE_EXPIRED_TIME_DEFAULT
  )

  // Use for controlling to build bhj hash table once.
  // key: hashtable id, value is hashtable backend pointer(long to string).
  private val buildSideRelationCache: Cache[String, BroadcastHashTable] =
    Caffeine.newBuilder
      .expireAfterAccess(expiredTime, TimeUnit.SECONDS)
      .removalListener(this)
      .build[String, BroadcastHashTable]()

  // Cache for driver-side serialized hash tables to avoid rebuilding for reuse exchange
  private val driverSerializedCache: Cache[String, SerializedBroadcastHashTable] =
    Caffeine.newBuilder
      .expireAfterAccess(expiredTime, TimeUnit.SECONDS)
      .build[String, SerializedBroadcastHashTable]()

  def getOrBuildBroadcastHashTable(
      broadcast: Broadcast[BuildSideRelation],
      broadcastContext: BroadcastHashJoinContext): BroadcastHashTable = {

    buildSideRelationCache
      .get(
        broadcastContext.buildHashTableId,
        (broadcast_id: String) => {
          val (pointer, relation) = broadcast.value match {
            case columnar: ColumnarBuildSideRelation =>
              columnar.buildHashTable(broadcastContext)
            case unsafe: UnsafeColumnarBuildSideRelation =>
              unsafe.buildHashTable(broadcastContext)
          }

          BroadcastHashTable(pointer, relation)
        }
      )
  }

  /**
   * Build hash table on driver and serialize for broadcasting. This version is called from
   * BroadcastExchangeExec and doesn't need a broadcast variable.
   *
   * This is the Spark-native approach where hash table is built in BroadcastExchangeExec.
   */
  def buildAndSerializeOnDriverInBroadcastExchange(
      relation: BuildSideRelation,
      broadcastContext: BroadcastHashJoinContext): SerializedBroadcastHashTable = {

    val broadcastId = broadcastContext.buildHashTableId

    val cached = driverSerializedCache.getIfPresent(broadcastId)
    if (cached != null) {
      logInfo(s"Reusing cached serialized hash table for broadcast ID: $broadcastId")
      return cached
    }

    def resetRelation(): Unit = relation match {
      case r: ColumnarBuildSideRelation => r.reset()
      case r: UnsafeColumnarBuildSideRelation => r.reset()
      case _ =>
    }

    relation.synchronized {
      val cachedAfterLock = driverSerializedCache.getIfPresent(broadcastId)
      if (cachedAfterLock != null) {
        logInfo(s"Reusing cached serialized hash table for broadcast ID: $broadcastId (after lock)")
        return cachedAfterLock
      }

      logInfo(
        s"Building hash table on driver in BroadcastExchangeExec " +
          s"for broadcast ID: $broadcastId")

      val backendName = BackendsApiManager.getBackendName

      val runtime = org.apache.gluten.runtime.Runtime.createStandalone(
        backendName,
        "DriverBroadcastHashTableBuild"
      )

      try {
        resetRelation()
        val startBuildTime = System.currentTimeMillis()
        val (hashTableHandle, _) = relation match {
          case r: ColumnarBuildSideRelation =>
            r.buildHashTableWithRuntime(broadcastContext, runtime)
          case r: UnsafeColumnarBuildSideRelation =>
            r.buildHashTableWithRuntime(broadcastContext, runtime)
          case other =>
            throw new IllegalArgumentException(
              s"Unsupported relation type for driver-side build: ${other.getClass.getName}")
        }
        try {
          val buildTimeMs = System.currentTimeMillis() - startBuildTime
          broadcastContext.buildHashTableTimeMetric.foreach(_ += buildTimeMs)
          val startSerializeTime = System.currentTimeMillis()
          val result = SerializedBroadcastHashTable.fromHashTable(hashTableHandle, relation)
          val serializeTimeMs = System.currentTimeMillis() - startSerializeTime

          logInfo(
            s"Built and serialized hash table on driver: " +
              s"size=${result.sizeInBytes} bytes, " +
              s"rows=${result.numRows}, " +
              s"serializeTime=${serializeTimeMs}ms " +
              s"for broadcast ID: $broadcastId")

          broadcastContext.serializeHashTableTimeMetric.foreach(_ += serializeTimeMs)
          broadcastContext.serializedHashTableSizeMetric.foreach(_ += result.sizeInBytes)

          driverSerializedCache.put(broadcastId, result)
          result
        } finally {
          resetRelation()
        }
      } finally {
        runtime.close()
      }
    }
  }

  /** Deserialize hash table on executor from broadcast data. */
  def deserializeOnExecutor(
      serialized: SerializedBroadcastHashTable,
      broadcastHashTableId: String,
      deserializeHashTableTimeMetric: Option[org.apache.spark.sql.execution.metric.SQLMetric] =
        None): BroadcastHashTable = {

    buildSideRelationCache.get(
      broadcastHashTableId,
      (_: String) => {
        logInfo(s"Deserializing hash table on executor for broadcast ID: $broadcastHashTableId")
        val startTime = System.currentTimeMillis()
        val hashTableHandle = serialized.deserialize()
        val timeMs = System.currentTimeMillis() - startTime
        deserializeHashTableTimeMetric.foreach(_ += timeMs)
        BroadcastHashTable(hashTableHandle, serialized.buildSideRelation)
      }
    )
  }

  /** This is callback from c++ backend. */
  def get(broadcastHashtableId: String): Long = {
    Option(buildSideRelationCache.getIfPresent(broadcastHashtableId))
      .map(_.pointer)
      .getOrElse(0)
  }

  def invalidateBroadcastHashtable(broadcastHashtableId: String): Unit = {
    // Cleanup operations on the backend are idempotent.
    buildSideRelationCache.invalidate(broadcastHashtableId)
  }

  /** Only used in UT. */
  def size(): Long = buildSideRelationCache.estimatedSize()

  def cleanAll(): Unit = {
    buildSideRelationCache.invalidateAll()
    driverSerializedCache.invalidateAll()
  }

  override def onRemoval(key: String, value: BroadcastHashTable, cause: RemovalCause): Unit = {
    synchronized {
      if (value.relation != null) {
        value.relation match {
          case columnar: ColumnarBuildSideRelation =>
            columnar.reset()
          case unsafe: UnsafeColumnarBuildSideRelation =>
            unsafe.reset()
        }
      }

      HashJoinBuilder.clearHashTable(value.pointer)
    }
  }
}
