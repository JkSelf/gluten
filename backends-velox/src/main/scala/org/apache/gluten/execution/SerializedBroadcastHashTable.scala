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

import org.apache.gluten.vectorized.HashJoinBuilder

import org.apache.spark.sql.execution.joins.BuildSideRelation

import java.io.{Externalizable, ObjectInput, ObjectOutput}

/**
 * Serialized broadcast hash table that can be efficiently broadcast to executors. This is built on
 * the driver and contains the serialized hash table data.
 */
case class SerializedBroadcastHashTable(
    serializedData: Array[Byte],
    numRows: Long,
    ignoreNullKeys: Boolean,
    joinHasNullKeys: Boolean,
    bloomFilterBlocksByteSize: Long,
    hashProbeDynamicFiltersProduced: Long,
    buildSideRelation: BuildSideRelation)
  extends Externalizable {

  def this() = this(null, 0, false, false, 0, 0, null) // Required for Externalizable

  override def writeExternal(out: ObjectOutput): Unit = {
    out.writeLong(numRows)
    out.writeBoolean(ignoreNullKeys)
    out.writeBoolean(joinHasNullKeys)
    out.writeLong(bloomFilterBlocksByteSize)
    out.writeLong(hashProbeDynamicFiltersProduced)
    out.writeInt(serializedData.length)
    out.write(serializedData)
    out.writeObject(buildSideRelation)
  }

  override def readExternal(in: ObjectInput): Unit = {
    val numRows = in.readLong()
    val ignoreNullKeys = in.readBoolean()
    val joinHasNullKeys = in.readBoolean()
    val bloomFilterBlocksByteSize = in.readLong()
    val hashProbeDynamicFiltersProduced = in.readLong()
    val dataLength = in.readInt()
    val data = new Array[Byte](dataLength)
    in.readFully(data)
    val relation = in.readObject().asInstanceOf[BuildSideRelation]

    // Use reflection to set final fields
    val numRowsField = classOf[SerializedBroadcastHashTable].getDeclaredField("numRows")
    numRowsField.setAccessible(true)
    numRowsField.set(this, numRows)

    val dataField = classOf[SerializedBroadcastHashTable].getDeclaredField("serializedData")
    dataField.setAccessible(true)
    dataField.set(this, data)

    val relationField = classOf[SerializedBroadcastHashTable].getDeclaredField("buildSideRelation")
    relationField.setAccessible(true)
    relationField.set(this, relation)

    val ignoreNullKeysField =
      classOf[SerializedBroadcastHashTable].getDeclaredField("ignoreNullKeys")
    ignoreNullKeysField.setAccessible(true)
    ignoreNullKeysField.set(this, ignoreNullKeys)

    val joinHasNullKeysField =
      classOf[SerializedBroadcastHashTable].getDeclaredField("joinHasNullKeys")
    joinHasNullKeysField.setAccessible(true)
    joinHasNullKeysField.set(this, joinHasNullKeys)

    val bloomFilterBlocksByteSizeField =
      classOf[SerializedBroadcastHashTable].getDeclaredField("bloomFilterBlocksByteSize")
    bloomFilterBlocksByteSizeField.setAccessible(true)
    bloomFilterBlocksByteSizeField.set(this, bloomFilterBlocksByteSize)

    val hashProbeDynamicFiltersProducedField =
      classOf[SerializedBroadcastHashTable].getDeclaredField("hashProbeDynamicFiltersProduced")
    hashProbeDynamicFiltersProducedField.setAccessible(true)
    hashProbeDynamicFiltersProducedField.set(this, hashProbeDynamicFiltersProduced)
  }

  /**
   * Deserialize the hash table on executor side. The serialized Velox hash table is already in a
   * prepared, probe-ready form, so executor side only needs deserialization without re-running
   * prepareJoinTable.
   *
   * @return
   *   Hash table builder handle
   */
  def deserialize(): Long = {
    HashJoinBuilder.deserializeHashTableWithIgnoreNullKeys(
      serializedData,
      ignoreNullKeys,
      joinHasNullKeys)
  }

  /** Get the size of serialized data in bytes. */
  def sizeInBytes: Long = serializedData.length.toLong
}

object SerializedBroadcastHashTable {

  /**
   * Build and serialize a hash table on the driver.
   *
   * @param hashTableHandle
   *   Handle to the built hash table
   * @param buildSideRelation
   *   The build side relation for metadata
   * @return
   *   Serialized broadcast hash table
   */
  def fromHashTable(
      hashTableHandle: Long,
      buildSideRelation: BuildSideRelation): SerializedBroadcastHashTable = {

    // Serialize the hash table
    val serializedHandle = HashJoinBuilder.serializeHashTable(hashTableHandle)

    try {
      // Get serialized data
      val serializedData = HashJoinBuilder
        .getSerializedData(serializedHandle)
      val numRows = HashJoinBuilder
        .getSerializedSize(serializedHandle)
      val ignoreNullKeys = HashJoinBuilder
        .getSerializedIgnoreNullKeys(serializedHandle)
      val joinHasNullKeys = HashJoinBuilder
        .getSerializedJoinHasNullKeys(serializedHandle)

      // Get bloom filter metrics
      val bloomFilterBlocksByteSize = HashJoinBuilder
        .getBloomFilterBlocksByteSize(serializedHandle)
      val hashProbeDynamicFiltersProduced = if (bloomFilterBlocksByteSize > 0) 1L else 0L

      SerializedBroadcastHashTable(
        serializedData,
        numRows,
        ignoreNullKeys,
        joinHasNullKeys,
        bloomFilterBlocksByteSize,
        hashProbeDynamicFiltersProduced,
        buildSideRelation)
    } finally {
      // Clean up serialized handle
      HashJoinBuilder.releaseSerializedData(serializedHandle)
      synchronized {
        // Clean up original hash table
        HashJoinBuilder.clearHashTable(hashTableHandle)
      }
    }
  }
}
