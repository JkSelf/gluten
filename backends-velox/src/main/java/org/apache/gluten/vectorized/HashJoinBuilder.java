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
package org.apache.gluten.vectorized;

import org.apache.gluten.runtime.Runtime;
import org.apache.gluten.runtime.RuntimeAware;

public class HashJoinBuilder implements RuntimeAware {
  private final Runtime runtime;

  private HashJoinBuilder(Runtime runtime) {
    this.runtime = runtime;
  }

  public static HashJoinBuilder create(Runtime runtime) {
    return new HashJoinBuilder(runtime);
  }

  @Override
  public long rtHandle() {
    return runtime.getHandle();
  }

  public static native void clearHashTable(long hashTableData);

  public static native long cloneHashTable(long hashTableData);

  /**
   * Serialize a hash table for broadcasting.
   *
   * @param hashTableHandle Handle to the hash table builder
   * @return Handle to the serialized hash table data
   */
  public static native long serializeHashTable(long hashTableHandle);

  /**
   * Deserialize a hash table from broadcast data with explicit ignoreNullKeys parameter.
   *
   * @param serializedData Byte array containing serialized hash table
   * @param ignoreNullKeys Whether to ignore null keys (must match the serialized hash table)
   * @param joinHasNullKeys Whether the build side has null keys (for null-aware anti join)
   * @return Handle to the deserialized hash table builder
   */
  public static native long deserializeHashTableWithIgnoreNullKeys(
      byte[] serializedData, boolean ignoreNullKeys, boolean joinHasNullKeys);

  /**
   * Get the size of serialized hash table data.
   *
   * @param serializedHandle Handle to serialized data
   * @return Size in bytes
   */
  public static native long getSerializedSize(long serializedHandle);

  /**
   * Get ignoreNullKeys parameter from serialized hash table metadata.
   *
   * @param serializedHandle Handle to serialized data
   * @return ignoreNullKeys flag used when building the hash table
   */
  public static native boolean getSerializedIgnoreNullKeys(long serializedHandle);

  /**
   * Get joinHasNullKeys parameter from serialized hash table metadata.
   *
   * @param serializedHandle Handle to serialized data
   * @return joinHasNullKeys flag indicating if build side has null keys
   */
  public static native boolean getSerializedJoinHasNullKeys(long serializedHandle);

  /**
   * Get bloom filter blocks byte size from serialized hash table metadata.
   *
   * @param serializedHandle Handle to serialized data
   * @return bloom filter blocks byte size
   */
  public static native long getBloomFilterBlocksByteSize(long serializedHandle);

  /**
   * Get serialized hash table data as byte array.
   *
   * @param serializedHandle Handle to serialized data
   * @return Byte array containing serialized data
   */
  public static native byte[] getSerializedData(long serializedHandle);

  /**
   * Release serialized hash table data.
   *
   * @param serializedHandle Handle to serialized data
   */
  public static native void releaseSerializedData(long serializedHandle);

  public native long nativeBuild(
      String buildHashTableId,
      long[] batchHandlers,
      String[] joinKeys,
      String[] filterBuildColumns,
      boolean filterPropagatesNulls,
      int joinType,
      boolean hasMixedFiltCondition,
      boolean isExistenceJoin,
      byte[] namedStruct,
      boolean isNullAwareAntiJoin,
      long bloomFilterPushdownSize,
      int broadcastHashTableBuildThreads);
}
