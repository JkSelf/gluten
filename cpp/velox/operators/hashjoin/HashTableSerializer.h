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

#pragma once

#include <cstdint>
#include <memory>
#include "velox/exec/HashTable.h"

namespace gluten {

/**
 * HashTableSerializer provides serialization and deserialization for Velox hash tables.
 * This is a thin wrapper around the HashTable's native serialize/deserialize methods
 * from IBM Velox's verified implementation.
 */
class HashTableSerializer {
 public:
  /**
   * Serialized hash table data structure.
   * Contains the serialized bytes that can be transmitted or stored.
   */
  struct SerializedHashTable {
    std::unique_ptr<uint8_t[]> data; // Serialized data buffer
    size_t size; // Total size in bytes
    bool ignoreNullKeys; // ignoreNullKeys used when building the hash table
    bool joinHasNullKeys; // Whether the build side has null keys (for null-aware anti join)
    int64_t bloomFilterBlocksByteSize; // Total size of bloom filter blocks in bytes

    SerializedHashTable() : size(0), ignoreNullKeys(false), joinHasNullKeys(false), bloomFilterBlocksByteSize(0) {}
  };

  /**
   * Serialize a hash table to a contiguous memory buffer.
   * Directly uses HashTable's serialize() method from IBM Velox.
   *
   * @param hashTable The hash table to serialize (must be a join build table)
   * @return Serialized hash table data
   */
  template <bool ignoreNullKeys>
  static SerializedHashTable serialize(const facebook::velox::exec::HashTable<ignoreNullKeys>* hashTable);

  /**
   * Deserialize a hash table from a memory buffer.
   * Directly uses HashTable's deserialize() method from IBM Velox.
   *
   * @param data Pointer to serialized data
   * @param size Size of serialized data
   * @param pool Memory pool for allocations
   * @return Deserialized hash table
   */
  template <bool ignoreNullKeys>
  static std::unique_ptr<facebook::velox::exec::HashTable<ignoreNullKeys>>
  deserialize(const uint8_t* data, size_t size, facebook::velox::memory::MemoryPool* pool);
};

} // namespace gluten
