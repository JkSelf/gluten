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

#include "operators/hashjoin/HashTableSerializer.h"
#include <cstring>
#include <sstream>
#include "velox/common/base/Exceptions.h"

namespace gluten {

template <bool ignoreNullKeys>
HashTableSerializer::SerializedHashTable HashTableSerializer::serialize(
    const facebook::velox::exec::HashTable<ignoreNullKeys>* hashTable) {
  VELOX_CHECK_NOT_NULL(hashTable, "Hash table cannot be null");

  std::ostringstream oss(std::ios::binary);

  hashTable->serialize(oss);

  SerializedHashTable result;
  std::string str = oss.str();
  result.size = str.size();
  result.data = std::make_unique<uint8_t[]>(result.size);
  std::memcpy(result.data.get(), str.data(), result.size);

  return result;
}

template <bool ignoreNullKeys>
std::unique_ptr<facebook::velox::exec::HashTable<ignoreNullKeys>>
HashTableSerializer::deserialize(const uint8_t* data, size_t size, facebook::velox::memory::MemoryPool* pool) {
  VELOX_CHECK_NOT_NULL(data, "Serialized data cannot be null");
  VELOX_CHECK_GT(size, 0, "Invalid serialized data size");
  VELOX_CHECK_NOT_NULL(pool, "Memory pool cannot be null");

  std::string str(reinterpret_cast<const char*>(data), size);
  std::istringstream iss(str, std::ios::binary);

  return facebook::velox::exec::HashTable<ignoreNullKeys>::deserialize(iss, pool);
}

template HashTableSerializer::SerializedHashTable HashTableSerializer::serialize<true>(
    const facebook::velox::exec::HashTable<true>*);

template HashTableSerializer::SerializedHashTable HashTableSerializer::serialize<false>(
    const facebook::velox::exec::HashTable<false>*);

template std::unique_ptr<facebook::velox::exec::HashTable<true>>
HashTableSerializer::deserialize<true>(const uint8_t*, size_t, facebook::velox::memory::MemoryPool*);

template std::unique_ptr<facebook::velox::exec::HashTable<false>>
HashTableSerializer::deserialize<false>(const uint8_t*, size_t, facebook::velox::memory::MemoryPool*);

} // namespace gluten
