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

#include "VeloxMemoryPool.h"
#include <sstream>
#include "compute/Backend.h"
#include "compute/VeloxBackend.h"
#include "compute/VeloxInitializer.h"
#include "velox/common/memory/MemoryAllocator.h"

using namespace facebook;

namespace gluten {
#define VELOX_MEM_POOL_CAP_EXCEEDED(errorMessage)                   \
  _VELOX_THROW(                                                     \
      ::facebook::velox::VeloxRuntimeError,                         \
      ::facebook::velox::error_source::kErrorSourceRuntime.c_str(), \
      ::facebook::velox::error_code::kMemCapExceeded.c_str(),       \
      /* isRetriable */ true,                                       \
      "{}",                                                         \
      (errorMessage));

// Check if memory operation is allowed and increment the named stats.
#define CHECK_AND_INC_MEM_OP_STATS(stats)                                                 \
  do {                                                                                    \
    if (FOLLY_UNLIKELY(kind_ != Kind::kLeaf)) {                                           \
      VELOX_FAIL("Memory operation is only allowed on leaf memory pool: {}", toString()); \
    }                                                                                     \
    ++num##stats##_;                                                                      \
  } while (0)

// Check if memory operation is allowed and increment the named stats.
#define INC_MEM_OP_STATS(stats) ++num##stats##_;

// Check if a memory pool management operation is allowed.
#define CHECK_POOL_MANAGEMENT_OP(opName)                                                                          \
  do {                                                                                                            \
    if (FOLLY_UNLIKELY(kind_ != Kind::kAggregate)) {                                                              \
      VELOX_FAIL("Memory pool {} operation is only allowed on aggregation memory pool: {}", #opName, toString()); \
    }                                                                                                             \
  } while (0)

// Collect the memory usage from memory pool for memory capacity exceeded error
// message generation.
struct MemoryUsage {
  std::string name;
  uint64_t currentUsage;
  uint64_t peakUsage;

  bool operator>(const MemoryUsage& other) const {
    return std::tie(currentUsage, peakUsage, name) > std::tie(other.currentUsage, other.peakUsage, other.name);
  }

  std::string toString() const {
    return fmt::format(
        "{} usage {} peak {}", name, velox::succinctBytes(currentUsage), velox::succinctBytes(peakUsage));
  }
};

struct MemoryUsageComp {
  bool operator()(const MemoryUsage& lhs, const MemoryUsage& rhs) const {
    return lhs > rhs;
  }
};
using MemoryUsageHeap = std::priority_queue<MemoryUsage, std::vector<MemoryUsage>, MemoryUsageComp>;

static constexpr size_t kCapMessageIndentSize = 4;

std::vector<MemoryUsage> sortMemoryUsages(MemoryUsageHeap& heap) {
  std::vector<MemoryUsage> usages;
  usages.reserve(heap.size());
  while (!heap.empty()) {
    usages.push_back(heap.top());
    heap.pop();
  }
  std::reverse(usages.begin(), usages.end());
  return usages;
}

// Invoked by visitChildren() to traverse the memory pool structure to build the
// memory capacity exceeded exception error message.
void capExceedingMessageVisitor(
    velox::memory::MemoryPool* pool,
    size_t indent,
    MemoryUsageHeap& topLeafMemUsages,
    std::stringstream& out) {
  const velox::memory::MemoryPool::Stats stats = pool->stats();
  // Avoid logging empty pools.
  if (stats.empty()) {
    return;
  }
  const MemoryUsage usage{
      .name = pool->name(),
      .currentUsage = stats.currentBytes,
      .peakUsage = stats.peakBytes,
  };
  out << std::string(indent, ' ') << usage.toString() << "\n";

  if (pool->kind() == velox::memory::MemoryPool::Kind::kLeaf) {
    static const size_t kTopNLeafMessages = 10;
    topLeafMemUsages.push(usage);
    if (topLeafMemUsages.size() > kTopNLeafMessages) {
      topLeafMemUsages.pop();
    }
    return;
  }
  pool->visitChildren([&, indent = indent + kCapMessageIndentSize](velox::memory::MemoryPool* pool) {
    capExceedingMessageVisitor(pool, indent, topLeafMemUsages, out);
    return true;
  });
}

std::string capacityToString(int64_t capacity) {
  return capacity == facebook::velox::memory::kMaxMemory ? "UNLIMITED" : velox::succinctBytes(capacity);
}

//  The code is originated from /velox/common/memory/Memory.h
//  Removed memory manager.
class WrappedVeloxMemoryPool final : public velox::memory::MemoryPool {
 public:
  using DestructionCallback = std::function<void(MemoryPool*)>;

  // Should perhaps make this method private so that we only create node through
  // parent.
  WrappedVeloxMemoryPool(
      velox::memory::MemoryAllocator* veloxAlloc,
      const std::string& name,
      Kind kind,
      std::shared_ptr<velox::memory::MemoryPool> parent,
      gluten::MemoryAllocator* glutenAlloc,
      int64_t spillThreshold,
      std::unique_ptr<velox::memory::MemoryReclaimer> reclaimer = nullptr,
      DestructionCallback destructionCb = nullptr,
      const Options& options = Options{})
      : velox::memory::MemoryPool{name, kind, parent, std::move(reclaimer), options},
        manager_{memoryManager},
        veloxAlloc_{&manager_->allocator()},
        glutenAlloc_{glutenAlloc},
        destructionCb_(std::move(destructionCb)),
        capacity_(parent_ == nullptr ? options.capacity : facebook::velox::memory::kMaxMemory) {
    VELOX_CHECK(options.threadSafe || isLeaf());
    if (parent == nullptr) {
      if (options.trackUsage) {
        this->setHighUsageCallback([=](velox::memory::MemoryPool& t) {
          if (t.reservedBytes() >= spillThreshold) {
            return true;
          }
          return false;
        });
      }
    }
  }

  ~WrappedVeloxMemoryPool() {
    if (checkUsageLeak_) {
      VELOX_CHECK(
          (usedReservationBytes_ == 0) && (reservationBytes_ == 0) && (minReservationBytes_ == 0),
          "Bad memory usage track state: {}",
          toString());
    }
    if (destructionCb_ != nullptr) {
      destructionCb_(this);
    }
  }

  void* allocate(int64_t size) {
    CHECK_AND_INC_MEM_OP_STATS(Allocs);
    const auto alignedSize = sizeAlign(size);
    reserve(alignedSize);
    void* buffer;
    bool succeed = glutenAlloc_->allocate(alignedSize, &buffer);
    VELOX_CHECK(succeed);
    if (FOLLY_UNLIKELY(buffer == nullptr)) {
      release(alignedSize);
      VELOX_MEM_ALLOC_ERROR(fmt::format("{} failed with {} bytes from {}", __FUNCTION__, size, toString()));
    }
    return buffer;
  }

  void* allocateZeroFilled(int64_t numEntries, int64_t sizeEach) {
    CHECK_AND_INC_MEM_OP_STATS(Allocs);
    const auto alignedSize = sizeAlign(sizeEach * numEntries);
    reserve(alignedSize);
    void* buffer;
    bool succeed = glutenAlloc_->allocateZeroFilled(alignedSize, 1, &buffer);
    VELOX_CHECK(succeed);
    if (FOLLY_UNLIKELY(buffer == nullptr)) {
      release(alignedSize);
      VELOX_MEM_ALLOC_ERROR(fmt::format(
          "{} failed with {} entries and {} bytes each from {}", __FUNCTION__, numEntries, sizeEach, toString()));
    }
    return buffer;
  }

  void* reallocate(void* p, int64_t size, int64_t newSize) {
    CHECK_AND_INC_MEM_OP_STATS(Allocs);
    const auto alignedSize = sizeAlign(size);
    const auto alignedNewSize = sizeAlign(newSize);
    reserve(alignedNewSize);
    void* newP;
    bool succeed = glutenAlloc_->allocate(alignedNewSize, &newP);
    VELOX_CHECK(succeed);
    if (FOLLY_UNLIKELY(newP == nullptr)) {
      release(alignedNewSize);
      VELOX_MEM_ALLOC_ERROR(
          fmt::format("{} failed with {} new bytes and {} old bytes from {}", __FUNCTION__, newSize, size, toString()));
    }
    if (p != nullptr) {
      ::memcpy(newP, p, std::min(size, newSize));
      free(p, alignedSize);
    }

    return newP;
  }

  void free(void* p, int64_t size) {
    CHECK_AND_INC_MEM_OP_STATS(Frees);
    const auto alignedSize = sizeAlign(size);
    if (!glutenAlloc_->free(p, alignedSize)) {
      VELOX_FAIL(fmt::format("WrappedVeloxMemoryPool: Failed to free {} bytes", alignedSize))
    }
    release(alignedSize);
  }

  void allocateNonContiguous(
      velox::memory::MachinePageCount numPages,
      velox::memory::Allocation& out,
      velox::memory::MachinePageCount minSizeClass) {
    CHECK_AND_INC_MEM_OP_STATS(Allocs);
    if (!out.empty()) {
      INC_MEM_OP_STATS(Frees);
    }
    VELOX_CHECK_GT(numPages, 0);
    if (!veloxAlloc_->allocateNonContiguous(
            numPages,
            out,
            [this](int64_t allocBytes, bool preAllocate) {
              bool succeed =
                  preAllocate ? glutenAlloc_->reserveBytes(allocBytes) : glutenAlloc_->unreserveBytes(allocBytes);
              VELOX_CHECK(succeed)
              if (preAllocate) {
                reserve(allocBytes);
              } else {
                release(allocBytes);
              }
            },
            minSizeClass)) {
      VELOX_CHECK(out.empty());
      VELOX_MEM_ALLOC_ERROR(fmt::format("{} failed with {} pages from {}", __FUNCTION__, numPages, toString()));
    }
    VELOX_CHECK(!out.empty());
    VELOX_CHECK_NULL(out.pool());
    out.setPool(this);
  }

  void freeNonContiguous(velox::memory::Allocation& allocation) {
    CHECK_AND_INC_MEM_OP_STATS(Frees);
    const int64_t freedBytes = veloxAlloc_->freeNonContiguous(allocation);
    VELOX_CHECK(allocation.empty());
    release(freedBytes);
    VELOX_CHECK(glutenAlloc_->unreserveBytes(freedBytes));
  }

  velox::memory::MachinePageCount largestSizeClass() const {
    return veloxAlloc_->largestSizeClass();
  }

  const std::vector<velox::memory::MachinePageCount>& sizeClasses() const {
    return veloxAlloc_->sizeClasses();
  }

  void allocateContiguous(velox::memory::MachinePageCount numPages, velox::memory::ContiguousAllocation& out) {
    CHECK_AND_INC_MEM_OP_STATS(Allocs);
    if (!out.empty()) {
      INC_MEM_OP_STATS(Frees);
    }
    VELOX_CHECK_GT(numPages, 0);

    if (!veloxAlloc_->allocateContiguous(numPages, nullptr, out, [this](int64_t allocBytes, bool preAlloc) {
          bool succeed = preAlloc ? glutenAlloc_->reserveBytes(allocBytes) : glutenAlloc_->unreserveBytes(allocBytes);
          VELOX_CHECK(succeed)
          if (preAlloc) {
            reserve(allocBytes);
          } else {
            release(allocBytes);
          }
        })) {
      VELOX_CHECK(out.empty());
      VELOX_MEM_ALLOC_ERROR(fmt::format("{} failed with {} pages from {}", __FUNCTION__, numPages, toString()));
    }
    VELOX_CHECK(!out.empty());
    VELOX_CHECK_NULL(out.pool());
    out.setPool(this);
  }

  void freeContiguous(velox::memory::ContiguousAllocation& allocation) {
    CHECK_AND_INC_MEM_OP_STATS(Frees);
    const int64_t bytesToFree = allocation.size();
    veloxAlloc_->freeContiguous(allocation);
    VELOX_CHECK(allocation.empty());
    release(bytesToFree);
    VELOX_CHECK(glutenAlloc_->unreserveBytes(bytesToFree))
  }

  int64_t capacity() const {
    if (parent_ != nullptr) {
      return parent_->capacity();
    }
    std::lock_guard<std::mutex> l(mutex_);
    return capacity_;
  }

  bool highUsage() {
    if (parent_ != nullptr) {
      return parent_->highUsage();
    }

    if (highUsageCallback_ != nullptr) {
      return highUsageCallback_(*this);
    }

    return false;
  }

  int64_t currentBytes() const override {
    std::lock_guard<std::mutex> l(mutex_);
    return currentBytesLocked();
  }

  int64_t peakBytes() const override {
    std::lock_guard<std::mutex> l(mutex_);
    return peakBytes_;
  }

  int64_t availableReservation() const override {
    std::lock_guard<std::mutex> l(mutex_);
    return availableReservationLocked();
  }

  int64_t reservedBytes() const override {
    std::lock_guard<std::mutex> l(mutex_);
    return reservationBytes_;
  }

  void setGlutenAllocator(gluten::MemoryAllocator* allocator) {
    glutenAlloc_ = allocator;
  }

  bool maybeReserve(uint64_t increment) {
    CHECK_AND_INC_MEM_OP_STATS(Reserves);
    // TODO: make this a configurable memory pool option.
    constexpr int32_t kGrowthQuantum = 8 << 20;
    const auto reservationToAdd = velox::bits::roundUp(increment, kGrowthQuantum);
    try {
      reserve(reservationToAdd, true);
    } catch (const std::exception& e) {
      return false;
    }
    return true;
  }

  void release() {
    CHECK_AND_INC_MEM_OP_STATS(Releases);
    release(0, true);
  }

  uint64_t freeBytes() const {
    if (parent_ != nullptr) {
      return parent_->freeBytes();
    }
    std::lock_guard<std::mutex> l(mutex_);
    if (capacity_ == facebook::velox::memory::kMaxMemory) {
      return 0;
    }
    VELOX_CHECK_GE(capacity_, reservationBytes_);
    return capacity_ - reservationBytes_;
  }

  uint64_t shrink(uint64_t targetBytes) {
    if (parent_ != nullptr) {
      return parent_->shrink(targetBytes);
    }
    std::lock_guard<std::mutex> l(mutex_);
    // We don't expect to shrink a memory pool without capacity limit.
    VELOX_CHECK_NE(capacity_, facebook::velox::memory::kMaxMemory);
    uint64_t freeBytes = std::max<uint64_t>(0, capacity_ - reservationBytes_);
    if (targetBytes != 0) {
      freeBytes = std::min(targetBytes, freeBytes);
    }
    capacity_ -= freeBytes;
    return freeBytes;
  }

  uint64_t grow(uint64_t bytes) {
    if (parent_ != nullptr) {
      return parent_->grow(bytes);
    }
    std::lock_guard<std::mutex> l(mutex_);
    // We don't expect to grow a memory pool without capacity limit.
    VELOX_CHECK_NE(capacity_, facebook::velox::memory::kMaxMemory);
    capacity_ += bytes;
    VELOX_CHECK_GE(capacity_, bytes);
    return capacity_;
  }

  std::string toString() const override {
    std::lock_guard<std::mutex> l(mutex_);
    return toStringLocked();
  }

  velox::memory::MemoryPool::Stats stats() const {
    std::lock_guard<std::mutex> l(mutex_);
    return statsLocked();
  }

 private:
  FOLLY_ALWAYS_INLINE static WrappedVeloxMemoryPool* toImpl(MemoryPool* pool) {
    return static_cast<WrappedVeloxMemoryPool*>(pool);
  }

  FOLLY_ALWAYS_INLINE static WrappedVeloxMemoryPool* toImpl(const std::shared_ptr<MemoryPool>& pool) {
    return static_cast<WrappedVeloxMemoryPool*>(pool.get());
  }

  std::shared_ptr<MemoryPool> genChild(
      std::shared_ptr<MemoryPool> parent,
      const std::string& name,
      Kind kind,
      bool threadSafe,
      std::unique_ptr<velox::memory::MemoryReclaimer> reclaimer) {
    return std::make_shared<WrappedVeloxMemoryPool>(
        manager_,
        name,
        kind,
        parent,
        glutenAlloc_,
        -1,
        std::move(reclaimer),
        nullptr,
        Options{.alignment = alignment_, .trackUsage = trackUsage_, .threadSafe = threadSafe});
  }

  FOLLY_ALWAYS_INLINE int64_t capacityLocked() const {
    return parent_ != nullptr ? toImpl(parent_)->capacity_ : capacity_;
  }

  FOLLY_ALWAYS_INLINE int64_t currentBytesLocked() const {
    return isLeaf() ? usedReservationBytes_ : reservationBytes_;
  }

  FOLLY_ALWAYS_INLINE int64_t availableReservationLocked() const {
    return !isLeaf() ? 0 : std::max<int64_t>(0, reservationBytes_ - usedReservationBytes_);
  }

  FOLLY_ALWAYS_INLINE int64_t sizeAlign(int64_t size) {
    const auto remainder = size % alignment_;
    return (remainder == 0) ? size : (size + alignment_ - remainder);
  }

  FOLLY_ALWAYS_INLINE static int64_t roundedDelta(int64_t size, int64_t delta) {
    return quantizedSize(size + delta) - size;
  }

  void reserve(uint64_t size, bool reserveOnly = false) {
    if (FOLLY_LIKELY(trackUsage_)) {
      if (FOLLY_LIKELY(threadSafe_)) {
        reserveThreadSafe(size, reserveOnly);
      } else {
        reserveNonThreadSafe(size, reserveOnly);
      }
    }
    if (reserveOnly) {
      return;
    }
    if (FOLLY_UNLIKELY(!manager_->reserve(size))) {
      // NOTE: If we can make the reserve and release a single transaction we
      // would have more accurate aggregates in intermediate states. However, this
      // is low-pri because we can only have inflated aggregates, and be on the
      // more conservative side.
      release(size);
      VELOX_MEM_POOL_CAP_EXCEEDED(toImpl(root())->capExceedingMessage(
          this,
          fmt::format(
              "Exceeded memory manager cap of {} when requesting {}, memory pool cap is {}",
              capacityToString(manager_->capacity()),
              velox::succinctBytes(size),
              capacityToString(capacity()))));
    }
  }

  FOLLY_ALWAYS_INLINE void reserveNonThreadSafe(uint64_t size, bool reserveOnly = false) {
    VELOX_CHECK(isLeaf());

    int32_t numAttempts{0};
    for (;; ++numAttempts) {
      int64_t increment = reservationSizeLocked(size);
      if (FOLLY_LIKELY(increment == 0)) {
        if (FOLLY_UNLIKELY(reserveOnly)) {
          minReservationBytes_.store(reservationBytes_);
        } else {
          usedReservationBytes_ += size;
          cumulativeBytes_ += size;
          maybeUpdatePeakBytesLocked(usedReservationBytes_);
        }
        sanityCheckLocked();
        break;
      }
      incrementReservationNonThreadSafe(this, increment);
    }

    // NOTE: in case of concurrent reserve requests to the same root memory pool
    // from the other leaf memory pools, we might have to retry
    // incrementReservation(). This should happen rarely in production
    // as the leaf tracker does quantized memory reservation so that we don't
    // expect high concurrency at the root memory pool.
    if (FOLLY_UNLIKELY(numAttempts > 1)) {
      numCollisions_ += numAttempts - 1;
    }
  }

  void reserveThreadSafe(uint64_t size, bool reserveOnly = false) {
    VELOX_CHECK(isLeaf());

    int32_t numAttempts = 0;
    int64_t increment = 0;
    for (;; ++numAttempts) {
      {
        std::lock_guard<std::mutex> l(mutex_);
        increment = reservationSizeLocked(size);
        if (increment == 0) {
          if (reserveOnly) {
            minReservationBytes_.store(reservationBytes_);
          } else {
            usedReservationBytes_ += size;
            cumulativeBytes_ += size;
            maybeUpdatePeakBytesLocked(usedReservationBytes_);
          }
          sanityCheckLocked();
          break;
        }
      }
      try {
        incrementReservationThreadSafe(this, increment);
      } catch (const std::exception& e) {
        // When race with concurrent memory reservation free, we might end up with
        // unused reservation but no used reservation if a retry memory
        // reservation attempt run into memory capacity exceeded error.
        releaseThreadSafe(0, false);
        std::rethrow_exception(std::current_exception());
      }
    }

    // NOTE: in case of concurrent reserve and release requests, we might see
    // potential conflicts as the quantized memory release might free up extra
    // reservation bytes so reserve might go extra round to reserve more bytes.
    // This should happen rarely in production as the leaf tracker updates are
    // mostly single thread executed.
    if (numAttempts > 1) {
      numCollisions_ += numAttempts - 1;
    }
  }

  bool incrementReservationThreadSafe(MemoryPool* requestor, uint64_t size) {
    VELOX_CHECK(threadSafe_);
    VELOX_CHECK_GT(size, 0);

    // Propagate the increment to the root memory pool to check the capacity limit
    // first. If it exceeds the capacity and can't grow, the root memory pool will
    // throw an exception to fail the request.
    if (parent_ != nullptr) {
      if (!toImpl(parent_)->incrementReservationThreadSafe(requestor, size)) {
        return false;
      }
    }

    {
      std::lock_guard<std::mutex> l(mutex_);
      if (maybeIncrementReservationLocked(size)) {
        return true;
      }
    }
    VELOX_CHECK_NULL(parent_);

    if (manager_->growPool(requestor, size)) {
      // NOTE: the memory reservation might still fail even if the memory grow
      // callback succeeds. The reason is that we don't hold the root tracker's
      // mutex lock while running the grow callback. Therefore, there is a
      // possibility in theory that a concurrent memory reservation request
      // might steal away the increased memory capacity after the grow callback
      // finishes and before we increase the reservation. If it happens, we can
      // simply fall back to retry the memory reservation from the leaf memory
      // pool which should happen rarely.
      return maybeIncrementReservation(size);
    }

    VELOX_MEM_POOL_CAP_EXCEEDED(capExceedingMessage(
        requestor,
        fmt::format(
            "Exceeded memory pool cap of {} when requesting {}, memory manager cap is {}",
            capacityToString(capacity_),
            velox::succinctBytes(size),
            capacityToString(manager_->capacity()))));
  }

  FOLLY_ALWAYS_INLINE bool incrementReservationNonThreadSafe(MemoryPool* requestor, uint64_t size) {
    VELOX_CHECK_NOT_NULL(parent_);
    VELOX_CHECK(isLeaf());

    if (!toImpl(parent_)->incrementReservationThreadSafe(requestor, size)) {
      return false;
    }

    reservationBytes_ += size;
    return true;
  }

  FOLLY_ALWAYS_INLINE int64_t reservationSizeLocked(int64_t size) {
    const int64_t neededSize = size - (reservationBytes_ - usedReservationBytes_);
    if (neededSize <= 0) {
      return 0;
    }
    return roundedDelta(reservationBytes_, neededSize);
  }

  FOLLY_ALWAYS_INLINE void maybeUpdatePeakBytesLocked(int64_t newPeak) {
    peakBytes_ = std::max<int64_t>(peakBytes_, newPeak);
  }

  bool maybeIncrementReservation(uint64_t size) {
    std::lock_guard<std::mutex> l(mutex_);
    return maybeIncrementReservationLocked(size);
  }

  bool maybeIncrementReservationLocked(uint64_t size) {
    if (!isRoot() || (reservationBytes_ + size <= capacity_)) {
      reservationBytes_ += size;
      if (!isLeaf()) {
        cumulativeBytes_ += size;
        maybeUpdatePeakBytesLocked(reservationBytes_);
      }
      return true;
    }
    return false;
  }

  void release(uint64_t size, bool releaseOnly = false) {
    if (!releaseOnly) {
      manager_->release(size);
    }
    if (FOLLY_LIKELY(trackUsage_)) {
      if (FOLLY_LIKELY(threadSafe_)) {
        releaseThreadSafe(size, releaseOnly);
      } else {
        releaseNonThreadSafe(size, releaseOnly);
      }
    }
  }

  void releaseThreadSafe(uint64_t size, bool releaseOnly) {
    VELOX_CHECK(isLeaf());
    VELOX_DCHECK_NOT_NULL(parent_);

    int64_t freeable = 0;
    {
      std::lock_guard<std::mutex> l(mutex_);
      int64_t newQuantized;
      if (FOLLY_UNLIKELY(releaseOnly)) {
        VELOX_DCHECK_EQ(size, 0);
        if (minReservationBytes_ == 0) {
          return;
        }
        newQuantized = quantizedSize(usedReservationBytes_);
        minReservationBytes_ = 0;
      } else {
        usedReservationBytes_ -= size;
        const int64_t newCap = std::max(minReservationBytes_, usedReservationBytes_);
        newQuantized = quantizedSize(newCap);
      }
      freeable = reservationBytes_ - newQuantized;
      if (freeable > 0) {
        reservationBytes_ = newQuantized;
      }
      sanityCheckLocked();
    }
    if (freeable > 0) {
      toImpl(parent_)->decrementReservation(freeable);
    }
  }

  FOLLY_ALWAYS_INLINE void releaseNonThreadSafe(uint64_t size, bool releaseOnly) {
    VELOX_CHECK(isLeaf());
    VELOX_DCHECK_NOT_NULL(parent_);

    int64_t newQuantized;
    if (FOLLY_UNLIKELY(releaseOnly)) {
      VELOX_DCHECK_EQ(size, 0);
      if (minReservationBytes_ == 0) {
        return;
      }
      newQuantized = quantizedSize(usedReservationBytes_);
      minReservationBytes_ = 0;
    } else {
      usedReservationBytes_ -= size;
      const int64_t newCap = std::max(minReservationBytes_, usedReservationBytes_);
      newQuantized = quantizedSize(newCap);
    }

    const int64_t freeable = reservationBytes_ - newQuantized;
    if (FOLLY_UNLIKELY(freeable > 0)) {
      reservationBytes_ = newQuantized;
      sanityCheckLocked();
      toImpl(parent_)->decrementReservation(freeable);
    }
  }

  void decrementReservation(uint64_t size) noexcept {
    VELOX_CHECK_GT(size, 0);

    if (parent_ != nullptr) {
      toImpl(parent_)->decrementReservation(size);
    }
    std::lock_guard<std::mutex> l(mutex_);
    reservationBytes_ -= size;
    sanityCheckLocked();
  }

  std::string capExceedingMessage(MemoryPool* requestor, const std::string& errorMessage) {
    VELOX_CHECK_NULL(parent_);

    std::stringstream out;
    out << "\n" << errorMessage << "\n";
    {
      std::lock_guard<std::mutex> l(mutex_);
      const Stats stats = statsLocked();
      const MemoryUsage usage{.name = name(), .currentUsage = stats.currentBytes, .peakUsage = stats.peakBytes};
      out << usage.toString() << "\n";
    }

    MemoryUsageHeap topLeafMemUsages;
    visitChildren([&, indent = kCapMessageIndentSize](MemoryPool* pool) {
      capExceedingMessageVisitor(pool, indent, topLeafMemUsages, out);
      return true;
    });

    if (!topLeafMemUsages.empty()) {
      out << "\nTop " << topLeafMemUsages.size() << " leaf memory pool usages:\n";
      std::vector<MemoryUsage> usages = sortMemoryUsages(topLeafMemUsages);
      for (const auto& usage : usages) {
        out << std::string(kCapMessageIndentSize, ' ') << usage.toString() << "\n";
      }
    }

    out << "\nFailed memory pool: " << requestor->name() << ": " << velox::succinctBytes(requestor->currentBytes())
        << "\n";
    return out.str();
  }

  FOLLY_ALWAYS_INLINE void sanityCheckLocked() const {
    if (FOLLY_UNLIKELY(
            (reservationBytes_ < usedReservationBytes_) || (reservationBytes_ < minReservationBytes_) ||
            (usedReservationBytes_ < 0))) {
      VELOX_FAIL("Bad memory usage track state: {}", toStringLocked());
    }
  }

  velox::memory::MemoryPool::Stats statsLocked() const {
    Stats stats;
    stats.currentBytes = currentBytesLocked();
    stats.peakBytes = peakBytes_;
    stats.cumulativeBytes = cumulativeBytes_;
    stats.numAllocs = numAllocs_;
    stats.numFrees = numFrees_;
    stats.numReserves = numReserves_;
    stats.numReleases = numReleases_;
    stats.numCollisions = numCollisions_;
    return stats;
  }

  FOLLY_ALWAYS_INLINE std::string toStringLocked() const {
    std::stringstream out;
    out << "Memory Pool[" << name_ << " " << kindString(kind_) << " "
        << velox::memory::MemoryAllocator::kindString(veloxAlloc_->kind())
        << (trackUsage_ ? " track-usage" : " no-usage-track") << (threadSafe_ ? " thread-safe" : " non-thread-safe")
        << "]<";
    if (capacityLocked() != facebook::velox::memory::kMaxMemory) {
      out << "capacity " << velox::succinctBytes(capacity()) << " ";
    } else {
      out << "unlimited capacity ";
    }
    out << "used " << velox::succinctBytes(currentBytesLocked()) << " available "
        << velox::succinctBytes(availableReservationLocked());
    out << " reservation [used " << velox::succinctBytes(usedReservationBytes_) << ", reserved "
        << velox::succinctBytes(reservationBytes_) << ", min " << velox::succinctBytes(minReservationBytes_);
    out << "] counters [allocs " << numAllocs_ << ", frees " << numFrees_ << ", reserves " << numReserves_
        << ", releases " << numReleases_ << ", collisions " << numCollisions_ << "])";
    out << ">";
    return out.str();
  }

  velox::memory::MemoryManager* const manager_;
  velox::memory::MemoryAllocator* const veloxAlloc_;
  gluten::MemoryAllocator* glutenAlloc_;
  const DestructionCallback destructionCb_;
  mutable std::mutex mutex_;
  int64_t capacity_;

  // The number of reservation bytes.
  std::atomic<int64_t> reservationBytes_{0};

  // The number of used reservation bytes which is maintained at the leaf
  // tracker and protected by mutex for consistent memory reservation/release
  // decisions.
  std::atomic<int64_t> usedReservationBytes_{0};

  // Minimum amount of reserved memory in bytes to hold until explicit
  // release().
  std::atomic<int64_t> minReservationBytes_{0};

  std::atomic<int64_t> peakBytes_{0};
  std::atomic<int64_t> cumulativeBytes_{0};

  std::atomic<uint64_t> numAllocs_{0};
  std::atomic<uint64_t> numFrees_{0};
  std::atomic<uint64_t> numReserves_{0};
  std::atomic<uint64_t> numReleases_{0};
  std::atomic<uint64_t> numCollisions_{0};
};

std::shared_ptr<velox::memory::MemoryPool> asWrappedVeloxAggregateMemoryPool(gluten::MemoryAllocator* allocator) {
  static std::atomic_uint32_t id = 0;
  auto pool = getDefaultVeloxAggregateMemoryPool()->addAggregateChild("wrapped_root" + std::to_string(id++));
  auto wrapped = std::dynamic_pointer_cast<WrappedVeloxMemoryPool>(pool);
  VELOX_CHECK_NOT_NULL(wrapped);
  wrapped->setGlutenAllocator(allocator);
  return pool;
}

std::shared_ptr<velox::memory::MemoryPool> getDefaultVeloxAggregateMemoryPool() {
  facebook::velox::memory::MemoryPool::Options options;
  int64_t spillThreshold;
  options = gluten::VeloxInitializer::get()->getMemoryPoolOptions();
  spillThreshold = gluten::VeloxInitializer::get()->getSpillThreshold();
  static auto veloxAlloc = facebook::velox::memory::MemoryAllocator::createDefaultInstance();
  static std::shared_ptr<WrappedVeloxMemoryPool> defaultPoolRoot = std::make_shared<WrappedVeloxMemoryPool>(
      veloxAlloc.get(),
      "root",
      velox::memory::MemoryPool::Kind::kAggregate,
      nullptr,
      defaultMemoryAllocator().get(),
      spillThreshold,
      nullptr,
      nullptr,
      options);
  return defaultPoolRoot;
}

std::shared_ptr<velox::memory::MemoryPool> getDefaultVeloxLeafMemoryPool() {
  static std::shared_ptr<velox::memory::MemoryPool> defaultPool =
      getDefaultVeloxAggregateMemoryPool()->addLeafChild("default_pool");
  return defaultPool;
}
} // namespace gluten
