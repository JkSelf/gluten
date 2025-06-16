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

#include <jni.h>

#include <numeric>

#include "jni/JniCommon.h"
#include "shuffle/LocalPartitionWriter.h"
#include "shuffle/PartitionWriter.h"
#include "shuffle/disaggregated/MapOutputWriter.h"

namespace gluten {

// This class is responsible for writing data to a Java OutputStream.
// It handles the conversion of data to byte arrays and manages the output stream lifecycle.
// The class is not thread-safe and should be used in a single-threaded context.
// It uses JNI to call Java methods for writing and closing the stream.
class DisaggregatedOutputStream final : public arrow::io::OutputStream {
 public:
  DisaggregatedOutputStream(JNIEnv* env, jobject outputStream)
      : env_(env), outputStream_(env->NewGlobalRef(outputStream)) {
    jclass outputStreamClass = env->GetObjectClass(outputStream);
    if (outputStreamClass == nullptr) {
      throw GlutenException("Failed to get class of outputStream");
    }
    writeMethod_ = getMethodIdOrError(env, outputStreamClass, "write", "([BII)V");
    closeMethod_ = getMethodIdOrError(env, outputStreamClass, "close", "()V");
  }

  bool closed() const override {
    return closed_;
  }

  arrow::Status Close() override {
    if (!closed_) {
      closed_ = true;
      env_->CallVoidMethod(outputStream_, closeMethod_);
      env_->DeleteGlobalRef(outputStream_);
    }
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> Tell() const override {
    if (closed_) {
      return arrow::Status::Invalid("Output stream is closed");
    }
    return bytesWritten_;
  }

  arrow::Status Write(const void* data, int64_t nbytes) override {
    constexpr int64_t kBufferSize = 4096;
    const auto buffer = env_->NewByteArray(kBufferSize);
    if (buffer == nullptr) {
      return arrow::Status::Invalid("Failed to allocate buffer");
    }

    auto bytes = static_cast<const jbyte*>(data);
    int64_t remaining = nbytes;
    while (remaining > 0) {
      const auto toWrite = std::min(remaining, kBufferSize);
      env_->SetByteArrayRegion(buffer, 0, toWrite, bytes);
      env_->CallVoidMethod(outputStream_, writeMethod_, buffer, 0, toWrite);
      if (env_->ExceptionCheck()) {
        return arrow::Status::Invalid("Failed to write data to output stream");
      }
      bytes += toWrite;
      remaining -= toWrite;
    }

    bytesWritten_ += nbytes;
    env_->DeleteLocalRef(buffer);

    return arrow::Status::OK();
  }

 private:
  JNIEnv* env_;
  jobject outputStream_;
  jmethodID writeMethod_;
  jmethodID closeMethod_;

  bool closed_{false};
  int64_t bytesWritten_{0};
};

class DisaggregatedShuffleMapOutputWriterWrapper : public MapOutputWriter {
 public:
  DisaggregatedShuffleMapOutputWriterWrapper(JNIEnv* env, jobject mapOutputWriter)
      : env_(env), mapOutputWriter_(env->NewGlobalRef(mapOutputWriter)) {
    jclass objClass = env->GetObjectClass(mapOutputWriter);
    if (objClass == nullptr) {
      throw GlutenException("Failed to get class of mapOutputWriter");
    }
    openStreamMethod_ = getMethodIdOrError(env, objClass, "openStream", "(I)Ljava/io/OutputStream;");
    transferMapSpillFileMethod_ = getMethodIdOrError(env, objClass, "transferMapSpillFile", "(Ljava/lang/String;[J)V");
    commitAllPartitionsMethod_ = getMethodIdOrError(env, objClass, "commitAllPartitions", "()[J");

    env_->DeleteLocalRef(objClass);
  }

  ~DisaggregatedShuffleMapOutputWriterWrapper() override {
    env_->DeleteGlobalRef(mapOutputWriter_);
  }

  std::shared_ptr<arrow::io::OutputStream> openStream(int32_t partitionId) override {
    jobject outputStream = env_->CallObjectMethod(mapOutputWriter_, openStreamMethod_, partitionId);
    if (env_->ExceptionCheck()) {
      throw GlutenException("Failed to call openStream on mapOutputWriter");
    }
    return std::make_shared<DisaggregatedOutputStream>(env_, outputStream);
  }

  void transferMapSpillFile(const std::string& spillFile, const std::vector<int64_t>& partitionLengths) override {
    jstring spillFileJstr = env_->NewStringUTF(spillFile.c_str());

    jlongArray partitionLengthsArr = env_->NewLongArray(partitionLengths.size());
    env_->SetLongArrayRegion(
        partitionLengthsArr, 0, partitionLengths.size(), reinterpret_cast<const jlong*>(partitionLengths.data()));

    env_->CallVoidMethod(mapOutputWriter_, transferMapSpillFileMethod_, spillFileJstr, partitionLengthsArr);
    if (env_->ExceptionCheck()) {
      throw GlutenException("Failed to call transferMapSpillFile on mapOutputWriter");
    }

    env_->DeleteLocalRef(spillFileJstr);
    env_->DeleteLocalRef(partitionLengthsArr);
  }

  void commitAllPartitions(std::vector<int64_t>& partitionLengths) override {
    auto partitionLengthArr =
        static_cast<jlongArray>(env_->CallObjectMethod(mapOutputWriter_, commitAllPartitionsMethod_));
    if (env_->ExceptionCheck()) {
      throw GlutenException("Failed to call commitAllPartitions on mapOutputWriter");
    }

    jsize length = env_->GetArrayLength(partitionLengthArr);
    if (length != partitionLengths.size()) {
      throw GlutenException(
          "Mismatch in partition lengths size: expected " + std::to_string(partitionLengths.size()) + ", got " +
          std::to_string(length));
    }

    jlong* elements = env_->GetLongArrayElements(partitionLengthArr, nullptr);
    if (elements == nullptr) {
      throw GlutenException("Failed to get elements from partitionLengths array");
    }

    for (auto i = 0; i < length; ++i) {
      partitionLengths[i] = static_cast<int64_t>(elements[i]);
    }

    env_->ReleaseLongArrayElements(partitionLengthArr, elements, JNI_ABORT);
  }

 private:
  JNIEnv* env_;
  jobject mapOutputWriter_;
  jmethodID openStreamMethod_;
  jmethodID transferMapSpillFileMethod_;
  jmethodID commitAllPartitionsMethod_;
};

class DisaggregatedShufflePartitionWriter final : public LocalPartitionWriter {
 public:
  DisaggregatedShufflePartitionWriter(
      uint32_t numPartitions,
      std::unique_ptr<arrow::util::Codec> codec,
      MemoryManager* memoryManager,
      const std::shared_ptr<LocalPartitionWriterOptions>& options,
      const std::vector<std::string>& localDirs,
      const std::shared_ptr<MapOutputWriter>& mapOutputWriterWrapper)
      : LocalPartitionWriter(numPartitions, std::move(codec), memoryManager, options, "placeholder", localDirs),
        mapOutputWriter_(mapOutputWriterWrapper) {}

  arrow::Status sortEvict(
      uint32_t partitionId,
      std::unique_ptr<InMemoryPayload> inMemoryPayload,
      bool isFinal,
      int64_t& evictBytes) override {
    if (!isFinal) {
      isOnlySpillFromSort_ = false;
    }
    if (isFinal && !isOnlySpillFromSort_.has_value()) {
      isOnlySpillFromSort_ = true;
    }
    return LocalPartitionWriter::sortEvict(partitionId, std::move(inMemoryPayload), false, evictBytes);
  }

  arrow::Status stop(ShuffleWriterMetrics* metrics, int64_t& evictBytes) override {
    if (stopped_) {
      return arrow::Status::OK();
    }
    stopped_ = true;

    RETURN_NOT_OK(finishSpill());
    RETURN_NOT_OK(finishMerger());

    if (isOnlySpillFromSort_.has_value() && isOnlySpillFromSort_.value()) {
      // Only one spill file. Use `SingleSpillShuffleMapOutputWriter`.
      GLUTEN_CHECK(spills_.size() == 1, "Expected only one spill file, but got " + std::to_string(spills_.size()));
      auto spill = spills_.front();
      spills_.pop_front();
      for (auto pid = 0; pid < numPartitions_; ++pid) {
        while (auto payload = spill->nextPayload(pid)) {
          partitionLengths_[pid] += payload->rawSize();
        }
      }
      mapOutputWriter_->transferMapSpillFile(spill->spillFile(), partitionLengths_);
    } else {
      // Merge all partitions from spills and write the cached payloads to the final data file.
      for (auto pid = 0; pid < numPartitions_; ++pid) {
        auto os = mapOutputWriter_->openStream(pid);
        RETURN_NOT_OK(mergeSpills(pid, os.get()));
        RETURN_NOT_OK(writeCachedPayloads(pid, os.get()));
        RETURN_NOT_OK(os->Close());
      }
      mapOutputWriter_->commitAllPartitions(partitionLengths_);
    }

    totalBytesWritten_ = std::accumulate(partitionLengths_.begin(), partitionLengths_.end(), 0L);

    RETURN_NOT_OK(clearResource());

    // Populate shuffle writer metrics.
    RETURN_NOT_OK(populateMetrics(metrics));
    return arrow::Status::OK();
  }

 private:
  std::shared_ptr<MapOutputWriter> mapOutputWriter_;
  std::optional<bool> isOnlySpillFromSort_{std::nullopt};
};
} // namespace gluten