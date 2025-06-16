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

#include <arrow/io/interfaces.h>

namespace gluten {

class MapOutputWriter {
 public:
  virtual ~MapOutputWriter() = default;

  virtual std::shared_ptr<arrow::io::OutputStream> openStream(int32_t partitionId) = 0;

  virtual void transferMapSpillFile(const std::string& spillFile, const std::vector<int64_t>& partitionLengths) = 0;

  virtual void commitAllPartitions(std::vector<int64_t>& partitionLengths) = 0;
};

class MockMapOutputWriter final : public MapOutputWriter {
 public:
  MockMapOutputWriter(int32_t numPartitions, const std::string& dataFile) : dataFile_(dataFile) {
    partitionLengths_.resize(numPartitions);
  }

  std::shared_ptr<arrow::io::OutputStream> openStream(int32_t partitionId) override {
    if (os_ == nullptr || os_->closed()) {
      GLUTEN_ASSIGN_OR_THROW(os_, arrow::io::FileOutputStream::Open(dataFile_, true));
    }
    GLUTEN_ASSIGN_OR_THROW(auto pos, os_->Tell());
    if (lastPartitionId_ != -1) {
      partitionLengths_[lastPartitionId_] = pos - lastPos_;
    }
    lastPartitionId_ = partitionId;
    lastPos_ = pos;

    return os_;
  }

  void transferMapSpillFile(const std::string& spillFile, const std::vector<int64_t>& partitionLengths) override {
    std::filesystem::copy(spillFile, dataFile_, std::filesystem::copy_options::overwrite_existing);
  }

  void commitAllPartitions(std::vector<int64_t>& partitionLengths) override {
    if (lastPartitionId_ != -1) {
      openStream(-1);
    }
    GLUTEN_THROW_NOT_OK(os_->Close());
    partitionLengths = std::move(partitionLengths_);
  }

 private:
  std::string dataFile_;

  std::shared_ptr<arrow::io::FileOutputStream> os_{nullptr};
  int64_t lastPos_{0};
  int32_t lastPartitionId_{-1};

  std::vector<int64_t> partitionLengths_;
};
} // namespace gluten