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

#include "VeloxColumnarToRowConverter.h"

#include <arrow/array/array_base.h>
#include <arrow/buffer.h>
#include <arrow/c/abi.h>
#include <arrow/type_traits.h>
#include <arrow/util/decimal.h>

#include "ArrowTypeUtils.h"
#include "arrow/c/bridge.h"
#include "arrow/c/helpers.h"
#include "velox/row/UnsafeRowDeserializers.h"

#include "velox/vector/arrow/Bridge.h"

#include <iostream>
using namespace facebook;

namespace gluten {

arrow::Status VeloxColumnarToRowConverter::init() {
  numRows_ = rv_->size();
  numCols_ = rv_->childrenSize();

  // The input is Arrow batch. We need to resume Velox Vector here.
  resumeVeloxVector();

  fast_ = std::make_unique<velox::row::UnsafeRowFast>(rv_);
  lengths_.resize(numRows_, 0);
  offsets_.resize(numRows_, 0);

  size_t totalSize = 0;
  if (auto fixedRowSize = velox::row::UnsafeRowFast::fixedRowSize(velox::asRowType(rv_->type()))) {
    totalSize += fixedRowSize.value() * numRows_;

    for (auto i = 0; i < numRows_; ++i) {
      lengths_[i] = fixedRowSize.value();
    }

  } else {
    for (auto i = 0; i < numRows_; ++i) {
      auto rowSize = fast_->rowSize(i);
      totalSize += rowSize;
      lengths_[i] = rowSize;
    }
  }

  for (auto rowIdx = 1; rowIdx < numRows_; rowIdx++) {
    offsets_[rowIdx] = offsets_[rowIdx - 1] + lengths_[rowIdx - 1];
  }

  ARROW_ASSIGN_OR_RAISE(buffer_, arrow::AllocateBuffer(totalSize, arrowPool_.get()));
  bufferAddress_ = buffer_->mutable_data();
  memset(bufferAddress_, 0, sizeof(int8_t) * totalSize);
  return arrow::Status::OK();
}

void VeloxColumnarToRowConverter::resumeVeloxVector() {
  vecs_.reserve(numCols_);
  for (int colIdx = 0; colIdx < numCols_; colIdx++) {
    vecs_.push_back(rv_->childAt(colIdx));
  }
}

arrow::Status VeloxColumnarToRowConverter::write() {

  size_t offset = 0;
  for (auto i = 0; i < numRows_; ++i) {
    auto rowSize = fast_->serialize(i, (char*)(bufferAddress_ + offset));
    offset += rowSize;
   }
  return arrow::Status::OK();
}

} // namespace gluten
