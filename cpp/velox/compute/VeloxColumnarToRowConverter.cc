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
#include <arrow/type_traits.h>

#include "ArrowTypeUtils.h"
#include "arrow/c/helpers.h"
#include "include/arrow/c/bridge.h"
#include "velox/row/UnsafeRowDynamicSerializer.h"
#include "velox/row/UnsafeRowSerializer.h"
#include "velox/vector/arrow/Bridge.h"
#if defined(__x86_64__)
#include <immintrin.h>
#endif
using namespace facebook;

namespace gluten {

uint32_t x_7[8] __attribute__((aligned(32))) = {0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7, 0x7};
uint32_t x_8[8] __attribute__((aligned(32))) = {0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8};
uint32_t x_seq[8] __attribute__((aligned(32))) = {0x0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70};

arrow::Status VeloxColumnarToRowConverter::Init() {
  support_avx512_ = false;
#if defined(__x86_64__)
  support_avx512_ = __builtin_cpu_supports("avx512bw");
#endif

  num_rows_ = rv_->size();
  num_cols_ = rv_->childrenSize();

  ArrowSchema c_schema{};
  velox::exportToArrow(rv_, c_schema);
  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Schema> schema, arrow::ImportSchema(&c_schema));
  if (num_cols_ != schema->num_fields()) {
    return arrow::Status::Invalid("Mismatch: num_cols_ != schema->num_fields()");
  }
  schema_ = schema;

  // The input is Arrow batch. We need to resume Velox Vector here.
  ResumeVeloxVector();

  // Calculate the initial size
  nullBitsetWidthInBytes_ = CalculateBitSetWidthInBytes(num_cols_);
  int64_t fixed_size_per_row = CalculatedFixeSizePerRow(schema_, num_cols_);

  // Initialize the offsets_ , lengths_, buffer_cursor_
  lengths_.resize(num_rows_, fixed_size_per_row);
  offsets_.resize(num_rows_ + 1, 0);
  buffer_cursor_.resize(num_rows_, nullBitsetWidthInBytes_ + 8 * num_cols_);

  // Calculated the lengths_
  for (int64_t col_idx = 0; col_idx < num_cols_; col_idx++) {
    std::shared_ptr<arrow::Field> field = schema_->field(col_idx);
    if (arrow::is_binary_like(field->type()->id())) {
      auto str_views = vecs_[col_idx]->asFlatVector<velox::StringView>()->rawValues();
      int j = 0;
      // StringView: {size; prefix; inline}
      int32_t* length_data = lengths_.data();
#ifdef __AVX512BW__
      if (ARROW_PREDICT_TRUE(support_avx512_)) {
        __m256i x7_8x = _mm256_load_si256((__m256i*)x_7);
        __m256i x8_8x = _mm256_load_si256((__m256i*)x_8);
        __m256i vidx_8x = _mm256_load_si256((__m256i*)x_seq);

        for (; j + 16 < num_rows_; j += 8) {
          __m256i length_8x = _mm256_i32gather_epi32((int*)&str_views[j], vidx_8x, 1);

          __m256i reminder_8x = _mm256_and_si256(length_8x, x7_8x);
          reminder_8x = _mm256_sub_epi32(x8_8x, reminder_8x);
          reminder_8x = _mm256_and_si256(reminder_8x, x7_8x);
          reminder_8x = _mm256_add_epi32(reminder_8x, length_8x);
          __m256i dst_length_8x = _mm256_load_si256((__m256i*)length_data);
          dst_length_8x = _mm256_add_epi32(dst_length_8x, reminder_8x);
          _mm256_store_si256((__m256i*)length_data, dst_length_8x);
          length_data += 8;
          _mm_prefetch(&str_views[j + (128 + 128) / sizeof(velox::StringView)], _MM_HINT_T0);
        }
      }
#endif
      for (; j < num_rows_; j++) {
        auto length = str_views[j].size();
        *length_data += RoundNumberOfBytesToNearestWord(length);
        length_data++;
      }
    }
  }

  // Calculated the offsets_  and total memory size based on lengths_
  int64_t total_memory_size = lengths_[0];
  offsets_[0] = 0;
  for (auto i = 1; i < num_rows_; i++) {
    offsets_[i] = total_memory_size;
    total_memory_size += lengths_[i];
  }
  offsets_[num_rows_] = total_memory_size;

  // allocate one more cache line to ease avx operations
  if (buffer_ == nullptr || buffer_->capacity() < total_memory_size + 64) {
    ARROW_ASSIGN_OR_RAISE(buffer_, AllocateBuffer(total_memory_size + 64, arrow_pool_.get()));
#ifdef __AVX512BW__
    if (ARROW_PREDICT_TRUE(support_avx512_)) {
      // only set the extra allocated bytes
      // will set the buffer during fillbuffer
      memset(buffer_->mutable_data() + total_memory_size, 0, buffer_->capacity() - total_memory_size);
    } else
#endif
    {
      memset(buffer_->mutable_data(), 0, buffer_->capacity());
    }
  }

  buffer_address_ = buffer_->mutable_data();

  return arrow::Status::OK();
}

void VeloxColumnarToRowConverter::ResumeVeloxVector() {
  vecs_.reserve(num_cols_);
  for (int col_idx = 0; col_idx < num_cols_; col_idx++) {
    auto& child = rv_->childAt(col_idx);
    if (child->isFlatEncoding()) {
      vecs_.push_back(rv_->childAt(col_idx));
    } else {
      // Perform copy to flatten dictionary vectors.
      velox::VectorPtr copy = velox::BaseVector::create(child->type(), child->size(), child->pool());
      copy->copy(child.get(), 0, 0, child->size());
      vecs_.push_back(copy);
    }
  }
}
arrow::Status VeloxColumnarToRowConverter::FillBuffer(
    int32_t& row_start,
    int32_t batch_rows,
    std::vector<const uint8_t*>& dataptrs,
    std::vector<uint8_t> nullvec,
    std::vector<arrow::Type::type>& typevec,
    std::vector<uint8_t>& typewidth) {
#ifdef __AVX512BW__
  if (ARROW_PREDICT_TRUE(support_avx512_)) {
    __m256i fill_0_8x = {0LL};
    fill_0_8x = _mm256_xor_si256(fill_0_8x, fill_0_8x);
    // memset 0, force to bring the row buffer into L1/L2 cache
    for (auto j = row_start; j < row_start + batch_rows; j++) {
      auto rowlength = offsets_[j + 1] - offsets_[j];
      for (auto p = 0; p < rowlength + 32; p += 32) {
        _mm256_storeu_si256((__m256i*)(buffer_address_ + offsets_[j] + p), fill_0_8x);
        _mm_prefetch(buffer_address_ + offsets_[j] + 128, _MM_HINT_T0);
      }
    }
  }
#endif
  // fill the row by one column each time
  for (auto col_index = 0; col_index < num_cols_; col_index++) {
    auto& array = vecs_[col_index];
    int64_t field_offset = nullBitsetWidthInBytes_ + (col_index << 3L);

    switch (typevec[col_index]) {
      // We should keep supported types consistent with that in #buildCheck of GlutenColumnarToRowExec.scala.
      case arrow::BooleanType::type_id: {
        // Boolean type
        auto bool_array = array->asFlatVector<bool>();

        for (auto j = row_start; j < row_start + batch_rows; j++) {
          if (nullvec[col_index] || (!array->isNullAt(j))) {
            auto value = bool_array->valueAt(j);
            memcpy(buffer_address_ + offsets_[j] + field_offset, &value, sizeof(bool));
          } else {
            SetNullAt(buffer_address_, offsets_[j], field_offset, col_index);
          }
        }
        break;
      }
      case arrow::StringType::type_id:
      case arrow::BinaryType::type_id: {
        // Binary type
        facebook::velox::StringView* valuebuffers = (facebook::velox::StringView*)(dataptrs[col_index]);
        for (auto j = row_start; j < row_start + batch_rows; j++) {
          if (nullvec[col_index] || (!array->isNullAt(j))) {
            size_t length = valuebuffers[j].size();
            const char* value = valuebuffers[j].data();

#ifdef __AVX512BW__
            if (ARROW_PREDICT_TRUE(support_avx512_)) {
              // write the variable value
              if (j < row_start + batch_rows - 2) {
                _mm_prefetch(valuebuffers[j + 1].data(), _MM_HINT_T0);
                _mm_prefetch(valuebuffers[j + 2].data(), _MM_HINT_T0);
              }
              size_t k;
              for (k = 0; k + 32 < length; k += 32) {
                __m256i v = _mm256_loadu_si256((const __m256i*)(value + k));
                _mm256_storeu_si256((__m256i*)(buffer_address_ + offsets_[j] + buffer_cursor_[j] + k), v);
              }
              // create some bits of "1", num equals length
              auto mask = (1L << (length - k)) - 1;
              __m256i v = _mm256_maskz_loadu_epi8(mask, value + k);
              _mm256_mask_storeu_epi8(buffer_address_ + offsets_[j] + buffer_cursor_[j] + k, mask, v);
              _mm_prefetch(&valuebuffers[j + 128 / sizeof(facebook::velox::StringView)], _MM_HINT_T0);
              _mm_prefetch(&offsets_[j], _MM_HINT_T1);
            } else
#endif
            {
              // write the variable value
              memcpy(buffer_address_ + offsets_[j] + buffer_cursor_[j], value, length);
            }

            // write the offset and size
            int64_t offsetAndSize = ((int64_t)buffer_cursor_[j] << 32) | length;
            *(int64_t*)(buffer_address_ + offsets_[j] + field_offset) = offsetAndSize;
            buffer_cursor_[j] += RoundNumberOfBytesToNearestWord(length);
          } else {
            SetNullAt(buffer_address_, offsets_[j], field_offset, col_index);
          }
        }
        break;
      }
      case arrow::Decimal128Type::type_id: {
        /*auto out_array = dynamic_cast<arrow::Decimal128Array*>(array.get());
        auto dtype = dynamic_cast<arrow::Decimal128Type*>(out_array->type().get());

        int32_t precision = dtype->precision();

        for (auto j = row_start; j < row_start + batch_rows; j++) {
          const arrow::Decimal128 out_value(out_array->GetValue(j));
          bool flag = out_array->IsNull(j);

          if (precision <= 18) {
            if (!flag) {
              // Get the long value and write the long value
              // Refer to the int64_t() method of Decimal128
              int64_t long_value = static_cast<int64_t>(out_value.low_bits());
              memcpy(buffer_address_ + offsets_[j] + field_offset, &long_value, sizeof(long));
            } else {
              SetNullAt(buffer_address_, offsets_[j], field_offset, col_index);
            }
          } else {
            if (flag) {
              SetNullAt(buffer_address_, offsets_[j], field_offset, col_index);
            } else {
              int32_t size;
              auto out = ToByteArray(out_value, &size);
              assert(size <= 16);

              // write the variable value
              memcpy(buffer_address_ + buffer_cursor_[j] + offsets_[j], &out[0], size);
              // write the offset and size
              int64_t offsetAndSize = ((int64_t)buffer_cursor_[j] << 32) | size;
              memcpy(buffer_address_ + offsets_[j] + field_offset, &offsetAndSize, sizeof(int64_t));
            }

            // Update the cursor of the buffer.
            int64_t new_cursor = buffer_cursor_[j] + 16;
            buffer_cursor_[j] = new_cursor;
          }
        }*/
        return arrow::Status::Invalid("Unsupported data type decimal ");
        break;
      }
      default: {
        if (typewidth[col_index] > 0) {
          auto dataptr = dataptrs[col_index];
          auto mask = (1L << (typewidth[col_index])) - 1;
          (void)mask;
#if defined(__x86_64__)
          auto shift = _tzcnt_u32(typewidth[col_index]);
#else
          auto shift = __builtin_ctz((uint32_t)typewidth[col_index]);
#endif
          auto buffer_address_tmp = buffer_address_ + field_offset;
          for (auto j = row_start; j < row_start + batch_rows; j++) {
            if (nullvec[col_index] || (!array->isNullAt(j))) {
              const uint8_t* srcptr = dataptr + (j << shift);
#ifdef __AVX512BW__
              if (ARROW_PREDICT_TRUE(support_avx512_)) {
                __m256i v = _mm256_maskz_loadu_epi8(mask, srcptr);
                _mm256_mask_storeu_epi8(buffer_address_tmp + offsets_[j], mask, v);
                _mm_prefetch(srcptr + 64, _MM_HINT_T0);
              } else
#endif
              {
                memcpy(buffer_address_tmp + offsets_[j], srcptr, typewidth[col_index]);
              }
            } else {
              SetNullAt(buffer_address_, offsets_[j], field_offset, col_index);
            }
          }
          break;
        } else {
          return arrow::Status::Invalid("Unsupported data type: " + typevec[col_index]);
        }
      }
    }
  }
  return arrow::Status::OK();
}

arrow::Status VeloxColumnarToRowConverter::Write() {
  std::vector<facebook::velox::VectorPtr>& arrays = vecs_;
  std::vector<const uint8_t*> dataptrs;
  dataptrs.resize(num_cols_);
  std::vector<uint8_t> nullvec;
  nullvec.resize(num_cols_, 0);

  std::vector<arrow::Type::type> typevec;
  std::vector<uint8_t> typewidth;

  typevec.resize(num_cols_);
  // Store bytes for different fixed width types
  typewidth.resize(num_cols_);

  for (auto col_index = 0; col_index < num_cols_; col_index++) {
    facebook::velox::VectorPtr array = arrays[col_index];
    auto buf = array->values();

    // If nullvec[col_index]  equals 1, means no null value in this column
    nullvec[col_index] = !(array->mayHaveNulls());
    typevec[col_index] = schema_->field(col_index)->type()->id();
    // calculate bytes from bit_num
    typewidth[col_index] = arrow::bit_width(typevec[col_index]) >> 3;

    if (arrow::bit_width(typevec[col_index]) > 1 || typevec[col_index] == arrow::StringType::type_id ||
        typevec[col_index] == arrow::BinaryType::type_id) {
      dataptrs[col_index] = buf->as<uint8_t>();
    } else {
      return arrow::Status::Invalid(
          "Type " + schema_->field(col_index)->type()->name() + " is not supported in VeloxToRow conversion.");
    }
  }

  int32_t i = 0;
#define BATCH_ROW_NUM 16
  // file BATCH_ROW_NUM rows each time, fill by column. Make sure the row is in L1/L2 cache
  for (; i + BATCH_ROW_NUM < num_rows_; i += BATCH_ROW_NUM) {
    RETURN_NOT_OK(FillBuffer(i, BATCH_ROW_NUM, dataptrs, nullvec, typevec, typewidth));
  }

  for (; i < num_rows_; i++) {
    RETURN_NOT_OK(FillBuffer(i, 1, dataptrs, nullvec, typevec, typewidth));
  }

  return arrow::Status::OK();
}
} // namespace gluten
