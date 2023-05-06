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

#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/dataset/file_parquet.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/io/interfaces.h>
#include <arrow/memory_pool.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
// #include <arrow/testing/gtest_util.h>
#include <arrow/dataset/file_base.h>
#include <arrow/dataset/scanner.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/io_util.h>
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>

#include <chrono>

#include "compute/VeloxColumnarToRowConverter.h"
#include "memory/ArrowMemoryPool.h"
#include "memory/VeloxMemoryPool.h"
#include "tests/TestUtils.h"
#include "velox/vector/arrow/Bridge.h"

#include "compute/VeloxBackend.h"
#include "velox/dwio/parquet/writer/Writer.h"

using namespace facebook;
using namespace arrow;
namespace gluten {

const int batch_buffer_size = 32768;

class GoogleBenchmarkParquetWrite {
 public:
  GoogleBenchmarkParquetWrite(std::string file_name) {
    GetRecordBatchReader(file_name);
  }

  void GetRecordBatchReader(const std::string& input_file) {
    std::unique_ptr<::parquet::arrow::FileReader> parquet_reader;
    std::shared_ptr<RecordBatchReader> record_batch_reader;

    std::shared_ptr<arrow::fs::FileSystem> fs;
    std::string file_name;
    ARROW_ASSIGN_OR_THROW(fs, arrow::fs::FileSystemFromUriOrPath(input_file, &file_name))

    ARROW_ASSIGN_OR_THROW(file, fs->OpenInputFile(file_name));

    properties.set_batch_size(batch_buffer_size);
    properties.set_pre_buffer(false);
    properties.set_use_threads(false);

    ASSERT_NOT_OK(::parquet::arrow::FileReader::Make(
        arrow::default_memory_pool(), ::parquet::ParquetFileReader::Open(file), properties, &parquet_reader));

    ASSERT_NOT_OK(parquet_reader->GetSchema(&schema));

    auto num_rowgroups = parquet_reader->num_row_groups();

    for (int i = 0; i < num_rowgroups; ++i) {
      row_group_indices.push_back(i);
    }

    auto num_columns = schema->num_fields();
    for (int i = 0; i < num_columns; ++i) {
      column_indices.push_back(i);
    }
  }

  virtual void operator()(benchmark::State& state) {}

 protected:
  long SetCPU(uint32_t cpuindex) {
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(cpuindex, &cs);
    return sched_setaffinity(0, sizeof(cs), &cs);
  }

  velox::VectorPtr recordBatch2RowVector(const arrow::RecordBatch& rb) {
    ArrowArray arrowArray;
    ArrowSchema arrowSchema;
    ASSERT_NOT_OK(arrow::ExportRecordBatch(rb, &arrowArray, &arrowSchema));
    return velox::importFromArrowAsOwner(arrowSchema, arrowArray, gluten::GetDefaultLeafWrappedVeloxMemoryPool().get());
  }

 protected:
  std::string file_name;
  std::shared_ptr<arrow::io::RandomAccessFile> file;
  std::vector<int> row_group_indices;
  std::vector<int> column_indices;
  std::shared_ptr<arrow::Schema> schema;
  parquet::ArrowReaderProperties properties;
};

class GoogleBenchmarkParquetWrite_IterateScan_Benchmark : public GoogleBenchmarkParquetWrite {
 public:
  arrow::Result<std::shared_ptr<arrow::Schema>> SchemaFromColumnNames(
      const std::shared_ptr<arrow::Schema>& input,
      const std::vector<std::string>& column_names) {
    std::vector<std::shared_ptr<arrow::Field>> columns;
    for (arrow::FieldRef ref : column_names) {
      auto maybe_field = ref.GetOne(*input);
      if (maybe_field.ok()) {
        columns.push_back(std::move(maybe_field).ValueOrDie());
      } else {
        return arrow::Status::Invalid("Partition column '", ref.ToString(), "' is not in dataset schema");
      }
    }

    return arrow::schema(std::move(columns))->WithMetadata(input->metadata());
  }

  GoogleBenchmarkParquetWrite_IterateScan_Benchmark(std::string filename) : GoogleBenchmarkParquetWrite(filename) {}
  void operator()(benchmark::State& state) {
    // // auto backend = std::dynamic_pointer_cast<gluten::VeloxBackend>(gluten::CreateBackend());
    // auto veloxPool =
    //     AsWrappedVeloxAggregateMemoryPool(gluten::DefaultMemoryAllocator().get(), nullptr);
    // auto pool = veloxPool->addLeafChild("velox_parquet_write");
    auto pool = GetDefaultWrappedVeloxAggregateMemoryPool()->addLeafChild("parquet_write_benchmark");

    auto final_path = "/tmp/velox-parquet-write";
    auto sink = std::make_unique<velox::dwio::common::LocalFileSink>(final_path);

    auto properities = ::parquet::WriterProperties::Builder().build();

    // auto properities =
    //     ::parquet::WriterProperties::Builder().write_batch_size(blockSize)->compression(compressionCodec)->build();

    auto parquetWriter = std::make_unique<velox::parquet::Writer>(std::move(sink), *(pool), 2048, properities);

    SetCPU(state.range(0));

    int64_t elapse_read = 0;
    int64_t num_batches = 0;
    int64_t num_rows = 0;
    // int64_t init_time = 0;
    int64_t write_time = 0;
    int64_t data_convert_time = 0;

    std::shared_ptr<arrow::RecordBatch> record_batch;

    std::unique_ptr<::parquet::arrow::FileReader> parquet_reader;
    std::shared_ptr<RecordBatchReader> record_batch_reader;
    ASSERT_NOT_OK(::parquet::arrow::FileReader::Make(
        arrow::default_memory_pool(), ::parquet::ParquetFileReader::Open(file), properties, &parquet_reader));

    // auto arrowPool = GetDefaultWrappedArrowMemoryPool();
    auto ctxPool = GetDefaultLeafWrappedVeloxMemoryPool();
    for (auto _ : state) {
      ASSERT_NOT_OK(parquet_reader->GetRecordBatchReader(row_group_indices, column_indices, &record_batch_reader));
      auto scanner_builder = arrow::dataset::ScannerBuilder::FromRecordBatchReader(record_batch_reader);

      TIME_NANO_OR_THROW(elapse_read, record_batch_reader->ReadNext(&record_batch));
      std::unique_ptr<::parquet::arrow::FileWriter> arrowWriter;

      while (record_batch) {
        num_batches += 1;
        num_rows += record_batch->num_rows();
        // std::cout << "the numrows is " << record_batch->num_rows() << "\n";
        // Convert arrow RecordBatch to velox row vector and then convert row vector to arrow record batch
        // auto vector = recordBatch2RowVector(*record_batch);
        // auto row_vector = std::dynamic_pointer_cast<velox::RowVector>(vector);

        // ArrowArray array;
        // ArrowSchema schema;
        // auto start = std::chrono::steady_clock::now();
        // facebook::velox::exportToArrow(row_vector, array, GetDefaultLeafWrappedVeloxMemoryPool().get());
        // facebook::velox::exportToArrow(row_vector, schema);

        // PARQUET_ASSIGN_OR_THROW(auto recordBatch, arrow::ImportRecordBatch(&array, &schema));
        // auto table = arrow::Table::Make(recordBatch->schema(), recordBatch->columns(), row_vector->size());
        // auto end = std::chrono::steady_clock::now();
        // data_convert_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        // if (!arrowWriter) {
        //   auto stream = std::make_shared<facebook::velox::parquet::DataBufferSink>(*pool);
        //   auto arrowProperties = ::parquet::ArrowWriterProperties::Builder().build();
        //   auto properties = ::parquet::WriterProperties::Builder().build();
        //   PARQUET_ASSIGN_OR_THROW(
        //       arrowWriter,
        //       ::parquet::arrow::FileWriter::Open(
        //           *recordBatch->schema(), arrow::default_memory_pool(), stream, properties, arrowProperties));
        // }

        if (!arrowWriter) {
          auto stream = std::make_shared<facebook::velox::parquet::DataBufferSink>(*pool);
          // stream->dataBuffer().reserve(20480000);
          // auto stream = ::arrow::io::BufferOutputStream::Create().ValueOrDie();
          auto arrowProperties = ::parquet::ArrowWriterProperties::Builder().build();
          auto properties = ::parquet::WriterProperties::Builder().build();
          PARQUET_ASSIGN_OR_THROW(
              arrowWriter,
              ::parquet::arrow::FileWriter::Open(
                  *record_batch->schema(), arrow::default_memory_pool(), stream, properties, arrowProperties));
        }

        auto table = arrow::Table::Make(record_batch->schema(), record_batch->columns(), record_batch->num_rows());

        auto start = std::chrono::steady_clock::now();
        PARQUET_THROW_NOT_OK(arrowWriter->WriteTable(*table, 10 * 1024 * 1024));
        // PARQUET_THROW_NOT_OK(arrowWriter->WriteRecordBatch(*record_batch));
        // parquetWriter->write(std::dynamic_pointer_cast<velox::RowVector>(vector));
        auto end = std::chrono::steady_clock::now();
        write_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        TIME_NANO_OR_THROW(elapse_read, record_batch_reader->ReadNext(&record_batch));
      }
    }

    // state.counters["rowgroups"] = benchmark::Counter(
    //     row_group_indices.size(), benchmark::Counter::kAvgThreads, benchmark::Counter::OneK::kIs1000);
    // state.counters["columns"] =
    //     benchmark::Counter(column_indices.size(), benchmark::Counter::kAvgThreads,
    //     benchmark::Counter::OneK::kIs1000);
    // state.counters["batches"] =
    //     benchmark::Counter(num_batches, benchmark::Counter::kAvgThreads, benchmark::Counter::OneK::kIs1000);
    // state.counters["num_rows"] =
    //     benchmark::Counter(num_rows, benchmark::Counter::kAvgThreads, benchmark::Counter::OneK::kIs1000);
    // state.counters["batch_buffer_size"] =
    //     benchmark::Counter(batch_buffer_size, benchmark::Counter::kAvgThreads, benchmark::Counter::OneK::kIs1024);

    // state.counters["parquet_parse"] =
    //     benchmark::Counter(elapse_read, benchmark::Counter::kAvgThreads, benchmark::Counter::OneK::kIs1000);
    // state.counters["init_time"] =
    //     benchmark::Counter(init_time, benchmark::Counter::kAvgThreads, benchmark::Counter::OneK::kIs1000);
    state.counters["write_time"] =
        benchmark::Counter(write_time, benchmark::Counter::kAvgThreads, benchmark::Counter::OneK::kIs1000);
    state.counters["data_convert_time"] =
        benchmark::Counter(data_convert_time, benchmark::Counter::kAvgThreads, benchmark::Counter::OneK::kIs1000);
  }
};

} // namespace gluten

// usage
// ./columnar_to_row_benchmark --threads=1 --file /mnt/DP_disk1/int.parquet
int main(int argc, char** argv) {
  uint32_t iterations = 1;
  uint32_t threads = 1;
  std::string datafile;
  uint32_t cpu = 0xffffffff;

  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "--iterations") == 0) {
      iterations = atol(argv[i + 1]);
    } else if (strcmp(argv[i], "--threads") == 0) {
      threads = atol(argv[i + 1]);
    } else if (strcmp(argv[i], "--file") == 0) {
      datafile = argv[i + 1];
    } else if (strcmp(argv[i], "--cpu") == 0) {
      cpu = atol(argv[i + 1]);
    }
  }
  std::cout << "iterations = " << iterations << std::endl;
  std::cout << "threads = " << threads << std::endl;
  std::cout << "datafile = " << datafile << std::endl;
  std::cout << "cpu = " << cpu << std::endl;

  datafile =
      "/mnt/DP_disk2/sparkuser/tpcds/tpcds_parquet_nopartition_date_decimal_1/store_sales/part-00000-2c4d479f-bb3e-46fc-9082-00c317ceee90-c000.snappy.parquet";

  gluten::GoogleBenchmarkParquetWrite_IterateScan_Benchmark bck(datafile);

  benchmark::RegisterBenchmark("GoogleBenchmarkParquetWrite::IterateScan", bck)
      ->Args({
          cpu,
      })
      ->Iterations(iterations)
      ->Threads(threads)
      ->ReportAggregatesOnly(false)
      ->MeasureProcessCPUTime()
      ->Unit(benchmark::kSecond);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}
