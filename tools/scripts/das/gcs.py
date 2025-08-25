# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from pyspark import SparkConf
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

from das import *


def default_conf(enable_das: bool = True) -> SparkConf:
    conf = SparkConf()

    conf.set("spark.plugins", "org.apache.gluten.GlutenPlugin") \
        .set("spark.memory.offHeap.enabled", "true") \
        .set("spark.memory.offHeap.size", "1g") \
        .set("spark.sql.adaptive.enabled", "false") \
        .set("spark.gluten.sql.debug", "true")

    if enable_das:
        conf = with_wxd_das_configuration(conf)
        conf.set("spark.hadoop.wxd.cas.sas.expiry.period", "1")

    return conf


def run_and_compare(conf: SparkConf, gcs_prefix: str):
    stop_active_session()

    spark = SparkSession.builder.master("local[1]").config(conf=conf).getOrCreate()
    try:
        spark.sparkContext.setLogLevel("DEBUG")

        path = f"{gcs_prefix}/gluten_test/t1.parquet"

        schema = StructType([
            StructField("col1", IntegerType(), True),
            StructField("col2", StringType(), True),
        ])

        df = spark.createDataFrame([Row(100, "cas-test")], schema=schema)

        df.write.mode("overwrite").parquet(path)

        result = spark.read.parquet(path).collect()
        assert len(result) == 1, f"Expected 1 row, got {len(result)}"
        assert result[0]["col1"] == 100 and result[0]["col2"] == "cas-test", f"Unexpected row: {result[0]}"

        print("GCS client test passed")
    finally:
        spark.stop()


def main():
    bucket = os.getenv("GCS_BUCKET_NAME", GCS_BUCKET_NAME)
    gcs_prefix = f"gs://{bucket}/default"

    conf = default_conf(enable_das=True)
    run_and_compare(conf, gcs_prefix=gcs_prefix)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"TEST FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
