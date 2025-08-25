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
from pyspark.sql import SparkSession

from das import *


def default_conf(enable_das: bool) -> SparkConf:
    conf = SparkConf()

    conf.set("spark.plugins", "org.apache.gluten.GlutenPlugin") \
        .set("spark.memory.offHeap.enabled", "true") \
        .set("spark.memory.offHeap.size", "1g") \
        .set("spark.gluten.sql.debug", "true") \
        .set("spark.sql.adaptive.enabled", "false") \
        .set("spark.hadoop.fs.s3a.access.key", os.getenv("S3_ACCESS_KEY", S3_ACCESS_KEY)) \
        .set("spark.hadoop.fs.s3a.secret.key", os.getenv("S3_SECRET_KEY", S3_SECRET_KEY)) \
        .set("spark.hadoop.fs.s3a.endpoint", os.getenv("S3_ENDPOINT", S3_ENDPOINT)) \
        .set("spark.hadoop.fs.s3a.endpoint.region", os.getenv("S3_REGION", S3_REGION)) \
        .set("spark.hadoop.fs.s3a.path.style.access", "true") \
        .set("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
        .set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .set("spark.gluten.velox.awsSdkLogLevel", "DEBUG")

    if enable_das:
        return with_wxd_das_configuration(conf)

    return conf


def run_lineitem_count_test(conf: SparkConf, bucket_name: str):
    stop_active_session()
    spark = SparkSession.builder.master("local[1]").config(conf=conf).getOrCreate()
    try:
        spark.sql("DROP TABLE IF EXISTS lineitem")
        spark.sql(f"""
            CREATE TABLE lineitem (
              l_orderkey      bigint,
              l_partkey       bigint,
              l_suppkey       bigint,
              l_linenumber    int,
              l_quantity      decimal(12,2),
              l_extendedprice decimal(12,2),
              l_discount      decimal(12,2),
              l_tax           decimal(12,2),
              l_returnflag    string,
              l_linestatus    string,
              l_shipdate      date,
              l_commitdate    date,
              l_receiptdate   date,
              l_shipinstruct  string,
              l_shipmode      string,
              l_comment       string)
            USING PARQUET
            LOCATION 's3a://{bucket_name}/test-das/lineitem'
        """.strip())

        df = spark.sql("select count(*) from lineitem where l_partkey in (1552, 674, 1062)")
        count = df.collect()[0][0]
        assert count == 122, f"Expected count=122, got {count}"
        print("lineitem count test passed")
    finally:
        spark.stop()


def run_iceberg_table_test(base_conf: SparkConf, bucket_name: str):
    # Clone conf.
    new_conf = SparkConf()
    for k, v in base_conf.getAll():
        new_conf.set(k, v)

    conf = (new_conf
            .set("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
            .set("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkCatalog")
            .set("spark.sql.catalog.spark_catalog.type", "hadoop")
            .set("spark.sql.catalog.spark_catalog.warehouse", f"s3a://{bucket_name}/iceberg_warehouse")
            )

    stop_active_session()
    spark = SparkSession.builder.master("local[1]").config(conf=conf).getOrCreate()
    try:
        spark.sql("""
            CREATE TABLE IF NOT EXISTS
            iceberg_tbl(id INTEGER, name VARCHAR(10), age INTEGER, salary DECIMAL(10, 2))
            USING iceberg
        """.strip())

        spark.sql("""
            INSERT INTO iceberg_tbl
            VALUES(1,'foo',23,3400.00),(2,'bar',30,5500.00),(3,'Baz',35,6500.00)
        """.strip())

        df = spark.sql("SELECT ID FROM iceberg_tbl WHERE id = 1")
        val = df.collect()[0][0]
        assert val == 1, f"Expected ID=1, got {val}"
        print("iceberg table test passed")
    finally:
        spark.stop()


def run_das_bucket_configuration(conf: SparkConf, bucket_name: str):
    conf.set(f"spark.hadoop.fs.s3a.bucket.{bucket_name}.aws.credentials.provider",
             "com.ibm.iae.s3.credentialprovider.WatsonxCredentialsProvider")
    run_lineitem_count_test(conf, bucket_name)


def run_das_fallback_configuration(conf: SparkConf, bucket_name: str):
    conf.set(f"spark.hadoop.fs.s3a.bucket.{bucket_name}.aws.credentials.provider",
             "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    run_lineitem_count_test(conf, bucket_name)


def run_disable_das(bucket_name: str):
    conf = default_conf(enable_das=False)
    run_lineitem_count_test(conf, bucket_name=bucket_name)
    run_iceberg_table_test(conf, bucket_name=bucket_name)


def run_enable_das(bucket_name: str):
    conf = default_conf(enable_das=True)
    run_lineitem_count_test(conf, bucket_name=bucket_name)
    run_iceberg_table_test(conf, bucket_name=bucket_name)
    run_das_bucket_configuration(conf, bucket_name=bucket_name)
    run_das_fallback_configuration(conf, bucket_name=bucket_name)


def main():
    bucket_name = os.getenv("S3_BUCKET_NAME", S3_BUCKET_NAME)

    run_disable_das(bucket_name)
    run_enable_das(bucket_name)

    print("All tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"TEST FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
