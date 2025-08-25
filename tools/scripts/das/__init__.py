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

from pyspark import SparkConf
from pyspark.sql import SparkSession

# Set the DAS URL here, e.g., "https://das.example.com"
DAS_URL: str = "https://127.0.0.1:8080"
DAS_ENDPOINT: str = DAS_URL + "/cas/v1/signature"
INSTANCE_ID: str = "0123456789"
API_KEY: str = "ZenApiKey "

# wxd.cas.ssl.no.verify
WXD_CAS_SSL_NO_VERIFY: str = "true"

# Minio end point. Replace with s3 endpoint url if you are using AWS S3.
S3_ENDPOINT: str = "http://127.0.0.1:9000/"
S3_REGION: str = "us-east-1"
S3_ACCESS_KEY: str = "minioadmin"
S3_SECRET_KEY: str = "minioadmin"
S3_BUCKET_NAME: str = "gluten-test-bucket"

ABFS_ACCOUNT_NAME: str = "gluten-account"
ABFS_CONTAINER_NAME: str = "gluten-container"

GCS_BUCKET_NAME: str = "gluten-test-bucket-gcs"


def with_wxd_das_configuration(conf: SparkConf) -> SparkConf:
    # Common configurations
    conf.set(
        "spark.hadoop.wxd.cas.endpoint",
        os.getenv("DAS_ENDPOINT", DAS_ENDPOINT),
    )
    conf.set(
        "spark.hadoop.wxd.instanceId",
        os.getenv("INSTANCE_ID", INSTANCE_ID),
    )
    conf.set(
        "spark.hadoop.wxd.apikey",
        os.getenv("API_KEY", API_KEY),
    )
    conf.set(
        "spark.hadoop.wxd.cas.ssl.no.verify",
        os.getenv("WXD_CAS_SSL_NO_VERIFY", WXD_CAS_SSL_NO_VERIFY),
    )

    # DAS S3 configurations
    conf.set(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        "com.ibm.iae.s3.credentialprovider.WatsonxCredentialsProvider",
    )
    conf.set("spark.hadoop.fs.s3a.s3.signing-algorithm", "WatsonxAWSV4Signer")
    conf.set(
        "spark.hadoop.fs.s3a.custom.signers",
        "WatsonxAWSV4Signer:com.ibm.iae.s3.credentialprovider.WatsonxAWSV4Signer",
    )

    # DAS ABFS configurations
    abfs_account_name = os.getenv("ABFS_ACCOUNT_NAME", ABFS_ACCOUNT_NAME)
    conf.set(
        f"spark.hadoop.fs.azure.account.auth.type.{abfs_account_name}.dfs.core.windows.net",
        "SAS",
    )
    conf.set(
        f"spark.hadoop.fs.azure.sas.token.provider.type.{abfs_account_name}.dfs.core.windows.net",
        "org.apache.hadoop.fs.azurebfs.sas.IbmlhcasSASTokenProvider",
    )

    # DAS GCS configurations
    conf.set(
        "spark.hadoop.fs.gs.impl",
        "org.apache.hadoop.fs.gcs.wxd.DasAwareGoogleHadoopFileSystem",
    )
    conf.set(
        "spark.hadoop.fs.gs.auth.access.token.provider.impl",
        "org.apache.hadoop.fs.gcs.wxd.DasAwareAccessTokenProvider",
    )

    return conf


def stop_active_session():
    active = SparkSession.getActiveSession()
    if active:
        try:
            active.stop()
        except Exception:
            pass
