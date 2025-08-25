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
package org.apache.spark.das

import org.apache.spark.SparkEnv
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.internal.Logging

import com.google.cloud.hadoop.util.AccessTokenProvider.AccessToken
import org.apache.commons.lang3.ClassUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

class DasGcsAccessTokenProvider(bucket: String) extends Logging {
  private val dasGcsClientClazz =
    ClassUtils.getClass("org.apache.hadoop.fs.wxd.cas.client.IbmlhDasGcsClient")
  private var dasGcsClient: Any = _

  def initialize(): Unit = {
    val hadoopConf = SparkHadoopUtil.get.newConfiguration(SparkEnv.get.conf)
    FileSystem.get(new Path(s"gs://$bucket").toUri, hadoopConf)

    dasGcsClient = dasGcsClientClazz.getMethod("getInstance").invoke(null)
    if (dasGcsClient == null) {
      throw new IllegalStateException(
        "IbmlhDasGcsClient is not initialized. " +
          "Please ensure the class is available in the classpath.")
    }

    dasGcsClientClazz
      .getMethod("initialize", classOf[Configuration])
      .invoke(dasGcsClient, hadoopConf)
  }

  def getAccessToken(): AccessToken = {
    if (dasGcsClient == null) {
      throw new IllegalStateException("DasGcsAccessTokenProvider is not initialized.")
    }

    val accessToken = dasGcsClientClazz
      .getMethod("getGcsAccessTokenFromIbmWxdCas", classOf[String])
      .invoke(dasGcsClient, bucket)
      .asInstanceOf[AccessToken]
    if (accessToken == null) {
      throw new IllegalStateException("Failed to retrieve access token for GCS bucket: " + bucket)
    }

    logDebug(
      s"Retrieved access token for GCS bucket: $bucket, token: ${accessToken.getToken}," +
        s"expiry in millis: ${accessToken.getExpirationTimeMilliSeconds}")
    accessToken
  }
}
