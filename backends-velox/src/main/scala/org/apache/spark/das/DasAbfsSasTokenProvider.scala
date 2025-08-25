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

import org.apache.commons.lang3.ClassUtils
import org.apache.hadoop.fs.azurebfs.extensions.SASTokenProvider

class DasAbfsSasTokenProvider extends Logging {
  private var account: String = _

  private val generatorClazz =
    ClassUtils.getClass("org.apache.hadoop.fs.azurebfs.sas.IbmlhcasSASTokenProvider")

  private lazy val generator =
    generatorClazz.getConstructor().newInstance().asInstanceOf[SASTokenProvider]

  def initialize(account: String): Unit = {
    if (this.account != null && this.account != account) {
      throw new IllegalStateException(
        s"SAS key generator is already initialized for account: ${this.account}, " +
          "but got another initialization request for account: " + account)
    }
    this.account = account

    val hadoopConf = SparkHadoopUtil.get.newConfiguration(SparkEnv.get.conf)
    val endPoint = s"$account.dfs.core.windows.net"
    generator.initialize(hadoopConf, endPoint)
  }

  def getSasToken(container: String, path: String, operation: String): String =
    synchronized {
      val sasKey = generator.getSASToken(account, container, "/" + path, operation)
      logDebug(s"Generated SAS key: $sasKey")
      sasKey
    }
}
