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
package org.apache.spark.shuffle

import org.apache.spark.SparkConf

import java.util.UUID

object GlutenDisaggregatedTestConfiguration {
  def newSparkConf(): SparkConf =
    new SparkConf()
      .setAppName("testApp")
      .setMaster(s"local[2]")
      .set(
        "spark.shuffle.disaggregated.useSparkShuffleFetch",
        scala.util.Properties.envOrElse("USE_SPARK_SHUFFLE_FETCH", "false")
      )
      .set("spark.ui.enabled", "false")
      //    .set("spark.shuffle.disaggregated.bufferSize", "1024")
      .set("spark.shuffle.disaggregated.bufferSize", "16384")
      .set("spark.driver.allowMultipleContexts", "true")
      .set("spark.app.id", "app-" + UUID.randomUUID())
      .set(
        "spark.shuffle.disaggregated.rootDir",
        "file:///tmp/spark-disaggregated-shuffle"
      )
      .set(
        "spark.storage.decommission.fallbackStorage.path",
        "file:///tmp/spark-disaggregated-shuffle/"
      )
      .set("spark.dynamicAllocation.enabled", "true")
      .set("spark.local.dir", "./spark-temp") // Configure the working dir.
      .set(
        "spark.shuffle.manager",
        "org.apache.spark.shuffle.sort.GlutenDisaggregatedShuffleManager"
      )
      .set(
        "spark.shuffle.sort.io.plugin.class",
        "org.apache.spark.shuffle.DisaggregatedShuffleDataIO"
      )
      .set("spark.plugins", "org.apache.gluten.GlutenPlugin")
      .set("spark.memory.offHeap.enabled", "true")
      .set("spark.memory.offHeap.size", "1g")
      .set("spark.gluten.sql.debug", "true")
      .set("spark.gluten.sql.columnar.backend.velox.glogSeverityLevel", "0")
      .set("spark.gluten.sql.columnar.shuffle.sort.partitions.threshold", "0")
}
