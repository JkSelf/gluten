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

import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession

import GlutenDisaggregatedTestConfiguration.newSparkConf
import ch.cern.sparkmeasure.StageMetrics
import org.junit.Test
import org.scalatest.Assertions._

case class KeyClass()

case class ValueClass()

case class CombinerClass()

/*
 * The test has been adapted from the following pull request
 *  https://github.com/apache/spark/pull/34864/files .
 */
class GlutenDisaggregatedShuffleManagerTest extends Logging {
  @Test
  def runWithSparkMeasure(): Unit = {
    val conf = newSparkConf()
    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
    val stageMetrics = StageMetrics(spark)
    val result = stageMetrics.runAndMeasure {
      spark
        .sql(
          "select count(*) from range(1000) cross join range(1000) cross join range(1000)"
        )
        .take(1)
    }
    assert(result.map(r => r.getLong(0)).head === 1000000000)

    val timestamp = System.currentTimeMillis()
    stageMetrics.createStageMetricsDF(s"spark_measure_test_$timestamp")
    val metrics =
      stageMetrics.aggregateStageMetrics(s"spark_measure_test_$timestamp")
    // get all of the stats
    val (runTime, bytesRead, recordsRead, bytesWritten, recordsWritten) =
      metrics
        .select(
          "elapsedTime",
          "bytesRead",
          "recordsRead",
          "bytesWritten",
          "recordsWritten"
        )
        .take(1)
        .map(r => (r.getLong(0), r.getLong(1), r.getLong(2), r.getLong(3), r.getLong(4)))
        .head
    logInfo(
      f"Elapsed: $runTime, bytesRead: $bytesRead, recordsRead: $recordsRead, " +
        f"bytesWritten $bytesWritten, recordsWritten: $recordsWritten"
    )
    spark.stop()
    spark.close()
  }
}
