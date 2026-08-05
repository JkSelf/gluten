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
package org.apache.spark.sql

import org.apache.spark.sql.catalyst.expressions.BloomFilterMightContain
import org.apache.spark.sql.catalyst.plans.logical.Filter
import org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanHelper
import org.apache.spark.sql.execution.exchange.ReusedExchangeExec
import org.apache.spark.sql.internal.{SQLConf => SparkSQLConf}

/**
 * Tests for [[org.apache.gluten.extension.RemoveBloomFilterToRecoverExchangeReuse]].
 *
 * The Q24-style pattern: a CTE is referenced twice - once in the main query and once in a
 * correlated HAVING subquery. Spark's InjectRuntimeFilter injects a bloom filter on the main
 * reference but skips the correlated subquery, making the two sub-plans structurally different and
 * preventing ReusedExchange. Our rule strips the bloom filter to recover exchange reuse.
 */
class GlutenRemoveBloomFilterToRecoverExchangeReuseSuite
  extends GlutenSQLTestsTrait
  with AdaptiveSparkPlanHelper {

  // Mimic Q24: a large "fact" table joined with a small "dim" table,
  // where the CTE is referenced twice (main query + HAVING subquery).
  private def setupTables(): Unit = {
    // fact: large enough to exceed the bloom-filter scan size threshold
    spark
      .range(0, 1000, 1, 4)
      .selectExpr("id as fact_id", "(id % 10) as dim_id", "id * 2 as val")
      .write
      .saveAsTable("bf_fact")

    // dim: small selective table - InjectRuntimeFilter will build a bloom filter from it
    spark
      .range(0, 5)
      .selectExpr("id as dim_id", "id * 3 as dim_val")
      .write
      .saveAsTable("bf_dim")
  }

  private def withBloomFilterDisabled[T](spark: SparkSession)(body: => T): T = {
    val key = SparkSQLConf.RUNTIME_BLOOM_FILTER_ENABLED.key
    val original = spark.conf.getOption(key)
    try {
      spark.conf.set(key, "false")
      body
    } finally {
      original match {
        case Some(value) => spark.conf.set(key, value)
        case None => spark.conf.unset(key)
      }
    }
  }

  testGluten(
    "RemoveBloomFilterToRecoverExchangeReuse: bloom filter is stripped when it breaks reuse") {
    withSQLConf(
      // Force bloom filter injection (low threshold so our small test tables qualify)
      SparkSQLConf.RUNTIME_BLOOM_FILTER_ENABLED.key -> "true",
      SparkSQLConf.RUNTIME_BLOOM_FILTER_CREATION_SIDE_THRESHOLD.key -> "10000000",
      SparkSQLConf.RUNTIME_BLOOM_FILTER_APPLICATION_SIDE_SCAN_SIZE_THRESHOLD.key -> "1",
      SparkSQLConf.AUTO_BROADCASTJOIN_THRESHOLD.key -> "-1",
      SparkSQLConf.DYNAMIC_PARTITION_PRUNING_ENABLED.key -> "false"
    ) {
      withTable("bf_fact", "bf_dim") {
        setupTables()

        // Q24-style: CTE used in both main query and HAVING subquery
        val q =
          """
            |WITH ssales AS (
            |  SELECT f.fact_id, f.val, d.dim_val
            |  FROM bf_fact f JOIN bf_dim d ON f.dim_id = d.dim_id
            |)
            |SELECT dim_val, sum(val) AS paid
            |FROM ssales
            |GROUP BY dim_val
            |HAVING sum(val) > (SELECT 0.05 * avg(val) FROM ssales)
            |ORDER BY dim_val
          """.stripMargin

        // 1. Correctness: result must match non-bloom-filtered execution
        val expected = withBloomFilterDisabled(spark) {
          sql(q).collect()
        }
        checkAnswer(sql(q), expected)

        // 2. The BF that was breaking reuse should have been removed from the main query side.
        //    We verify this indirectly: the physical plan must contain a ReusedExchangeExec,
        //    which is only possible when the two ssales sub-plans are structurally identical.
        val physicalPlan = sql(q).queryExecution.executedPlan
        val hasReusedExchange = collect(physicalPlan) {
          case r: ReusedExchangeExec => r
        }.nonEmpty
        assert(hasReusedExchange, "ReusedExchangeExec should be present after bloom filter removal")
      }
    }
  }

  testGluten(
    "RemoveBloomFilterToRecoverExchangeReuse: keep bloom filter when there is no reuse candidate") {
    withSQLConf(
      SparkSQLConf.RUNTIME_BLOOM_FILTER_ENABLED.key -> "true",
      SparkSQLConf.RUNTIME_BLOOM_FILTER_CREATION_SIDE_THRESHOLD.key -> "10000000",
      SparkSQLConf.RUNTIME_BLOOM_FILTER_APPLICATION_SIDE_SCAN_SIZE_THRESHOLD.key -> "1",
      SparkSQLConf.AUTO_BROADCASTJOIN_THRESHOLD.key -> "-1",
      SparkSQLConf.DYNAMIC_PARTITION_PRUNING_ENABLED.key -> "false"
    ) {
      withTable("bf_fact", "bf_dim") {
        setupTables()

        val q =
          """
            |SELECT sum(f.val)
            |FROM bf_fact f JOIN bf_dim d
            |  ON f.dim_id = d.dim_id
          """.stripMargin

        val optimizedPlan = sql(q).queryExecution.optimizedPlan
        val hasBloomFilter = optimizedPlan.exists {
          case Filter(condition, _) =>
            condition.exists(_.isInstanceOf[BloomFilterMightContain])
          case _ => false
        }

        assert(hasBloomFilter, "BloomFilterMightContain should be preserved without reuse target")
      }
    }
  }
}
