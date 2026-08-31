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
package org.apache.gluten.extension

import org.apache.gluten.config.VeloxConfig
import org.apache.gluten.execution.VeloxWholeStageTransformerSuite

import org.apache.spark.SparkConf
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.plans.logical.{HintInfo, Join, JoinHint}

class PinPartitionKeyJoinsSuite extends VeloxWholeStageTransformerSuite {

  override protected val resourcePath: String = "N/A"
  override protected val fileFormat: String = "N/A"

  private val ruleKey = VeloxConfig.PIN_PARTITION_KEY_JOIN_ENABLED.key

  /** The strategy-free hint the rule marks a pinned join with. */
  private val pinHint = JoinHint(Some(HintInfo()), None)

  override protected def sparkConf: SparkConf = {
    super.sparkConf
      .set("spark.sql.cbo.enabled", "true")
      .set("spark.sql.cbo.joinReorder.enabled", "true")
      // Between the size of a dimension below and the size of a fact table below, so that the
      // dimensions are broadcastable and the fact tables are not.
      .set("spark.sql.autoBroadcastJoinThreshold", "100000")
      .set("spark.sql.shuffle.partitions", "4")
  }

  test("pins the join between a partition column and a broadcastable dimension") {
    withFactAndDims {
      val query =
        """SELECT count(*) FROM fact f, dim_ds d, dim_id i
          |WHERE f.f_ds = d.d_ds AND f.f_id = i.i_id AND d.d_flag > 0
          |""".stripMargin

      // Only the edge on the partition column is pinned, the one on f_id is left to the reorder.
      assert(pinnedConditions(query) == Seq(Set("f_ds", "d_ds")))
      checkSameAnswerWithoutTheRule(query)
    }
  }

  test("pins one dimension per fact table") {
    withFactAndDims {
      // The shape of TPC-DS q72: two partitioned tables, each with the dimension that prunes it,
      // joined to each other. Both pins have to survive, and the edge between the two pinned items
      // has to stay a join condition.
      val query =
        """SELECT count(*) FROM fact f, dim_ds d, fact2 g, dim_ds2 e
          |WHERE f.f_ds = d.d_ds AND g.g_ds = e.e_ds AND f.f_id = g.g_id AND d.d_flag > 0
          |""".stripMargin

      assert(
        pinnedConditions(query).toSet ==
          Set(Set("f_ds", "d_ds"), Set("g_ds", "e_ds")))
      checkSameAnswerWithoutTheRule(query)
    }
  }

  test("does not pin a join key that is not a partition column") {
    withFactAndDims {
      // fact_flat holds the same data as fact, it is just not partitioned, so pruning its
      // partitions is not a thing and there is nothing to protect.
      val query =
        """SELECT count(*) FROM fact_flat f, dim_ds d, dim_id i
          |WHERE f.f_ds = d.d_ds AND f.f_id = i.i_id AND d.d_flag > 0
          |""".stripMargin

      assert(pinnedConditions(query).isEmpty)
    }
  }

  test("does not pin a dimension that is too big to be broadcast") {
    withFactAndDims {
      // Both sides are fact tables here: joining them early would not turn into the broadcast a
      // pruning subquery could reuse.
      val query =
        """SELECT count(*) FROM fact f, fact2 g, dim_id i
          |WHERE f.f_ds = g.g_ds AND f.f_id = i.i_id
          |""".stripMargin

      assert(pinnedConditions(query).isEmpty)
    }
  }

  test("does nothing when disabled") {
    withFactAndDims {
      val query =
        """SELECT count(*) FROM fact f, dim_ds d, dim_id i
          |WHERE f.f_ds = d.d_ds AND f.f_id = i.i_id AND d.d_flag > 0
          |""".stripMargin

      withSQLConf(ruleKey -> "false") {
        assert(pinnedConditions(query).isEmpty)
      }
      withSQLConf("spark.sql.cbo.joinReorder.enabled" -> "false") {
        // Nothing can take the two sides apart, so there is nothing to pin.
        assert(pinnedConditions(query).isEmpty)
      }
    }
  }

  /** The join keys of every pinned join of the optimized plan of `query`. */
  private def pinnedConditions(query: String): Seq[Set[String]] = {
    spark.sql(query).queryExecution.optimizedPlan.collect {
      case j: Join if j.hint == pinHint =>
        j.condition.get.references.map(_.name).toSet
    }
  }

  private def checkSameAnswerWithoutTheRule(query: String): Unit = {
    var expected: Seq[Row] = Nil
    withSQLConf(ruleKey -> "false") {
      expected = spark.sql(query).collect().toSeq
    }
    checkAnswer(spark.sql(query), expected)
  }

  private def withFactAndDims(f: => Unit): Unit = {
    withTable("fact", "fact2", "fact_flat", "dim_ds", "dim_ds2", "dim_id") {
      createFact("fact", "f", partitioned = true)
      createFact("fact2", "g", partitioned = true)
      createFact("fact_flat", "f", partitioned = false)
      createDim("dim_ds", "d")
      createDim("dim_ds2", "e")
      createIdDim("dim_id", "i")
      f
    }
  }

  /**
   * A table of a few megabytes - the random column does not compress, so it stays well above the
   * broadcast threshold of this suite - partitioned by `<prefix>_ds` into eight partitions.
   */
  private def createFact(name: String, prefix: String, partitioned: Boolean): Unit = {
    val df = spark
      .range(0, 500000, 1, 4)
      .selectExpr(
        s"id AS ${prefix}_id",
        s"CAST(id % 8 AS INT) AS ${prefix}_ds",
        s"rand(1) AS ${prefix}_v")
    val writer = df.write.format("parquet").mode("overwrite")
    if (partitioned) {
      writer.partitionBy(s"${prefix}_ds").saveAsTable(name)
    } else {
      writer.saveAsTable(name)
    }
  }

  private def createDim(name: String, prefix: String): Unit = {
    spark
      .range(0, 5)
      .selectExpr(
        s"CAST(id AS INT) AS ${prefix}_ds",
        s"CAST(id + 1 AS INT) AS ${prefix}_flag")
      .write
      .format("parquet")
      .mode("overwrite")
      .saveAsTable(name)
  }

  private def createIdDim(name: String, prefix: String): Unit = {
    spark
      .range(0, 5)
      .selectExpr(s"id AS ${prefix}_id", s"CAST(id AS STRING) AS ${prefix}_name")
      .write
      .format("parquet")
      .mode("overwrite")
      .saveAsTable(name)
  }
}
