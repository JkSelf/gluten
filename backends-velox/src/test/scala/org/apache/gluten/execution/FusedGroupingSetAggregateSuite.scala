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
package org.apache.gluten.execution

import org.apache.gluten.config.VeloxConfig

import org.apache.spark.SparkConf
import org.apache.spark.sql.Row
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, QueryStageExec}

/** Planning and Linux execution coverage for direct raw-input grouping-set fusion. */
class FusedGroupingSetAggregateSuite extends VeloxWholeStageTransformerSuite {
  private val isMacOS = System.getProperty("os.name").startsWith("Mac OS X") ||
    System.getProperty("os.name").startsWith("macOS")

  override protected val resourcePath: String = "/tpch-data-parquet"
  override protected val fileFormat: String = "parquet"

  override def beforeAll(): Unit = {
    super.beforeAll()
    createTPCHNotNullTables()
  }

  override protected def sparkConf: SparkConf = {
    super.sparkConf
      .set("spark.shuffle.manager", "org.apache.spark.shuffle.sort.ColumnarShuffleManager")
      .set("spark.sql.shuffle.partitions", "2")
      .set("spark.memory.offHeap.size", "2g")
      // Spark 4 enables ANSI mode by default, which the Velox backend rejects.
      .set("spark.sql.ansi.enabled", "false")
      // Most planner tests inspect the plan without executing it. AQE has a dedicated test below.
      .set("spark.sql.adaptive.enabled", "false")
  }

  private val q67ShapeQuery =
    """
      |with input as (
      |  select
      |    cast(case when id < 3 then 1 else 2 end as int) as k1,
      |    cast(case when id < 3 then 1 else 2 end as int) as k2,
      |    cast(case when id < 2 then 1 else 2 end as int) as k3,
      |    cast(case when id < 2 then 1 else 2 end as int) as k4,
      |    case when id < 2 then 'R' when id = 2 then 'A' else null end as s1,
      |    case when id < 2 then 'O' else 'F' end as s2,
      |    case when id < 2 then 'AIR' else 'SHIP' end as s3,
      |    case
      |      when id < 2 then 'DELIVER IN PERSON'
      |      when id = 2 then null
      |      else 'TAKE BACK RETURN'
      |    end as s4,
      |    cast(
      |      case id when 0 then 10.25 when 1 then 20.50 when 2 then 30.75 else null end
      |      as decimal(12, 2)) as price,
      |    cast(
      |      case id when 0 then 0.90 when 2 then 0.80 when 3 then 0.75 else null end
      |      as decimal(5, 2)) as discount
      |  from range(0, 5)
      |)
      |select k1, k2, k3, k4, s1, s2, s3, s4,
      |  sum(price * coalesce(discount, cast(1.00 as decimal(5, 2)))) as revenue
      |from input
      |group by rollup(k1, k2, k3, k4, s1, s2, s3, s4)
      |""".stripMargin

  private val groupingNullQuery =
    """
      |with input as (
      |  select
      |    cast(
      |      case when id = 0 then null when id < 3 then 1 else 2 end
      |      as int) as k1,
      |    cast(
      |      case when id < 2 then null else 2 end
      |      as int) as k2,
      |    cast(id + 1 as bigint) as v
      |  from range(0, 5)
      |)
      |select
      |  k1,
      |  k2,
      |  cast(grouping(k1) as int) as g1,
      |  cast(grouping(k2) as int) as g2,
      |  cast(grouping_id(k1, k2) as bigint) as gid,
      |  count(v) as n,
      |  sum(v) as total
      |from input
      |group by rollup(k1, k2)
      |order by gid, k1 nulls first, k2 nulls first
      |""".stripMargin

  private def plannedNodes(query: String): Seq[SparkPlan] = {
    val plan = spark.sql(query).queryExecution.executedPlan
    plan.collect { case node => node }
  }

  // Spark 3.5's QueryTest.withSQLConf returns Unit, while Spark 4 preserves the block's result.
  private def withSQLConfResult[T](pairs: (String, String)*)(block: => T): T = {
    var result = Option.empty[T]
    withSQLConf(pairs: _*) {
      result = Some(block)
    }
    result.getOrElse(fail("withSQLConf did not evaluate its block"))
  }

  private def rawFusionConfs(
      adaptiveEnabled: Boolean = false,
      maxGroupingSets: Int = 16,
      flushableEnabled: Boolean = false): Seq[(String, String)] = {
    Seq(
      "spark.sql.adaptive.enabled" -> adaptiveEnabled.toString,
      VeloxConfig.VELOX_FLUSHABLE_PARTIAL_AGGREGATION_ENABLED.key ->
        flushableEnabled.toString,
      VeloxConfig.VELOX_FUSED_GROUPING_SET_AGGREGATE_ENABLED.key -> "true",
      VeloxConfig.VELOX_FUSED_GROUPING_SET_AGGREGATE_MAX_GROUPING_SETS.key ->
        maxGroupingSets.toString
    )
  }

  private def rawFusionExpands(nodes: Seq[SparkPlan]): Seq[ExpandExecTransformer] = {
    nodes.collect {
      case expand: ExpandExecTransformer if RawGroupingSetFusion.isMarked(expand) => expand
    }
  }

  private def nativePlans(nodes: Seq[SparkPlan]): Seq[String] = {
    nodes.collect {
      case stage: WholeStageTransformer => stage.nativePlanString()
    }
  }

  private def adaptivePlanNodes(plan: SparkPlan): Seq[SparkPlan] = {
    val nested = plan match {
      case adaptive: AdaptiveSparkPlanExec => adaptivePlanNodes(adaptive.executedPlan)
      case queryStage: QueryStageExec => adaptivePlanNodes(queryStage.plan)
      case other => other.children.flatMap(adaptivePlanNodes)
    }
    plan +: nested
  }

  private val ordinaryAggregation =
    "(?m)^\\s*(?:--\\s*)?Aggregation\\[\\d+\\]".r

  private def assertRawFusedNativePlan(
      nodes: Seq[SparkPlan],
      expectedGroupingSets: Int): Unit = {
    val rawExpands = rawFusionExpands(nodes)
    assert(rawExpands.length == 1, s"expected one raw-fusion Expand:\n${nodes.head}")

    val plans = nativePlans(nodes)
    val fusedPlans =
      plans.filter(plan => plan.contains("GroupingSetAggregation") && plan.contains("input: raw"))
    assert(
      fusedPlans.nonEmpty,
      s"no raw native GroupingSetAggregation found:\n${plans.mkString("\n")}")
    assert(
      fusedPlans.forall(!_.contains("Expand")),
      s"raw fused native stage still contains Expand:\n${fusedPlans.mkString("\n")}")
    assert(
      fusedPlans.forall(plan => ordinaryAggregation.findFirstIn(plan).isEmpty),
      s"raw fused native stage still contains an ordinary aggregate:\n" +
        fusedPlans.mkString("\n")
    )

    val nativeGroupingSetCount =
      "/gid=".r.findAllIn(fusedPlans.mkString("\n")).length
    assert(
      nativeGroupingSetCount == expectedGroupingSets,
      s"expected $expectedGroupingSets raw native grouping sets, found " +
        s"$nativeGroupingSetCount:\n${fusedPlans.mkString("\n")}"
    )
  }

  private def assertRawFusionMetrics(
      nodes: Seq[SparkPlan],
      expectedGroupingSets: Int): Unit = {
    val rawExpands = rawFusionExpands(nodes)
    assert(rawExpands.length == 1, s"expected one executed raw-fusion Expand:\n${nodes.head}")
    val metrics = rawExpands.head.metrics
    val fusedInstances = metrics("fusedGroupingSetOperatorInstances").value
    val groupingSets = metrics("fusedGroupingSetsAcrossInstances").value
    assert(fusedInstances > 0, s"raw fused operator metric was not reported: $metrics")
    assert(
      groupingSets == fusedInstances * expectedGroupingSets,
      s"expected $expectedGroupingSets sets per raw operator instance, got " +
        s"sets=$groupingSets instances=$fusedInstances"
    )
  }

  test("direct raw grouping-set fusion is disabled by default") {
    assert(
      VeloxConfig.VELOX_FUSED_GROUPING_SET_AGGREGATE_ENABLED.defaultValue.contains(false))
    withSQLConf(
      VeloxConfig.VELOX_FLUSHABLE_PARTIAL_AGGREGATION_ENABLED.key -> "false",
      VeloxConfig.VELOX_FUSED_GROUPING_SET_AGGREGATE_ENABLED.key -> "false") {
      val nodes = plannedNodes(q67ShapeQuery)
      assert(
        rawFusionExpands(nodes).isEmpty,
        s"raw grouping-set fusion must require its explicit opt-in:\n${nodes.head}")
    }
  }

  test("direct raw fusion re-grounds the q67 pre-project") {
    withSQLConf(rawFusionConfs(): _*) {
      val nodes = plannedNodes(q67ShapeQuery)
      val rawExpands = rawFusionExpands(nodes)
      assert(rawExpands.length == 1, s"expected one raw-fusion Expand:\n${nodes.head}")
      val rawExpand = rawExpands.head
      assert(
        rawExpand.child.isInstanceOf[ProjectExecTransformer],
        s"q67 measure projection was not re-grounded below Expand:\n${nodes.head}")
      assert(
        nodes.exists {
          case aggregate: RegularHashAggregateExecTransformer => aggregate.child eq rawExpand
          case _ => false
        },
        s"the original raw partial aggregate is not adjacent to Expand:\n${nodes.head}"
      )
      assert(
        !nodes.exists {
          case project: ProjectExecTransformer => project.child eq rawExpand
          case _ => false
        },
        s"the old q67 pre-project still sits between aggregate and Expand:\n${nodes.head}"
      )
    }
  }

  test("direct raw marker survives flushable aggregate conversion") {
    withSQLConf(rawFusionConfs(flushableEnabled = true): _*) {
      val nodes = plannedNodes(q67ShapeQuery)
      val rawExpands = rawFusionExpands(nodes)
      assert(rawExpands.length == 1, s"expected one raw-fusion Expand:\n${nodes.head}")
      assert(
        nodes.exists {
          case aggregate: FlushableHashAggregateExecTransformer =>
            aggregate.child eq rawExpands.head
          case _ => false
        },
        s"raw marker did not survive flushable aggregate conversion:\n${nodes.head}"
      )
    }
  }

  test("direct raw fusion rejects duplicate grouping sets") {
    val query =
      "select l_orderkey, l_partkey, sum(l_suppkey) from lineitem " +
        "group by grouping sets ((l_orderkey, l_partkey), (l_orderkey, l_partkey))"
    withSQLConf(rawFusionConfs(): _*) {
      val nodes = plannedNodes(query)
      assert(
        rawFusionExpands(nodes).isEmpty,
        s"duplicate grouping sets must not carry the raw-fusion marker:\n${nodes.head}")
    }
  }

  test("direct raw fusion rejects distinct aggregation look-alikes") {
    val query =
      "select l_orderkey, l_partkey, count(distinct l_suppkey) from lineitem " +
        "group by rollup(l_orderkey, l_partkey)"
    withSQLConf(rawFusionConfs(): _*) {
      val nodes = plannedNodes(query)
      assert(
        rawFusionExpands(nodes).isEmpty,
        s"distinct aggregation must not carry the raw-fusion marker:\n${nodes.head}")
    }
  }

  test("direct raw fusion rejects grouping sets with multiple lattice roots") {
    val query =
      "select l_orderkey, l_partkey, sum(l_suppkey) from lineitem " +
        "group by grouping sets ((l_orderkey), (l_partkey))"
    withSQLConf(rawFusionConfs(): _*) {
      val nodes = plannedNodes(query)
      assert(
        rawFusionExpands(nodes).isEmpty,
        s"multi-root grouping sets must not carry the raw-fusion marker:\n${nodes.head}")
    }
  }

  test("configured grouping-set limit rejects an oversized raw fusion") {
    withSQLConf(rawFusionConfs(maxGroupingSets = 8): _*) {
      val nodes = plannedNodes(q67ShapeQuery)
      assert(
        rawFusionExpands(nodes).isEmpty,
        s"nine grouping sets must exceed the configured maximum of eight:\n${nodes.head}")
    }
  }

  test("strict floating-point mode rejects an integral average") {
    val query =
      "select l_orderkey, l_partkey, avg(l_suppkey) from lineitem " +
        "group by rollup(l_orderkey, l_partkey)"
    withSQLConf(
      (rawFusionConfs() :+
        (VeloxConfig.FLOATING_POINT_MODE.key -> "strict")): _*) {
      val nodes = plannedNodes(query)
      assert(
        rawFusionExpands(nodes).isEmpty,
        s"strict mode must not reassociate an average's DOUBLE accumulator:\n${nodes.head}")
    }
  }

  test("q67-shaped rollup executes through direct raw fusion and reports metrics") {
    if (isMacOS) {
      cancel("raw grouping-set native conversion is covered by Linux CI")
    }

    withSQLConf(rawFusionConfs(): _*) {
      assertRawFusedNativePlan(plannedNodes(q67ShapeQuery), expectedGroupingSets = 9)
      runQueryAndCompare(q67ShapeQuery) {
        df =>
          val nodes = df.queryExecution.executedPlan.collect { case node => node }
          assertRawFusedNativePlan(nodes, expectedGroupingSets = 9)
          assertRawFusionMetrics(nodes, expectedGroupingSets = 9)
      }
    }
  }

  test("raw fusion preserves genuine null keys separately from grouping nulls") {
    if (isMacOS) {
      cancel("raw grouping-set native conversion is covered by Linux CI")
    }

    val expected = Seq(
      Row(null, null, 0, 0, 0L, 1L, 1L),
      Row(1, null, 0, 0, 0L, 1L, 2L),
      Row(1, 2, 0, 0, 0L, 1L, 3L),
      Row(2, 2, 0, 0, 0L, 2L, 9L),
      Row(null, null, 0, 1, 1L, 1L, 1L),
      Row(1, null, 0, 1, 1L, 2L, 5L),
      Row(2, null, 0, 1, 1L, 2L, 9L),
      Row(null, null, 1, 1, 3L, 5L, 15L)
    )

    withSQLConf(rawFusionConfs(): _*) {
      assertRawFusedNativePlan(plannedNodes(groupingNullQuery), expectedGroupingSets = 3)
      runQueryAndCompare(groupingNullQuery) {
        df =>
          assert(df.collect().toSeq == expected)
          val nodes = df.queryExecution.executedPlan.collect { case node => node }
          assertRawFusedNativePlan(nodes, expectedGroupingSets = 3)
          assertRawFusionMetrics(nodes, expectedGroupingSets = 3)
      }
    }
  }

  test("AQE preserves and executes direct raw grouping-set fusion") {
    if (isMacOS) {
      cancel("raw grouping-set native conversion is covered by Linux CI")
    }

    withSQLConf(rawFusionConfs(adaptiveEnabled = true): _*) {
      val initialPlan = spark.sql(q67ShapeQuery).queryExecution.executedPlan
      assert(
        initialPlan.isInstanceOf[AdaptiveSparkPlanExec],
        s"expected an adaptive plan:\n$initialPlan")
      assertRawFusedNativePlan(
        adaptivePlanNodes(initialPlan),
        expectedGroupingSets = 9)

      runQueryAndCompare(q67ShapeQuery) {
        df =>
          val adaptivePlan = df.queryExecution.executedPlan
          assert(
            adaptivePlan.isInstanceOf[AdaptiveSparkPlanExec],
            s"expected an executed adaptive plan:\n$adaptivePlan")
          val nodes = adaptivePlanNodes(adaptivePlan)
          assertRawFusedNativePlan(nodes, expectedGroupingSets = 9)
          assertRawFusionMetrics(nodes, expectedGroupingSets = 9)
      }
    }
  }

  test("native rejection executes both original operators and reports no fused metrics") {
    if (isMacOS) {
      cancel("raw grouping-set native conversion is covered by Linux CI")
    }

    val taggedDfAndNodes = withSQLConfResult(rawFusionConfs(): _*) {
      val df = spark.sql(q67ShapeQuery)
      val nodes = df.queryExecution.executedPlan.collect { case node => node }
      assertRawFusedNativePlan(nodes, expectedGroupingSets = 9)
      (df, nodes)
    }
    val (taggedDf, taggedNodes) = taggedDfAndNodes

    // The Spark plan was tagged under the permissive limit. Lower only the native conversion
    // limit so conversion must restore Aggregate over Expand without replanning the JVM tree.
    withSQLConf(rawFusionConfs(maxGroupingSets = 8): _*) {
      val fallbackPlans = nativePlans(taggedNodes)
      assert(
        fallbackPlans.exists(_.contains("Expand")),
        s"raw rejection did not restore native Expand:\n${fallbackPlans.mkString("\n")}")
      assert(
        fallbackPlans.exists(plan => ordinaryAggregation.findFirstIn(plan).nonEmpty),
        s"raw rejection did not restore Aggregate over Expand:\n" +
          fallbackPlans.mkString("\n")
      )
      assert(
        fallbackPlans.forall(!_.contains("GroupingSetAggregation")),
        s"raw rejection retained a fused native node:\n${fallbackPlans.mkString("\n")}")

      val expected = withSQLConfResult(vanillaSparkConfs(): _*) {
        spark.sql(q67ShapeQuery).collect().toSeq
      }
      checkAnswer(taggedDf, expected)

      val rawExpand = rawFusionExpands(taggedNodes).head
      val fusionMetrics = rawExpand.metrics
      assert(fusionMetrics("fusedGroupingSetOperatorInstances").value == 0)
      assert(fusionMetrics("fusedGroupingSetsAcrossInstances").value == 0)
      assert(
        fusionMetrics("numOutputRows").value > 0,
        s"restored native Expand did not report output rows: $fusionMetrics")

      val aggregate = taggedNodes.collectFirst {
        case node: HashAggregateExecBaseTransformer if node.child eq rawExpand => node
      }.getOrElse(fail(s"no restored aggregate owns the marked Expand:\n${taggedNodes.head}"))
      assert(
        aggregate.metrics("aggOutputRows").value > 0,
        s"restored native aggregate did not report output rows: ${aggregate.metrics}")
    }
  }
}
