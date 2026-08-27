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
import org.apache.gluten.execution.{BatchScanExecTransformerBase, FilterExecTransformer}
import org.apache.gluten.expression.VeloxBloomFilterMightContain
import org.apache.gluten.utils.FileIndexUtil

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.{Alias, And, Attribute, AttributeReference, BloomFilterMightContain, DynamicPruningExpression, Expression, ExprId, NamedExpression, PlanExpression, PredicateHelper}
import org.apache.spark.sql.catalyst.plans.{Cross, Inner, LeftSemi}
import org.apache.spark.sql.catalyst.plans.physical.{HashPartitioning, Partitioning, RangePartitioning, SinglePartition}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{DataSourceScanExec, ExpandExec, FilterExec, GlobalLimitExec, LocalLimitExec, ProjectExec, SortExec, SparkPlan}
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, QueryStageExec}
import org.apache.spark.sql.execution.aggregate.BaseAggregateExec
import org.apache.spark.sql.execution.datasources.HadoopFsRelation
import org.apache.spark.sql.execution.datasources.v2.BatchScanExec
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeLike, ReusedExchangeExec, ShuffleExchangeLike}
import org.apache.spark.sql.execution.joins.BaseJoinExec
import org.apache.spark.sql.execution.window.WindowExec
import org.apache.spark.sql.types.DataType

import java.util.IdentityHashMap
import java.util.concurrent.ConcurrentHashMap

import scala.collection.mutable

/**
 * Recovers exchange reuse between two shuffles that read the same tables and produce the same
 * output, but where one of them carries an EXTRA scan-level predicate the other does not have.
 *
 * The motivating shape is TPC-DS Q24a/Q24b once `InlineCTE` has inlined the `ssales` CTE. The CTE
 * body is materialized twice:
 *
 *   - the main branch adds `i_color = '<color>'`, which gets pushed all the way down to the `item`
 *     scan, and which additionally makes the CBO pick a DIFFERENT join order;
 *   - the HAVING correlated scalar subquery has no such predicate.
 *
 * The two shuffles therefore do not canonicalize equal and `store_sales` / `store_returns` are each
 * scanned twice -- roughly 54% of the wall clock of Q24a at sf10000.
 *
 * This rule does NOT try to make the two subtrees grow into the same shape: with the join order
 * differing as well that is not reachable from a physical rule. Instead it exploits that one output
 * is a row-wise SUBSET of the other:
 *
 * {{{
 *   mainShuffle.output == filter(i_color = 'rose', subqueryShuffle.output)
 * }}}
 *
 * so the subset side can be thrown away entirely and rebuilt as the superset side plus the extra
 * predicate PULLED UP above the shuffle:
 *
 * {{{
 *   Project(rename to the subset side's exprIds)
 *     Filter(i_color = 'rose')
 *       <the superset side's shuffle, byte for byte unchanged>
 * }}}
 *
 * Both references then point at one and the same exchange, and Spark's own
 * `AdaptiveExecutionContext.stageCache` turns the second one into a `ReusedExchange`. The rule
 * never builds a `ReusedExchangeExec` by hand.
 *
 * The superset side is left byte for byte unchanged, but the rewrite is NOT free: the subset side's
 * consumers now read the superset side's row count instead of their own. On Q24a at sf10000 that
 * turned the main branch's build side from 5.8 MiB into 194.8 MiB, which pushed its join past
 * `autoBroadcastJoinThreshold` and demoted it from a broadcast to a shuffled hash join. That was
 * still a large net win there -- 84s to 49s, because a whole scan of store_sales and of
 * store_returns goes away -- but the trade can invert when the shared tables are small and the
 * predicate is very selective.
 *
 * The rule runs in 4 steps:
 *
 * STEP1 Collect: describe every shuffle in the main plan tree and in every nested plan (subquery
 * expressions and `AdaptiveSparkPlanExec.initialPlan`, see [[nestedPlans]]) by its leaf tables, its
 * positional output signature, its partitioning, its join-key set and its predicate set. Shuffles
 * containing an operator outside the whitelist -- anything that is not a vanilla join / project /
 * filter / aggregate / exchange / sort / recognised scan -- are dropped individually.
 *
 * STEP2 Group: cluster the descriptions by (leafTables, outputSignature, partitioning, joinKeys) --
 * the dimensions that have to be equal for the two sides to produce the same rows in the same
 * layout under the same partitioning.
 *
 * STEP3 Match: inside a group, look for a pair whose predicate sets are in a strict superset
 * relation and where the subset side lives in the main plan tree (only that side can be grafted
 * into). Everything the rewrite depends on is validated here; any failed check drops the pair.
 *
 * STEP4 Rewrite: replace the subset side's shuffle with Project(Filter(supersetShuffle)).
 *
 * IMPORTANT -- a mis-fire of this rule does not merely produce a slower plan, it produces WRONG
 * RESULTS. It is therefore off by default and the checks in STEP3 are deliberately biased towards
 * giving up. See
 * `spark.gluten.sql.columnar.backend.velox.pullUpExtraPredicateToRecoverExchangeReuse`.
 */
case class PullUpExtraPredicateToRecoverExchangeReuse(spark: SparkSession)
  extends Rule[SparkPlan]
  with PredicateHelper
  with Logging {

  import PullUpExtraPredicateToRecoverExchangeReuse._

  // ==========================================================================
  // Expression / predicate identity
  // ==========================================================================

  /**
   * Identity key of an expression that is stable across the two sides being compared. Attribute
   * exprIds differ between the two copies of an inlined CTE body -- and so do alias-generated names
   * such as `_joinagg_buf_23971_0` -- so exprIds are flattened to a constant and only (name, type)
   * survives. Both sides read the same base tables, so base-column names do match.
   *
   * Spark's own `Expression.canonicalized` is deliberately NOT used: `AttributeReference` renders
   * as `none#<exprId>` there, dropping the name and KEEPING the exprId. That is the right
   * normalization inside one plan, where `QueryPlan.canonicalized` has already rewritten every
   * exprId to an ordinal, and exactly the wrong one for comparing two separate plan trees -- every
   * pair of attributes would then look different and no two shuffles would ever match.
   */
  private def exprKey(e: Expression): String =
    e.transformUp {
      case a: AttributeReference => a.withExprId(FlatExprId).withQualifier(Nil)
    }.toString

  /**
   * Predicates that are implied by a join this subtree performs anyway, rather than restricting the
   * result on their own: runtime BloomFilters and dynamic partition pruning. They are approximate
   * pre-filters -- the join downstream removes whatever they let through -- so they cannot change
   * the result and are excluded from the superset comparison. That matters in practice, because a
   * BloomFilter built from `item where i_color = 'rose'` and one built from an unfiltered `item`
   * would otherwise look like two unrelated predicates and make every pair incomparable.
   */
  private def isJoinImplied(e: Expression): Boolean = e match {
    case _: BloomFilterMightContain => true
    case _: VeloxBloomFilterMightContain => true
    case _: DynamicPruningExpression => true
    case _ => false
  }

  /** A physical filter, vanilla or columnar, reduced to its predicate. */
  private object PhysicalFilter {
    def unapply(p: SparkPlan): Option[Expression] = p match {
      case f: FilterExec => Some(f.condition)
      case f: FilterExecTransformer => Some(f.condition)
      case _ => None
    }
  }

  // ==========================================================================
  // Partitioning
  // ==========================================================================

  /**
   * Name-based signature of a shuffle's partitioning. Returns None for anything not listed here,
   * which covers both partitionings that cannot be shared even between two identical subtrees --
   * notably `RoundRobinPartitioning`, whose row-to-partition assignment is not a function of the
   * data -- and partitionings this rule simply has not reasoned about.
   */
  private def partitioningKey(p: Partitioning): Option[String] = p match {
    case h: HashPartitioning =>
      Some(s"hash(${h.expressions.map(exprKey).mkString(",")})/${h.numPartitions}")
    case r: RangePartitioning =>
      Some(s"range(${r.ordering.map(exprKey).mkString(",")})/${r.numPartitions}")
    case SinglePartition => Some("single")
    case _ => None
  }

  // ==========================================================================
  // Leaf scans
  // ==========================================================================
  private def tableNameOf(leaf: SparkPlan): Option[String] = leaf match {
    case scan: DataSourceScanExec =>
      scan.tableIdentifier.map(_.table).orElse(fileScanRootPaths(scan))
    case scan: BatchScanExecTransformerBase => Option(scan.table).map(t => stripCatalog(t.name()))
    case scan: BatchScanExec =>
      reflectField[AnyRef](scan, "table").map(t => stripCatalog(t.toString))
    case _ => None
  }

  /**
   * Identity of a file scan that has no catalog table behind it. `tableIdentifier` is set only for
   * a scan of a catalog table; reading a path with `spark.read.parquet(...)` and registering it as
   * a temp view -- how the TPC-DS harness sets its tables up -- leaves it empty, which used to drop
   * every single shuffle of the query. The root paths identify what is being read at least as
   * precisely as a table name does, and are compared between the two sides only for equality.
   *
   * `relation` is read reflectively because vanilla `FileSourceScanExec` and Gluten's
   * `FileSourceScanExecTransformer` declare it on unrelated class hierarchies.
   */
  private def fileScanRootPaths(leaf: SparkPlan): Option[String] =
    reflectField[AnyRef](leaf, "relation")
      .collect { case r: HadoopFsRelation => FileIndexUtil.getRootPath(r.location).sorted }
      .collect { case paths if paths.nonEmpty => paths.mkString(",") }

  /**
   * Everything about a leaf scan that could make it return a different set of rows than another
   * scan of the same table: its output columns and its static predicates. Dynamic filters
   * (`runtimeFilters`, DPP) are excluded on purpose, see [[isJoinImplied]].
   *
   * For V2 scans the pushed-down predicates are not enumerable in general -- an Iceberg
   * `SparkBatchQueryScan` keeps them inside the `Scan` object -- so `Scan.description()`, which
   * does render them, is folded into the signature as an opaque string. Two signatures being equal
   * is then still a sound "these two scans read the same rows" test; it is only less precise about
   * WHY two signatures differ, which is fine because a difference always means "give up".
   *
   * @return
   *   None when the leaf is of a type whose predicates cannot be accounted for at all, which drops
   *   the enclosing shuffle from consideration.
   */
  private def leafSignature(leaf: SparkPlan): Option[LeafSignature] = {
    val outputSig = leaf.output.map(a => (a.name, a.dataType, a.nullable))
    val filterSig: Option[Seq[String]] = leaf match {
      case _: DataSourceScanExec =>
        // partitionFilters / dataFilters moved from FileSourceScanExec up to FileSourceScanLike in
        // Spark 3.4, so they are read reflectively to keep compiling against 3.3.
        for {
          pf <- reflectField[Seq[Expression]](leaf, "partitionFilters")
          df <- reflectField[Seq[Expression]](leaf, "dataFilters")
        } yield (pf ++ df).filterNot(isJoinImplied).map(exprKey).sorted
      case scan: BatchScanExecTransformerBase =>
        val enumerable = (scan.scanFilters ++ scan.pushDownFilters.getOrElse(Nil))
          .filterNot(isJoinImplied)
          .map(exprKey)
          .sorted
        Some(enumerable :+ s"scanDesc=${scan.scan.description()}")
      case scan: BatchScanExec =>
        Some(Seq(s"scanDesc=${scan.scan.description()}"))
      case _ => None
    }
    for {
      table <- tableNameOf(leaf)
      filters <- filterSig
    } yield LeafSignature(table, outputSig, filters)
  }

  // ==========================================================================
  // STEP1 -- describe one shuffle
  // ==========================================================================

  /**
   * Operators allowed inside a shareable subtree. This is a whitelist rather than a blacklist on
   * purpose: an operator this rule has not reasoned about must make it give up, not silently pass.
   *
   * Aggregates ARE whitelisted, but only conditionally: pulling a predicate up above an aggregate
   * is valid just when the predicate's columns are all grouping keys of it, which is checked per
   * pair by [[aggregatesAllowPullUp]] rather than here. Q24a needs this, because Gluten's own
   * aggregate push-down (`PushAggregateThroughJoin`) puts a partial aggregate underneath the join,
   * so requiring an aggregate-free subtree would reject every shuffle of the query.
   */
  private def isWhitelisted(p: SparkPlan): Boolean = p match {
    case j: BaseJoinExec => j.joinType == Inner || j.joinType == LeftSemi || j.joinType == Cross
    case _: ProjectExec => true
    case _: FilterExec => true
    case _: SortExec => true
    case _: BaseAggregateExec => true
    case _: ShuffleExchangeLike => true
    case _: BroadcastExchangeLike => true
    case _: GlobalLimitExec | _: LocalLimitExec | _: WindowExec | _: ExpandExec => false
    case leaf if leaf.children.isEmpty => tableNameOf(leaf).isDefined
    // Gluten's own columnar nodes only appear inside stages that are already materialized, and
    // those subtrees are rejected by mkDescriptor before we get here.
    case _ => false
  }

  /**
   * Set of equi-join key pairs and residual conditions of every join in the subtree, as an
   * unordered set. Comparing this as a SET is what makes the differing join order between the two
   * Q24a subtrees a non-issue: inner joins commute and associate, so only the key set has to match.
   */
  private def joinKeys(root: SparkPlan): Set[String] = {
    val keys = Set.newBuilder[String]
    root.foreach {
      case j: BaseJoinExec =>
        // Sort each pair so that a join whose sides got swapped still yields the same key.
        j.leftKeys.zip(j.rightKeys).foreach {
          case (l, r) => keys += Seq(exprKey(l), exprKey(r)).sorted.mkString("=")
        }
        j.condition.foreach(c => splitConjunctivePredicates(c).foreach(p => keys += exprKey(p)))
      case _ =>
    }
    keys.result()
  }

  /**
   * Describes one shuffle for STEP2/STEP3, or None if it is not a candidate at all.
   *
   * @param inMainTree
   *   whether STEP4's traversal of the plan handed to `apply` can actually reach and replace this
   *   shuffle. Shuffles found in nested plans are collected as superset candidates only.
   */
  private def mkDescriptor(
      exchange: ShuffleExchangeLike,
      inMainTree: Boolean): Option[ShuffleDescriptor] = {
    // Anything already materialized is off limits: its rows exist, its inputs are frozen, and its
    // parent holds a QueryStageExec rather than the exchange itself.
    if (
      exchange.exists(p => p.isInstanceOf[QueryStageExec] || p.isInstanceOf[ReusedExchangeExec])
    ) {
      return None
    }
    // The root exchange is the sharing point, not part of what has to be whitelisted.
    val blockers = exchange.child.collect { case p if !isWhitelisted(p) => p.nodeName }.distinct
    if (blockers.nonEmpty) {
      logDebug(
        s"Skip a shuffle over" +
          s" ${exchange.child.collectLeaves().flatMap(tableNameOf).mkString(",")}" +
          s" because its subtree contains [${blockers.mkString(",")}]," +
          " which this rule does not reason about.")
      return None
    }

    val leafSigs = exchange.child.collectLeaves().map(leafSignature)
    if (leafSigs.isEmpty || leafSigs.contains(None)) {
      return None
    }
    val partKey = partitioningKey(exchange.outputPartitioning)
    if (partKey.isEmpty) {
      return None
    }

    val predicates = mutable.LinkedHashMap.empty[String, Expression]
    exchange.child.foreach {
      case PhysicalFilter(cond) =>
        splitConjunctivePredicates(cond)
          .filterNot(isJoinImplied)
          .foreach(p => predicates.getOrElseUpdate(exprKey(p), p))
      case _ =>
    }

    Some(
      ShuffleDescriptor(
        exchange = exchange,
        inMainTree = inMainTree,
        // Types only, deliberately NOT nullability: `FilterExec.output` marks an attribute
        // non-nullable once an `IsNotNull` conjunct constrains it, so the very predicate this rule
        // is looking for makes the two sides' nullability differ. `i_color` is non-nullable on
        // Q24a's main branch and nullable on its subquery branch for exactly that reason.
        // buildReplacement re-checks nullability on the plan it actually produces.
        outputSignature = exchange.output.map(_.dataType),
        partitioningKey = partKey.get,
        leafSignatures = leafSigs.flatten,
        predicates = predicates.toMap,
        joinKeys = joinKeys(exchange.child)
      ))
  }

  /**
   * Plans that plain traversal does not reach: subquery expression plans, and the plans behind an
   * `AdaptiveSparkPlanExec` -- an AQE node is a leaf and keeps its plan in a field.
   *
   * `initialPlan` is used rather than `inputPlan`, because the shuffle we want to share is inserted
   * by `EnsureRequirements`, which runs as part of the nested AQE's own query-stage preparation and
   * is therefore visible only in `initialPlan`. `InsertAdaptiveSparkPlan` plans subqueries through
   * `PlanAdaptiveSubqueries` BEFORE it builds the enclosing `AdaptiveSparkPlanExec`, so by the time
   * this rule runs on the main query, the subquery's `initialPlan` is fully formed.
   */
  private def nestedPlans(root: SparkPlan): Seq[SparkPlan] = {
    val visited = new IdentityHashMap[SparkPlan, java.lang.Boolean]()
    val collected = mutable.ArrayBuffer.empty[SparkPlan]

    def collectFrom(plan: SparkPlan): Unit = {
      plan.foreach {
        case aqe: AdaptiveSparkPlanExec =>
          if (visited.put(aqe.initialPlan, true) == null) {
            collected += aqe.initialPlan
            collectFrom(aqe.initialPlan)
          }
        case p =>
          p.subqueries.foreach {
            sub =>
              if (visited.put(sub, true) == null) {
                collected += sub
                collectFrom(sub)
              }
          }
      }
    }

    collectFrom(root)
    collected.toSeq
  }

  private def shufflesIn(plan: SparkPlan, inMainTree: Boolean): Seq[ShuffleDescriptor] =
    plan.collect { case s: ShuffleExchangeLike => s }.flatMap(mkDescriptor(_, inMainTree))

  // ==========================================================================
  // STEP3 -- validate one (subset, superset) pair
  // ==========================================================================

  /**
   * Whether moving a predicate over `extraNames` from below every aggregate of this subtree to
   * above the shuffle keeps the result unchanged.
   *
   * `filter(agg(x)) == agg(filter(x))` holds exactly when the filter's columns are grouping keys:
   * dropping WHOLE groups cannot change the buffer any surviving group accumulates, whereas
   * dropping rows WITHIN a group would. So an aggregate only has to be checked when it actually
   * carries one of those columns through -- and when it does, that column must be a grouping key of
   * it. An aggregate that does not output the column at all sits below the join that introduces it
   * (Q24a's pushed-down `partial_sum` over store_sales is exactly that) and is thus unaffected.
   */
  private def aggregatesAllowPullUp(d: ShuffleDescriptor, extraNames: Set[String]): Boolean =
    !d.exchange.child.exists {
      case agg: BaseAggregateExec =>
        val groupingNames = agg.groupingExpressions.map(_.name).toSet
        agg.output.exists(a => extraNames.contains(a.name) && !groupingNames.contains(a.name))
      case _ => false
    }

  /**
   * Builds the replacement for `subset` out of `superset`, or None if any of the 5 preconditions
   * (numbered in the body, in the order they are checked) fails.
   */
  private def buildReplacement(
      subset: ShuffleDescriptor,
      superset: ShuffleDescriptor): Option[SparkPlan] = {
    // (1) the predicate sets must be in a strict superset relation: everything the superset side
    // filters on the subset side filters too, plus at least one extra conjunct.
    val extraKeys = subset.predicates.keySet -- superset.predicates.keySet
    if (extraKeys.isEmpty || (superset.predicates.keySet -- subset.predicates.keySet).nonEmpty) {
      logDebug(
        "Give up recovering exchange reuse: predicate sets are not in strict superset relation.")
      return None
    }
    val extraPreds = extraKeys.toSeq.sorted.map(subset.predicates)

    // (2) a predicate that is non-deterministic, or that hides a subquery, is not the same filter
    // when it is evaluated once per shuffle output row instead of once per scan row.
    if (extraPreds.exists(p => !p.deterministic || p.exists(_.isInstanceOf[PlanExpression[_]]))) {
      logDebug("Give up recovering exchange reuse: extra predicate contains non-deterministic" +
        " or subquery expression.")
      return None
    }

    // (3) resolve each referenced column against the shared shuffle's output, by (name, type).
    val sharedByName = superset.exchange.output.groupBy(a => (a.name, a.dataType))
    val remapped = extraPreds.map {
      pred =>
        pred.transformUp {
          case a: Attribute =>
            sharedByName.get((a.name, a.dataType)) match {
              case Some(Seq(single)) => single
              // Ambiguous or absent: leave it alone, it is caught right below.
              case _ => a
            }
        }
    }
    if (remapped.exists(_.references.exists(a => !superset.exchange.outputSet.contains(a)))) {
      logDebug(
        "Give up recovering exchange reuse: the extra predicate references a column that is not" +
          " uniquely available in the shared shuffle's output.")
      return None
    }

    // (4) the extra predicates must all come from one table, and every other table must match.
    val extraTables = extraPreds
      .flatMap(_.references)
      .flatMap(a => subset.leafSignatures.find(_.outputSignature.exists(_._1 == a.name)))
      .map(_.table)
      .toSet
    if (extraTables.size != 1) {
      logDebug(
        s"Give up recovering exchange reuse: extra predicates reference" +
          s" ${extraTables.size} tables (expected exactly 1).")
      return None
    }
    val stable = (d: ShuffleDescriptor) =>
      d.leafSignatures.filterNot(l => extraTables.contains(l.table)).sortBy(_.toString)
    if (stable(subset) != stable(superset)) {
      logDebug(
        "Give up recovering exchange reuse: the two sides' scans differ on a table the extra" +
          " predicate does not reference, so one is not a subset of the other.")
      return None
    }

    // (5) every aggregate between the extra predicate and the shuffle must group BY the columns
    // that predicate reads, on BOTH sides -- see aggregatesAllowPullUp.
    val extraNames = extraPreds.flatMap(_.references).map(_.name).toSet
    if (!Seq(subset, superset).forall(aggregatesAllowPullUp(_, extraNames))) {
      logDebug(
        "Give up recovering exchange reuse: an aggregate below the shuffle does not group by the" +
          s" column(s) [${extraNames.mkString(",")}] the extra predicate reads, so filtering" +
          " above the shuffle is not the same as filtering below it.")
      return None
    }

    // Filter goes ABOVE the shared shuffle -- putting it below would rebuild the very subtree we
    // are trying to get rid of. The Project renames back to the subset side's exprIds so that no
    // ancestor of the replaced shuffle has to be touched, and it is what propagates the shared
    // shuffle's HashPartitioning back onto the subset side's attributes for EnsureRequirements
    // (ProjectExec is alias aware).
    val filtered = FilterExec(remapped.reduce(And), superset.exchange)
    val projectList: Seq[NamedExpression] =
      superset.exchange.output.zip(subset.exchange.output).map {
        case (from, to) =>
          // `from` is resolved against `filtered`, not against the exchange, so that the
          // non-nullability `FilterExec` derives from the pulled-up IsNotNull conjuncts carries
          // into the alias.
          val resolved = filtered.output.find(_.exprId == from.exprId).getOrElse(from)
          if (resolved.exprId == to.exprId && resolved.nullable == to.nullable) to
          else Alias(resolved, to.name)(exprId = to.exprId, qualifier = to.qualifier)
      }
    val replacement = ProjectExec(projectList, filtered)

    // The replacement stands where an attribute of the subset side's output used to be, so it may
    // not be MORE nullable than what the ancestors were promised. In the Q24a shape the pulled-up
    // `isnotnull(i_color)` restores it; if some other shape does not, give up rather than hand a
    // nullable column to a consumer that has been told it cannot be null.
    val nullabilityOk = replacement.output.zip(subset.exchange.output).forall {
      case (produced, expected) => expected.nullable || !produced.nullable
    }
    if (!nullabilityOk) {
      logDebug(
        "Give up recovering exchange reuse: pulling the extra predicate up would widen the" +
          " nullability of the shuffle's output.")
      return None
    }
    Some(replacement)
  }

  // ==========================================================================
  // apply
  // ==========================================================================

  override def apply(plan: SparkPlan): SparkPlan = {
    if (!VeloxConfig.get.pullUpExtraPredicateToRecoverExchangeReuse) {
      return plan
    }

    // STEP1
    val mainShuffles = shufflesIn(plan, inMainTree = true)
    if (mainShuffles.isEmpty) {
      return plan
    }
    val nestedShuffles = nestedPlans(plan).flatMap(shufflesIn(_, inMainTree = false))
    val all = mainShuffles ++ nestedShuffles

    // STEP2
    val groups = all.groupBy(d => (d.leafTables, d.outputSignature, d.partitioningKey, d.joinKeys))
    // Rendered lazily: on a plan the size of Q24a's the group dump is a few kB of string building.
    logDebug {
      val comparable = groups.values.filter(_.length > 1)
      s"${mainShuffles.size} candidate shuffle(s) in the main tree," +
        s" ${nestedShuffles.size} in nested plans, forming ${groups.size} group(s) of which" +
        s" ${comparable.size} have more than one member." +
        comparable
          .map(
            members =>
              s"\n  group over ${members.head.leafTables.mkString(",")}:" +
                members
                  .map(
                    m =>
                      s"\n    inMainTree=${m.inMainTree}" +
                        s" predicates=${m.predicates.keySet.mkString(" AND ")}")
                  .mkString)
          .mkString
    }

    // STEP3
    val replacements = new IdentityHashMap[SparkPlan, SparkPlan]()
    groups.values.filter(_.length > 1).foreach {
      members =>
        // Only a shuffle in the main tree can be replaced. Both sorts push the pairing towards the
        // two ends of the group -- most restrictive as the subset side, least restrictive as the
        // superset side -- so that a chain {a,b,c} / {a,b} / {} collapses onto the one superset
        // rather than pairing up in the middle and recovering nothing.
        val supersetCandidates = members.sortBy(_.predicates.size)
        members
          .filter(_.inMainTree)
          .sortBy(-_.predicates.size)
          .foreach {
            sub =>
              if (!replacements.containsKey(sub.exchange)) {
                supersetCandidates.iterator
                  .filterNot(_.exchange eq sub.exchange)
                  .flatMap(sup => buildReplacement(sub, sup).map(sup -> _))
                  .find(_ => true)
                  .foreach {
                    case (sup, replacement) =>
                      val extra =
                        (sub.predicates.keySet -- sup.predicates.keySet).mkString(" AND ")
                      // Warn, not debug: this rewrite is the one thing in this rule that can
                      // change query results, so it should be visible without turning on DEBUG.
                      logWarning(
                        s"Recovering exchange reuse over ${sub.leafTables.mkString(",")}:" +
                          " dropping one materialization of the subtree and rebuilding it as" +
                          s" the less restrictive one plus [$extra] evaluated above the shuffle.")
                      replacements.put(sub.exchange, replacement)
                  }
              }
          }
    }

    if (replacements.isEmpty) {
      logDebug("No (subset, superset) pair survived the checks, plan left untouched.")
      return plan
    }

    // STEP4
    //
    // transformDown, NOT transformUp. Q24a's main branch has a second shareable shuffle -- the
    // BHJ's build side, over the same tables minus store_returns -- nested inside the one we care
    // about. transformUp would rewrite that inner one first, which rebuilds every ancestor and thus
    // hands the outer exchange to the rule as a NEW object, so the IdentityHashMap lookup misses
    // and the big rewrite is silently skipped. Going top-down replaces the outermost match and the
    // inner one disappears with the subtree it lived in.
    plan.transformDown {
      case s: ShuffleExchangeLike if replacements.containsKey(s) => replacements.get(s)
    }
  }
}

object PullUpExtraPredicateToRecoverExchangeReuse {

  /** Constant exprId every attribute is flattened to before comparing expressions by string. */
  private val FlatExprId = ExprId(0)

  private def stripCatalog(name: String): String = {
    val i = name.lastIndexOf('.')
    if (i >= 0) name.substring(i + 1) else name
  }

  /**
   * Reflective invocation of a no-arg method, with the `Method` lookup cached per (class, name) so
   * that a plan-wide traversal does not pay for `getMethod` on every leaf. Any failure -- no such
   * method, or a throwing invocation -- is reported as None, which every caller treats as "give
   * up".
   */
  private object MethodCache {
    private val cache =
      new ConcurrentHashMap[(Class[_], String), Option[java.lang.reflect.Method]]()

    def invoke[T](obj: AnyRef, name: String): Option[T] = {
      val cls = obj.getClass
      val methodOpt = cache.computeIfAbsent(
        (cls, name),
        k => {
          try { Option(k._1.getMethod(k._2)) }
          catch { case _: Exception => None }
        })
      methodOpt.flatMap {
        m =>
          try { Option(m.invoke(obj).asInstanceOf[T]) }
          catch { case _: Exception => None }
      }
    }
  }

  /** Reads a Scala field through its generated accessor, see [[MethodCache]]. */
  private def reflectField[T](obj: AnyRef, name: String): Option[T] =
    MethodCache.invoke[T](obj, name)

  /**
   * What a leaf scan reads. `table` feeds the STEP2 grouping key; the whole signature is compared
   * by equality between the two sides for every table the extra predicate does not touch.
   */
  private case class LeafSignature(
      table: String,
      outputSignature: Seq[(String, DataType, Boolean)],
      filterSignature: Seq[String])

  private case class ShuffleDescriptor(
      exchange: ShuffleExchangeLike,
      inMainTree: Boolean,
      outputSignature: Seq[DataType],
      partitioningKey: String,
      leafSignatures: Seq[LeafSignature],
      predicates: Map[String, Expression],
      joinKeys: Set[String]) {
    def leafTables: Set[String] = leafSignatures.map(_.table).toSet
  }
}
