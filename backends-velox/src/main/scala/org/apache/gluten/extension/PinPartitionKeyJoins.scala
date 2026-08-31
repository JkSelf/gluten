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

import org.apache.spark.sql.catalyst.catalog.HiveTableRelation
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.optimizer.ReorderJoin
import org.apache.spark.sql.catalyst.plans.{Inner, InnerLike}
import org.apache.spark.sql.catalyst.plans.logical.{Filter, HintInfo, Join, JoinHint, LeafNode}
import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, Project}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.trees.TreePattern.INNER_LIKE_JOIN
import org.apache.spark.sql.execution.datasources.{HadoopFsRelation, LogicalRelation}
import org.apache.spark.sql.execution.datasources.v2.DataSourceV2Relation
import org.apache.spark.sql.execution.datasources.v2.DataSourceV2ScanRelation

import scala.collection.mutable

/**
 * Keeps the cost based join reorder from separating a partitioned fact table from the dimension
 * that prunes its partitions.
 *
 * `CostBasedJoinReorder` costs a join order by the cardinality and the size of the intermediate
 * results only. It has no notion of dynamic partition pruning, of runtime filters or of the fan out
 * a join can cause, because all of those are added by later rules (`PartitionPruning`,
 * `InjectRuntimeFilter`) that only see the order the reorder has already committed to. So the
 * reorder happily moves a dimension away from its fact table, and the pruning that would have been
 * derived from that edge is silently lost. Column statistics do not fix this: they make the
 * cardinalities right, they do not tell the cost function that one of the orders reads a fifth of
 * the fact table.
 *
 * TPC-DS q72 on partitioned tables is the motivating case. It joins `catalog_sales` (partitioned by
 * `cs_sold_date_sk`) to `date_dim d1` filtered to a single year, and `inventory` (partitioned by
 * `inv_date_sk`) to `date_dim d2`, with `d1.d_week_seq = d2.d_week_seq` tying the two together. The
 * reordered plan builds `d1 join d2` first and joins that to `inventory`, which costs twice:
 *   - `cs_sold_date_sk` now meets its join partner on the far side of a shuffle join, so with
 *     `spark.sql.optimizer.dynamicPartitionPruning.reuseBroadcastOnly` (on by default) there is no
 *     broadcast left to reuse and `catalog_sales` is read in full - five times the rows it needs,
 *     four fifths of which are then carried through the shuffle before the year filter is applied,
 *   - `d1 join d2` holds seven rows per `d2.d_date_sk` (the days of a week), so joining it to
 *     `inventory` duplicates every inventory row seven times. The pruned and fanned out inventory
 *     side ends up bigger than the unpruned one.
 *
 * This rule pins such an edge instead: the fact table and the dimension are joined into a single
 * item before the reorder runs, so the reorder can only order that item as a whole. The pin is a
 * join carrying [[pinHint]], which `CostBasedJoinReorder` treats as a leaf. In q72 both edges above
 * are pinned, which restores the partition pruning on `catalog_sales`, removes the fan out on
 * `inventory`, and leaves the reorder free to place the remaining dimensions. The equality that
 * ties the two pinned items together (`d_week_seq`) is still a join condition, so the join between
 * them keeps both keys.
 *
 * An edge is pinned when all of the following hold, which keeps the rule to the shape it is meant
 * for - a large partitioned table joined to a dimension small enough to be broadcast:
 *   - the two items are connected by a single equality between two plain attributes,
 *   - the attribute on the fact side is a partition column of the underlying relation,
 *   - the fact side is too big to be broadcast and the dimension side is not, so the dimension can
 *     become the broadcast that the pruning subquery reuses,
 *   - both items are single relations, not join trees, and neither was brought in by a
 *     `CROSS JOIN`.
 *
 * At most one dimension is pinned per fact table. A dimension that filters is preferred, that is
 * the one partition pruning needs, and the smaller one wins otherwise.
 *
 * Pinning also takes away the reorder's other way of building a join the backend cannot execute: a
 * join driven by a non-equi ("theta") predicate, which has no keys and is planned as a
 * `BroadcastNestedLoopJoin`. Contracting two items into one group can only add equalities to the
 * edges of a group, so a predicate such as q72's `d3.d_date > d1.d_date + 5` - which connects two
 * `date_dim`s that hold no equality between them - ends up on an edge that does have a join key
 * once `d1` is pinned to `catalog_sales`. The rule verifies this rather than assuming it, see
 * [[groupsCanBeJoined]], and leaves a tree alone when it does not hold.
 *
 * The rule has to run after the operator optimization batch and before the join reorder, so it is
 * injected as a pre CBO rule, see `VeloxRuleApi`. It is switched off by
 * `spark.gluten.sql.columnar.backend.velox.pinPartitionKeyJoin`.
 */
class PinPartitionKeyJoins extends Rule[LogicalPlan] with PredicateHelper {

  /**
   * The hint that marks a pinned join. Any hint that is not [[JoinHint.NONE]] stops
   * `CostBasedJoinReorder` from flattening the join, and a [[HintInfo]] without a strategy is inert
   * everywhere else: `JoinSelection` only looks at `HintInfo.strategy`, so the join is still
   * planned by size as it would be without the hint.
   */
  private val pinHint: JoinHint = JoinHint(Some(HintInfo()), None)

  /**
   * The rule only has something to do when the cost based join reorder is on, because that is the
   * only thing that takes the two sides of such a join apart.
   */
  private def isEnabled: Boolean =
    VeloxConfig.get.pinPartitionKeyJoinEnabled && conf.cboEnabled && conf.joinReorderEnabled

  override def apply(plan: LogicalPlan): LogicalPlan = {
    if (!isEnabled) {
      plan
    } else {
      plan.transformDownWithPruning(_.containsPattern(INNER_LIKE_JOIN)) {
        case j @ Join(_, _, _: InnerLike, Some(_), JoinHint.NONE) => rewriteJoin(j)
      }
    }
  }

  private def rewriteJoin(join: Join): LogicalPlan = {
    val (items, extractedConditions) = extractInnerJoins(join, Inner)
    val conditions = ExpressionSet(extractedConditions).toSeq
    // Only join trees that the reorder rule would actually touch are interesting: less than three
    // items cannot be reordered at all and `CostBasedJoinReorder` gives up above the DP threshold.
    if (items.size < 3 || items.size > conf.joinReorderDPThreshold || conditions.isEmpty) {
      return join
    }
    // Moving a non-deterministic predicate or a predicate holding a subquery around changes how
    // often it is evaluated, so such plans are left alone.
    if (conditions.exists(c => !c.deterministic || SubqueryExpression.hasSubquery(c))) {
      return join
    }

    val outputSets = items.map(_._1.outputSet)
    def itemsOf(e: Expression): Set[Int] = {
      outputSets.indices.filter(i => e.references.exists(outputSets(i).contains)).toSet
    }

    // Every condition, with the items it references. A condition on a single item is a predicate of
    // that item rather than something joining two of them; one referencing no item at all holds
    // only literals or outer references, which cannot connect two items either, but
    // `createOrderedJoin` keeps it, so it is safe to leave it among the tree's conditions.
    val (singleItem, crossConditions) =
      conditions.map(cond => (cond, itemsOf(cond))).partition { case (_, refs) => refs.size == 1 }
    val itemConditions: Map[Int, Seq[Expression]] =
      singleItem.groupBy { case (_, refs) => refs.head }.map {
        case (item, conds) => item -> conds.map { case (cond, _) => cond }
      }

    // A condition spanning three items or more can never become the join key of a pair of groups,
    // see [[canDriveJoin]], so it cannot be shown to be harmless once the pins are known, and
    // rebuilding a tree that holds one risks handing `createOrderedJoin` a join without any key.
    // Bail out on those here, before any statistics are looked at. The two item ones are decided
    // after the grouping, because that is what can make them harmless.
    if (crossConditions.exists { case (_, refs) => refs.size > 2 }) {
      logDebug(
        "Not pinning the partition key joins of this tree: it holds a condition spanning three " +
          "items or more.")
      return join
    }

    // The single item predicates belong to their item, both to keep them out of the way of the
    // reorder and because the pruning side needs its filter to be recognized as selective.
    val itemInfos = items.zipWithIndex.map {
      case ((item, joinType), i) =>
        val filtered = itemConditions
          .get(i)
          .map(cs => Filter(cs.reduceLeft(And), item).asInstanceOf[LogicalPlan])
          .getOrElse(item)
        analyzeItem(filtered, joinType)
    }

    val pins = selectPins(itemInfos, crossConditions)
    if (pins.isEmpty) {
      return join
    }

    // The items of a pin become one group, every other item is a group of its own. The groups keep
    // the order the items had, so a tree the reorder does not touch stays close to the original.
    val absorbed = pins.values.toSet
    val groups = itemInfos.indices.filterNot(absorbed.contains).map {
      i =>
        pins.get(i) match {
          case Some(dim) =>
            val indices = Set(i, dim)
            val pinConditions = crossConditions.collect {
              case (cond, refs) if refs.nonEmpty && refs.subsetOf(indices) => cond
            }
            val pinned = Join(
              itemInfos(i).plan,
              itemInfos(dim).plan,
              Inner,
              pinConditions.reduceLeftOption(And),
              pinHint)
            Group(indices, pinned, itemInfos(i).joinType)
          case None =>
            Group(Set(i), itemInfos(i).plan, itemInfos(i).joinType)
        }
    }
    if (groups.size < 2) {
      return join
    }

    // The conditions keep the items they reference, which is what the check below maps onto groups.
    // Recomputing that from the expressions would be redundant, it was needed to sort the
    // conditions into `itemConditions` and `crossConditions` in the first place.
    val remaining = crossConditions.filter {
      case (_, refs) => !groups.exists(g => refs.nonEmpty && refs.subsetOf(g.items))
    }
    if (!groupsCanBeJoined(groups.map(_.items), remaining, itemsOf)) {
      return join
    }

    val joined =
      ReorderJoin.createOrderedJoin(groups.map(g => (g.plan, g.joinType)), remaining.map(_._1))
    // Grouping the items may have changed the order of the columns, restore it.
    val result = if (join.sameOutput(joined)) joined else Project(join.output, joined)
    logDebug(s"Pinned ${pins.size} partition key join(s) of a join tree of " +
      s"${items.size} items:\n$result")
    result
  }

  /**
   * Picks the dimension to pin for every fact table, as a map from the index of the fact item to
   * the index of the dimension item. Every item takes part in at most one pin.
   */
  private def selectPins(
      itemInfos: Seq[ItemInfo],
      crossConditions: Seq[(Expression, Set[Int])]): Map[Int, Int] = {
    val threshold = BigInt(conf.autoBroadcastJoinThreshold)
    val candidates = crossConditions.flatMap {
      case (cond, refs) if refs.size == 2 =>
        val sorted = refs.toSeq.sorted
        val (l, r) = (sorted.head, sorted.last)
        pinCandidate(cond, l, r, itemInfos, threshold) ++
          pinCandidate(cond, r, l, itemInfos, threshold)
      case _ => Nil
    }
    // A dimension that filters something is the one partition pruning can use, and a smaller
    // dimension is the cheaper broadcast. Both are only tie breakers between candidates of the
    // same fact table, the pin itself is worth it either way.
    val ordered = candidates.sortBy {
      case (fact, dim) =>
        (
          if (itemInfos(dim).hasSelectivePredicate) 0 else 1,
          itemInfos(dim).sizeInBytes,
          fact,
          dim)
    }
    val pins = mutable.Map.empty[Int, Int]
    val used = mutable.Set.empty[Int]
    ordered.foreach {
      case (fact, dim) =>
        if (!used.contains(fact) && !used.contains(dim)) {
          pins += fact -> dim
          used += fact
          used += dim
        }
    }
    pins.toMap
  }

  /** Returns `(fact, dim)` if `cond` is an edge that is worth pinning, in that direction. */
  private def pinCandidate(
      cond: Expression,
      fact: Int,
      dim: Int,
      itemInfos: Seq[ItemInfo],
      threshold: BigInt): Option[(Int, Int)] = {
    val factInfo = itemInfos(fact)
    val dimInfo = itemInfos(dim)
    lazy val keys = joinKeys(cond, factInfo.plan.outputSet, dimInfo.plan.outputSet)
    if (
      factInfo.joinType == Inner && dimInfo.joinType == Inner &&
      factInfo.isSingleRelation && dimInfo.isSingleRelation &&
      // The dimension has to be broadcastable for its broadcast to be the one the pruning
      // subquery reuses, and the fact table has to be big enough for pruning it to matter.
      dimInfo.sizeInBytes <= threshold && factInfo.sizeInBytes > threshold &&
      keys.exists { case (factKey, _) => factInfo.partitionColumns.contains(factKey) }
    ) {
      Some((fact, dim))
    } else {
      None
    }
  }

  /**
   * Returns the two sides of `cond` as `(left key, right key)` if it is an equality between a plain
   * attribute of `leftOutput` and a plain attribute of `rightOutput`.
   */
  private def joinKeys(
      cond: Expression,
      leftOutput: AttributeSet,
      rightOutput: AttributeSet): Option[(Attribute, Attribute)] = cond match {
    case EqualTo(l: Attribute, r: Attribute) if leftOutput.contains(l) && rightOutput.contains(r) =>
      Some((l, r))
    case EqualTo(l: Attribute, r: Attribute) if leftOutput.contains(r) && rightOutput.contains(l) =>
      Some((r, l))
    case _ => None
  }

  /**
   * Whether `cond` can drive the join between the two sides it spans, that is whether
   * `ExtractEquiJoinKeys` would turn it into a join key rather than leaving the join with no key at
   * all. It mirrors what that rule accepts: an equality is a key only when each of its two sides
   * can be evaluated on one side of the join and neither side is reference free. Being an `EqualTo`
   * is not enough - `f_ds + d_ds = 5` spans both sides and is still not a key, and a pair of groups
   * that has nothing else would come out as a `BroadcastNestedLoopJoin`.
   *
   * `sidesOf` maps an expression to the sides it reads from, which is the items for the check
   * before the pins are picked and the groups for the checks after.
   */
  private def canDriveJoin(cond: Expression, sidesOf: Expression => Set[Int]): Boolean =
    cond match {
      case Equality(l, r) if canEvaluateWithinJoin(cond) =>
        val (left, right) = (sidesOf(l), sidesOf(r))
        // A side that reads from no item at all holds only literals, which is the
        // `l.references.isEmpty` case `ExtractEquiJoinKeys` rejects, and a side that reads from
        // both cannot be evaluated before the join.
        left.size == 1 && right.size == 1 && left != right
      case _ => false
    }

  /** The partition columns of the relation `leaf` reads, empty for anything else. */
  private def partitionColumnsOfRelation(leaf: LogicalPlan): AttributeSet = leaf match {
    case r: DataSourceV2ScanRelation => byName(r.output, partitionColumnNames(r.relation))
    case r: DataSourceV2Relation => byName(r.output, partitionColumnNames(r))
    case r: LogicalRelation =>
      r.relation match {
        case fsRelation: HadoopFsRelation =>
          byName(r.output, fsRelation.partitionSchema.fieldNames.toSeq)
        case _ => AttributeSet.empty
      }
    case r: HiveTableRelation => AttributeSet(r.partitionCols)
    case _ => AttributeSet.empty
  }

  /**
   * The top level columns the partitioning of a v2 table is derived from. A transform such as
   * `days(ts)` counts, the pruning happens on the column it references.
   */
  private def partitionColumnNames(relation: DataSourceV2Relation): Seq[String] = {
    relation.table
      .partitioning()
      .toSeq
      .flatMap(_.references().toSeq)
      .map(_.fieldNames())
      .collect { case Array(name) => name }
  }

  private def byName(output: Seq[Attribute], names: Seq[String]): AttributeSet = {
    AttributeSet(output.filter(a => names.exists(conf.resolver(a.name, _))))
  }

  /**
   * Walks down to the relation an item reads, mapping the partition columns of that relation onto
   * the attributes the item itself outputs. Returns None when the item is not a single relation but
   * a join tree of its own, or when the projections in between are not simple enough to follow -
   * which amounts to the same thing here, since the rule only ever pins single relations.
   *
   * A [[Project]] that renames a column is followed through its [[Alias]], so an item such as
   * `SELECT f_ds AS ds FROM fact` still offers a partition column, under the name `ds`. Insisting
   * on plain attributes instead would leave the rule blind to any item that renames its columns,
   * and silently so: the pin would just never be considered.
   */
  private def partitionColumnsOfSingleRelation(plan: LogicalPlan): Option[AttributeSet] =
    plan match {
      case Filter(_, child) => partitionColumnsOfSingleRelation(child)
      case Project(projectList, child) =>
        partitionColumnsOfSingleRelation(child).flatMap {
          childColumns =>
            val followed: Seq[Option[Seq[Attribute]]] = projectList.map {
              case a: Attribute =>
                Some(if (childColumns.contains(a)) Seq(a) else Nil)
              case alias @ Alias(a: Attribute, _) =>
                Some(if (childColumns.contains(a)) Seq(alias.toAttribute) else Nil)
              // A computed column is not a renaming, and following it would mean reasoning about
              // whether the pruning still applies to what it computes.
              case _ => None
            }
            if (followed.forall(_.isDefined)) {
              Some(AttributeSet(followed.flatten.flatten))
            } else {
              None
            }
        }
      case leaf: LeafNode => Some(partitionColumnsOfRelation(leaf))
      case _ => None
    }

  /**
   * Everything the rule needs to know about one item of the join tree, gathered in one pass.
   *
   * Whether the item is a single relation and which of its columns are partition columns come out
   * of the same walk down its subtree, and the two properties [[selectPins]] sorts by are computed
   * here rather than in the sort itself, because `sortBy` evaluates its key on every comparison.
   * Note that `LogicalPlan.stats` needs no such treatment, Spark memoizes it per node.
   */
  private case class ItemInfo(
      plan: LogicalPlan,
      joinType: InnerLike,
      isSingleRelation: Boolean,
      partitionColumns: AttributeSet,
      hasSelectivePredicate: Boolean) {
    def sizeInBytes: BigInt = plan.stats.sizeInBytes
  }

  private def analyzeItem(plan: LogicalPlan, joinType: InnerLike): ItemInfo = {
    val partitionColumns = partitionColumnsOfSingleRelation(plan)
    ItemInfo(
      plan = plan,
      joinType = joinType,
      isSingleRelation = partitionColumns.isDefined,
      partitionColumns = partitionColumns.getOrElse(AttributeSet.empty),
      hasSelectivePredicate = hasSelectivePredicate(plan)
    )
  }

  /**
   * One item of the rebuilt tree: the items it absorbed - two of them for a pin, one otherwise -
   * its plan, and the type of the inner join that brought its first item in.
   */
  private case class Group(items: Set[Int], plan: LogicalPlan, joinType: InnerLike)

  /**
   * Whether `plan` filters on something else than nullability, the same notion of a selective
   * predicate `PartitionPruning` uses to decide that a pruning side is worth pruning with.
   */
  private def hasSelectivePredicate(plan: LogicalPlan): Boolean = plan.exists {
    case Filter(condition, _) =>
      splitConjunctivePredicates(condition).exists {
        case _: IsNotNull | _: IsNull => false
        case _ => true
      }
    case _ => false
  }

  /**
   * Flattens the items and the conditions of an inner join tree, the same way
   * `CostBasedJoinReorder` does. Besides the inner joins themselves this looks through the
   * [[Project]]s that `ColumnPruning` leaves between the joins - without that the tree would look
   * like a two item join to this rule while the reorder rule still sees all of the items - and it
   * collects the conditions of the [[Filter]]s in between, which are put back where they belong.
   *
   * Every item is returned together with the type of the inner join that brought it in, so that an
   * explicit `CROSS JOIN` stays a cross join. A join that is already pinned is an item, so the rule
   * is idempotent.
   *
   * A [[Project]] or a [[Filter]] may only be looked through when there is a join underneath it,
   * otherwise the node is an item of its own and unwrapping it would drop its predicate or widen
   * its output. That is decided by descending and looking at what comes back rather than by probing
   * the subtree first: a chain that does not reach a join yields exactly one item, in which case
   * the answer is thrown away in favour of the node itself. Probing would walk the same subtree
   * twice.
   */
  private def extractInnerJoins(
      plan: LogicalPlan,
      parentJoinType: InnerLike): (Seq[(LogicalPlan, InnerLike)], Seq[Expression]) = {
    lazy val asItem: (Seq[(LogicalPlan, InnerLike)], Seq[Expression]) =
      (Seq((plan, parentJoinType)), Nil)
    plan match {
      case Join(left, right, joinType: InnerLike, cond, JoinHint.NONE) =>
        val (leftItems, leftConditions) = extractInnerJoins(left, joinType)
        val (rightItems, rightConditions) = extractInnerJoins(right, joinType)
        (
          leftItems ++ rightItems,
          leftConditions ++ rightConditions ++ cond.toSeq.flatMap(splitConjunctivePredicates))

      case Project(projectList, child) if projectList.forall(_.isInstanceOf[Attribute]) =>
        // A pruning project, the columns it removes are pruned again after the join order is fixed.
        val flattened = extractInnerJoins(child, parentJoinType)
        if (flattened._1.size > 1) flattened else asItem

      case Filter(condition, child) =>
        val (items, conditions) = extractInnerJoins(child, parentJoinType)
        if (items.size > 1) {
          (items, conditions ++ splitConjunctivePredicates(condition))
        } else {
          asItem
        }

      case _ => asItem
    }
  }

  /**
   * Whether the groups can be joined on `conditions` without producing a join the backend cannot
   * execute. Both things that can go wrong are properties of the same graph - the groups as nodes,
   * the pairs of groups a condition spans as edges - so both are decided here:
   *
   *   - every edge that carries any condition has to carry one that can drive a join.
   *     `createOrderedJoin` joins the plan it has built so far to the first remaining group holding
   *     a condition it can evaluate, and `CostBasedJoinReorder` does the same when it costs a pair,
   *     so either of them can end up with a condition that is no join key as the only condition of
   *     a join - a join without keys, planned as a nested loop join, with no Velox offload. It is
   *     harmless when the same edge also carries a real key, because both of them collect every
   *     condition that fits the pair: the join keeps its keys and the other condition rides along.
   *   - the edges that do carry a key have to connect every group, or the rebuilt tree holds a
   *     cartesian product. The pins only contract edges, which cannot disconnect the graph, but a
   *     tree that was held together by a condition that is no key can come apart here.
   *
   * The first is what pinning buys on top of the pruning it is named for. In TPC-DS q72
   * `d3.d_date > d1.d_date + 5` spans `date_dim d1` and `date_dim d3`, which hold no equality
   * between them - `d1` is reached through `cs_sold_date_sk` and `d3` through `cs_ship_date_sk` -
   * so the reorder is free to join the two `date_dim`s on that predicate alone. Pinning `d1` to
   * `catalog_sales` puts it in the same group, and that group does hold an equality with `d3`, so
   * the predicate can no longer drive a join of its own.
   *
   * @param conditions
   *   every condition together with the items it references, so that the edge it spans is a lookup
   *   rather than another walk over the item outputs
   * @param itemsOf
   *   still needed for the two sides of an equality, which are sub expressions of a condition and
   *   therefore not covered by the item sets carried above
   */
  private def groupsCanBeJoined(
      groups: Seq[Set[Int]],
      conditions: Seq[(Expression, Set[Int])],
      itemsOf: Expression => Set[Int]): Boolean = {
    val groupOf = groups.zipWithIndex.flatMap { case (items, g) => items.map(_ -> g) }.toMap
    def groupsOf(e: Expression): Set[Int] = itemsOf(e).map(groupOf)

    // The edges, each with whether the condition on it can drive a join. A condition that stays
    // inside one group is evaluated there and is no edge at all: it neither drives a join between
    // groups nor needs one. None can span more than two groups, because the ones spanning more
    // than two items were rejected before the pins were picked and grouping only merges items.
    val (drivers, passengers) = conditions
      .flatMap {
        case (cond, refs) =>
          val touched = refs.map(groupOf).toSeq.sorted
          if (touched.size == 2) {
            Some(((touched.head, touched.last), canDriveJoin(cond, groupsOf)))
          } else {
            None
          }
      }
      .partition { case (_, drives) => drives }

    val drivenEdges = drivers.map { case (edge, _) => edge }.toSet
    val keylessEdges = passengers.map { case (edge, _) => edge }.toSet -- drivenEdges
    if (keylessEdges.nonEmpty) {
      logDebug(
        "Not pinning the partition key joins of this tree: a condition that is no join key would " +
          "be the only condition available for a pair of groups.")
      return false
    }

    val parent = Array.tabulate(groups.size)(identity)
    def find(i: Int): Int = {
      if (parent(i) != i) {
        parent(i) = find(parent(i))
      }
      parent(i)
    }
    drivenEdges.foreach {
      case (l, r) =>
        val (lRoot, rRoot) = (find(l), find(r))
        if (lRoot != rRoot) {
          parent(lRoot) = rRoot
        }
    }
    val connected = parent.indices.map(find).distinct.size == 1
    if (!connected) {
      logDebug(
        "Not pinning the partition key joins of this tree: the groups are not connected by " +
          "conditions that can drive a join.")
    }
    connected
  }
}
