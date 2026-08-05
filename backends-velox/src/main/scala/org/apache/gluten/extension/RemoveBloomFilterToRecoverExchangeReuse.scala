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

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.{BloomFilterMightContain, Expression, PredicateHelper}
import org.apache.spark.sql.catalyst.plans.logical.{Filter, Join, LogicalPlan}
import org.apache.spark.sql.catalyst.rules.Rule

import java.util.IdentityHashMap

import scala.collection.mutable.ArrayBuffer

/**
 * When Spark's [[org.apache.spark.sql.catalyst.optimizer.InjectRuntimeFilter]] injects a bloom
 * filter on one reference of a duplicated sub-plan but not another, the two sub-plans are no longer
 * structurally identical. Spark's ReuseExchangeAndSubquery then fails its `canonicalized` check and
 * the underlying table is scanned twice.
 *
 * This rule only inspects direct join inputs because
 * [[org.apache.spark.sql.catalyst.optimizer.InjectRuntimeFilter]] only injects bloom filters on
 * join probe sides. For a join input of the form [[Filter]](BF(...), child), the rule computes the
 * exact rewritten join input that would remain after removing the bloom-filter conjuncts. The bloom
 * filter is removed only if that rewritten join input has the same `canonicalized` form as another
 * unfiltered join input anywhere in the plan. This keeps the rule scoped to actual exchange-reuse
 * candidates and avoids dropping bloom filters for unrelated join branches.
 *
 * Trade-off: we lose the per-scan bloom filter selectivity, but we recover the exchange reuse which
 * avoids scanning the large table a second time - typically a much larger saving.
 */
case class RemoveBloomFilterToRecoverExchangeReuse(spark: SparkSession)
  extends Rule[LogicalPlan]
  with PredicateHelper {

  private case class FilteredJoinInput(
      child: LogicalPlan,
      predicates: Seq[Expression],
      rewrittenCanonicalized: LogicalPlan)

  override def apply(plan: LogicalPlan): LogicalPlan = {
    val (bfFilteredJoinInputs, unfilteredJoinInputCanonicalized) = collectJoinInputs(plan)
    if (bfFilteredJoinInputs.isEmpty) {
      return plan
    }

    val childrenToStrip = new IdentityHashMap[LogicalPlan, Seq[Expression]]()
    bfFilteredJoinInputs.foreach {
      filtered =>
        if (unfilteredJoinInputCanonicalized.contains(filtered.rewrittenCanonicalized)) {
          childrenToStrip.put(filtered.child, filtered.predicates)
        }
    }

    if (childrenToStrip.isEmpty) {
      return plan
    }

    plan.transformWithSubqueries {
      case f @ Filter(condition, child)
          if childrenToStrip.containsKey(child) =>
        val newCondition =
          removeOffendingBloomFilterPredicates(condition, childrenToStrip.get(child))
        newCondition match {
          case None =>
            child
          case Some(cond) if cond.fastEquals(condition) =>
            f
          case Some(cond) =>
            Filter(cond, child)
        }
    }
  }

  private def collectJoinInputs(
      plan: LogicalPlan): (Seq[FilteredJoinInput], Set[LogicalPlan]) = {
    val filteredJoinInputs = ArrayBuffer[FilteredJoinInput]()
    val unfilteredJoinInputCanonicalized = scala.collection.mutable.HashSet[LogicalPlan]()

    def processJoinInput(input: LogicalPlan): Unit = input match {
      case filter @ Filter(condition, child)
          if splitConjunctivePredicates(condition).exists(isBloomFilter) =>
        val predicatesToRemove = splitConjunctivePredicates(condition).filter(isBloomFilter)
        val rewrittenPlan =
          removeOffendingBloomFilterPredicates(condition, predicatesToRemove) match {
            case None => child
            case Some(cond) if cond.fastEquals(condition) => filter
            case Some(cond) => Filter(cond, child)
          }
        filteredJoinInputs += FilteredJoinInput(
          child,
          predicatesToRemove,
          rewrittenPlan.canonicalized)
      case other =>
        unfilteredJoinInputCanonicalized += other.canonicalized
    }

    plan.collectWithSubqueries {
      case j: Join =>
        processJoinInput(j.left)
        processJoinInput(j.right)
    }

    (filteredJoinInputs.toSeq, unfilteredJoinInputCanonicalized.toSet)
  }

  private def isBloomFilter(expr: Expression): Boolean = expr match {
    case _: BloomFilterMightContain => true
    case _ => false
  }

  private def removeOffendingBloomFilterPredicates(
      condition: Expression,
      predicatesToRemove: Seq[Expression]): Option[Expression] = {
    val remaining = splitConjunctivePredicates(condition).filterNot {
      expr => predicatesToRemove.exists(_.fastEquals(expr))
    }
    remaining.reduceOption(org.apache.spark.sql.catalyst.expressions.And)
  }
}
