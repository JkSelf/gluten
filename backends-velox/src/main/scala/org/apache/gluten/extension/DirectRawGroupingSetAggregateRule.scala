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
import org.apache.gluten.execution._

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.types._

import scala.util.control.NonFatal

/**
 * Marks an eligible raw-input partial aggregate over Expand for direct native grouping-set fusion.
 *
 * This rule does not introduce a finest-grain partial aggregate or a partial-merge aggregate. The
 * original aggregate-over-Expand pair remains in the Spark plan, and the marker on Expand is only a
 * native optimization hint. A native library that does not recognize the marker therefore executes
 * the original pair.
 *
 * PullOutPreProject may place a deterministic measure projection between the aggregate and Expand.
 * This rule moves those computed aliases below Expand and rebuilds the Expand projections so the
 * aggregate and Expand are adjacent. Filters are deliberately not moved by this first version.
 */
object DirectRawGroupingSetAggregateRule
  extends Rule[SparkPlan]
  with Logging {

  override def apply(plan: SparkPlan): SparkPlan = {
    if (!VeloxConfig.get.enableVeloxFusedGroupingSetAggregate) {
      return plan
    }

    plan.transformUp {
      case agg: RegularHashAggregateExecTransformer if isEligibleAggregate(agg) =>
        agg.child match {
          case expand: ExpandExecTransformer if isUnmarked(expand) =>
            rewrite(agg, expand, preProject = None).getOrElse(agg)
          case project @ ProjectExecTransformer(_, expand: ExpandExecTransformer)
              if isUnmarked(expand) =>
            rewrite(agg, expand, preProject = Some(project)).getOrElse(agg)
          // Moving a filter changes which raw rows contribute to each grouping set.
          case _: FilterExecTransformer => agg
          case _ => agg
        }
    }
  }

  private def isUnmarked(expand: ExpandExecTransformer): Boolean = {
    !RawGroupingSetFusion.isMarked(expand)
  }

  private def isEligibleAggregate(agg: RegularHashAggregateExecTransformer): Boolean = {
    agg.initialInputBufferOffset == 0 &&
    agg.groupingExpressions.forall(_.isInstanceOf[Attribute]) &&
    agg.aggregateExpressions.nonEmpty &&
    agg.aggregateExpressions.forall(_.mode == Partial) &&
    agg.aggregateExpressions.forall(isSupportedAggregateExpression) &&
    !hasUnsafeFloatingPointAggregate(agg.aggregateExpressions)
  }

  private def isSupportedAggregateExpression(aggExpr: AggregateExpression): Boolean = {
    if (aggExpr.filter.isDefined || aggExpr.isDistinct) {
      return false
    }
    aggExpr.aggregateFunction match {
      case s: Sum => !s.prettyName.equals("try_sum")
      case a: Average => !a.prettyName.equals("try_avg")
      case _: Count => true
      case _: Min => true
      case _: Max => true
      case _ => false
    }
  }

  // Direct raw fusion can change the order in which partial states are combined. Match the
  // floating-point policy used by the existing aggregation rewrites.
  private def hasUnsafeFloatingPointAggregate(aggExprs: Seq[AggregateExpression]): Boolean = {
    if (VeloxConfig.get.floatingPointMode == "loose") {
      return false
    }

    def isFloatingPointType(dataType: DataType): Boolean = {
      dataType == DoubleType || dataType == FloatType
    }

    aggExprs.exists {
      aggExpr =>
        aggExpr.aggregateFunction match {
          case s: Sum => isFloatingPointType(s.child.dataType)
          case a: Average => isFloatingPointType(a.sumDataType)
          case _ => false
        }
    }
  }

  private def rewrite(
      agg: RegularHashAggregateExecTransformer,
      expand: ExpandExecTransformer,
      preProject: Option[ProjectExecTransformer]): Option[SparkPlan] = {
    if (
      expand.projections.isEmpty ||
      expand.projections.exists(_.length != expand.output.length)
    ) {
      logDebug("Direct raw grouping-set fusion: malformed Expand projections")
      return None
    }

    val numKeys = agg.groupingExpressions.length
    val groupingAttributes = agg.groupingExpressions.map(_.toAttribute)
    val numBufferAttributes =
      agg.aggregateExpressions.map(_.aggregateFunction.aggBufferAttributes.length).sum
    if (
      agg.resultExpressions.length != numKeys + numBufferAttributes ||
      !agg.resultExpressions.forall(_.isInstanceOf[Attribute]) ||
      !agg.resultExpressions
        .take(numKeys)
        .zip(groupingAttributes)
        .forall { case (result, key) => result.toAttribute.semanticEquals(key) }
    ) {
      logDebug(
        s"Direct raw grouping-set fusion: unexpected partial aggregate output shape: " +
          s"${agg.resultExpressions}")
      return None
    }

    val expandChildOutput = expand.child.output
    val preProjectAliases = preProject.toSeq.flatMap(_.projectList.collect { case a: Alias => a })
    if (
      !preProject.forall(
        _.projectList.forall {
          case attr: Attribute => uniqueOutputIndex(attr, expand.output).isDefined
          case alias: Alias =>
            alias.child.deterministic && resolvableFrom(alias.references, expandChildOutput)
          case _ => false
        })
    ) {
      logDebug(
        "Direct raw grouping-set fusion: pre-project cannot be re-grounded below Expand")
      return None
    }

    // This also rejects RewriteDistinctAggregates look-alikes: their aggregate functions bind to
    // attributes created by Expand instead of raw pass-through columns.
    val aggregateInputCandidates =
      expandChildOutput ++ preProjectAliases.map(_.toAttribute)
    if (
      !agg.aggregateExpressions.forall(
        expression =>
          resolvableFrom(expression.aggregateFunction.references, aggregateInputCandidates))
    ) {
      logDebug(
        "Direct raw grouping-set fusion: aggregate inputs are not raw pass-through columns")
      return None
    }

    val groupingSlots = groupingAttributes.map(uniqueOutputIndex(_, expand.output))
    if (groupingSlots.exists(_.isEmpty)) {
      logDebug("Direct raw grouping-set fusion: a grouping attribute has no unique Expand slot")
      return None
    }
    val groupingSlotIndexes = groupingSlots.flatten
    if (groupingSlotIndexes.distinct.length != groupingSlotIndexes.length) {
      logDebug("Direct raw grouping-set fusion: grouping attributes share an Expand slot")
      return None
    }

    val passThroughBySlot = buildPassThroughBySlot(expand)
    val literalGroupingSlots = groupingSlotIndexes.zipWithIndex.filter {
      case (slot, _) => passThroughBySlot(slot).isEmpty
    }
    val hasSingleLongGid = literalGroupingSlots.length == 1 &&
      literalGroupingSlots.head._2 == numKeys - 1 &&
      groupingAttributes.lastOption.exists(_.dataType == LongType)
    if (!hasSingleLongGid) {
      logDebug(
        "Direct raw grouping-set fusion: expected one trailing LONG literal grouping-id slot")
      return None
    }

    val gidSlot = literalGroupingSlots.head._1
    if (
      !expand.projections.forall(
        projection =>
          projection(gidSlot) match {
            case Literal(_, LongType) => true
            case _ => false
          })
    ) {
      logDebug("Direct raw grouping-set fusion: grouping ids are not LONG literals")
      return None
    }

    val keySlots = groupingSlotIndexes.dropRight(1)
    val rawGroupingKeys = keySlots.flatMap(slot => passThroughBySlot(slot))
    if (
      rawGroupingKeys.length != numKeys - 1 ||
      semanticDistinct(rawGroupingKeys).length != rawGroupingKeys.length ||
      rawGroupingKeys.isEmpty ||
      rawGroupingKeys.length > 64 ||
      !rawGroupingKeys.forall(key => expandChildOutput.exists(_.semanticEquals(key))) ||
      !rawGroupingKeys.forall(key => isSupportedGroupingKeyType(key.dataType))
    ) {
      logDebug(
        s"Direct raw grouping-set fusion: unsupported raw grouping keys: $rawGroupingKeys")
      return None
    }

    val masks = expand.projections.map {
      projection =>
        keySlots.zip(rawGroupingKeys).zip(groupingAttributes.dropRight(1)).map {
          case ((slot, rawKey), outputKey) =>
            projection(slot) match {
              case attr: Attribute if attr.semanticEquals(rawKey) => Some(true)
              case literal: Literal
                  if literal.value == null && literal.dataType == outputKey.dataType =>
                Some(false)
              case _ => None
            }
        }
    }
    if (masks.flatten.exists(_.isEmpty)) {
      logDebug(
        "Direct raw grouping-set fusion: grouping-key slots are not attribute-or-null projections")
      return None
    }
    val groupingMasks = masks.map(_.flatten)
    val distinctMasks = groupingMasks.distinct.length == groupingMasks.length
    val withinSetLimit =
      groupingMasks.length <= VeloxConfig.get.maxVeloxFusedGroupingSets
    val singleRoot =
      distinctMasks && withinSetLimit && numLatticeRoots(groupingMasks) == 1
    if (!distinctMasks) {
      logDebug("Direct raw grouping-set fusion: duplicate grouping sets")
    }
    if (!withinSetLimit) {
      logDebug(
        s"Direct raw grouping-set fusion: ${groupingMasks.length} grouping sets exceed the " +
          s"configured maximum ${VeloxConfig.get.maxVeloxFusedGroupingSets}")
    }
    if (distinctMasks && withinSetLimit && !singleRoot) {
      logDebug("Direct raw grouping-set fusion: grouping sets have multiple lattice roots")
    }
    if (!distinctMasks || !withinSetLimit || !singleRoot) {
      return None
    }

    val reGroundedPreProject = preProject.map {
      project =>
        val reGrounded =
          ProjectExecTransformer(expandChildOutput ++ preProjectAliases, expand.child)
        reGrounded.copyTagsFrom(project)
        reGrounded
    }
    val newExpandProjections = preProject match {
      case Some(project) =>
        val rebuilt = expand.projections.map {
          projection =>
            project.projectList.flatMap {
              case attr: Attribute =>
                uniqueOutputIndex(attr, expand.output).map(projection)
              case alias: Alias => Some(alias.toAttribute)
              case _ => None
            }
        }
        if (rebuilt.exists(_.length != project.projectList.length)) {
          logDebug(
            "Direct raw grouping-set fusion: pre-project reconstruction lost an expression")
          return None
        }
        rebuilt
      case None => expand.projections
    }
    val newExpandOutput = preProject.map(_.output).getOrElse(expand.output)
    val newExpand = ExpandExecTransformer(
      newExpandProjections,
      newExpandOutput,
      reGroundedPreProject.getOrElse(expand.child)
    )
    newExpand.copyTagsFrom(expand)
    RawGroupingSetFusion.mark(newExpand)

    val newAggregate = agg.copy(child = newExpand)
    newAggregate.copyTagsFrom(agg)

    val rewrittenNodes: Seq[SparkPlan] =
      reGroundedPreProject.toSeq ++ Seq(newExpand, newAggregate)
    if (!rewrittenNodes.forall(passesNativeValidation)) {
      logDebug("Direct raw grouping-set fusion: rewritten plan failed native validation")
      return None
    }

    logDebug(s"Direct raw grouping-set fusion tagged aggregate over Expand: $newAggregate")
    Some(newAggregate)
  }

  private def uniqueOutputIndex(
      attribute: Attribute,
      output: Seq[Attribute]): Option[Int] = {
    val matching = output.indices.filter(index => output(index).semanticEquals(attribute))
    if (matching.length == 1) Some(matching.head) else None
  }

  private def resolvableFrom(references: AttributeSet, candidates: Seq[Attribute]): Boolean = {
    references.forall(reference => candidates.exists(_.semanticEquals(reference)))
  }

  private def buildPassThroughBySlot(
      expand: ExpandExecTransformer): IndexedSeq[Option[Attribute]] = {
    expand.output.indices.map {
      slot =>
        val attributes = expand.projections.collect {
          case projection if projection(slot).isInstanceOf[Attribute] =>
            projection(slot).asInstanceOf[Attribute]
        }
        attributes.headOption.filter(head => attributes.forall(_.semanticEquals(head)))
    }
  }

  private def semanticDistinct(attributes: Seq[Attribute]): Seq[Attribute] = {
    attributes.foldLeft(Seq.empty[Attribute]) {
      case (distinct, attribute)
          if distinct.exists(_.semanticEquals(attribute)) =>
        distinct
      case (distinct, attribute) => distinct :+ attribute
    }
  }

  private def numLatticeRoots(masks: Seq[Seq[Boolean]]): Int = {
    if (masks.isEmpty) {
      return 0
    }

    val candidatesContainingKey = Array.fill[Long](masks.head.length)(0L)
    masks.zipWithIndex.foreach {
      case (mask, setIndex) =>
        val candidateBit = 1L << setIndex
        mask.zipWithIndex.foreach {
          case (true, keyIndex) =>
            candidatesContainingKey(keyIndex) |= candidateBit
          case _ => ()
        }
    }

    val allCandidates =
      if (masks.length == 64) -1L else (1L << masks.length) - 1L
    masks.zipWithIndex.count {
      case (mask, setIndex) =>
        var candidates = allCandidates & ~(1L << setIndex)
        var keyIndex = 0
        while (keyIndex < mask.length && candidates != 0L) {
          if (mask(keyIndex)) {
            candidates &= candidatesContainingKey(keyIndex)
          }
          keyIndex += 1
        }
        candidates == 0L
    }
  }

  private def isSupportedGroupingKeyType(dataType: DataType): Boolean = {
    dataType match {
      case dt if dt.typeName == "timestamp_ntz" => true
      case BooleanType | StringType | DateType | TimestampType | BinaryType =>
        true
      case _: NumericType => true
      case _ => false
    }
  }

  private def passesNativeValidation(plan: SparkPlan): Boolean = {
    plan match {
      case validatable: ValidatablePlan =>
        try {
          validatable.doValidate().ok()
        } catch {
          case NonFatal(exception) =>
            logDebug(
              s"Direct raw grouping-set fusion: validation threw for ${plan.nodeName}: " +
                s"${exception.getMessage}")
            false
        }
      case _ => true
    }
  }
}
