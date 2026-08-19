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

import org.apache.gluten.backendsapi.BackendsApiManager
import org.apache.gluten.expression.ConverterUtils
import org.apache.gluten.substrait.`type`.{TypeBuilder, TypeNode}
import org.apache.gluten.substrait.SubstraitContext
import org.apache.gluten.substrait.expression.ExpressionNode
import org.apache.gluten.substrait.extensions.ExtensionBuilder
import org.apache.gluten.substrait.rel.{RelBuilder, RelNode}

import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.catalyst.trees.TreeNodeTag

import com.google.protobuf.StringValue

import java.util.{List => JList}

/**
 * Carries the raw grouping-set fusion hint from the Spark plan into Substrait.
 *
 * The hint is deliberately isolated from [[ExpandExecTransformer]] so the ordinary Expand
 * implementation remains unchanged. It is advisory: native rejection reconstructs the complete
 * Aggregate-over-Expand pair.
 */
private[gluten] object RawGroupingSetFusion {
  private val MarkerTag: TreeNodeTag[Boolean] =
    TreeNodeTag[Boolean]("org.apache.gluten.execution.RawGroupingSetAggregation")

  // Keep this payload in sync with RawGroupingSetPlanConverter.cc.
  private val Optimization = "rawGroupingSetFusion=1\n"

  def isMarked(expand: ExpandExecTransformer): Boolean =
    expand.getTagValue(MarkerTag).getOrElse(false)

  def mark(expand: ExpandExecTransformer): Unit =
    expand.setTagValue(MarkerTag, true)

  def makeExpandRel(
      input: RelNode,
      projections: JList[JList[ExpressionNode]],
      originalInputAttributes: Seq[Attribute],
      context: SubstraitContext,
      operatorId: Long,
      validation: Boolean): RelNode = {
    val optimization = BackendsApiManager.getTransformerApiInstance.packPBMessage(
      StringValue.newBuilder.setValue(Optimization).build)
    val enhancement = if (validation) {
      val inputTypes = new java.util.ArrayList[TypeNode]()
      originalInputAttributes.foreach {
        attribute =>
          inputTypes.add(ConverterUtils.getTypeNode(attribute.dataType, attribute.nullable))
      }
      BackendsApiManager.getTransformerApiInstance.packPBMessage(
        TypeBuilder.makeStruct(false, inputTypes).toProtobuf)
    } else {
      null
    }

    val extension = ExtensionBuilder.makeAdvancedExtension(optimization, enhancement)
    RelBuilder.makeExpandRel(input, projections, extension, context, operatorId)
  }
}
