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

import org.apache.spark.sql.catalyst.planning.{ExtractEquiJoinKeys, ExtractSingleColumnNullAwareAntiJoin}
import org.apache.spark.sql.catalyst.plans.logical.{Join, LogicalPlan}
import org.apache.spark.sql.execution.{SparkPlan, SparkStrategy}

/**
 * Strategy to capture join keys from logical plan before Spark's JoinSelection transforms them.
 * This strategy runs early in the planning phase to preserve the original join keys before any
 * transformations like rewriteKeyExpr.
 *
 * This strategy does not generate any physical plans (returns Nil), allowing Spark's standard
 * JoinSelection strategy to continue with normal planning. It only serves to capture and store join
 * keys using TreeNodeTag for later retrieval during physical plan execution.
 *
 * The captured keys are used by backends (e.g., Velox) to optimize broadcast operations by avoiding
 * performance regressions caused by key transformations.
 *
 * Performance optimizations:
 *   1. Quick type check before expensive pattern matching 2. Early return for non-join nodes 3.
 *      Uses TreeNodeTag instead of global registry for thread safety
 *
 * Note: This strategy is only registered when enableJoinKeysRewrite() is false, meaning the backend
 * does not support join key rewriting and needs the original keys.
 */
case class GlutenJoinKeysCapture() extends SparkStrategy {

  /**
   * Apply the strategy to a logical plan. Captures join keys from equi-join nodes and stores them
   * in JoinKeysRegistry.
   *
   * @param plan
   *   The logical plan to analyze.
   * @return
   *   Empty sequence (Nil) to allow other strategies to continue.
   */
  def apply(plan: LogicalPlan): Seq[SparkPlan] = {
    // Quick type check: only process Join nodes to avoid expensive pattern matching
    // on non-join nodes (most nodes in a plan are not joins)
    if (!plan.isInstanceOf[Join]) {
      return Nil
    }

    plan match {
      // Use Spark's ExtractEquiJoinKeys extractor to match equi-join patterns
      // and extract the join keys, join type, and other join information.
      case ExtractEquiJoinKeys(_, leftKeys, rightKeys, _, _, left, right, _) =>
        // Store keys using TreeNodeTag on left and right child plans
        // This allows BroadcastExchange nodes to find the correct keys
        // based on which side (left or right) they are broadcasting.
        // TreeNodeTag provides thread-safe storage without global registry.
        if (leftKeys.nonEmpty) {
          left.setTagValue(JoinKeysTag.ORIGINAL_JOIN_KEYS, leftKeys)
        }
        if (rightKeys.nonEmpty) {
          right.setTagValue(JoinKeysTag.ORIGINAL_JOIN_KEYS, rightKeys)
        }
        // Return Nil to indicate this strategy does not produce a physical plan
        // This allows Spark's JoinSelection strategy to handle the actual planning.
        Nil

      case j @ ExtractSingleColumnNullAwareAntiJoin(leftKeys, rightKeys) =>
        // Store keys using TreeNodeTag on left and right child plans
        // This allows BroadcastExchange nodes to find the correct keys
        // based on which side (left or right) they are broadcasting.
        if (leftKeys.nonEmpty) {
          j.left.setTagValue(JoinKeysTag.ORIGINAL_JOIN_KEYS, leftKeys)
        }
        if (rightKeys.nonEmpty) {
          j.right.setTagValue(JoinKeysTag.ORIGINAL_JOIN_KEYS, rightKeys)
        }
        // Return Nil to indicate this strategy does not produce a physical plan
        // This allows Spark's JoinSelection strategy to handle the actual planning.
        Nil

      // For non-equi-join or other plan nodes, return Nil.
      case _ => Nil
    }
  }
}
