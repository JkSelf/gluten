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

import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan

import scala.collection.mutable

/**
 * Registry to store join keys extracted during query planning phase. This registry captures join
 * keys from logical plan before Spark's transformations (like rewriteKeyExpr) modify them, allowing
 * backends to access the original keys during physical plan execution.
 *
 * Ultra-fast ThreadLocal design with identity-based lookup:
 *   1. Each thread (query) has its own isolated storage - zero contention 2. Uses
 *      System.identityHashCode for O(1) lookup - no expensive canonicalization 3. Automatic cleanup
 *      when thread completes - no memory leaks 4. No synchronization overhead - much faster than
 *      concurrent maps 5. Natural isolation between concurrent queries
 *
 * This approach is ideal for Spark's query planning which happens on a single thread per query.
 */
object JoinKeysRegistry {

  /**
   * Container for join key information. Stores keys for a specific side (left or right) of the
   * join.
   * @param keys
   *   The join keys for this side
   */
  case class JoinKeysInfo(keys: Seq[Expression])

  /**
   * ThreadLocal storage for join keys. Each thread gets its own HashMap. Key: identity hash of
   * LogicalPlan (System.identityHashCode), Value: JoinKeysInfo
   *
   * Using identity hash provides:
   *   - O(1) lookup with no computation overhead
   *   - No expensive canonicalization or toString() calls
   *   - Direct object identity matching
   *
   * Using ThreadLocal provides:
   *   - Zero contention between concurrent queries
   *   - No synchronization overhead
   *   - Automatic memory cleanup when thread completes
   *   - Perfect isolation for query planning phase
   */
  private val threadLocalRegistry = new ThreadLocal[mutable.HashMap[Int, JoinKeysInfo]] {
    override def initialValue(): mutable.HashMap[Int, JoinKeysInfo] = {
      new mutable.HashMap[Int, JoinKeysInfo]()
    }
  }

  /**
   * Register join keys for a logical plan node (left or right child of join). Uses object identity
   * for ultra-fast O(1) registration.
   * @param plan
   *   The logical plan node (left or right child)
   * @param keys
   *   The join keys for this side
   */
  def register(plan: LogicalPlan, keys: Seq[Expression]): Unit = {
    // Use object identity hash - fastest possible key generation
    val key = System.identityHashCode(plan)
    threadLocalRegistry.get().put(key, JoinKeysInfo(keys))
  }

  /**
   * Lookup join keys by logical plan. Uses object identity for ultra-fast O(1) lookup.
   * @param plan
   *   The logical plan node
   * @return
   *   Optional JoinKeysInfo if found
   */
  def lookup(plan: LogicalPlan): Option[JoinKeysInfo] = {
    // Use object identity hash - fastest possible lookup
    val key = System.identityHashCode(plan)
    threadLocalRegistry.get().get(key)
  }

  /**
   * Clear all registered join keys for the current thread. Should be called after query execution
   * completes.
   */
  def clear(): Unit = {
    threadLocalRegistry.get().clear()
  }

}
