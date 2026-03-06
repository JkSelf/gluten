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
import org.apache.spark.sql.catalyst.trees.TreeNodeTag

/**
 * TreeNodeTag for storing original join keys before Spark's transformations. This approach uses
 * Spark's native tag mechanism instead of a global registry, providing better thread safety and
 * supporting cross-thread scenarios like lateral joins.
 */
object JoinKeysTag {

  /**
   * Tag to store original join keys on logical plan nodes. The keys are stored on the left and
   * right child nodes of a join.
   */
  val ORIGINAL_JOIN_KEYS: TreeNodeTag[Seq[Expression]] =
    TreeNodeTag[Seq[Expression]]("gluten.originalJoinKeys")
}

// Made with Bob
