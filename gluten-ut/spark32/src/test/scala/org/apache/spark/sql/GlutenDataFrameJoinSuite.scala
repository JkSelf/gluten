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

package org.apache.spark.sql

import scala.collection.JavaConverters._
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.catalyst.plans.{Inner, InnerLike, LeftOuter, RightOuter}
import org.apache.spark.sql.catalyst.plans.logical.{BROADCAST, Filter, HintInfo, Join, JoinHint, LogicalPlan, Project}
import org.apache.spark.sql.connector.catalog.CatalogManager
import org.apache.spark.sql.execution.FileSourceScanExec
import org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanHelper
import org.apache.spark.sql.execution.analysis.DetectAmbiguousSelfJoin.LogicalPlanWithDatasetId
import org.apache.spark.sql.execution.datasources.LogicalRelation
import org.apache.spark.sql.execution.exchange.BroadcastExchangeExec
import org.apache.spark.sql.execution.joins.BroadcastHashJoinExec
import org.apache.spark.sql.functions._
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, ShuffledHashJoinExec}

class GlutenDataFrameJoinSuite extends DataFrameJoinSuite with GlutenSQLTestsTrait {

  override def testNameBlackList: Seq[String] = Seq(
    "Supports multi-part names for broadcast hint resolution"
  )
  import testImplicits._

  test("gluten join - join using multiple columns and specifying join type") {
    val df = Seq((1, 2, "1"), (3, 4, "3")).toDF("int", "int2", "str")
    val df2 = Seq((1, 3, "1"), (5, 6, "5")).toDF("int", "int2", "str")

    checkAnswer(
      df.join(df2, Seq("int", "str"), "inner"),
      Row(1, "1", 2, 3) :: Nil)
  }

}
