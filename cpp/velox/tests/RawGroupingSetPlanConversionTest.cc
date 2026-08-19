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

#include <google/protobuf/wrappers.pb.h>

#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "operators/plannodes/GroupingSetAggregationNode.h"
#include "substrait/RawGroupingSetPlanConverter.h"
#include "substrait/SubstraitToVeloxPlan.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/type/Type.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace gluten {
namespace {

::substrait::Expression fieldSelection(int32_t channel) {
  ::substrait::Expression expression;
  expression.mutable_selection()->mutable_direct_reference()->mutable_struct_field()->set_field(channel);
  return expression;
}

::substrait::Expression i64Literal(int64_t value) {
  ::substrait::Expression expression;
  expression.mutable_literal()->set_i64(value);
  return expression;
}

::substrait::Expression nullI64Literal() {
  ::substrait::Expression expression;
  expression.mutable_literal()->mutable_null()->mutable_i64()->set_nullability(
      ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
  return expression;
}

void setNullableI64(::substrait::Type* type) {
  type->mutable_i64()->set_nullability(::substrait::Type_Nullability_NULLABILITY_NULLABLE);
}

void setNullableI32(::substrait::Type* type) {
  type->mutable_i32()->set_nullability(::substrait::Type_Nullability_NULLABILITY_NULLABLE);
}

void setOptimization(::substrait::extensions::AdvancedExtension* extension, const std::string& value) {
  google::protobuf::StringValue optimization;
  optimization.set_value(value);
  extension->add_optimization()->PackFrom(optimization);
}

void addExpandProjection(::substrait::ExpandRel* expand, std::initializer_list<::substrait::Expression> expressions) {
  auto* duplicates = expand->add_fields()->mutable_switching_field()->mutable_duplicates();
  for (const auto& expression : expressions) {
    duplicates->Add()->CopyFrom(expression);
  }
}

/// Builds the raw shape produced by DirectRawGroupingSetAggregateRule:
///
///   Aggregate(group by key, gid; sum(measure))
///     Expand(measure passthrough, key|null, gid literal, unused)
///       [optional Project that re-grounds the measure]
///         Read(v, key, unused)
::substrait::Rel makeRawGroupingSetRel(bool unstableMeasure, bool withPreProject) {
  ::substrait::Rel rel;
  auto* aggregate = rel.mutable_aggregate();
  aggregate->mutable_common()->mutable_direct();
  setOptimization(aggregate->mutable_advanced_extension(), "allowFlush=1\n");

  auto* expand = aggregate->mutable_input()->mutable_expand();
  expand->mutable_common()->mutable_direct();
  setOptimization(expand->mutable_advanced_extension(), "rawGroupingSetFusion=1\n");

  ::substrait::Rel* readRel;
  int32_t rawMeasureChannel;
  if (withPreProject) {
    auto* project = expand->mutable_input()->mutable_project();
    project->mutable_common()->mutable_direct();
    project->add_expressions()->CopyFrom(fieldSelection(0));
    readRel = project->mutable_input();
    // A direct Substrait Project appends its expression after all input
    // columns. Expand passes that re-grounded measure through.
    rawMeasureChannel = 3;
  } else {
    readRel = expand->mutable_input();
    rawMeasureChannel = 0;
  }

  auto* read = readRel->mutable_read();
  read->mutable_common()->mutable_direct();
  auto* schema = read->mutable_base_schema();
  for (const auto* name : {"v", "k", "unused"}) {
    schema->add_names(name);
    setNullableI64(schema->mutable_struct_()->add_types());
  }

  addExpandProjection(expand, {fieldSelection(rawMeasureChannel), fieldSelection(1), i64Literal(0), fieldSelection(2)});
  addExpandProjection(expand, {fieldSelection(rawMeasureChannel), nullI64Literal(), i64Literal(1), fieldSelection(2)});

  auto* grouping = aggregate->add_groupings();
  aggregate->add_grouping_expressions()->CopyFrom(fieldSelection(1));
  grouping->add_expression_references(0);
  aggregate->add_grouping_expressions()->CopyFrom(fieldSelection(2));
  grouping->add_expression_references(1);

  auto* function = aggregate->add_measures()->mutable_measure();
  function->set_function_reference(1);
  function->set_phase(::substrait::AGGREGATION_PHASE_INITIAL_TO_INTERMEDIATE);
  function->set_invocation(::substrait::AggregateFunction::AGGREGATION_INVOCATION_ALL);
  function->add_arguments()->mutable_value()->CopyFrom(fieldSelection(unstableMeasure ? 1 : 0));
  setNullableI64(function->mutable_output_type());
  return rel;
}

} // namespace

class RawGroupingSetPlanConversionTest : public exec::test::HiveConnectorTestBase {
 protected:
  std::shared_ptr<SubstraitToVeloxPlanConverter> makeRawPlanConverter(
      std::unordered_map<std::string, std::string> configs = {}) {
    rawVeloxCfg_ = std::make_shared<facebook::velox::config::ConfigBase>(std::move(configs));
    auto converter = std::make_shared<SubstraitToVeloxPlanConverter>(
        pool(),
        rawVeloxCfg_.get(),
        std::vector<std::shared_ptr<ResultIterator>>{},
        VeloxConnectorIds{.hive = facebook::velox::exec::test::kHiveConnectorId},
        std::nullopt,
        std::nullopt,
        /*validationMode=*/true);
    converter->constructFunctionMap(std::unordered_map<uint64_t, std::string>{{1, "sum:opt_i64"}});
    return converter;
  }

  void expectOrdinaryExpandAndAggregation(const core::PlanNodePtr& plan) {
    const auto aggregate = std::dynamic_pointer_cast<const core::AggregationNode>(plan);
    ASSERT_NE(aggregate, nullptr);
    ASSERT_EQ(aggregate->sources().size(), 1);
    const auto expand = std::dynamic_pointer_cast<const core::ExpandNode>(aggregate->sources().front());
    ASSERT_NE(expand, nullptr);
    ASSERT_EQ(expand->sources().size(), 1);
    EXPECT_NE(std::dynamic_pointer_cast<const core::TableScanNode>(expand->sources().front()), nullptr);
  }

  std::shared_ptr<facebook::velox::config::ConfigBase> rawVeloxCfg_;
};

TEST_F(RawGroupingSetPlanConversionTest, rawGroupingSetPairFusesAndRestoresAggregateSchema) {
  auto converter = makeRawPlanConverter();
  auto plan = converter->toVeloxPlan(makeRawGroupingSetRel(
      /*unstableMeasure=*/false,
      /*withPreProject=*/true));

  const auto schemaProject = std::dynamic_pointer_cast<const core::ProjectNode>(plan);
  ASSERT_NE(schemaProject, nullptr);
  ASSERT_EQ(schemaProject->sources().size(), 1);

  const auto fused = std::dynamic_pointer_cast<const facebook::velox::exec::GroupingSetAggregationNode>(
      schemaProject->sources().front());
  ASSERT_NE(fused, nullptr);
  ASSERT_EQ(fused->groupingSets().size(), 2);
  EXPECT_EQ(fused->groupingSets()[0].activeKeysMask, 1);
  EXPECT_EQ(fused->groupingSets()[0].groupingId, 0);
  EXPECT_EQ(fused->groupingSets()[1].activeKeysMask, 0);
  EXPECT_EQ(fused->groupingSets()[1].groupingId, 1);

  // The pre-project remains below the two-node fused replacement.
  ASSERT_EQ(fused->sources().size(), 1);
  EXPECT_NE(std::dynamic_pointer_cast<const core::ProjectNode>(fused->sources().front()), nullptr);

  // AggregateRel output is groupings in original order followed by packed
  // accumulators. Unused Expand columns are deliberately absent.
  ASSERT_EQ(plan->outputType()->size(), 3);
  EXPECT_EQ(plan->outputType()->childAt(0), BIGINT());
  EXPECT_EQ(plan->outputType()->childAt(1), BIGINT());
  EXPECT_EQ(plan->outputType()->childAt(2), BIGINT());
  EXPECT_EQ(plan->outputType()->nameOf(0), "n2_1");
  EXPECT_EQ(plan->outputType()->nameOf(1), "n2_2");
  EXPECT_EQ(plan->outputType()->nameOf(2), "n3_2");
}

TEST_F(RawGroupingSetPlanConversionTest, rawGroupingSetPairRestoresBothOperatorsOnRejection) {
  auto converter = makeRawPlanConverter();
  auto plan = converter->toVeloxPlan(makeRawGroupingSetRel(
      /*unstableMeasure=*/true,
      /*withPreProject=*/false));

  const auto aggregate = std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  ASSERT_NE(aggregate, nullptr);
  ASSERT_EQ(aggregate->sources().size(), 1);
  const auto expand = std::dynamic_pointer_cast<const core::ExpandNode>(aggregate->sources().front());
  ASSERT_NE(expand, nullptr);
  ASSERT_EQ(expand->sources().size(), 1);
  EXPECT_NE(std::dynamic_pointer_cast<const core::TableScanNode>(expand->sources().front()), nullptr);

  // Direct raw fusion has a two-for-two slot contract. A rejected hint must
  // restore Expand and Aggregate without identity-Project padding.
  EXPECT_EQ(plan->id(), "2");
  EXPECT_EQ(expand->id(), "1");
}

TEST_F(RawGroupingSetPlanConversionTest, rawGroupingSetDuplicateMasksRestoreBothOperators) {
  auto rel = makeRawGroupingSetRel(
      /*unstableMeasure=*/false,
      /*withPreProject=*/false);
  auto* expand = rel.mutable_aggregate()->mutable_input()->mutable_expand();
  expand->mutable_fields(1)->mutable_switching_field()->mutable_duplicates(1)->CopyFrom(fieldSelection(1));

  auto converter = makeRawPlanConverter();
  expectOrdinaryExpandAndAggregation(converter->toVeloxPlan(rel));
}

TEST_F(RawGroupingSetPlanConversionTest, rawGroupingSetMultipleRootsRestoreBothOperators) {
  auto rel = makeRawGroupingSetRel(
      /*unstableMeasure=*/false,
      /*withPreProject=*/false);
  auto* aggregate = rel.mutable_aggregate();
  auto* expand = aggregate->mutable_input()->mutable_expand();
  expand->mutable_fields(0)->mutable_switching_field()->mutable_duplicates(3)->CopyFrom(nullI64Literal());
  aggregate->add_grouping_expressions()->CopyFrom(fieldSelection(3));
  aggregate->mutable_groupings(0)->add_expression_references(2);

  auto converter = makeRawPlanConverter();
  expectOrdinaryExpandAndAggregation(converter->toVeloxPlan(rel));
}

TEST_F(RawGroupingSetPlanConversionTest, rawGroupingSetConfiguredMaximumRestoresBothOperators) {
  auto rel = makeRawGroupingSetRel(
      /*unstableMeasure=*/false,
      /*withPreProject=*/false);
  auto converter = makeRawPlanConverter({{kFusedGroupingSetAggregateMaxGroupingSets, "1"}});

  expectOrdinaryExpandAndAggregation(converter->toVeloxPlan(rel));
}

TEST_F(RawGroupingSetPlanConversionTest, rawGroupingSetUnsupportedPhaseRestoresBothOperators) {
  auto rel = makeRawGroupingSetRel(
      /*unstableMeasure=*/false,
      /*withPreProject=*/false);
  rel.mutable_aggregate()->mutable_measures(0)->mutable_measure()->set_phase(
      ::substrait::AGGREGATION_PHASE_INITIAL_TO_RESULT);

  auto converter = makeRawPlanConverter();
  expectOrdinaryExpandAndAggregation(converter->toVeloxPlan(rel));
}

TEST_F(RawGroupingSetPlanConversionTest, rawGroupingSetLateRejectionReusesConvertedSource) {
  auto rel = makeRawGroupingSetRel(
      /*unstableMeasure=*/false,
      /*withPreProject=*/false);
  auto* measureType = rel.mutable_aggregate()
                          ->mutable_input()
                          ->mutable_expand()
                          ->mutable_input()
                          ->mutable_read()
                          ->mutable_base_schema()
                          ->mutable_struct_()
                          ->mutable_types(0);
  measureType->Clear();
  setNullableI32(measureType);

  auto converter = makeRawPlanConverter();
  auto plan = converter->toVeloxPlan(rel);
  const auto aggregate = std::dynamic_pointer_cast<const core::AggregationNode>(plan);
  ASSERT_NE(aggregate, nullptr);
  const auto expand = std::dynamic_pointer_cast<const core::ExpandNode>(aggregate->sources().front());
  ASSERT_NE(expand, nullptr);
  const auto scan = std::dynamic_pointer_cast<const core::TableScanNode>(expand->sources().front());
  ASSERT_NE(scan, nullptr);

  // The source-schema mismatch is discovered only after Read conversion.
  // Reusing that TableScan keeps the original source/Expand/Aggregate ids
  // contiguous and, more importantly, avoids consuming a stream twice.
  EXPECT_EQ(scan->id(), "0");
  EXPECT_EQ(expand->id(), "1");
  EXPECT_EQ(aggregate->id(), "2");
}

} // namespace gluten
