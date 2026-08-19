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

#include "RawGroupingSetPlanConverter.h"

#include <glog/logging.h>
#include <algorithm>
#include <optional>
#include <unordered_set>
#include <utility>

#include "SubstraitParser.h"
#include "SubstraitToVeloxPlan.h"
#include "operators/plannodes/GroupingSetAggregationNode.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/AggregateFunctionRegistry.h"

namespace gluten {
namespace {

// Keep this key in sync with
// RawGroupingSetFusion.Optimization. This is only an
// optimization hint: absent or false values retain the ordinary
// Aggregate -> Expand plan.
constexpr char kRawGroupingSetFusionConfig[] = "rawGroupingSetFusion=";

} // namespace

RawGroupingSetPlanConverter::RawGroupingSetPlanConverter(SubstraitToVeloxPlanConverter& owner) : owner_(owner) {}

core::PlanNodePtr RawGroupingSetPlanConverter::tryConvert(const ::substrait::AggregateRel& aggRel) {
  VELOX_CHECK(aggRel.has_input(), "Child Rel is expected in AggregateRel.");

  if (!aggRel.input().has_expand()) {
    return nullptr;
  }

  const auto& expandRel = aggRel.input().expand();
  const bool isRawGroupingSetFusion = expandRel.has_advanced_extension() &&
      SubstraitParser::configSetInOptimization(expandRel.advanced_extension(), kRawGroupingSetFusionConfig);
  if (!isRawGroupingSetFusion) {
    return nullptr;
  }

  VELOX_CHECK(expandRel.has_input(), "Child Rel is expected in a raw-fusion ExpandRel.");

  const auto logFallback = [](const VeloxException& e) {
    LOG(WARNING) << "Raw-input fused grouping-set aggregation is not applicable to this "
                    "Aggregate -> Expand pair; falling back to both ordinary operators. Reason: "
                 << e.what();
  };

  // Reject unsupported protobuf shapes and aggregate implementations before
  // converting the source. This avoids consuming scans or input iterators for
  // an optimization that is already known not to apply.
  try {
    validate(aggRel, expandRel);
  } catch (const VeloxException& e) {
    logFallback(e);
    auto expandInputNode = owner_.toVeloxPlan(expandRel.input());
    auto expandNode = owner_.makeExpandNode(expandRel, expandInputNode);
    return owner_.makeAggregateNode(aggRel, std::move(expandNode));
  }

  // Convert the source below Expand exactly once. Any remaining rejection is
  // type- or source-schema-dependent; reuse this converted node to construct
  // the complete original pair.
  auto expandInputNode = owner_.toVeloxPlan(expandRel.input());
  const auto pairPlanNodeId = owner_.planNodeId_;
  try {
    return convert(aggRel, expandRel, expandInputNode);
  } catch (const VeloxException& e) {
    logFallback(e);
    // Raw conversion may have reserved one of the pair's two native plan node
    // ids before a schema-dependent check failed. Restore the first id so
    // fallback retains the ordinary Expand/Aggregate metric slots.
    owner_.planNodeId_ = pairPlanNodeId;
    auto expandNode = owner_.makeExpandNode(expandRel, expandInputNode);
    return owner_.makeAggregateNode(aggRel, std::move(expandNode));
  }
}

namespace {

/// Returns the referenced field index if 'expr' is a direct struct-field selection.
std::optional<uint32_t> selectionFieldIndex(const ::substrait::Expression& expr) {
  if (!expr.has_selection() || !expr.selection().has_direct_reference()) {
    return std::nullopt;
  }
  uint32_t fieldIndex = 0;
  if (!SubstraitParser::parseReferenceSegment(expr.selection().direct_reference(), fieldIndex)) {
    return std::nullopt;
  }
  return fieldIndex;
}

/// Returns the value if 'expr' is an i64 literal. GroupingSetAggregationNode's
/// gid channel is BIGINT, so accepting a narrower literal here would change the
/// tagged Expand's output schema. Older or narrowed plans use the plain-Expand
/// fallback instead.
std::optional<int64_t> groupingIdLiteralValue(const ::substrait::Expression& expr) {
  if (!expr.has_literal() ||
      expr.literal().literal_type_case() != ::substrait::Expression_Literal::LiteralTypeCase::kI64) {
    return std::nullopt;
  }
  return expr.literal().i64();
}

bool isNullLiteral(const ::substrait::Expression& expr) {
  return expr.has_literal() &&
      expr.literal().literal_type_case() == ::substrait::Expression_Literal::LiteralTypeCase::kNull;
}

void validateRawGroupingSetAggregateName(const std::string& name, int32_t measureIndex) {
  VELOX_CHECK(
      name == "sum" || name == "avg" || name == "count" || name == "min" || name == "max",
      "Raw grouping-set fusion does not support aggregate function '{}' in measure {}.",
      name,
      measureIndex);
}

struct RawGroupingSlot {
  uint32_t expandChannel;
  std::optional<uint32_t> rawChannel;

  bool isGroupingId() const {
    return !rawChannel.has_value();
  }
};

struct RawGroupingSetShape {
  int32_t projectionWidth;
  std::vector<RawGroupingSlot> groupingSlots;
  std::vector<facebook::velox::exec::GroupingSetSpec> groupingSets;

  int32_t numKeys() const {
    return static_cast<int32_t>(groupingSlots.size()) - 1;
  }
};

/// Returns whether one Expand output slot is the same raw field or the same
/// literal in every projection. Raw aggregate arguments may use only these
/// stable pass-through slots; a grouping-key slot that switches between a
/// field and NULL must remain set-specific and therefore cannot be fused.
bool isStableRawExpandSlot(const ::substrait::ExpandRel& expandRel, uint32_t channel) {
  const auto& first = expandRel.fields(0).switching_field().duplicates(channel);
  if (const auto firstField = selectionFieldIndex(first)) {
    for (int32_t setIdx = 1; setIdx < expandRel.fields_size(); ++setIdx) {
      const auto field = selectionFieldIndex(expandRel.fields(setIdx).switching_field().duplicates(channel));
      if (!field.has_value() || *field != *firstField) {
        return false;
      }
    }
    return true;
  }
  if (first.has_literal()) {
    const auto serialized = first.literal().SerializeAsString();
    for (int32_t setIdx = 1; setIdx < expandRel.fields_size(); ++setIdx) {
      const auto& expression = expandRel.fields(setIdx).switching_field().duplicates(channel);
      if (!expression.has_literal() || expression.literal().SerializeAsString() != serialized) {
        return false;
      }
    }
    return true;
  }
  return false;
}

RawGroupingSetShape parseRawGroupingSetShape(
    const ::substrait::AggregateRel& aggRel,
    const ::substrait::ExpandRel& expandRel,
    int32_t maxGroupingSets) {
  using GroupingSetMask = facebook::velox::exec::GroupingSetMask;
  using GroupingSetSpec = facebook::velox::exec::GroupingSetSpec;

  VELOX_CHECK(
      !aggRel.has_common() || aggRel.common().emit_kind_case() == ::substrait::RelCommon::EmitKindCase::kDirect,
      "Raw grouping-set fusion requires direct AggregateRel output.");
  VELOX_CHECK(
      !expandRel.has_common() || expandRel.common().emit_kind_case() == ::substrait::RelCommon::EmitKindCase::kDirect,
      "Raw grouping-set fusion requires direct ExpandRel output.");
  VELOX_CHECK_EQ(aggRel.groupings_size(), 1, "Raw grouping-set fusion requires exactly one AggregateRel grouping.");
  VELOX_CHECK_GT(aggRel.measures_size(), 0, "Raw grouping-set fusion requires at least one aggregate measure.");

  const int32_t numSets = expandRel.fields_size();
  VELOX_CHECK_GT(numSets, 0, "Raw grouping-set fusion requires at least one Expand projection.");
  VELOX_CHECK_GE(maxGroupingSets, 1, "The fused grouping-set maximum must be positive.");
  VELOX_CHECK_LE(
      maxGroupingSets,
      facebook::velox::exec::kMaxGroupingSets,
      "The fused grouping-set maximum must not exceed {}.",
      facebook::velox::exec::kMaxGroupingSets);
  VELOX_CHECK_LE(
      numSets,
      maxGroupingSets,
      "Raw grouping-set fusion has {} sets, exceeding the configured maximum {}.",
      numSets,
      maxGroupingSets);

  VELOX_CHECK(expandRel.fields(0).has_switching_field(), "Raw grouping-set fusion requires switching Expand fields.");
  const int32_t projectionWidth = expandRel.fields(0).switching_field().duplicates_size();
  VELOX_CHECK_GT(projectionWidth, 0, "Raw grouping-set fusion requires non-empty Expand projections.");
  for (int32_t setIdx = 0; setIdx < numSets; ++setIdx) {
    VELOX_CHECK(
        expandRel.fields(setIdx).has_switching_field(),
        "Raw grouping-set fusion projection {} is not a switching field.",
        setIdx);
    VELOX_CHECK_EQ(
        expandRel.fields(setIdx).switching_field().duplicates_size(),
        projectionWidth,
        "Raw grouping-set fusion projection {} has a different width.",
        setIdx);
  }

  RawGroupingSetShape shape;
  shape.projectionWidth = projectionWidth;
  std::unordered_set<uint32_t> seenGroupingChannels;
  std::unordered_set<uint32_t> seenRawKeyChannels;
  int32_t numGroupingIds = 0;

  const auto& groupingExpressionRefs = aggRel.groupings(0).expression_references();
  VELOX_CHECK_GT(
      groupingExpressionRefs.size(), 0, "Raw grouping-set fusion requires AggregateRel grouping expressions.");
  shape.groupingSlots.reserve(groupingExpressionRefs.size());
  for (int32_t groupingIdx = 0; groupingIdx < groupingExpressionRefs.size(); ++groupingIdx) {
    const auto expressionRef = groupingExpressionRefs.Get(groupingIdx);
    VELOX_CHECK_LT(
        expressionRef,
        static_cast<uint32_t>(aggRel.grouping_expressions_size()),
        "Raw grouping-set fusion grouping expression {} references expression {}, but the pool has {} expressions.",
        groupingIdx,
        expressionRef,
        aggRel.grouping_expressions_size());
    const auto expandChannel = selectionFieldIndex(aggRel.grouping_expressions(expressionRef));
    VELOX_CHECK(
        expandChannel.has_value(),
        "Raw grouping-set fusion grouping expression {} must be a top-level field.",
        groupingIdx);
    VELOX_CHECK_LT(
        *expandChannel,
        static_cast<uint32_t>(projectionWidth),
        "Raw grouping-set fusion grouping expression {} references Expand channel {}, but the width is {}.",
        groupingIdx,
        *expandChannel,
        projectionWidth);
    VELOX_CHECK(
        seenGroupingChannels.insert(*expandChannel).second,
        "Raw grouping-set fusion has duplicate grouping channel {}.",
        *expandChannel);

    bool allGroupingIdLiterals = true;
    for (int32_t setIdx = 0; setIdx < numSets; ++setIdx) {
      if (!groupingIdLiteralValue(expandRel.fields(setIdx).switching_field().duplicates(*expandChannel)).has_value()) {
        allGroupingIdLiterals = false;
        break;
      }
    }
    if (allGroupingIdLiterals) {
      ++numGroupingIds;
      shape.groupingSlots.push_back(RawGroupingSlot{*expandChannel, std::nullopt});
      continue;
    }

    std::optional<uint32_t> rawChannel;
    for (int32_t setIdx = 0; setIdx < numSets; ++setIdx) {
      const auto& expression = expandRel.fields(setIdx).switching_field().duplicates(*expandChannel);
      if (const auto selectedChannel = selectionFieldIndex(expression)) {
        if (rawChannel.has_value()) {
          VELOX_CHECK_EQ(
              *rawChannel,
              *selectedChannel,
              "Raw grouping key {} reads different source channels across sets.",
              groupingIdx);
        } else {
          rawChannel = *selectedChannel;
        }
      } else {
        VELOX_CHECK(
            isNullLiteral(expression),
            "Raw grouping key {} is neither a top-level field nor a null literal in set {}.",
            groupingIdx,
            setIdx);
      }
    }
    VELOX_CHECK(rawChannel.has_value(), "Raw grouping key {} is null in every grouping set.", groupingIdx);
    VELOX_CHECK(
        seenRawKeyChannels.insert(*rawChannel).second,
        "Raw grouping-set fusion maps multiple grouping keys to source channel {}.",
        *rawChannel);
    shape.groupingSlots.push_back(RawGroupingSlot{*expandChannel, rawChannel});
  }

  VELOX_CHECK_EQ(
      numGroupingIds,
      1,
      "Raw grouping-set fusion requires exactly one all-i64-literal grouping-id channel, but found {}.",
      numGroupingIds);
  VELOX_CHECK_GT(shape.numKeys(), 0, "Raw grouping-set fusion requires at least one grouping key.");
  VELOX_CHECK_LE(shape.numKeys(), 64, "Raw grouping-set fusion supports at most 64 grouping keys.");

  shape.groupingSets.reserve(numSets);
  for (int32_t setIdx = 0; setIdx < numSets; ++setIdx) {
    GroupingSetSpec set;
    int32_t keyIdx = 0;
    for (const auto& groupingSlot : shape.groupingSlots) {
      const auto& expression = expandRel.fields(setIdx).switching_field().duplicates(groupingSlot.expandChannel);
      if (groupingSlot.isGroupingId()) {
        const auto gid = groupingIdLiteralValue(expression);
        VELOX_CHECK(gid.has_value(), "Raw grouping-set fusion grouping-id must be an i64 literal in set {}.", setIdx);
        set.groupingId = *gid;
      } else {
        if (selectionFieldIndex(expression).has_value()) {
          set.activeKeysMask |= GroupingSetMask{1} << keyIdx;
        } else {
          VELOX_CHECK(
              isNullLiteral(expression), "Raw grouping key {} is neither a field nor null in set {}.", keyIdx, setIdx);
        }
        ++keyIdx;
      }
    }
    shape.groupingSets.push_back(std::move(set));
  }

  // A raw aggregate argument can be a literal itself, or it can select an
  // Expand slot that is a stable raw-field/literal pass-through. Any
  // set-varying slot would require one aggregate instance per set and is not
  // equivalent to the raw cascade.
  for (int32_t measureIdx = 0; measureIdx < aggRel.measures_size(); ++measureIdx) {
    const auto& measure = aggRel.measures(measureIdx);
    VELOX_CHECK(
        !measure.has_filter() || measure.filter().ByteSizeLong() == 0,
        "Raw grouping-set fusion does not support masked aggregates.");
    const auto& aggregate = measure.measure();
    VELOX_CHECK(
        aggregate.phase() == ::substrait::AGGREGATION_PHASE_INITIAL_TO_INTERMEDIATE,
        "Raw grouping-set fusion requires INITIAL_TO_INTERMEDIATE measures.");
    VELOX_CHECK(
        aggregate.invocation() != ::substrait::AggregateFunction::AGGREGATION_INVOCATION_DISTINCT,
        "Raw grouping-set fusion does not support DISTINCT measures.");
    VELOX_CHECK_EQ(aggregate.sorts_size(), 0, "Raw grouping-set fusion does not support ordered measures.");
    VELOX_CHECK(aggregate.has_output_type(), "Raw grouping-set fusion measure {} has no output type.", measureIdx);

    for (int32_t argIdx = 0; argIdx < aggregate.arguments_size(); ++argIdx) {
      const auto& argument = aggregate.arguments(argIdx);
      VELOX_CHECK(argument.has_value(), "Raw grouping-set fusion supports only value arguments.");
      const auto& value = argument.value();
      if (value.has_literal()) {
        continue;
      }
      const auto expandChannel = selectionFieldIndex(value);
      VELOX_CHECK(
          expandChannel.has_value(),
          "Raw grouping-set fusion aggregate argument {} of measure {} must be a literal or top-level field.",
          argIdx,
          measureIdx);
      VELOX_CHECK_LT(
          *expandChannel,
          static_cast<uint32_t>(projectionWidth),
          "Raw grouping-set fusion aggregate argument {} of measure {} references Expand channel {}, "
          "but the width is {}.",
          argIdx,
          measureIdx,
          *expandChannel,
          projectionWidth);
      VELOX_CHECK(
          isStableRawExpandSlot(expandRel, *expandChannel),
          "Raw grouping-set fusion aggregate argument {} of measure {} reads an Expand slot that varies across sets.",
          argIdx,
          measureIdx);
    }
  }

  std::vector<GroupingSetMask> masks;
  masks.reserve(shape.groupingSets.size());
  for (const auto& set : shape.groupingSets) {
    masks.push_back(set.activeKeysMask);
  }

  const auto derivationPlan = facebook::velox::exec::buildDerivationPlan(masks);
  const auto numRoots =
      std::count(derivationPlan.parent.begin(), derivationPlan.parent.end(), facebook::velox::exec::kRawInputParent);
  VELOX_CHECK_EQ(numRoots, 1, "Raw grouping-set fusion requires one derivation root, but this shape has {}.", numRoots);

  return shape;
}

} // namespace

void RawGroupingSetPlanConverter::validate(
    const ::substrait::AggregateRel& aggRel,
    const ::substrait::ExpandRel& expandRel) const {
  const auto maxGroupingSets = owner_.veloxCfg_->get<int32_t>(
      kFusedGroupingSetAggregateMaxGroupingSets, kFusedGroupingSetAggregateMaxGroupingSetsDefault);
  parseRawGroupingSetShape(aggRel, expandRel, maxGroupingSets);

  // Driver construction of MultiGroupingSetAggregation creates the same
  // aggregate instances. Probe the factories now so an unsupported function,
  // signature, or partial result type rejects the hint while the original pair
  // is still available as a fallback.
  const core::QueryConfig queryConfig(owner_.veloxCfg_->rawConfigs());
  for (int32_t measureIdx = 0; measureIdx < aggRel.measures_size(); ++measureIdx) {
    const auto& aggregate = aggRel.measures(measureIdx).measure();
    const auto baseName = SubstraitParser::findVeloxFunction(owner_.functionMap_, aggregate.function_reference());
    validateRawGroupingSetAggregateName(baseName, measureIdx);
    auto rawInputTypes = SubstraitParser::sigToTypes(
        SubstraitParser::findFunctionSpec(owner_.functionMap_, aggregate.function_reference()));
    VELOX_CHECK_EQ(
        rawInputTypes.size(),
        aggregate.arguments_size(),
        "Raw grouping-set fusion measure {} has {} arguments, but its function signature has {}.",
        measureIdx,
        aggregate.arguments_size(),
        rawInputTypes.size());

    const auto intermediateType = facebook::velox::exec::resolveIntermediateType(baseName, rawInputTypes);
    const auto declaredType = SubstraitParser::parseType(aggregate.output_type());
    VELOX_CHECK(
        declaredType->equivalent(*intermediateType),
        "Raw grouping-set fusion measure {} declares output type {}, but {} produces intermediate type {}.",
        measureIdx,
        declaredType->toString(),
        baseName,
        intermediateType->toString());
    facebook::velox::exec::Aggregate::create(
        baseName, core::AggregationNode::Step::kPartial, rawInputTypes, intermediateType, queryConfig);
  }
}

core::PlanNodePtr RawGroupingSetPlanConverter::convert(
    const ::substrait::AggregateRel& aggRel,
    const ::substrait::ExpandRel& expandRel,
    const core::PlanNodePtr& expandInputNode) {
  using GroupingSetNode = facebook::velox::exec::GroupingSetAggregationNode;

  VELOX_CHECK_NOT_NULL(expandInputNode, "Raw grouping-set fusion requires an Expand source.");
  const auto maxGroupingSets = owner_.veloxCfg_->get<int32_t>(
      kFusedGroupingSetAggregateMaxGroupingSets, kFusedGroupingSetAggregateMaxGroupingSetsDefault);
  auto shape = parseRawGroupingSetShape(aggRel, expandRel, maxGroupingSets);
  const auto& rawInputType = expandInputNode->outputType();

  std::vector<core::FieldAccessTypedExprPtr> groupingKeys;
  groupingKeys.reserve(shape.numKeys());
  for (const auto& groupingSlot : shape.groupingSlots) {
    if (groupingSlot.isGroupingId()) {
      continue;
    }
    VELOX_CHECK_LT(
        *groupingSlot.rawChannel,
        rawInputType->size(),
        "Raw grouping key references source channel {}, but the source has {} columns.",
        *groupingSlot.rawChannel,
        rawInputType->size());
    const auto& keyType = rawInputType->childAt(*groupingSlot.rawChannel);
    for (int32_t setIdx = 0; setIdx < expandRel.fields_size(); ++setIdx) {
      const auto& expression = expandRel.fields(setIdx).switching_field().duplicates(groupingSlot.expandChannel);
      const auto projectedType = expression.has_selection()
          ? rawInputType->childAt(*groupingSlot.rawChannel)
          : owner_.exprConverter_->toVeloxExpr(expression.literal())->type();
      VELOX_CHECK(
          projectedType->equivalent(*keyType),
          "Raw grouping key source type {} does not match Expand projection type {} in set {}.",
          keyType->toString(),
          projectedType->toString(),
          setIdx);
    }
    groupingKeys.emplace_back(
        std::make_shared<core::FieldAccessTypedExpr>(keyType, rawInputType->nameOf(*groupingSlot.rawChannel)));
  }

  std::vector<std::string> aggregateNames;
  std::vector<core::AggregationNode::Aggregate> aggregates;
  aggregateNames.reserve(aggRel.measures_size());
  aggregates.reserve(aggRel.measures_size());
  for (int32_t measureIdx = 0; measureIdx < aggRel.measures_size(); ++measureIdx) {
    const auto& aggregate = aggRel.measures(measureIdx).measure();
    const auto baseName = SubstraitParser::findVeloxFunction(owner_.functionMap_, aggregate.function_reference());
    validateRawGroupingSetAggregateName(baseName, measureIdx);
    auto rawInputTypes = SubstraitParser::sigToTypes(
        SubstraitParser::findFunctionSpec(owner_.functionMap_, aggregate.function_reference()));

    std::vector<core::TypedExprPtr> inputs;
    inputs.reserve(aggregate.arguments_size());
    for (int32_t argIdx = 0; argIdx < aggregate.arguments_size(); ++argIdx) {
      const auto& value = aggregate.arguments(argIdx).value();
      core::TypedExprPtr input;
      if (value.has_literal()) {
        input = owner_.exprConverter_->toVeloxExpr(value.literal());
      } else {
        const auto expandChannel = selectionFieldIndex(value);
        VELOX_CHECK(expandChannel.has_value(), "Raw grouping-set fusion aggregate argument must be a top-level field.");
        const auto& rawExpression = expandRel.fields(0).switching_field().duplicates(*expandChannel);
        input = owner_.exprConverter_->toVeloxExpr(rawExpression, rawInputType);
      }
      VELOX_CHECK_LT(
          static_cast<size_t>(argIdx),
          rawInputTypes.size(),
          "Raw grouping-set fusion aggregate has more arguments than its signature.");
      VELOX_CHECK(
          input->type()->equivalent(*rawInputTypes[argIdx]),
          "Raw grouping-set fusion measure {} argument {} has source type {}, but the function signature expects {}.",
          measureIdx,
          argIdx,
          input->type()->toString(),
          rawInputTypes[argIdx]->toString());
      inputs.emplace_back(std::move(input));
    }
    VELOX_CHECK_EQ(
        inputs.size(),
        rawInputTypes.size(),
        "Raw grouping-set fusion measure {} argument count does not match its function signature.",
        measureIdx);

    const auto declaredType = SubstraitParser::parseType(aggregate.output_type());
    aggregates.emplace_back(core::AggregationNode::Aggregate{
        std::make_shared<const core::CallTypedExpr>(declaredType, std::move(inputs), baseName),
        std::move(rawInputTypes),
        /*mask=*/nullptr,
        {},
        {}});
    aggregateNames.emplace_back(fmt::format("gsagg_{}_{}", owner_.planNodeId_, measureIdx));
  }

  // In the ordinary pair these two ids belong to Expand and Aggregate. The
  // raw branch uses the same two slots for GroupingSetAggregation and its
  // schema-restoring Project; no identity padding is needed.
  const int32_t ordinaryExpandId = owner_.planNodeId_;
  auto groupingSetNode = std::make_shared<GroupingSetNode>(
      owner_.nextPlanNodeId(),
      std::move(groupingKeys),
      std::move(shape.groupingSets),
      std::move(aggregateNames),
      std::move(aggregates),
      expandInputNode);

  const auto& fusedType = groupingSetNode->outputType();
  std::vector<std::string> names;
  std::vector<core::TypedExprPtr> expressions;
  const int32_t numOutputColumns = static_cast<int32_t>(shape.groupingSlots.size()) + aggRel.measures_size();
  names.reserve(numOutputColumns);
  expressions.reserve(numOutputColumns);
  const auto addField = [&](int32_t channel) {
    expressions.emplace_back(
        std::make_shared<core::FieldAccessTypedExpr>(fusedType->childAt(channel), fusedType->nameOf(channel)));
  };

  int32_t keyChannel = 0;
  for (const auto& groupingSlot : shape.groupingSlots) {
    if (groupingSlot.isGroupingId()) {
      addField(static_cast<int32_t>(groupingSetNode->gidChannel()));
    } else {
      addField(keyChannel++);
    }
    // Match the field name the ordinary AggregateNode inherits from Expand.
    names.emplace_back(SubstraitParser::makeNodeName(ordinaryExpandId, groupingSlot.expandChannel));
  }

  const int32_t aggregateProjectId = owner_.planNodeId_;
  for (int32_t measureIdx = 0; measureIdx < aggRel.measures_size(); ++measureIdx) {
    const int32_t channel = shape.numKeys() + measureIdx;
    const auto declaredType = SubstraitParser::parseType(aggRel.measures(measureIdx).measure().output_type());
    VELOX_CHECK(
        fusedType->childAt(channel)->equivalent(*declaredType),
        "Raw grouping-set fusion measure {} produced {}, but AggregateRel expects {}.",
        measureIdx,
        fusedType->childAt(channel)->toString(),
        declaredType->toString());
    addField(channel);
    names.emplace_back(SubstraitParser::makeNodeName(
        aggregateProjectId, static_cast<int32_t>(shape.groupingSlots.size()) + measureIdx));
  }

  return std::make_shared<core::ProjectNode>(
      owner_.nextPlanNodeId(), std::move(names), std::move(expressions), std::move(groupingSetNode));
}

} // namespace gluten
