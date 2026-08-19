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

// Tests for the Gluten-local native grouping-set aggregation operator.
//
// Most correctness tests compare the custom operator with Expand followed by
// intermediate aggregation, then apply the same final aggregation to both.
//
// OperatorTestBase registers Presto aggregate functions. The main tests cover
// fixed-width, row-typed, and external-memory accumulator states:
//
//   sum(BIGINT)   -> BIGINT
//   avg(DOUBLE)   -> ROW(DOUBLE, BIGINT)
//   array_agg(T)  -> ARRAY(T)
//
// Spark-specific decimal and average buffers are covered separately below.

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <optional>

#include <folly/Random.h>
#include <folly/ScopeGuard.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/exec/AggregateFunctionRegistry.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/Operator.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

#include "operators/plannodes/GroupingSetAggregationNode.h"
#include "operators/plannodes/GroupingSetLattice.h"
#include "operators/plannodes/MultiGroupingSetAggregation.h"
#include "velox/functions/sparksql/aggregates/Register.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

namespace facebook::velox::exec {

namespace {

using Set = GroupingSetSpec;

class MultiGroupingSetAggregationTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    // Registration is process-wide, so pair each registration with TearDown.
    registerMultiGroupingSetAggregation();
  }

  void TearDown() override {
    Operator::unregisterAllOperators();
    OperatorTestBase::TearDown();
  }

  // -------------------------------------------------------------------------
  // Grouping-set shapes
  // -------------------------------------------------------------------------

  /// ROLLUP(k1..kn): a chain in the lattice.
  static std::vector<Set> rollupSets(int32_t numKeys) {
    std::vector<Set> sets;
    for (auto i = 0; i <= numKeys; ++i) {
      Set set;
      set.groupingId = i;
      for (auto j = 0; j < numKeys; ++j) {
        if (j < numKeys - i) {
          set.activeKeysMask |= GroupingSetMask{1} << j;
        }
      }
      sets.push_back(std::move(set));
    }
    return sets;
  }

  /// CUBE(k1..kn): the full power set, with 2^n grouping sets.
  static std::vector<Set> cubeSets(int32_t numKeys) {
    std::vector<Set> sets;
    const int32_t total = 1 << numKeys;
    // Descending so that set 0 is the all-keys-active lattice root.
    for (int32_t m = total - 1; m >= 0; --m) {
      Set set;
      set.groupingId = m;
      set.activeKeysMask = static_cast<GroupingSetMask>(m);
      sets.push_back(std::move(set));
    }
    return sets;
  }

  /// Grouping sets with no containment form an antichain and a flat fan-out.
  static std::vector<Set> antichainSets() {
    std::vector<Set> sets;
    // {k1, k2}
    Set a;
    a.activeKeysMask = 0b011;
    a.groupingId = 11;
    sets.push_back(std::move(a));
    // {k3}
    Set b;
    b.activeKeysMask = 0b100;
    b.groupingId = 22;
    sets.push_back(std::move(b));
    return sets;
  }

  // -------------------------------------------------------------------------
  // Input generation
  // -------------------------------------------------------------------------

  /// Raw input rows: three grouping keys of decreasing cardinality (so the
  /// hierarchy actually reduces, which is the shape the operator is designed
  /// for), plus the aggregate inputs.
  std::vector<RowVectorPtr> makeInput(
      int32_t numBatches,
      vector_size_t batchSize,
      int32_t k1Cardinality,
      int32_t k2Cardinality,
      int32_t k3Cardinality,
      uint32_t seed = 1234) {
    std::vector<RowVectorPtr> batches;
    batches.reserve(numBatches);
    folly::Random::DefaultGenerator rng(seed);
    for (auto b = 0; b < numBatches; ++b) {
      std::vector<int64_t> k1(batchSize), k2(batchSize), k3(batchSize), v(batchSize);
      std::vector<double> w(batchSize);
      for (auto i = 0; i < batchSize; ++i) {
        k1[i] = folly::Random::rand32(k1Cardinality, rng);
        k2[i] = folly::Random::rand32(k2Cardinality, rng);
        k3[i] = folly::Random::rand32(k3Cardinality, rng);
        v[i] = folly::Random::rand32(1000, rng);
        w[i] = static_cast<double>(folly::Random::rand32(1000, rng)) / 7.0;
      }
      batches.push_back(makeRowVector(
          {"k1", "k2", "k3", "v", "w"},
          {makeFlatVector<int64_t>(k1),
           makeFlatVector<int64_t>(k2),
           makeFlatVector<int64_t>(k3),
           makeFlatVector<int64_t>(v),
           makeFlatVector<double>(w)}));
    }
    return batches;
  }

  /// ROLLUP-shaped raw input whose finest key is unique while the next level
  /// collapses to at most 64 groups. This models q67's ineffective finest
  /// partial aggregation without making every level high-cardinality.
  std::vector<RowVectorPtr> makeNearUniqueFinestInput(int32_t numBatches, vector_size_t batchSize) {
    std::vector<RowVectorPtr> batches;
    batches.reserve(numBatches);
    for (auto batch = 0; batch < numBatches; ++batch) {
      std::vector<int64_t> k1(batchSize), k2(batchSize), k3(batchSize), v(batchSize);
      std::vector<double> w(batchSize);
      for (auto row = 0; row < batchSize; ++row) {
        const auto id = static_cast<int64_t>(batch) * batchSize + row;
        k1[row] = id % 4;
        k2[row] = (id / 4) % 16;
        k3[row] = id;
        v[row] = id % 1'000;
        w[row] = static_cast<double>(id % 1'000) / 7.0;
      }
      batches.push_back(makeRowVector(
          {"k1", "k2", "k3", "v", "w"},
          {makeFlatVector<int64_t>(k1),
           makeFlatVector<int64_t>(k2),
           makeFlatVector<int64_t>(k3),
           makeFlatVector<int64_t>(v),
           makeFlatVector<double>(w)}));
    }
    return batches;
  }

  /// Raw ROLLUP input with a controlled finest-level reduction ratio. Every
  /// `rowsPerFinestGroup` consecutive rows share the full key tuple.
  std::vector<RowVectorPtr>
  makeFinestReductionInput(int32_t numBatches, vector_size_t batchSize, int32_t rowsPerFinestGroup) {
    VELOX_CHECK_GT(rowsPerFinestGroup, 0);
    std::vector<RowVectorPtr> batches;
    batches.reserve(numBatches);
    for (auto batch = 0; batch < numBatches; ++batch) {
      std::vector<int64_t> k1(batchSize), k2(batchSize), k3(batchSize), v(batchSize);
      std::vector<double> w(batchSize);
      for (auto row = 0; row < batchSize; ++row) {
        const auto id = static_cast<int64_t>(batch) * batchSize + row;
        const auto group = id / rowsPerFinestGroup;
        k1[row] = group % 4;
        k2[row] = (group / 4) % 16;
        k3[row] = group;
        v[row] = id % 1'000;
        w[row] = static_cast<double>(id % 1'000) / 7.0;
      }
      batches.push_back(makeRowVector(
          {"k1", "k2", "k3", "v", "w"},
          {makeFlatVector<int64_t>(k1),
           makeFlatVector<int64_t>(k2),
           makeFlatVector<int64_t>(k3),
           makeFlatVector<int64_t>(v),
           makeFlatVector<double>(w)}));
    }
    return batches;
  }

  /// One raw ROLLUP batch with an exact number of finest-level groups. The
  /// repeated groups are interleaved so `numFinestGroups / batchSize` is the
  /// exact reduction ratio seen by the root table.
  std::vector<RowVectorPtr> makeFinestCardinalityInput(vector_size_t batchSize, vector_size_t numFinestGroups) {
    VELOX_CHECK_GT(numFinestGroups, 0);
    VELOX_CHECK_LE(numFinestGroups, batchSize);
    std::vector<int64_t> k1(batchSize), k2(batchSize), k3(batchSize), v(batchSize);
    std::vector<double> w(batchSize);
    for (auto row = 0; row < batchSize; ++row) {
      const auto group = row % numFinestGroups;
      k1[row] = group % 4;
      k2[row] = (group / 4) % 16;
      k3[row] = group;
      v[row] = row % 1'000;
      w[row] = static_cast<double>(row % 1'000) / 7.0;
    }
    return {makeRowVector(
        {"k1", "k2", "k3", "v", "w"},
        {makeFlatVector<int64_t>(k1),
         makeFlatVector<int64_t>(k2),
         makeFlatVector<int64_t>(k3),
         makeFlatVector<int64_t>(v),
         makeFlatVector<double>(w)})};
  }

  static RowTypePtr rawInputType() {
    return ROW({"k1", "k2", "k3", "v", "w"}, {BIGINT(), BIGINT(), BIGINT(), BIGINT(), DOUBLE()});
  }

  // -------------------------------------------------------------------------
  // Plan construction
  // -------------------------------------------------------------------------
  //
  // The production node consumes raw rows. The fused side therefore has the
  // intended physical shape:
  //
  //     Values(raw) -> GroupingSetAggregation -> final aggregation
  //
  // The reference side retains Spark's original shape:
  //
  //     Values(raw) -> Expand -> partial aggregation -> final aggregation

  struct AggSpec {
    /// Partial-aggregation expressions over the raw input columns.
    std::vector<std::string> partial;
    /// Raw argument types per aggregate, same order as `partial`.
    std::vector<std::vector<TypePtr>> rawTypes;
    /// Final-aggregation expressions over the intermediate columns.
    std::vector<std::string> final;
    /// Output columns of the closing projection, keys and gid excluded.
    std::vector<std::string> outputs;
  };

  static AggSpec defaultAggSpec() {
    return AggSpec{
        {"sum(v) as s", "avg(w) as m", "array_agg(v) as l"},
        {{BIGINT()}, {DOUBLE()}, {BIGINT()}},
        {"sum(s) as s", "avg(m) as m", "array_agg(l) as l"},
        {"s", "m", "array_sort(l) as l"}};
  }

  static std::vector<std::string> keyNames(int32_t numKeys) {
    std::vector<std::string> keys;
    for (auto i = 0; i < numKeys; ++i) {
      keys.push_back(fmt::format("k{}", i + 1));
    }
    return keys;
  }

  struct AggregateMetadata {
    std::vector<std::string> names;
    std::vector<core::AggregationNode::Aggregate> aggregates;
  };

  /// PlanBuilder owns the SQL-expression parsing used by production plans.
  /// Build a metadata-only partial aggregation, then copy its raw aggregate
  /// calls into the custom node. The partial aggregation is not in the fused
  /// execution plan.
  AggregateMetadata
  makeAggregateMetadata(const std::vector<RowVectorPtr>& input, int32_t numKeys, const AggSpec& spec) {
    PlanBuilder metadataBuilder(pool());
    const auto metadataNode =
        metadataBuilder.values(input).partialAggregation(keyNames(numKeys), spec.partial).planNode();
    const auto* metadata = dynamic_cast<const core::AggregationNode*>(metadataNode.get());
    VELOX_CHECK_NOT_NULL(metadata);

    auto aggregates = metadata->aggregates();
    for (auto& aggregate : aggregates) {
      aggregate.rawInputTypes.clear();
      for (const auto& inputExpression : aggregate.call->inputs()) {
        aggregate.rawInputTypes.push_back(inputExpression->type());
      }
    }
    return {metadata->aggregateNames(), std::move(aggregates)};
  }

  static std::vector<core::FieldAccessTypedExprPtr> makeGroupingKeys(const core::PlanNodePtr& source, int32_t numKeys) {
    std::vector<core::FieldAccessTypedExprPtr> keys;
    const auto& type = source->outputType();
    for (auto i = 0; i < numKeys; ++i) {
      const auto name = fmt::format("k{}", i + 1);
      keys.push_back(std::make_shared<core::FieldAccessTypedExpr>(type->findChild(name), name));
    }
    return keys;
  }

  /// ROW(k1..kn, acc1..accm, gid BIGINT).
  static RowTypePtr
  makeExpectedOutputType(const core::PlanNodePtr& source, int32_t numKeys, const AggregateMetadata& metadata) {
    std::vector<std::string> names;
    std::vector<TypePtr> types;
    const auto& sourceType = source->outputType();
    for (auto i = 0; i < numKeys; ++i) {
      names.push_back(sourceType->nameOf(i));
      types.push_back(sourceType->childAt(i));
    }
    for (size_t i = 0; i < metadata.names.size(); ++i) {
      names.push_back(metadata.names[i]);
      types.push_back(
          resolveIntermediateType(metadata.aggregates[i].call->name(), metadata.aggregates[i].rawInputTypes));
    }
    names.push_back("gid");
    types.push_back(BIGINT());
    return ROW(std::move(names), std::move(types));
  }

  /// The raw-input fused plan used in production.
  core::PlanNodePtr makeFusedPlan(
      const std::vector<RowVectorPtr>& input,
      int32_t numKeys,
      const std::vector<Set>& sets,
      const AggSpec& spec = defaultAggSpec(),
      bool reorderSource = false) {
    auto metadata = makeAggregateMetadata(input, numKeys, spec);

    PlanBuilder builder(pool());
    builder.values(input);
    if (reorderSource) {
      VELOX_CHECK_EQ(numKeys, 3, "The raw-input channel-reordering fixture expects three keys");
      builder.project({"v", "k2", "w", "k1", "k3"});
    }
    auto groupingKeys = makeGroupingKeys(builder.planNode(), numKeys);

    return builder
        .addNode([&](std::string id, core::PlanNodePtr source) {
          return std::make_shared<GroupingSetAggregationNode>(
              std::move(id), groupingKeys, sets, metadata.names, metadata.aggregates, std::move(source));
        })
        .finalAggregation(appendGid(keyNames(numKeys)), spec.final, spec.rawTypes)
        .project(finalProjection(numKeys, spec))
        .planNode();
  }

  /// Original-shape reference: Expand raw rows, then build partial states.
  core::PlanNodePtr makeReferencePlan(
      const std::vector<RowVectorPtr>& input,
      int32_t numKeys,
      const std::vector<Set>& sets,
      const AggSpec& spec = defaultAggSpec()) {
    PlanBuilder builder(pool());
    builder.values(input);
    const auto& sourceType = builder.planNode()->outputType();

    std::vector<std::vector<std::string>> projections;
    for (size_t s = 0; s < sets.size(); ++s) {
      const auto& set = sets[s];
      std::vector<std::string> projection;
      projection.reserve(sourceType->size() + 1);
      for (column_index_t channel = 0; channel < sourceType->size(); ++channel) {
        const auto& name = sourceType->nameOf(channel);
        if (channel < numKeys && !set.containsKey(channel)) {
          const auto typedNull = fmt::format("null::{}", sourceType->childAt(channel)->toString());
          projection.push_back(s == 0 ? fmt::format("{} as {}", typedNull, name) : typedNull);
        } else {
          projection.push_back(s == 0 ? fmt::format("{0} as {0}", name) : name);
        }
      }
      projection.push_back(s == 0 ? fmt::format("{} as gid", set.groupingId) : fmt::format("{}", set.groupingId));
      projections.push_back(std::move(projection));
    }

    return builder.expand(projections)
        .partialAggregation(appendGid(keyNames(numKeys)), spec.partial)
        .finalAggregation(appendGid(keyNames(numKeys)), spec.final, spec.rawTypes)
        .project(finalProjection(numKeys, spec))
        .planNode();
  }

  static std::vector<std::string> appendGid(std::vector<std::string> keys) {
    keys.push_back("gid");
    return keys;
  }

  /// array_agg's result ORDER is not stable across flush timing -- the fused
  /// operator makes no promise about it, and neither does Velox's own
  /// partial/final split. Sorting the array before comparison tests the
  /// multiset of collected values, which is the property that actually has to
  /// hold.
  static std::vector<std::string> finalProjection(int32_t numKeys, const AggSpec& spec) {
    auto columns = appendGid(keyNames(numKeys));
    for (const auto& out : spec.outputs) {
      columns.push_back(out);
    }
    return columns;
  }

  // -------------------------------------------------------------------------
  // Differential driver
  // -------------------------------------------------------------------------

  /// Runs both plans under the same config and asserts multiset equality.
  /// assertEqualResults is multiset-based and has the epsilon handling for the
  /// floating-point avg column, which applies here because the final
  /// aggregation yields exactly one row per group.
  void assertFusedMatchesReference(
      const std::vector<RowVectorPtr>& input,
      int32_t numKeys,
      const std::unordered_map<std::string, std::string>& configs = {},
      const AggSpec& spec = defaultAggSpec()) {
    assertFusedMatchesReferenceForSets(input, numKeys, rollupSets(numKeys), configs, spec);
  }

  void assertFusedMatchesReferenceForSets(
      const std::vector<RowVectorPtr>& input,
      int32_t numKeys,
      const std::vector<Set>& sets,
      const std::unordered_map<std::string, std::string>& configs = {},
      const AggSpec& spec = defaultAggSpec()) {
    auto fused = AssertQueryBuilder(makeFusedPlan(input, numKeys, sets, spec)).configs(configs).copyResults(pool());
    auto reference =
        AssertQueryBuilder(makeReferencePlan(input, numKeys, sets, spec)).configs(configs).copyResults(pool());
    ASSERT_TRUE(assertEqualResults({reference}, {fused}));
  }

  void assertFusedMatchesReferenceWithAggregates(
      const std::vector<RowVectorPtr>& input,
      int32_t numKeys,
      const std::vector<std::string>& partial,
      const std::vector<std::vector<TypePtr>>& rawTypes,
      const std::vector<std::string>& final,
      const std::vector<std::string>& outputs,
      const std::unordered_map<std::string, std::string>& configs = {}) {
    assertFusedMatchesReference(input, numKeys, configs, AggSpec{partial, rawTypes, final, outputs});
  }

  static int64_t groupingSetRuntimeStat(const std::shared_ptr<Task>& task, std::string_view name) {
    int64_t result{0};
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& stats : pipeline.operatorStats) {
        if (stats.operatorType != "GroupingSetAggregation") {
          continue;
        }
        if (const auto it = stats.runtimeStats.find(std::string(name)); it != stats.runtimeStats.end()) {
          result += it->second.sum;
        }
      }
    }
    return result;
  }

  static int32_t groupingSetRuntimeStatOccurrences(const std::shared_ptr<Task>& task, std::string_view name) {
    int32_t result{0};
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& stats : pipeline.operatorStats) {
        if (stats.operatorType == "GroupingSetAggregation" && stats.runtimeStats.contains(std::string(name))) {
          ++result;
        }
      }
    }
    return result;
  }

  static void assertGroupingSetLifetimeAccounting(const std::shared_ptr<Task>& task, int32_t numSets) {
    ASSERT_NE(task, nullptr);
    for (int32_t set = 0; set < numSets; ++set) {
      const auto prefix = fmt::format("gsagg.set{}", set);
      const auto inputRows = prefix + ".inputRows";
      const auto outputRows = prefix + ".outputRows";
      const auto totalInputRows = prefix + ".totalInputRows";
      const auto totalOutputRows = prefix + ".totalOutputRows";

      ASSERT_GT(groupingSetRuntimeStatOccurrences(task, inputRows), 0) << inputRows;
      ASSERT_GT(groupingSetRuntimeStatOccurrences(task, outputRows), 0) << outputRows;
      ASSERT_GT(groupingSetRuntimeStatOccurrences(task, totalInputRows), 0) << totalInputRows;
      ASSERT_GT(groupingSetRuntimeStatOccurrences(task, totalOutputRows), 0) << totalOutputRows;
      EXPECT_EQ(groupingSetRuntimeStat(task, inputRows), groupingSetRuntimeStat(task, totalInputRows))
          << "set " << set << " input deltas do not cover the full lifetime";
      EXPECT_EQ(groupingSetRuntimeStat(task, outputRows), groupingSetRuntimeStat(task, totalOutputRows))
          << "set " << set << " output deltas do not cover the full lifetime";
    }
  }

  void assertRawConverterFallbackAfterEarlyAbandon(const std::vector<RowVectorPtr>& input, const AggSpec& spec) {
    const auto sets = rollupSets(3);
    const std::unordered_map<std::string, std::string> configs = {
        {core::QueryConfig::kMaxPartialAggregationMemory, fmt::format("{}", 1LL << 30)},
        {core::QueryConfig::kMaxExtendedPartialAggregationMemory, fmt::format("{}", 1LL << 30)},
        {core::QueryConfig::kAbandonPartialAggregationMinRows, "512"},
        {core::QueryConfig::kAbandonPartialAggregationMinPct, "90"},
    };

    std::shared_ptr<Task> task;
    auto fused = AssertQueryBuilder(makeFusedPlan(input, 3, sets, spec)).configs(configs).copyResults(pool(), task);
    auto reference = AssertQueryBuilder(makeReferencePlan(input, 3, sets, spec)).copyResults(pool());
    ASSERT_TRUE(assertEqualResults({reference}, {fused}));
    ASSERT_NE(task, nullptr);
    EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.earlyAbandoned"), 1);
    EXPECT_GT(groupingSetRuntimeStat(task, "gsagg.rawConvertedRows"), 0);
    assertGroupingSetLifetimeAccounting(task, sets.size());
  }
};

// Lattice construction.
// buildDerivationPlan is a pure function of grouping masks.

TEST_F(MultiGroupingSetAggregationTest, latticeChainIsACascade) {
  // ROLLUP(k1,k2,k3): masks 111, 110, 100, 000 form a chain.
  const std::vector<GroupingSetMask> masks = {0b111, 0b110, 0b100, 0b000};
  auto plan = buildDerivationPlan(masks);

  EXPECT_EQ(plan.parent[0], kRawInputParent);
  EXPECT_EQ(plan.parent[1], 0);
  EXPECT_EQ(plan.parent[2], 1);
  EXPECT_EQ(plan.parent[3], 2);
  // Only the lattice root reads the input.
  EXPECT_EQ(std::count(plan.parent.begin(), plan.parent.end(), kRawInputParent), 1);
  // Parents strictly before children.
  std::vector<int32_t> rank(masks.size());
  for (size_t i = 0; i < plan.order.size(); ++i) {
    rank[plan.order[i]] = static_cast<int32_t>(i);
  }
  for (size_t i = 0; i < masks.size(); ++i) {
    if (plan.parent[i] != kRawInputParent) {
      EXPECT_LT(rank[plan.parent[i]], rank[i]);
    }
  }
}

TEST_F(MultiGroupingSetAggregationTest, latticeAntichainIsAFlatFanOut) {
  // No containment means every set is a root.
  const std::vector<GroupingSetMask> masks = {0b1100, 0b0011};
  auto plan = buildDerivationPlan(masks);
  EXPECT_EQ(plan.parent[0], kRawInputParent);
  EXPECT_EQ(plan.parent[1], kRawInputParent);
  EXPECT_TRUE(plan.children[0].empty());
  EXPECT_TRUE(plan.children[1].empty());
}

TEST_F(MultiGroupingSetAggregationTest, latticeCubePicksSmallestParent) {
  // CUBE(a,b,c), masks descending: index 0 == 111 ... index 7 == 000.
  const std::vector<GroupingSetMask> masks = {0b111, 0b110, 0b101, 0b100, 0b011, 0b010, 0b001, 0b000};
  auto plan = buildDerivationPlan(masks);

  // Only the top of the lattice reads input; the other seven sets are derived.
  EXPECT_EQ(std::count(plan.parent.begin(), plan.parent.end(), kRawInputParent), 1);
  EXPECT_EQ(plan.parent[0], kRawInputParent);

  // Every chosen parent must be a strict superset.
  for (size_t i = 0; i < masks.size(); ++i) {
    const auto p = plan.parent[i];
    if (p == kRawInputParent) {
      continue;
    }
    EXPECT_EQ(masks[i] & ~masks[p], 0u) << "set " << i;
    EXPECT_NE(masks[i], masks[p]) << "set " << i;
    // CUBE provides a parent with exactly one additional active key.
    EXPECT_EQ(__builtin_popcountll(masks[p]), __builtin_popcountll(masks[i]) + 1)
        << "set " << i << " should derive from a minimal superset";
  }
}

TEST_F(MultiGroupingSetAggregationTest, latticeParentPreferenceTiesAreStable) {
  // The two two-key candidates are equally desirable parents of the empty
  // set. Preserve the original-input-order tie break after comparing active-key
  // counts.
  const std::vector<GroupingSetMask> masks = {0b1111, 0b0011, 0b1100, 0b0000};

  const auto fallback = buildDerivationPlan(masks);
  EXPECT_EQ(fallback.parent, (std::vector<int32_t>{kRawInputParent, 0, 0, 1}));
  EXPECT_EQ(fallback.order, (std::vector<int32_t>{0, 1, 2, 3}));
  EXPECT_EQ(fallback.children[0], (std::vector<int32_t>{1, 2}));
  EXPECT_EQ(fallback.children[1], (std::vector<int32_t>{3}));
}

TEST_F(MultiGroupingSetAggregationTest, latticeRejectsDuplicateMasks) {
  // The public lattice API validates masks independently of the plan node.
  const std::vector<GroupingSetMask> masks = {0b111, 0b110, 0b110};
  VELOX_ASSERT_THROW(buildDerivationPlan(masks), "Duplicate grouping-set mask");
}

TEST_F(MultiGroupingSetAggregationTest, latticeNonAdjacentParent) {
  // The nearest available superset can differ by more than one key.
  const std::vector<GroupingSetMask> masks = {0b111, 0b001};
  auto plan = buildDerivationPlan(masks);
  EXPECT_EQ(plan.parent[0], kRawInputParent);
  EXPECT_EQ(plan.parent[1], 0);
  EXPECT_EQ(std::count(plan.parent.begin(), plan.parent.end(), kRawInputParent), 1);
}

TEST_F(MultiGroupingSetAggregationTest, latticeWideMasks) {
  // GroupingSetMask is uint64_t and the node caps keys at 64. Exercise the top
  // bit: a chain over bits 63, 62, 61 must still be a cascade, with no shift
  // UB and no sign-extension surprise in popcount.
  const GroupingSetMask top = GroupingSetMask{1} << 63;
  const GroupingSetMask second = GroupingSetMask{1} << 62;
  const GroupingSetMask third = GroupingSetMask{1} << 61;
  const std::vector<GroupingSetMask> masks = {top | second | third, top | second, top, 0};
  auto plan = buildDerivationPlan(masks);
  EXPECT_EQ(plan.parent[0], kRawInputParent);
  EXPECT_EQ(plan.parent[1], 0);
  EXPECT_EQ(plan.parent[2], 1);
  EXPECT_EQ(plan.parent[3], 2);
}

TEST_F(MultiGroupingSetAggregationTest, latticeSupportsMaximumGroupingSetCount) {
  // A 64-level chain over 63 keys exercises every candidate-bitmap bit,
  // including rank 63, and the empty-set query with allCandidates == UINT64_MAX.
  std::vector<GroupingSetMask> masks;
  masks.reserve(kMaxGroupingSets);
  const auto allKeys = std::numeric_limits<GroupingSetMask>::max() >> 1;
  for (int32_t removedKeys = 0; removedKeys < kMaxGroupingSets; ++removedKeys) {
    masks.push_back(allKeys >> removedKeys);
  }

  const auto plan = buildDerivationPlan(masks);
  EXPECT_EQ(plan.parent[0], kRawInputParent);
  for (int32_t set = 1; set < kMaxGroupingSets; ++set) {
    EXPECT_EQ(plan.parent[set], set - 1);
  }
}

TEST_F(MultiGroupingSetAggregationTest, latticeRejectsMoreThanMaximumGroupingSetCount) {
  std::vector<GroupingSetMask> masks(kMaxGroupingSets + 1);
  std::iota(masks.begin(), masks.end(), GroupingSetMask{0});
  VELOX_ASSERT_THROW(
      buildDerivationPlan(masks),
      fmt::format("A grouping-set derivation plan supports at most {} sets", kMaxGroupingSets));
}

TEST_F(MultiGroupingSetAggregationTest, groupingSetSpecSupportsHighestBit) {
  const GroupingSetSpec set{GroupingSetMask{1} << 63, 7};
  EXPECT_EQ(set.numActiveKeys(), 1);
  EXPECT_TRUE(set.containsKey(63));
  EXPECT_FALSE(set.containsKey(62));
  EXPECT_FALSE(set.containsKey(64));
}

TEST_F(MultiGroupingSetAggregationTest, latticeRejectsEmptyMasks) {
  VELOX_ASSERT_THROW(buildDerivationPlan({}), "A grouping-set aggregation needs at least one set");
}

// Plan-node validation.

TEST_F(MultiGroupingSetAggregationTest, duplicateMaskRejected) {
  // Two sets at the same grain with different gids would each be a lattice root
  // and each scan the raw input for identical work. A planner bug; reject at
  // plan build time rather than paper over it.
  auto input = makeInput(1, 10, 2, 2, 2);
  PlanBuilder builder(pool());
  auto source = builder.values(input).planNode();
  auto metadata = makeAggregateMetadata(input, 3, defaultAggSpec());

  std::vector<Set> sets = {Set{0b011, 1}, Set{0b011, 2}};

  VELOX_ASSERT_THROW(
      std::make_shared<GroupingSetAggregationNode>(
          "gsagg-0", makeGroupingKeys(source, 3), sets, metadata.names, metadata.aggregates, source),
      "Duplicate grouping-set mask");

  std::vector<Set> tooManySets(kMaxGroupingSets + 1, Set{0b011, 1});
  VELOX_ASSERT_THROW(
      std::make_shared<GroupingSetAggregationNode>(
          "gsagg-too-many", makeGroupingKeys(source, 3), tooManySets, metadata.names, metadata.aggregates, source),
      fmt::format("supports at most {} grouping sets", kMaxGroupingSets));
}

TEST_F(MultiGroupingSetAggregationTest, zeroAggregatesRejected) {
  // Velox treats a GroupingSet with no aggregate functions as distinct
  // aggregation, whose extraction path differs from the merge-only contract
  // implemented by this operator.
  auto input = makeInput(1, 10, 2, 2, 2);
  PlanBuilder builder(pool());
  auto source = builder.values(input).planNode();

  VELOX_ASSERT_THROW(
      std::make_shared<GroupingSetAggregationNode>(
          "gsagg-no-measures",
          makeGroupingKeys(source, 3),
          rollupSets(3),
          std::vector<std::string>{},
          std::vector<core::AggregationNode::Aggregate>{},
          source),
      "needs at least one aggregate");
}

TEST_F(MultiGroupingSetAggregationTest, maskOutsideGroupingKeyRangeRejected) {
  auto input = makeInput(1, 10, 2, 2, 2);
  PlanBuilder builder(pool());
  auto source = builder.values(input).planNode();
  auto metadata = makeAggregateMetadata(input, 3, defaultAggSpec());

  // Bit 63 is representable by the mask, but this node has only keys 0..2.
  const std::vector<Set> sets = {Set{GroupingSetMask{1} << 63, 1}};
  VELOX_ASSERT_THROW(
      std::make_shared<GroupingSetAggregationNode>(
          "gsagg-0", makeGroupingKeys(source, 3), sets, metadata.names, metadata.aggregates, source),
      "contains keys outside the 3 grouping keys");
}

TEST_F(MultiGroupingSetAggregationTest, outputTypeShape) {
  // Guards the output-type contract independently of execution, so that a
  // regression fails here with a readable message rather than as a type
  // mismatch deep inside a GroupingSet.
  auto input = makeInput(1, 100, 3, 5, 7);
  PlanBuilder builder(pool());
  auto source = builder.values(input).planNode();
  auto metadata = makeAggregateMetadata(input, 3, defaultAggSpec());

  auto node = std::make_shared<GroupingSetAggregationNode>(
      "gsagg-0", makeGroupingKeys(source, 3), rollupSets(3), metadata.names, metadata.aggregates, source);

  auto expected = makeExpectedOutputType(source, 3, metadata);
  EXPECT_EQ(*node->outputType(), *expected);
  EXPECT_EQ(node->gidChannel(), expected->size() - 1);
}

TEST_F(MultiGroupingSetAggregationTest, rawAggregateSourceTypeMismatchRejected) {
  auto input = makeInput(1, 100, 3, 5, 7);
  const AggSpec spec{{"sum(v) as s"}, {{BIGINT()}}, {"sum(s) as s"}, {"s"}};
  auto metadata = makeAggregateMetadata(input, 1, spec);
  PlanBuilder builder(pool());
  auto source = builder.values(input).project({"k1", "cast(v as double) as v"}).planNode();

  VELOX_ASSERT_THROW(
      std::make_shared<GroupingSetAggregationNode>(
          "gsagg-0", makeGroupingKeys(source, 1), rollupSets(1), metadata.names, metadata.aggregates, source),
      "field v expects type BIGINT, but the source provides DOUBLE");
}

TEST_F(MultiGroupingSetAggregationTest, rawAggregateMissingSourceFieldRejected) {
  auto input = makeInput(1, 100, 3, 5, 7);
  auto metadata = makeAggregateMetadata(input, 3, defaultAggSpec());
  auto aggregates = metadata.aggregates;
  ASSERT_FALSE(aggregates.empty());
  aggregates[0].rawInputTypes = {BIGINT()};
  aggregates[0].call = std::make_shared<core::CallTypedExpr>(
      aggregates[0].call->type(),
      std::vector<core::TypedExprPtr>{std::make_shared<core::FieldAccessTypedExpr>(BIGINT(), "missing_value")},
      aggregates[0].call->name());

  PlanBuilder builder(pool());
  const auto rawSource = builder.values(input).planNode();
  VELOX_ASSERT_THROW(
      std::make_shared<GroupingSetAggregationNode>(
          "gsagg-raw-invalid",
          makeGroupingKeys(rawSource, 3),
          rollupSets(3),
          metadata.names,
          std::move(aggregates),
          rawSource),
      "references missing source field missing_value");
}

TEST_F(MultiGroupingSetAggregationTest, planNodeSerdeRoundTrip) {
  // SetUp registers the custom node. Register the Values source in the
  // serialized subtree, then verify fields and derived output type explicitly.
  Type::registerSerDe();
  core::ITypedExpr::registerSerDe();
  {
    auto& reg = DeserializationWithContextRegistryForSharedPtr();
    reg.Register("ValuesNode", core::ValuesNode::create);
  }
  auto input = makeInput(1, 100, 3, 5, 7);
  PlanBuilder builder(pool());
  auto source = builder.values(input).planNode();
  auto metadata = makeAggregateMetadata(input, 3, defaultAggSpec());

  // Cover both a rollup chain and a genuine CUBE lattice so masks with
  // interior holes survive the round-trip.
  const std::vector<std::vector<Set>> shapes = {rollupSets(3), cubeSets(3)};

  for (size_t s = 0; s < shapes.size(); ++s) {
    SCOPED_TRACE(fmt::format("shape={}", s));
    const auto& sets = shapes[s];

    core::PlanNodePtr node = std::make_shared<GroupingSetAggregationNode>(
        "gsagg-0", makeGroupingKeys(source, 3), sets, metadata.names, metadata.aggregates, source);

    const auto serialized = node->serialize();
    ASSERT_EQ(serialized["groupingSets"].size(), sets.size());
    for (size_t i = 0; i < sets.size(); ++i) {
      const auto& serializedSet = serialized["groupingSets"][i];
      EXPECT_EQ(serializedSet["keyIsActive"].size(), 3);
      EXPECT_EQ(serializedSet.count("activeKeysMask"), 0);
    }
    if (s == 0) {
      auto malformed = serialized;
      malformed["groupingSets"][0]["keyIsActive"] = folly::dynamic::array(true, false);
      VELOX_ASSERT_THROW(
          ISerializable::deserialize<core::PlanNode>(malformed, pool()),
          "Every serialized grouping-set mask must match the grouping-key count");

      auto oversizedSets = serialized;
      oversizedSets["groupingSets"] = folly::dynamic::array;
      for (int32_t i = 0; i <= kMaxGroupingSets; ++i) {
        oversizedSets["groupingSets"].push_back(serialized["groupingSets"][0]);
      }
      VELOX_ASSERT_THROW(
          ISerializable::deserialize<core::PlanNode>(oversizedSets, pool()),
          fmt::format("supports at most {} grouping sets", kMaxGroupingSets));
    }
    const auto copy = ISerializable::deserialize<core::PlanNode>(serialized, pool());
    ASSERT_EQ(node->toString(true, true), copy->toString(true, true));

    // Also verify fields that identify grouping sets and the derived output.
    auto gsCopy = std::dynamic_pointer_cast<const GroupingSetAggregationNode>(copy);
    ASSERT_NE(gsCopy, nullptr);
    ASSERT_EQ(gsCopy->groupingSets().size(), sets.size());
    for (size_t i = 0; i < sets.size(); ++i) {
      EXPECT_EQ(gsCopy->groupingSets()[i].groupingId, sets[i].groupingId);
      EXPECT_EQ(gsCopy->groupingSets()[i].activeKeysMask, sets[i].activeKeysMask);
    }
    EXPECT_EQ(*gsCopy->outputType(), *node->outputType());
  }

  // The boolean-array wire format must also preserve bit 63. A signed numeric
  // JSON field could not do this safely.
  std::vector<std::string> wideNames = keyNames(64);
  wideNames.push_back("v");
  std::vector<VectorPtr> wideChildren;
  wideChildren.reserve(65);
  for (int32_t key = 0; key < 64; ++key) {
    wideChildren.push_back(makeFlatVector<int64_t>({key}));
  }
  wideChildren.push_back(makeFlatVector<int64_t>({1}));
  auto wideInput = makeRowVector(std::move(wideNames), std::move(wideChildren));
  PlanBuilder wideBuilder(pool());
  auto wideSource = wideBuilder.values({wideInput}).planNode();
  const AggSpec wideSpec{{"sum(v) as s"}, {{BIGINT()}}, {"sum(s) as s"}, {"s"}};
  auto wideMetadata = makeAggregateMetadata({wideInput}, 64, wideSpec);

  core::PlanNodePtr wideNode = std::make_shared<GroupingSetAggregationNode>(
      "gsagg-wide",
      makeGroupingKeys(wideSource, 64),
      std::vector<Set>{{GroupingSetMask{1} << 63, 7}},
      wideMetadata.names,
      wideMetadata.aggregates,
      wideSource);
  const auto wideSerialized = wideNode->serialize();
  ASSERT_EQ(wideSerialized["groupingSets"][0]["keyIsActive"].size(), 64);
  EXPECT_TRUE(wideSerialized["groupingSets"][0]["keyIsActive"][63].asBool());
  const auto wideCopy = ISerializable::deserialize<core::PlanNode>(wideSerialized, pool());
  const auto wideGroupingSetCopy = std::dynamic_pointer_cast<const GroupingSetAggregationNode>(wideCopy);
  ASSERT_NE(wideGroupingSetCopy, nullptr);
  EXPECT_EQ(wideGroupingSetCopy->groupingSets()[0].activeKeysMask, GroupingSetMask{1} << 63);
}

// Differential correctness.

TEST_F(MultiGroupingSetAggregationTest, differentialCorrectness) {
  auto input = makeInput(
      /*numBatches=*/10,
      /*batchSize=*/1'000,
      /*k1Cardinality=*/5,
      /*k2Cardinality=*/23,
      /*k3Cardinality=*/97);
  assertFusedMatchesReference(input, /*numKeys=*/3);
}

TEST_F(MultiGroupingSetAggregationTest, differentialCorrectnessAndChannelResolution) {
  auto input = makeInput(
      /*numBatches=*/8,
      /*batchSize=*/1'000,
      /*k1Cardinality=*/5,
      /*k2Cardinality=*/23,
      /*k3Cardinality=*/97);
  const auto sets = rollupSets(3);

  auto fused = AssertQueryBuilder(makeFusedPlan(
                                      input,
                                      3,
                                      sets,
                                      defaultAggSpec(),
                                      /*reorderSource=*/true))
                   .copyResults(pool());
  auto reference = AssertQueryBuilder(makeReferencePlan(input, 3, sets)).copyResults(pool());
  ASSERT_TRUE(assertEqualResults({reference}, {fused}));
}

TEST_F(MultiGroupingSetAggregationTest, resolvesConstantAggregateArguments) {
  auto input = makeInput(
      /*numBatches=*/4,
      /*batchSize=*/512,
      /*k1Cardinality=*/4,
      /*k2Cardinality=*/12,
      /*k3Cardinality=*/200);
  const auto sets = rollupSets(3);
  const AggSpec spec{{"count(1) as c"}, {{BIGINT()}}, {"count(c) as c"}, {"c"}};

  auto fused = AssertQueryBuilder(makeFusedPlan(input, 3, sets, spec)).copyResults(pool());
  auto reference = AssertQueryBuilder(makeReferencePlan(input, 3, sets, spec)).copyResults(pool());
  ASSERT_TRUE(assertEqualResults({reference}, {fused}));
}

TEST_F(MultiGroupingSetAggregationTest, differentialCorrectnessManySeeds) {
  // Cheap fuzzing over the key-cardinality shape: a steeply reducing hierarchy
  // (the case fusion is designed for) and a barely reducing one (the case the
  // abandon valve is designed for) exercise very different code paths.
  const std::vector<std::array<int32_t, 3>> shapes = {{2, 4, 8}, {3, 100, 5000}, {1, 1, 2}, {50, 50, 50}};
  for (uint32_t seed = 0; seed < shapes.size(); ++seed) {
    SCOPED_TRACE(fmt::format("seed={}", seed));
    auto input = makeInput(4, 500, shapes[seed][0], shapes[seed][1], shapes[seed][2], seed + 1);
    assertFusedMatchesReference(input, 3);
  }
}

TEST_F(MultiGroupingSetAggregationTest, differentialCube) {
  // CUBE(k1,k2,k3): eight sets with multiple candidate parents.
  auto input = makeInput(8, 1'000, 4, 12, 40);
  assertFusedMatchesReferenceForSets(input, 3, cubeSets(3));
}

TEST_F(MultiGroupingSetAggregationTest, differentialAntichainGroupingSets) {
  // GROUPING SETS ((k1,k2),(k3)) forms a flat fan-out.
  auto input = makeInput(8, 1'000, 4, 12, 40);
  assertFusedMatchesReferenceForSets(input, 3, antichainSets());
}

TEST_F(MultiGroupingSetAggregationTest, differentialSingleKey) {
  // A single key is the two-set chain {k1} -> {}.
  auto input = makeInput(5, 1'000, 6, 1, 1);
  assertFusedMatchesReferenceForSets(input, 1, rollupSets(1));
}

TEST_F(MultiGroupingSetAggregationTest, finalRuntimeStatsRecorded) {
  auto input = makeInput(2, 100, 4, 12, 40);
  std::shared_ptr<Task> task;
  AssertQueryBuilder(makeFusedPlan(input, 3, rollupSets(3))).copyResults(pool(), task);

  bool foundOperator = false;
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& stats : pipeline.operatorStats) {
      if (stats.operatorType != "GroupingSetAggregation") {
        continue;
      }
      foundOperator = true;
      EXPECT_EQ(stats.runtimeStats.count("gsagg.flushes"), 1);
      EXPECT_EQ(stats.runtimeStats.count("gsagg.nestedDrains"), 1);
      EXPECT_EQ(stats.runtimeStats.count("gsagg.operatorTargetFlushes"), 1);
      EXPECT_EQ(stats.runtimeStats.count("gsagg.flushTargetBytes"), 1);
      EXPECT_EQ(stats.runtimeStats.count("gsagg.peakSampledFlushableBytes"), 1);
      EXPECT_EQ(stats.runtimeStats.count("gsagg.maxTargetOvershootBytes"), 1);
      EXPECT_EQ(stats.runtimeStats.count("gsagg.growthReservationCalls"), 1);
      EXPECT_EQ(stats.runtimeStats.count("gsagg.growthReservationBytes"), 1);
      EXPECT_EQ(stats.runtimeStats.count("abandonedPartialAggregationRows"), 1);
    }
  }
  EXPECT_TRUE(foundOperator);
}

TEST_F(MultiGroupingSetAggregationTest, globalVariableWidthAccumulatorFlushesAndResets) {
  // A global array_agg has no hash table, but its external accumulator memory
  // must still participate in pressure. Compare repeated global cycles with a
  // one-cycle run, then require evidence that pressure produced extra drains.
  auto input = makeInput(20, 1'000, 5, 40, 400);
  const AggSpec spec{{"array_agg(v) as l"}, {{BIGINT()}}, {"array_agg(l) as l"}, {"array_sort(l) as l"}};
  const std::vector<Set> globalOnly = {Set{0, 0}};

  auto expected = AssertQueryBuilder(makeFusedPlan(input, 3, globalOnly, spec))
                      .config(core::QueryConfig::kMaxPartialAggregationMemory, fmt::format("{}", 1LL << 30))
                      .config(core::QueryConfig::kMaxExtendedPartialAggregationMemory, fmt::format("{}", 1LL << 30))
                      .copyResults(pool());

  std::shared_ptr<Task> task;
  auto actual = AssertQueryBuilder(makeFusedPlan(input, 3, globalOnly, spec))
                    .config(core::QueryConfig::kMaxPartialAggregationMemory, "4096")
                    .config(core::QueryConfig::kMaxExtendedPartialAggregationMemory, "4096")
                    .copyResults(pool(), task);

  EXPECT_TRUE(assertEqualResults({expected}, {actual}));

  bool foundOperator = false;
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& stats : pipeline.operatorStats) {
      if (stats.operatorType != "GroupingSetAggregation") {
        continue;
      }
      foundOperator = true;
      const auto flushes = stats.runtimeStats.find("gsagg.flushes");
      ASSERT_NE(flushes, stats.runtimeStats.end());
      EXPECT_GT(flushes->second.sum, 1) << "global state never pressure-drained before the final sweep";
    }
  }
  EXPECT_TRUE(foundOperator);
}

TEST_F(MultiGroupingSetAggregationTest, multiLevelPressureFlushes) {
  // Exercise repeated pressure with every lattice level owning a GroupingSet.
  auto input = makeInput(20, 1'000, 5, 80, 1'000);
  const auto sets = rollupSets(3);

  auto expected = AssertQueryBuilder(makeFusedPlan(input, 3, sets))
                      .config(core::QueryConfig::kMaxPartialAggregationMemory, fmt::format("{}", 1LL << 30))
                      .config(core::QueryConfig::kMaxExtendedPartialAggregationMemory, fmt::format("{}", 1LL << 30))
                      .copyResults(pool());

  std::shared_ptr<Task> task;
  auto actual = AssertQueryBuilder(makeFusedPlan(input, 3, sets))
                    .config(core::QueryConfig::kMaxPartialAggregationMemory, "4096")
                    .config(core::QueryConfig::kMaxExtendedPartialAggregationMemory, "4096")
                    .copyResults(pool(), task);

  EXPECT_TRUE(assertEqualResults({expected}, {actual}));

  int64_t flushes{0};
  int64_t pressureFlushedRows{0};
  bool foundOperator = false;
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& stats : pipeline.operatorStats) {
      if (stats.operatorType != "GroupingSetAggregation") {
        continue;
      }
      foundOperator = true;
      const auto it = stats.runtimeStats.find("gsagg.flushes");
      ASSERT_NE(it, stats.runtimeStats.end());
      flushes += it->second.sum;
      const auto flushedRows = stats.runtimeStats.find("flushRowCount");
      ASSERT_NE(flushedRows, stats.runtimeStats.end());
      pressureFlushedRows += flushedRows->second.sum;
    }
  }
  EXPECT_TRUE(foundOperator);
  EXPECT_GT(flushes, static_cast<int64_t>(sets.size()))
      << "the operator reached only its final sweep and never pressure-drained";
  EXPECT_GT(pressureFlushedRows, 0) << "pressure rows were not exported through the standard metric";
}

TEST_F(MultiGroupingSetAggregationTest, adaptiveAbandonConvertsSubsequentRows) {
  constexpr int32_t kNumBatches = 16;
  constexpr vector_size_t kBatchSize = 256;
  auto input = makeInput(kNumBatches, kBatchSize, 2, 4, 100'000);
  const auto sets = rollupSets(3);
  const AggSpec spec{{"sum(v) as s"}, {{BIGINT()}}, {"sum(s) as s"}, {"s"}};
  const std::unordered_map<std::string, std::string> forceMaxMemoryAbandon = {
      {core::QueryConfig::kMaxPartialAggregationMemory, "4096"},
      {core::QueryConfig::kMaxExtendedPartialAggregationMemory, "4096"},
      // Keep the per-batch early valve closed so this test exercises the
      // independent full-table/max-memory transition into converter mode.
      {core::QueryConfig::kAbandonPartialAggregationMinRows, "1000000"},
      {core::QueryConfig::kAbandonPartialAggregationMinPct, "90"},
  };

  std::shared_ptr<Task> task;
  auto fused =
      AssertQueryBuilder(makeFusedPlan(input, 3, sets, spec)).configs(forceMaxMemoryAbandon).copyResults(pool(), task);
  auto reference = AssertQueryBuilder(makeReferencePlan(input, 3, sets, spec)).copyResults(pool());
  ASSERT_TRUE(assertEqualResults({reference}, {fused}));
  ASSERT_NE(task, nullptr);

  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.dynamicallyAbandoned"), 1);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.earlyAbandoned"), 0);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.maxMemoryAbandoned"), 1);
  EXPECT_GT(groupingSetRuntimeStat(task, "gsagg.set0.dynamicAbandonRows"), 0);
  EXPECT_EQ(
      groupingSetRuntimeStat(task, "gsagg.rawConvertedRows"),
      groupingSetRuntimeStat(task, "gsagg.set0.dynamicAbandonRows"));
  EXPECT_GE(
      groupingSetRuntimeStat(task, "gsagg.dynamicAbandonRows"), groupingSetRuntimeStat(task, "gsagg.rawConvertedRows"));
  assertGroupingSetLifetimeAccounting(task, sets.size());
}

TEST_F(MultiGroupingSetAggregationTest, earlyAbandonCountUsesRawConverterFallback) {
  auto input = makeNearUniqueFinestInput(/*numBatches=*/8, /*batchSize=*/256);
  const AggSpec spec{{"count(v) as c"}, {{BIGINT()}}, {"count(c) as c"}, {"c"}};
  assertRawConverterFallbackAfterEarlyAbandon(input, spec);
}

TEST_F(MultiGroupingSetAggregationTest, earlyAbandonNullableStringMinMaxUsesRawConverterFallback) {
  auto input = makeNearUniqueFinestInput(/*numBatches=*/8, /*batchSize=*/256);
  for (auto& batch : input) {
    auto children = batch->children();
    children.push_back(makeFlatVector<StringView>(
        batch->size(),
        [](auto row) {
          return StringView(row % 3 == 0 ? "pear" : row % 3 == 1 ? "apple" : "orange");
        },
        [](auto row) { return row % 7 == 0; }));
    batch = makeRowVector({"k1", "k2", "k3", "v", "w", "text"}, std::move(children));
  }
  const AggSpec spec{
      {"min(text) as lo", "max(text) as hi"},
      {{VARCHAR()}, {VARCHAR()}},
      {"min(lo) as lo", "max(hi) as hi"},
      {"lo", "hi"}};
  assertRawConverterFallbackAfterEarlyAbandon(input, spec);
}

TEST_F(MultiGroupingSetAggregationTest, abandonsNearUniqueRootWithoutWaitingForPressure) {
  constexpr int32_t kNumBatches = 24;
  constexpr vector_size_t kBatchSize = 512;
  constexpr int32_t kMinRows = 4'096;
  constexpr int64_t kTotalRows = static_cast<int64_t>(kNumBatches) * kBatchSize;
  auto input = makeNearUniqueFinestInput(kNumBatches, kBatchSize);
  const auto sets = rollupSets(3);
  const AggSpec spec{{"sum(v) as s"}, {{BIGINT()}}, {"sum(s) as s"}, {"s"}};
  const std::unordered_map<std::string, std::string> configs = {
      {core::QueryConfig::kMaxPartialAggregationMemory, fmt::format("{}", 1LL << 30)},
      {core::QueryConfig::kMaxExtendedPartialAggregationMemory, fmt::format("{}", 1LL << 30)},
      {core::QueryConfig::kAbandonPartialAggregationMinRows, fmt::format("{}", kMinRows)},
      {core::QueryConfig::kAbandonPartialAggregationMinPct, "90"},
      {core::QueryConfig::kPreferredOutputBatchRows, "128"},
  };

  std::shared_ptr<Task> task;
  auto fused = AssertQueryBuilder(makeFusedPlan(input, 3, sets, spec)).configs(configs).copyResults(pool(), task);
  auto reference = AssertQueryBuilder(makeReferencePlan(input, 3, sets, spec)).copyResults(pool());
  ASSERT_TRUE(assertEqualResults({reference}, {fused}));
  ASSERT_NE(task, nullptr);

  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.earlyAbandonedSets"), 1);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.earlyAbandoned"), 1);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.dynamicallyAbandoned"), 1);
  const auto rowsBeforeAbandon = groupingSetRuntimeStat(task, "gsagg.set0.abandonInputRows");
  EXPECT_GT(rowsBeforeAbandon, kMinRows);
  EXPECT_LE(rowsBeforeAbandon, kMinRows + kBatchSize)
      << "raw root waited beyond the first batch that crossed the evidence threshold";
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.rawConvertedRows"), kTotalRows - rowsBeforeAbandon);
  EXPECT_GT(groupingSetRuntimeStat(task, "gsagg.peakSampledFlushableBytes"), 0)
      << "the effectiveness drain skipped the operator's memory sample";
  for (int32_t set = 1; set < sets.size(); ++set) {
    EXPECT_EQ(groupingSetRuntimeStat(task, fmt::format("gsagg.set{}.earlyAbandoned", set)), 0);
    EXPECT_EQ(groupingSetRuntimeStat(task, fmt::format("gsagg.set{}.dynamicallyAbandoned", set)), 0);
  }
  assertGroupingSetLifetimeAccounting(task, sets.size());
}

TEST_F(MultiGroupingSetAggregationTest, earlyAbandonPctBoundary) {
  constexpr vector_size_t kBatchSize = 1'000;
  const auto sets = rollupSets(3);
  const AggSpec spec{{"sum(v) as s"}, {{BIGINT()}}, {"sum(s) as s"}, {"s"}};
  const std::unordered_map<std::string, std::string> configs = {
      {core::QueryConfig::kMaxPartialAggregationMemory, fmt::format("{}", 1LL << 30)},
      {core::QueryConfig::kMaxExtendedPartialAggregationMemory, fmt::format("{}", 1LL << 30)},
      {core::QueryConfig::kAbandonPartialAggregationMinRows, "999"},
      {core::QueryConfig::kAbandonPartialAggregationMinPct, "90"},
  };

  const auto checkBoundary = [&](vector_size_t numFinestGroups, bool shouldAbandon) {
    SCOPED_TRACE(fmt::format("numFinestGroups={}", numFinestGroups));
    auto input = makeFinestCardinalityInput(kBatchSize, numFinestGroups);
    std::shared_ptr<Task> task;
    auto fused = AssertQueryBuilder(makeFusedPlan(input, 3, sets, spec)).configs(configs).copyResults(pool(), task);
    auto reference = AssertQueryBuilder(makeReferencePlan(input, 3, sets, spec)).copyResults(pool());
    ASSERT_TRUE(assertEqualResults({reference}, {fused}));
    ASSERT_NE(task, nullptr);

    EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.earlyAbandoned"), shouldAbandon ? 1 : 0);
    EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.dynamicallyAbandoned"), shouldAbandon ? 1 : 0);
    if (shouldAbandon) {
      EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.abandonInputRows"), kBatchSize);
      EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.abandonOutputRows"), numFinestGroups);
      EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.abandonPct"), 90);
    }
    assertGroupingSetLifetimeAccounting(task, sets.size());
  };

  checkBoundary(/*numFinestGroups=*/900, /*shouldAbandon=*/true);
  checkBoundary(/*numFinestGroups=*/890, /*shouldAbandon=*/false);
}

TEST_F(MultiGroupingSetAggregationTest, abandonsModerateReductionAtMaxMemory) {
  constexpr int32_t kNumBatches = 16;
  constexpr vector_size_t kBatchSize = 512;
  // Two rows per full key gives 50% output: below the 90% early valve, but
  // above HashAggregation's 40% final valve at maximum extended memory.
  auto input = makeFinestReductionInput(kNumBatches, kBatchSize, /*rowsPerFinestGroup=*/2);
  const auto sets = rollupSets(3);
  const AggSpec spec{{"sum(v) as s"}, {{BIGINT()}}, {"sum(s) as s"}, {"s"}};
  const std::unordered_map<std::string, std::string> configs = {
      {core::QueryConfig::kMaxPartialAggregationMemory, "4096"},
      // Match production's 0.10:0.15 initial-to-extended ratio. The first
      // full-table drain must grow the budget; only a later drain at the
      // extended maximum may apply the 40% final valve.
      {core::QueryConfig::kMaxExtendedPartialAggregationMemory, "6144"},
      {core::QueryConfig::kAbandonPartialAggregationMinRows, "1000000"},
      {core::QueryConfig::kAbandonPartialAggregationMinPct, "90"},
  };

  std::shared_ptr<Task> task;
  auto fused = AssertQueryBuilder(makeFusedPlan(input, 3, sets, spec)).configs(configs).copyResults(pool(), task);
  auto reference = AssertQueryBuilder(makeReferencePlan(input, 3, sets, spec)).copyResults(pool());
  ASSERT_TRUE(assertEqualResults({reference}, {fused}));
  ASSERT_NE(task, nullptr);

  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.earlyAbandoned"), 0);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.maxMemoryAbandoned"), 1);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.dynamicallyAbandoned"), 1);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.budgetGrowths"), 1);
  EXPECT_GE(groupingSetRuntimeStat(task, "gsagg.set0.flushTimes"), 2);
  EXPECT_GT(groupingSetRuntimeStat(task, "gsagg.rawConvertedRows"), 0);
  assertGroupingSetLifetimeAccounting(task, sets.size());
}

TEST_F(MultiGroupingSetAggregationTest, retainsExactlyFortyPctAtMaxMemory) {
  constexpr vector_size_t kBatchSize = 1'000;
  constexpr vector_size_t kNumFinestGroups = 400;
  auto input = makeFinestCardinalityInput(kBatchSize, kNumFinestGroups);
  const auto sets = rollupSets(3);
  const AggSpec spec{{"sum(v) as s"}, {{BIGINT()}}, {"sum(s) as s"}, {"s"}};
  const std::unordered_map<std::string, std::string> configs = {
      {core::QueryConfig::kMaxPartialAggregationMemory, "4096"},
      {core::QueryConfig::kMaxExtendedPartialAggregationMemory, "4096"},
      {core::QueryConfig::kAbandonPartialAggregationMinRows, "1000000"},
      {core::QueryConfig::kAbandonPartialAggregationMinPct, "90"},
  };

  std::shared_ptr<Task> task;
  auto fused = AssertQueryBuilder(makeFusedPlan(input, 3, sets, spec)).configs(configs).copyResults(pool(), task);
  auto reference = AssertQueryBuilder(makeReferencePlan(input, 3, sets, spec)).copyResults(pool());
  ASSERT_TRUE(assertEqualResults({reference}, {fused}));
  ASSERT_NE(task, nullptr);

  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.inputRows"), kBatchSize);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.outputRows"), kNumFinestGroups);
  EXPECT_GT(groupingSetRuntimeStat(task, "gsagg.set0.flushTimes"), 0);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.earlyAbandoned"), 0);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.maxMemoryAbandoned"), 0);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.dynamicallyAbandoned"), 0);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.rawConvertedRows"), 0);
  assertGroupingSetLifetimeAccounting(task, sets.size());
}

TEST_F(MultiGroupingSetAggregationTest, retainsUsefulReductionAtMaxMemory) {
  constexpr int32_t kNumBatches = 16;
  constexpr vector_size_t kBatchSize = 512;
  // Four rows per full key gives 25% output and must stay aggregated even at
  // the maximum extended memory budget.
  auto input = makeFinestReductionInput(kNumBatches, kBatchSize, /*rowsPerFinestGroup=*/4);
  const auto sets = rollupSets(3);
  const AggSpec spec{{"sum(v) as s"}, {{BIGINT()}}, {"sum(s) as s"}, {"s"}};
  const std::unordered_map<std::string, std::string> configs = {
      {core::QueryConfig::kMaxPartialAggregationMemory, "4096"},
      {core::QueryConfig::kMaxExtendedPartialAggregationMemory, "4096"},
      {core::QueryConfig::kAbandonPartialAggregationMinRows, "1000000"},
      {core::QueryConfig::kAbandonPartialAggregationMinPct, "90"},
  };

  std::shared_ptr<Task> task;
  auto fused = AssertQueryBuilder(makeFusedPlan(input, 3, sets, spec)).configs(configs).copyResults(pool(), task);
  auto reference = AssertQueryBuilder(makeReferencePlan(input, 3, sets, spec)).copyResults(pool());
  ASSERT_TRUE(assertEqualResults({reference}, {fused}));
  ASSERT_NE(task, nullptr);

  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.earlyAbandoned"), 0);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.maxMemoryAbandoned"), 0);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.dynamicallyAbandoned"), 0);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.rawConvertedRows"), 0);
  EXPECT_GT(groupingSetRuntimeStat(task, "gsagg.set0.flushTimes"), 1);
  assertGroupingSetLifetimeAccounting(task, sets.size());
}

TEST_F(MultiGroupingSetAggregationTest, retainsProductiveFinestAggregation) {
  constexpr int32_t kNumBatches = 24;
  constexpr vector_size_t kBatchSize = 512;
  auto input = makeInput(kNumBatches, kBatchSize, 4, 16, 4);
  const auto sets = rollupSets(3);
  const AggSpec spec{{"sum(v) as s"}, {{BIGINT()}}, {"sum(s) as s"}, {"s"}};
  const std::unordered_map<std::string, std::string> configs = {
      {core::QueryConfig::kMaxPartialAggregationMemory, fmt::format("{}", 1LL << 30)},
      {core::QueryConfig::kMaxExtendedPartialAggregationMemory, fmt::format("{}", 1LL << 30)},
      {core::QueryConfig::kAbandonPartialAggregationMinRows, "4096"},
      {core::QueryConfig::kAbandonPartialAggregationMinPct, "90"},
  };

  std::shared_ptr<Task> task;
  auto fused = AssertQueryBuilder(makeFusedPlan(input, 3, sets, spec)).configs(configs).copyResults(pool(), task);
  auto reference = AssertQueryBuilder(makeReferencePlan(input, 3, sets, spec)).copyResults(pool());
  ASSERT_TRUE(assertEqualResults({reference}, {fused}));
  ASSERT_NE(task, nullptr);

  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.earlyAbandonedSets"), 0);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.dynamicallyAbandonedSets"), 0);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.rawConvertedRows"), 0);
  EXPECT_LE(groupingSetRuntimeStat(task, "gsagg.set0.totalOutputRows"), 4 * 16 * 4);
  assertGroupingSetLifetimeAccounting(task, sets.size());
}

TEST_F(MultiGroupingSetAggregationTest, abandonedInternalLevelStillEnforcesDescendantHardCap) {
  // Each raw batch contains 16,384 rows over 4,096 full keys:
  //
  //   set 0 {k1,k2,k3}: 25% output/input
  //   set 1 {k1,k2}:    100% output/input (k3 is constant)
  //   set 2 {k1}:        25% output/input
  //
  // With a 100% abandon threshold, only internal set 1 opens its adaptive
  // pass-through valve. Later set-0 drains must then discover and hard-cap set
  // 2 through that abandoned level.
  constexpr int32_t kNumBatches = 6;
  constexpr vector_size_t kBatchSize = 16'384;
  constexpr int32_t kFullKeys = 4'096;
  std::vector<RowVectorPtr> input;
  input.reserve(kNumBatches);
  for (auto batch = 0; batch < kNumBatches; ++batch) {
    input.push_back(makeRowVector(
        {"k1", "k2", "k3", "v"},
        {makeFlatVector<int64_t>(kBatchSize, [](auto row) { return (row % kFullKeys) / 4; }),
         makeFlatVector<int64_t>(kBatchSize, [](auto row) { return (row % kFullKeys) % 4; }),
         makeFlatVector<int64_t>(kBatchSize, [](auto) { return 0; }),
         makeFlatVector<int64_t>(kBatchSize, [](auto) { return 1; })}));
  }

  const auto sets = rollupSets(3);
  const AggSpec spec{{"sum(v) as s"}, {{BIGINT()}}, {"sum(s) as s"}, {"s"}};

  auto expected = AssertQueryBuilder(makeFusedPlan(input, 3, sets, spec))
                      .config(core::QueryConfig::kMaxPartialAggregationMemory, fmt::format("{}", 1LL << 30))
                      .config(core::QueryConfig::kMaxExtendedPartialAggregationMemory, fmt::format("{}", 1LL << 30))
                      .copyResults(pool());

  std::shared_ptr<Task> task;
  auto actual = AssertQueryBuilder(makeFusedPlan(input, 3, sets, spec))
                    .config(core::QueryConfig::kMaxPartialAggregationMemory, "4096")
                    .config(core::QueryConfig::kMaxExtendedPartialAggregationMemory, "4096")
                    .config(core::QueryConfig::kAbandonPartialAggregationMinRows, "0")
                    .config(core::QueryConfig::kAbandonPartialAggregationMinPct, "100")
                    .config(core::QueryConfig::kPreferredOutputBatchRows, "8192")
                    .copyResults(pool(), task);

  ASSERT_TRUE(assertEqualResults({expected}, {actual}));
  ASSERT_NE(task, nullptr);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.dynamicallyAbandonedSets"), 1);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.dynamicallyAbandoned"), 0);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set1.dynamicallyAbandoned"), 1);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set1.abandonPct"), 100);
  EXPECT_GE(groupingSetRuntimeStat(task, "gsagg.set1.flushTimes"), 3);
  EXPECT_GT(groupingSetRuntimeStat(task, "gsagg.set1.dynamicAbandonRows"), 0);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set2.dynamicallyAbandoned"), 0);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set3.dynamicallyAbandoned"), 0);
  const auto transitiveDrains = groupingSetRuntimeStat(task, "gsagg.transitiveHardCapDrains");
  EXPECT_GT(transitiveDrains, 0) << "a live descendant behind the abandoned internal level escaped its hard cap";
  EXPECT_GE(groupingSetRuntimeStat(task, "gsagg.nestedDrains"), transitiveDrains);
  assertGroupingSetLifetimeAccounting(task, sets.size());
}

// ===========================================================================
// Boundary shapes
// ===========================================================================

TEST_F(MultiGroupingSetAggregationTest, boundaryTwoKeys) {
  auto input = makeInput(5, 1'000, 4, 50, 1);
  assertFusedMatchesReference(input, /*numKeys=*/2);
}

TEST_F(MultiGroupingSetAggregationTest, boundaryAllRowsIdentical) {
  // Every set collapses to a single group, so every set is maximally reducing
  // and each lattice level emits a single state.
  const vector_size_t size = 2'000;
  auto batch = makeRowVector(
      {"k1", "k2", "k3", "v", "w"},
      {makeFlatVector<int64_t>(size, [](auto) { return 7; }),
       makeFlatVector<int64_t>(size, [](auto) { return 7; }),
       makeFlatVector<int64_t>(size, [](auto) { return 7; }),
       makeFlatVector<int64_t>(size, [](auto) { return 1; }),
       makeFlatVector<double>(size, [](auto) { return 1.5; })});
  assertFusedMatchesReference({batch, batch, batch}, /*numKeys=*/3);
}

TEST_F(MultiGroupingSetAggregationTest, boundaryAllRowsDistinctAtEveryLevel) {
  // The worst case for fusion: G_i == G_{i+1} at every level, so no set
  // reduces, memory grows monotonically and (with default config) the abandon
  // valve is the only thing that saves it. Correctness must not depend on which
  // way that goes.
  const vector_size_t size = 2'000;
  auto batch = makeRowVector(
      {"k1", "k2", "k3", "v", "w"},
      {makeFlatVector<int64_t>(size, [](auto row) { return row; }),
       makeFlatVector<int64_t>(size, [](auto row) { return row; }),
       makeFlatVector<int64_t>(size, [](auto row) { return row; }),
       makeFlatVector<int64_t>(size, [](auto row) { return row; }),
       makeFlatVector<double>(size, [](auto row) { return static_cast<double>(row); })});
  assertFusedMatchesReference({batch}, /*numKeys=*/3);
}

TEST_F(MultiGroupingSetAggregationTest, boundarySingleBatch) {
  // One batch, so noMoreInput() arrives immediately after a single addInput()
  // and the entire final sweep runs from a cold start with no pressure drain
  // ever having happened.
  auto input = makeInput(/*numBatches=*/1, /*batchSize=*/1'000, 4, 20, 200);
  assertFusedMatchesReference(input, /*numKeys=*/3);
}

TEST_F(MultiGroupingSetAggregationTest, boundaryZeroInputRows) {
  // Spark grouping sets emit no rows for empty input. Guard against a global
  // GroupingSet synthesizing an identity row during the final sweep.
  auto empty = makeRowVector(rawInputType(), 0);
  assertFusedMatchesReference({empty}, /*numKeys=*/3);

  // Pin the absolute cardinality in addition to differential equality.
  auto fused = AssertQueryBuilder(makeFusedPlan({empty}, 3, rollupSets(3))).copyResults(pool());
  ASSERT_EQ(fused->size(), 0) << "empty input must produce zero rows, per Spark's grouping-sets "
                                 "semantics; got a synthesised grand-total row";
}

// ===========================================================================
// Aggregate coverage
// ===========================================================================
//
// The main differential tests already carry all three shapes together. These
// isolate each one so a failure names the accumulator rather than the plan.

TEST_F(MultiGroupingSetAggregationTest, aggregateFixedWidthOnly) {
  auto input = makeInput(8, 1'000, 5, 40, 400);
  assertFusedMatchesReferenceWithAggregates(input, 3, {"sum(v) as s"}, {{BIGINT()}}, {"sum(s) as s"}, {"s"});
}

TEST_F(MultiGroupingSetAggregationTest, aggregateStructIntermediate) {
  // avg(DOUBLE): intermediate is ROW(DOUBLE, BIGINT). A struct intermediate is
  // where a wrong column index shows up as a type error rather than a silent
  // wrong number.
  auto input = makeInput(8, 1'000, 5, 40, 400);
  assertFusedMatchesReferenceWithAggregates(input, 3, {"avg(w) as m"}, {{DOUBLE()}}, {"avg(m) as m"}, {"m"});
}

TEST_F(MultiGroupingSetAggregationTest, aggregateVariableWidthExternalMemory) {
  // array_agg(BIGINT) covers variable-width external-memory state.
  auto input = makeInput(8, 1'000, 5, 40, 400);
  assertFusedMatchesReferenceWithAggregates(
      input, 3, {"array_agg(v) as l"}, {{BIGINT()}}, {"array_agg(l) as l"}, {"array_sort(l) as l"});
}

// Spark partial buffers.
//
// Spark represents multi-field aggregate buffers as one ROW-typed Velox
// intermediate column:
//
//   spark sum(DECIMAL(p,s)) -> intermediate ROW(DECIMAL(min(38,p+10),s), boolean)
//                              == (sum, isEmpty)
//   spark avg(DOUBLE)       -> intermediate ROW(DOUBLE, BIGINT)
//                              == (sum, count)
//   spark avg(DECIMAL(p,s)) -> intermediate ROW(DECIMAL(...), BIGINT)
//
// Register them under a prefix so they coexist with the Presto functions above.
class MultiGroupingSetSparkBufferTest : public MultiGroupingSetAggregationTest {
 protected:
  void SetUp() override {
    MultiGroupingSetAggregationTest::SetUp();
    functions::aggregate::sparksql::registerAggregateFunctions(
        "spark_", /*withCompanionFunctions=*/true, /*overwrite=*/true);
  }

  // Add a physical decimal column because partialAggregation requires a field
  // input. Values remain within the declared precision.
  std::vector<RowVectorPtr> makeDecimalInput(
      int32_t numBatches,
      vector_size_t batchSize,
      int32_t k1Cardinality,
      int32_t k2Cardinality,
      int32_t k3Cardinality,
      uint32_t seed = 1234,
      const TypePtr& decimalType = DECIMAL(7, 2)) {
    std::vector<RowVectorPtr> batches;
    batches.reserve(numBatches);
    folly::Random::DefaultGenerator rng(seed);
    for (auto b = 0; b < numBatches; ++b) {
      std::vector<int64_t> k1(batchSize), k2(batchSize), k3(batchSize), v(batchSize), d(batchSize);
      std::vector<double> w(batchSize);
      for (auto i = 0; i < batchSize; ++i) {
        k1[i] = folly::Random::rand32(k1Cardinality, rng);
        k2[i] = folly::Random::rand32(k2Cardinality, rng);
        k3[i] = folly::Random::rand32(k3Cardinality, rng);
        v[i] = folly::Random::rand32(1000, rng);
        w[i] = static_cast<double>(folly::Random::rand32(1000, rng)) / 7.0;
        // Unscaled decimal(7,2): 0..999999 -> 0.00..9999.99.
        d[i] = folly::Random::rand32(1'000'000, rng);
      }
      batches.push_back(makeRowVector(
          {"k1", "k2", "k3", "v", "w", "d"},
          {makeFlatVector<int64_t>(k1),
           makeFlatVector<int64_t>(k2),
           makeFlatVector<int64_t>(k3),
           makeFlatVector<int64_t>(v),
           makeFlatVector<double>(w),
           makeFlatVector<int64_t>(d, decimalType)}));
    }
    return batches;
  }

  /// Eight-key q67-shaped input: four integer keys followed by four string
  /// keys. The string columns mix inline, external-buffer, and null values.
  /// DECIMAL(18,4) matches the widened product that feeds the query's Spark
  /// decimal sum and produces a two-field ROW accumulator.
  std::vector<RowVectorPtr> makeQ67ShapeInput(int32_t numBatches, vector_size_t batchSize) {
    const auto decimalType = DECIMAL(18, 4);
    const std::array<std::optional<std::string>, 4> s1Values = {"R", "A", "RETURN", std::nullopt};
    const std::array<std::optional<std::string>, 3> s2Values = {"O", "F", std::nullopt};
    const std::array<std::optional<std::string>, 4> s3Values = {"AIR", "SHIP", "TRUCK", std::nullopt};
    const std::array<std::optional<std::string>, 4> s4Values = {
        "DELIVER IN PERSON", "TAKE BACK RETURN", "EXPRESS PRIORITY COURIER", std::nullopt};

    std::vector<RowVectorPtr> batches;
    batches.reserve(numBatches);
    for (auto batch = 0; batch < numBatches; ++batch) {
      std::vector<int32_t> k1(batchSize), k2(batchSize), k3(batchSize), k4(batchSize);
      std::vector<std::optional<std::string>> s1(batchSize), s2(batchSize), s3(batchSize), s4(batchSize);
      std::vector<std::optional<int64_t>> d(batchSize);

      for (auto row = 0; row < batchSize; ++row) {
        const int64_t id = static_cast<int64_t>(batch) * batchSize + row;
        // A mixed-radix prefix creates progressively larger rollup levels and
        // more than one output batch at the finest grain.
        k1[row] = id % 3;
        k2[row] = (id / 3) % 5;
        k3[row] = (id / 15) % 7;
        k4[row] = (id / 105) % 11;
        s1[row] = s1Values[(id / 2) % s1Values.size()];
        s2[row] = s2Values[(id / 5) % s2Values.size()];
        s3[row] = s3Values[(id / 11) % s3Values.size()];
        s4[row] = s4Values[(id / 17) % s4Values.size()];

        // Keep values within DECIMAL(18,4), with nulls interspersed across
        // batches so the Spark sum's isEmpty field is exercised.
        if (id % 13 == 0) {
          d[row] = std::nullopt;
        } else {
          d[row] = (id * 10'007) % 100'000'000;
        }
      }

      batches.push_back(makeRowVector(
          {"k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "d"},
          {makeFlatVector<int32_t>(k1),
           makeFlatVector<int32_t>(k2),
           makeFlatVector<int32_t>(k3),
           makeFlatVector<int32_t>(k4),
           makeNullableFlatVector<std::string>(s1),
           makeNullableFlatVector<std::string>(s2),
           makeNullableFlatVector<std::string>(s3),
           makeNullableFlatVector<std::string>(s4),
           makeNullableFlatVector<int64_t>(d, decimalType)}));
    }
    return batches;
  }
};

TEST_F(MultiGroupingSetSparkBufferTest, decimalSum) {
  // Spark decimal sum uses the two-field (sum, isEmpty) state.
  auto input = makeDecimalInput(10, 1'000, 5, 23, 97);
  assertFusedMatchesReferenceWithAggregates(
      input, 3, {"spark_sum(d) as s"}, {{DECIMAL(7, 2)}}, {"spark_sum(s) as s"}, {"s"});
}

TEST_F(MultiGroupingSetSparkBufferTest, avgDouble) {
  // Spark avg(DOUBLE) uses the (sum, count) state.
  auto input = makeInput(10, 1'000, 5, 23, 97);
  assertFusedMatchesReferenceWithAggregates(
      input, 3, {"spark_avg(w) as m"}, {{DOUBLE()}}, {"spark_avg(m) as m"}, {"m"});
}

TEST_F(MultiGroupingSetSparkBufferTest, earlyAbandonAvgDoubleUsesRawConverterFallback) {
  auto input = makeNearUniqueFinestInput(/*numBatches=*/8, /*batchSize=*/256);
  const AggSpec spec{{"spark_avg(w) as m"}, {{DOUBLE()}}, {"spark_avg(m) as m"}, {"m"}};
  assertRawConverterFallbackAfterEarlyAbandon(input, spec);
}

TEST_F(MultiGroupingSetSparkBufferTest, avgDecimal) {
  // Use DECIMAL(12,2) because Velox rejects lower-precision decimal averages.
  auto input = makeDecimalInput(10, 1'000, 5, 23, 97, /*seed=*/1234, /*decimalType=*/DECIMAL(12, 2));
  assertFusedMatchesReferenceWithAggregates(
      input, 3, {"spark_avg(d) as m"}, {{DECIMAL(12, 2)}}, {"spark_avg(m) as m"}, {"m"});
}

TEST_F(MultiGroupingSetSparkBufferTest, mixedMultiAndSingleColumn) {
  // Mix row-typed and scalar states to catch per-aggregate channel errors.
  auto input = makeDecimalInput(10, 1'000, 5, 23, 97);
  assertFusedMatchesReferenceWithAggregates(
      input,
      3,
      {"spark_sum(d) as s", "spark_avg(w) as m", "spark_sum(v) as t"},
      {{DECIMAL(7, 2)}, {DOUBLE()}, {BIGINT()}},
      {"spark_sum(s) as s", "spark_avg(m) as m", "spark_sum(t) as t"},
      {"s", "m", "t"});
}

TEST_F(MultiGroupingSetSparkBufferTest, mixedMultiColumnCube) {
  // Exercise row-typed states across a CUBE derivation lattice.
  auto input = makeDecimalInput(8, 1'000, 4, 12, 40);
  const auto spec = AggSpec{
      {"spark_sum(d) as s", "spark_avg(w) as m"},
      {{DECIMAL(7, 2)}, {DOUBLE()}},
      {"spark_sum(s) as s", "spark_avg(m) as m"},
      {"s", "m"}};
  assertFusedMatchesReferenceForSets(input, 3, cubeSets(3), {}, spec);
}

TEST_F(MultiGroupingSetSparkBufferTest, q67ShapeEightMixedKeysDecimalRollup) {
  auto input = makeQ67ShapeInput(/*numBatches=*/6, /*batchSize=*/257);
  const auto sets = rollupSets(/*numKeys=*/8);
  ASSERT_EQ(sets.size(), 9);
  const auto spec = AggSpec{{"spark_sum(d) as s"}, {{DECIMAL(18, 4)}}, {"spark_sum(s) as s"}, {"s"}};

  assertFusedMatchesReferenceForSets(input, 8, sets, {}, spec);
}

TEST_F(MultiGroupingSetSparkBufferTest, q67ShapeEarlyAbandonPreservesDecimalState) {
  constexpr int32_t kNumBatches = 8;
  constexpr vector_size_t kBatchSize = 257;
  constexpr int32_t kMinRows = 512;
  constexpr int64_t kTotalRows = static_cast<int64_t>(kNumBatches) * kBatchSize;
  auto input = makeQ67ShapeInput(kNumBatches, kBatchSize);
  const auto sets = rollupSets(/*numKeys=*/8);
  const auto spec = AggSpec{{"spark_sum(d) as s"}, {{DECIMAL(18, 4)}}, {"spark_sum(s) as s"}, {"s"}};
  const std::unordered_map<std::string, std::string> configs = {
      {core::QueryConfig::kMaxPartialAggregationMemory, fmt::format("{}", 1LL << 30)},
      {core::QueryConfig::kMaxExtendedPartialAggregationMemory, fmt::format("{}", 1LL << 30)},
      {core::QueryConfig::kAbandonPartialAggregationMinRows, fmt::format("{}", kMinRows)},
      {core::QueryConfig::kAbandonPartialAggregationMinPct, "90"},
      {core::QueryConfig::kPreferredOutputBatchRows, "128"},
  };

  std::shared_ptr<Task> task;
  auto fused = AssertQueryBuilder(makeFusedPlan(input, 8, sets, spec)).configs(configs).copyResults(pool(), task);
  auto reference = AssertQueryBuilder(makeReferencePlan(input, 8, sets, spec)).copyResults(pool());
  ASSERT_TRUE(assertEqualResults({reference}, {fused}));
  ASSERT_NE(task, nullptr);

  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.earlyAbandoned"), 1);
  const auto rowsBeforeAbandon = groupingSetRuntimeStat(task, "gsagg.set0.abandonInputRows");
  EXPECT_GT(rowsBeforeAbandon, kMinRows);
  EXPECT_LE(rowsBeforeAbandon, kMinRows + kBatchSize);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.rawConvertedRows"), kTotalRows - rowsBeforeAbandon);
  assertGroupingSetLifetimeAccounting(task, sets.size());
}

TEST_F(MultiGroupingSetSparkBufferTest, q67ShapeEightMixedKeysDecimalRollupUnderPressure) {
  // Close the q67-specific pressure cross-product: nine cascade levels,
  // nullable/string keys, Spark's ROW(decimal, isEmpty) state, and the
  // raw-input path. Compare against the same fused plan at a large budget
  // because the pinned Expand reference is not pressure-safe.
  auto input = makeQ67ShapeInput(/*numBatches=*/6, /*batchSize=*/257);
  const auto sets = rollupSets(/*numKeys=*/8);
  ASSERT_EQ(sets.size(), 9);
  const auto spec = AggSpec{{"spark_sum(d) as s"}, {{DECIMAL(18, 4)}}, {"spark_sum(s) as s"}, {"s"}};

  auto expected = AssertQueryBuilder(makeFusedPlan(input, 8, sets, spec))
                      .config(core::QueryConfig::kMaxPartialAggregationMemory, fmt::format("{}", 1LL << 30))
                      .config(core::QueryConfig::kMaxExtendedPartialAggregationMemory, fmt::format("{}", 1LL << 30))
                      .copyResults(pool());

  std::shared_ptr<Task> task;
  auto actual = AssertQueryBuilder(makeFusedPlan(input, 8, sets, spec))
                    .config(core::QueryConfig::kMaxPartialAggregationMemory, "4096")
                    .config(core::QueryConfig::kMaxExtendedPartialAggregationMemory, "4096")
                    .copyResults(pool(), task);

  ASSERT_TRUE(assertEqualResults({expected}, {actual}));
  ASSERT_NE(task, nullptr);
  EXPECT_GT(groupingSetRuntimeStat(task, "gsagg.flushes"), static_cast<int64_t>(sets.size()))
      << "the nine-level q67 shape reached only final drains and never pressure-drained";
  EXPECT_GT(groupingSetRuntimeStat(task, "flushRowCount"), 0)
      << "q67-shaped pressure rows were not exported through the standard metric";
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.numLatticeRoots"), 1);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.dynamicallyAbandonedSets"), 1);
  EXPECT_EQ(groupingSetRuntimeStat(task, "gsagg.set0.maxMemoryAbandoned"), 1);
  EXPECT_GT(groupingSetRuntimeStat(task, "gsagg.rawConvertedRows"), 0);
  assertGroupingSetLifetimeAccounting(task, sets.size());
}

TEST_F(MultiGroupingSetSparkBufferTest, decimalSumAbsoluteCorrectness) {
  // Compare the rollup grand total with an independent global aggregation.
  auto input = makeDecimalInput(6, 1'000, 4, 20, 200);

  auto fused =
      AssertQueryBuilder(
          makeFusedPlan(
              input, 3, rollupSets(3), AggSpec{{"spark_sum(d) as s"}, {{DECIMAL(7, 2)}}, {"spark_sum(s) as s"}, {"s"}}))
          .copyResults(pool());

  // Independent grand total: sum over all rows, no grouping.
  auto grandTotal =
      AssertQueryBuilder(PlanBuilder(pool()).values(input).singleAggregation({}, {"spark_sum(d) as s"}).planNode())
          .copyResults(pool());
  ASSERT_EQ(grandTotal->size(), 1);

  // The fused output is ROW(k1,k2,k3,gid,s). The grand-total set is gid == 3
  // (all keys masked out); there must be exactly one such row and its s must
  // equal the independent grand total.
  auto* gidVec = fused->childAt(3)->asFlatVector<int64_t>();
  ASSERT_NE(gidVec, nullptr);
  const auto& fusedS = fused->childAt(4);
  const auto& expectedS = grandTotal->childAt(0);
  int32_t grandRows = 0;
  for (auto i = 0; i < fused->size(); ++i) {
    if (gidVec->valueAt(i) == 3) {
      ++grandRows;
      ASSERT_TRUE(fusedS->equalValueAt(expectedS.get(), i, 0)) << "grand-total decimal sum mismatch at fused row " << i;
    }
  }
  EXPECT_EQ(grandRows, 1) << "rollup must emit exactly one grand-total row";
}

} // namespace

// Arbitration safety.
//
// Run the operator without a blocking final aggregation, pause after each
// output batch, and reclaim its leaf pool. Finalizing the collected partial
// batches must match a large-budget run.

namespace {
// Locate the operator's leaf memory pool. Its name is
// "op.<planNodeId>.<pipelineId>.<driverId>.GroupingSetAggregation"
// (Task::addOperatorPool), so the operator type is the stable suffix.
memory::MemoryPool* findGsaggPool(memory::MemoryPool* pool) {
  static const std::string kSuffix = ".GroupingSetAggregation";
  memory::MemoryPool* found = nullptr;
  pool->visitChildren([&](memory::MemoryPool* child) {
    if (child->isLeaf()) {
      const auto& name = child->name();
      if (name.size() >= kSuffix.size() && name.compare(name.size() - kSuffix.size(), kSuffix.size(), kSuffix) == 0) {
        found = child;
        return false;
      }
    } else if ((found = findGsaggPool(child)) != nullptr) {
      return false;
    }
    return true;
  });
  return found;
}

struct ArbitrationRun {
  std::vector<RowVectorPtr> partialBatches;
  int64_t reclaimCalls{0};
  int32_t reclaimPoints{0};
};
} // namespace

class MultiGroupingSetArbitrationTest : public MultiGroupingSetAggregationTest {
 protected:
  // Stop at the custom operator so partial batches reach the cursor directly.
  core::PlanNodePtr makeOperatorOnlyPlan(
      const std::vector<RowVectorPtr>& input,
      int32_t numKeys,
      const std::vector<Set>& sets,
      const AggSpec& spec) {
    auto metadata = makeAggregateMetadata(input, numKeys, spec);
    PlanBuilder builder(pool());
    auto source = builder.values(input).planNode();
    auto groupingKeys = makeGroupingKeys(source, numKeys);
    return builder
        .addNode([&](std::string id, core::PlanNodePtr source) {
          return std::make_shared<GroupingSetAggregationNode>(
              std::move(id), groupingKeys, sets, metadata.names, metadata.aggregates, std::move(source));
        })
        .planNode();
  }

  // Re-aggregates collected partial batches into the final answer, so an
  // arbitration run's output can be compared to a plain run's.
  RowVectorPtr finalize(const std::vector<RowVectorPtr>& partialBatches, int32_t numKeys, const AggSpec& spec) {
    return AssertQueryBuilder(PlanBuilder(pool())
                                  .values(partialBatches)
                                  .finalAggregation(appendGid(keyNames(numKeys)), spec.final, spec.rawTypes)
                                  .project(finalProjection(numKeys, spec))
                                  .planNode())
        .copyResults(pool());
  }

  ArbitrationRun runWithReclaimEachBatch(
      const std::vector<RowVectorPtr>& input,
      int32_t numKeys,
      const std::vector<Set>& sets,
      const AggSpec& spec,
      int64_t budget) {
    // A root reclaimer lets Task attach a reclaimer to the operator pool.
    auto queryCtx = core::QueryCtx::create(driverExecutor_.get());
    queryCtx->testingOverrideMemoryPool(memory::memoryManager()->addRootPool(
        queryCtx->queryId(),
        /*capacity=*/1LL << 30,
        exec::MemoryReclaimer::create()));

    CursorParameters params;
    params.planNode = makeOperatorOnlyPlan(input, numKeys, sets, spec);
    params.queryCtx = queryCtx;
    params.maxDrivers = 1;
    params.queryConfigs = {
        {core::QueryConfig::kMaxPartialAggregationMemory, fmt::format("{}", budget)},
        {core::QueryConfig::kMaxExtendedPartialAggregationMemory, fmt::format("{}", budget)},
    };
    auto cursor = TaskCursor::create(params);

    // The cursor copies each batch into the task's own pool, which dies with
    // the cursor at the end of this function. Deep-copy into the test's pool so
    // the collected batches outlive the arbitration task.
    auto deepCopy = [&](const RowVectorPtr& src) {
      auto dst = std::static_pointer_cast<RowVector>(BaseVector::create(src->type(), src->size(), pool()));
      dst->copy(src.get(), 0, 0, src->size());
      return dst;
    };

    ArbitrationRun run;
    memory::MemoryPool* opPool = nullptr;
    while (cursor->moveNext()) {
      run.partialBatches.push_back(deepCopy(cursor->current()));
      auto task = cursor->task();
      if (opPool == nullptr) {
        opPool = findGsaggPool(task->pool());
      }
      if (opPool == nullptr || opPool->reclaimer() == nullptr) {
        continue;
      }
      // Reclaim requires a paused task and an arbitration context.
      task->requestPause().wait();
      {
        memory::ScopedMemoryArbitrationContext arbCtx{opPool};
        memory::MemoryReclaimer::Stats stats;
        opPool->reclaim(/*targetBytes=*/0, /*maxWaitMs=*/0, stats);
      }
      Task::resume(task);
      ++run.reclaimPoints;
    }

    const auto taskStats = cursor->task()->taskStats();
    for (const auto& pipeline : taskStats.pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        if (op.operatorType != "GroupingSetAggregation") {
          continue;
        }
        if (auto it = op.runtimeStats.find("gsagg.reclaim.count"); it != op.runtimeStats.end()) {
          run.reclaimCalls = std::max<int64_t>(run.reclaimCalls, it->second.sum);
        }
      }
    }
    return run;
  }

  void assertArbitrationSafe(const AggSpec& spec, const char* label) {
    SCOPED_TRACE(label);
    auto input = makeInput(10, 1'000, 5, 40, 400);
    const auto sets = rollupSets(3);

    // Reference: the full plan at a large budget, without reclaim.
    auto expected = AssertQueryBuilder(makeFusedPlan(input, 3, sets, spec))
                        .config(core::QueryConfig::kMaxPartialAggregationMemory, fmt::format("{}", 1LL << 30))
                        .copyResults(pool());

    // A 16 KiB budget forces repeated flushes and reclaimer callbacks.
    auto run = runWithReclaimEachBatch(input, 3, sets, spec, /*budget=*/16 << 10);

    EXPECT_GT(run.reclaimPoints, 0) << "operator pool never reclaimed";
    EXPECT_GT(run.reclaimCalls, 0) << "operator reclaim() body never ran (no reclaimer wired?)";
    ASSERT_FALSE(run.partialBatches.empty());
    auto finalized = finalize(run.partialBatches, 3, spec);
    EXPECT_TRUE(assertEqualResults({expected}, {finalized})) << "results changed under memory-pressure reclaim";
  }
};

TEST_F(MultiGroupingSetArbitrationTest, reclaimFixedWidth) {
  // Fixed-width accumulator.
  assertArbitrationSafe(AggSpec{{"sum(v) as s"}, {{BIGINT()}}, {"sum(s) as s"}, {"s"}}, "sum(bigint)");
}

TEST_F(MultiGroupingSetArbitrationTest, reclaimStructIntermediate) {
  // Row-typed accumulator.
  assertArbitrationSafe(AggSpec{{"avg(w) as m"}, {{DOUBLE()}}, {"avg(m) as m"}, {"m"}}, "avg(double)");
}

TEST_F(MultiGroupingSetArbitrationTest, reclaimVariableWidthExternalMemory) {
  // Variable-width external-memory accumulator.
  assertArbitrationSafe(
      AggSpec{{"array_agg(v) as l"}, {{BIGINT()}}, {"array_agg(l) as l"}, {"array_sort(l) as l"}}, "array_agg(bigint)");
}

TEST_F(MultiGroupingSetArbitrationTest, reclaimsExplicitUnusedReservation) {
  auto input = makeInput(/*numBatches=*/10, /*batchSize=*/1'000, 5, 40, 400);
  const auto sets = rollupSets(3);
  const AggSpec spec{{"sum(v) as s"}, {{BIGINT()}}, {"sum(s) as s"}, {"s"}};

  auto queryCtx = core::QueryCtx::create(driverExecutor_.get());
  queryCtx->testingOverrideMemoryPool(memory::memoryManager()->addRootPool(
      queryCtx->queryId(),
      /*capacity=*/1LL << 30,
      exec::MemoryReclaimer::create()));

  const auto plan = makeOperatorOnlyPlan(input, 3, sets, spec);
  CursorParameters params;
  params.planNode = plan;
  params.queryCtx = queryCtx;
  params.maxDrivers = 1;
  params.breakpoints = {{plan->id(), nullptr}};
  params.queryConfigs = {
      {core::QueryConfig::kMaxPartialAggregationMemory, fmt::format("{}", 1LL << 30)},
      {core::QueryConfig::kMaxExtendedPartialAggregationMemory, fmt::format("{}", 1LL << 30)},
      {core::QueryConfig::kPreferredOutputBatchRows, "128"},
  };
  auto cursor = TaskCursor::create(params);
  ASSERT_TRUE(cursor->moveStep(plan->id()));
  ASSERT_EQ(cursor->at(), plan->id());

  const auto task = cursor->task();
  auto* opPool = findGsaggPool(task->pool());
  ASSERT_NE(opPool, nullptr);
  ASSERT_NE(opPool->reclaimer(), nullptr);

  task->requestPause().wait();
  bool resumed{false};
  auto resumeGuard = folly::makeGuard([&]() {
    if (!resumed) {
      Task::resume(task);
    }
  });

  // Model maybeGrowBudget's explicit reservation after the operator has
  // initialized, while its driver is held live at a deterministic input
  // boundary. The minimum reservation is unused headroom, not GroupingSet
  // allocation, and must still make this pool a candidate for parent
  // arbitration.
  ASSERT_TRUE(opPool->maybeReserve(1));
  auto reservationGuard = folly::makeGuard([&]() { opPool->release(); });
  const auto releasable = opPool->releasableReservation();
  ASSERT_GT(releasable, 0);
  const auto reclaimable = opPool->reclaimableBytes();
  ASSERT_TRUE(reclaimable.has_value());
  EXPECT_GE(reclaimable.value(), static_cast<uint64_t>(releasable));

  const auto reservedBefore = opPool->reservedBytes();
  uint64_t reclaimedBytes{0};
  {
    memory::ScopedMemoryArbitrationContext arbCtx{opPool};
    memory::MemoryReclaimer::Stats stats;
    reclaimedBytes = opPool->reclaim(/*targetBytes=*/0, /*maxWaitMs=*/0, stats);
  }
  EXPECT_GE(reclaimedBytes, static_cast<uint64_t>(releasable));
  EXPECT_LT(opPool->reservedBytes(), reservedBefore);
  EXPECT_EQ(opPool->releasableReservation(), 0);

  Task::resume(task);
  resumed = true;
  while (cursor->moveNext()) {
  }
}

} // namespace facebook::velox::exec
