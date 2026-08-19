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

#pragma once

#include <cstdint>

#include "velox/core/PlanNode.h"

namespace substrait {
class AggregateRel;
class ExpandRel;
} // namespace substrait

namespace gluten {

class SubstraitToVeloxPlanConverter;

inline constexpr char kFusedGroupingSetAggregateMaxGroupingSets[] =
    "spark.gluten.sql.columnar.backend.velox.fusedGroupingSetAggregate.maxGroupingSets";
inline constexpr int32_t kFusedGroupingSetAggregateMaxGroupingSetsDefault = 16;

/// Converts a planner-tagged AggregateRel -> ExpandRel pair into a raw-input
/// grouping-set aggregation. A rejected hint is restored to both ordinary
/// operators using the owning Substrait converter.
class RawGroupingSetPlanConverter {
 public:
  explicit RawGroupingSetPlanConverter(SubstraitToVeloxPlanConverter& owner);

  /// Returns nullptr when aggRel is not a tagged AggregateRel -> ExpandRel
  /// pair. Tagged pairs return either the fused plan or the ordinary fallback.
  facebook::velox::core::PlanNodePtr tryConvert(const ::substrait::AggregateRel& aggRel);

 private:
  void validate(const ::substrait::AggregateRel& aggRel, const ::substrait::ExpandRel& expandRel) const;

  facebook::velox::core::PlanNodePtr convert(
      const ::substrait::AggregateRel& aggRel,
      const ::substrait::ExpandRel& expandRel,
      const facebook::velox::core::PlanNodePtr& expandInputNode);

  SubstraitToVeloxPlanConverter& owner_;
};

} // namespace gluten
