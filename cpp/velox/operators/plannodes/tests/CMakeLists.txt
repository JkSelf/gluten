# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Link the operator test against Velox's static libraries instead of Gluten's
# libvelox shared library. Loading both would duplicate Velox's global
# registries and make aggregate/operator registration depend on link order.
add_executable(
  velox_grouping_set_aggregation_test
  MultiGroupingSetAggregationTest.cc
  ../GroupingSetAggregationNode.cc
  ../GroupingSetLattice.cc
  ../MultiGroupingSetAggregation.cc)

get_target_property(VELOX_INCLUDE_DIRECTORIES velox
                    INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(VELOX_LINK_LIBRARIES velox LINK_LIBRARIES)

target_include_directories(
  velox_grouping_set_aggregation_test
  PRIVATE ${CMAKE_SOURCE_DIR}/velox ${CMAKE_SOURCE_DIR}/src
          ${VELOX_INCLUDE_DIRECTORIES})

if(GLUTEN_VELOX_ENABLE_PARQUET)
  target_compile_definitions(velox_grouping_set_aggregation_test
                             PRIVATE VELOX_ENABLE_PARQUET)
endif()

target_link_libraries(
  velox_grouping_set_aggregation_test
  PRIVATE ${VELOX_LINK_LIBRARIES}
          ${GLUTEN_ROARING_LINK_LIBRARY}
          GTest::gtest
          GTest::gtest_main)

gtest_discover_tests(velox_grouping_set_aggregation_test DISCOVERY_MODE PRE_TEST
                     DISCOVERY_TIMEOUT 60)
