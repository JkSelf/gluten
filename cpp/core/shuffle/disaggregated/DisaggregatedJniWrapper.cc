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

#include <jni.h>

#include "jni/JniCommon.h"
#include "jni/JniError.h"
#include "shuffle/disaggregated/DisaggregatedShufflePartitionWriter.h"
#include "utils/StringUtil.h"

using namespace gluten;

#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT jlong JNICALL
Java_org_apache_gluten_vectorized_DisaggregatedShufflePartitionWriterJniWrapper_createPartitionWriter( // NOLINT
    JNIEnv* env,
    jobject wrapper,
    jint numPartitions,
    jstring codecJstr,
    jstring codecBackendJstr,
    jint compressionLevel,
    jint compressionBufferSize,
    jint compressionThreshold,
    jint mergeBufferSize,
    jdouble mergeThreshold,
    jint numSubDirs,
    jint shuffleFileBufferSize,
    jstring localDirsJstr,
    jobject mapOutputWriter) {
  JNI_METHOD_START

  const auto ctx = getRuntime(env, wrapper);

  auto localDirs = splitPaths(jStringToCString(env, localDirsJstr));

  auto partitionWriterOptions = std::make_shared<LocalPartitionWriterOptions>(
      shuffleFileBufferSize,
      compressionBufferSize,
      compressionThreshold,
      mergeBufferSize,
      mergeThreshold,
      numSubDirs,
      false);

  auto mapOutputWriterWrapper = std::make_shared<DisaggregatedShuffleMapOutputWriterWrapper>(env, mapOutputWriter);
  auto partitionWriter = std::make_shared<DisaggregatedShufflePartitionWriter>(
      numPartitions,
      createCompressionCodec(getCompressionType(env, codecJstr), getCodecBackend(env, codecBackendJstr), compressionLevel),
      ctx->memoryManager(),
      partitionWriterOptions,
      std::move(localDirs),
      mapOutputWriterWrapper);

  return ctx->saveObject(partitionWriter);
  JNI_METHOD_END(kInvalidObjectHandle)
}

#ifdef __cplusplus
}
#endif
