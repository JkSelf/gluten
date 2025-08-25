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

#include "jni/JniCommon.h"
#include "velox/connectors/hive/storage_adapters/abfs/DynamicSasTokenClientProvider.h"
#include "velox/connectors/hive/storage_adapters/abfs/RegisterAbfsFileSystem.h"

#include <jni.h>
#include <string>

namespace gluten {

class DasAbfsSasTokenProvider : public facebook::velox::filesystems::SasTokenProvider {
 public:
  static void init(JavaVM* vm, JNIEnv* env) {
    vm_ = vm;

    jclass threadClass = env->FindClass("java/lang/Thread");
    jmethodID currentThreadMethod = env->GetStaticMethodID(threadClass, "currentThread", "()Ljava/lang/Thread;");
    jobject currentThread = env->CallStaticObjectMethod(threadClass, currentThreadMethod);

    jmethodID getContextClassLoader =
        env->GetMethodID(threadClass, "getContextClassLoader", "()Ljava/lang/ClassLoader;");
    jobject classLoader = env->CallObjectMethod(currentThread, getContextClassLoader);

    // Load the Scala object class using the class loader
    jclass classLoaderClass = env->FindClass("java/lang/ClassLoader");
    jmethodID loadClassMethod =
        env->GetMethodID(classLoaderClass, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;");

    jstring className = env->NewStringUTF("org.apache.spark.das.DasAbfsSasTokenProvider");
    auto targetClass = static_cast<jclass>(env->CallObjectMethod(classLoader, loadClassMethod, className));
    class_ = static_cast<jclass>(env->NewGlobalRef(targetClass));
    env->DeleteLocalRef(targetClass);

    constructor_ = getMethodIdOrError(env, class_, "<init>", "()V");
    initalizeMethod_ = getMethodIdOrError(env, class_, "initialize", "(Ljava/lang/String;)V");
    getSasTokenMethod_ = getMethodIdOrError(
        env, class_, "getSasToken", "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;");
  }

  static void tearDown(JNIEnv* env) {
    env->DeleteGlobalRef(class_);
  }

  DasAbfsSasTokenProvider() = default;

  ~DasAbfsSasTokenProvider() override {
    if (generatorObj_ == nullptr) {
      return;
    }

    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm_, &env);
    env->DeleteGlobalRef(generatorObj_);
  }

  void initialize(const std::string& account) {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm_, &env);

    jobject instance = env->NewObject(class_, constructor_);
    generatorObj_ = env->NewGlobalRef(instance);
    env->DeleteLocalRef(instance);

    env->CallVoidMethod(generatorObj_, initalizeMethod_, env->NewStringUTF(account.c_str()));
  }

  std::string getSasToken(const std::string& fileSystem, const std::string& path, const std::string& operation)
      override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm_, &env);

    auto result = static_cast<jstring>(env->CallObjectMethod(
        generatorObj_,
        getSasTokenMethod_,
        env->NewStringUTF(fileSystem.c_str()),
        env->NewStringUTF(path.c_str()),
        env->NewStringUTF(operation.c_str())));
    checkException(env);
    return jStringToCString(env, result);
  }

  static void registerProvider(const facebook::velox::config::ConfigBase& conf) {
    // "spark.hadoop" prefix has been stripped.
    std::string_view kSasTokenProviderPrefix = "fs.azure.sas.token.provider.type.";
    std::string_view kAzureAccountSuffix = "dfs.core.windows.net";
    std::string_view kIbmlhcasSasTokenProvider = "org.apache.hadoop.fs.azurebfs.sas.IbmlhcasSASTokenProvider";
    for (const auto& [key, value] : conf.rawConfigsCopy()) {
      if (key.find(kSasTokenProviderPrefix) == 0) {
        std::string_view skey = key;
        auto remaining = skey.substr(kSasTokenProviderPrefix.size());
        auto dot = remaining.find(".");
        auto accountName = std::string(remaining.substr(0, dot));
        GLUTEN_CHECK(!accountName.empty(), "Account name cannot be empty in sas token provider config");
        auto suffix = remaining.substr(dot + 1);
        LOG(INFO) << "Found SAS token provider for account: " << accountName << ", suffix: " << suffix
                  << ", value: " << value;
        if (suffix == kAzureAccountSuffix && value == kIbmlhcasSasTokenProvider) {
          LOG(INFO) << "Registering DAS SAS key generator for account: " << accountName;
          facebook::velox::filesystems::registerAzureClientProviderFactory(accountName, [](const std::string& account) {
            auto sasTokenProvider = std::make_shared<DasAbfsSasTokenProvider>();
            sasTokenProvider->initialize(account);
            return std::make_unique<facebook::velox::filesystems::DynamicSasTokenClientProvider>(sasTokenProvider);
          });
        }
      }
    }
  }

 private:
  jobject generatorObj_{nullptr};

  static inline JavaVM* vm_;
  static inline jclass class_;
  static inline jmethodID constructor_{nullptr};
  static inline jmethodID initalizeMethod_{nullptr};
  static inline jmethodID getSasTokenMethod_{nullptr};
};

} // namespace gluten
