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
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/storage_adapters/gcs/GcsOAuthCredentialsProvider.h"
#include "velox/connectors/hive/storage_adapters/gcs/RegisterGcsFileSystem.h"

#include <jni.h>
#include <string>

#include <google/cloud/internal/oauth2_cached_credentials.h>
#include <google/cloud/internal/oauth2_credentials.h>
#include <google/cloud/storage/oauth2/credentials.h>

namespace gluten {

namespace gcs = ::google::cloud::storage;
namespace gc = ::google::cloud;

class DasAwareAccessTokenProviderWrapper final : public gc::oauth2_internal::Credentials {
 public:
  static void init(JavaVM* vm, JNIEnv*) {
    vm_ = vm;

    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm, &env);

    longClass_ = createGlobalClassReferenceOrError(env, "java/lang/Long");
    longValueMethod_ = env->GetMethodID(longClass_, "longValue", "()J");

    jclass threadClass = env->FindClass("java/lang/Thread");
    jmethodID currentThreadMethod = env->GetStaticMethodID(threadClass, "currentThread", "()Ljava/lang/Thread;");
    jobject currentThread = env->CallStaticObjectMethod(threadClass, currentThreadMethod);

    jmethodID getContextClassLoader =
        env->GetMethodID(threadClass, "getContextClassLoader", "()Ljava/lang/ClassLoader;");
    jobject classLoader = env->CallObjectMethod(currentThread, getContextClassLoader);

    jclass classLoaderClass = env->FindClass("java/lang/ClassLoader");
    jmethodID loadClassMethod =
        env->GetMethodID(classLoaderClass, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;");

    // Load DasAwareAccessTokenProvider using the class loader.
    jstring providerClassName = env->NewStringUTF("org.apache.spark.das.DasGcsAccessTokenProvider");
    auto providerClass = static_cast<jclass>(env->CallObjectMethod(classLoader, loadClassMethod, providerClassName));
    providerClass_ = static_cast<jclass>(env->NewGlobalRef(providerClass));
    env->DeleteLocalRef(providerClass);

    // Load AccessToken class using the class loader.
    jstring accessTokenClassName = env->NewStringUTF("com.google.cloud.hadoop.util.AccessTokenProvider$AccessToken");
    auto accessTokenClass =
        static_cast<jclass>(env->CallObjectMethod(classLoader, loadClassMethod, accessTokenClassName));
    if (accessTokenClass == nullptr) {
      throw std::runtime_error("Failed to load AccessToken class");
    }
    getTokenMethod_ = getMethodIdOrError(env, accessTokenClass, "getToken", "()Ljava/lang/String;");
    getExpirationTimeMilliSecondsMethod_ =
        getMethodIdOrError(env, accessTokenClass, "getExpirationTimeMilliSeconds", "()Ljava/lang/Long;");

    constructor_ = getMethodIdOrError(env, providerClass_, "<init>", "(Ljava/lang/String;)V");
    initalizeMethod_ = getMethodIdOrError(env, providerClass_, "initialize", "()V");
    getAccessTokenMethod_ = getMethodIdOrError(
        env, providerClass_, "getAccessToken", "()Lcom/google/cloud/hadoop/util/AccessTokenProvider$AccessToken;");
  }

  static void tearDown(JNIEnv* env) {
    env->DeleteGlobalRef(longClass_);
    env->DeleteGlobalRef(providerClass_);
  }

  explicit DasAwareAccessTokenProviderWrapper(const std::string& bucket) {
    initialize(bucket);
  }

  // `initialize` should be called when creating the FileSystem.
  void initialize(const std::string& bucket) {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm_, &env);

    jobject instance = env->NewObject(providerClass_, constructor_, env->NewStringUTF(bucket.c_str()));
    providerObj_ = env->NewGlobalRef(instance);

    env->CallVoidMethod(providerObj_, initalizeMethod_);
    checkException(env);
  }

  google::cloud::StatusOr<google::cloud::AccessToken> GetToken(
      std::chrono::system_clock::time_point tp /* unused */) override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm_, &env);

    GLUTEN_CHECK(providerObj_ != nullptr, "DasAwareAccessTokenProviderWrapper is not initialized");

    jobject accessToken = env->CallObjectMethod(providerObj_, getAccessTokenMethod_);
    checkException(env);

    jstring token = static_cast<jstring>(env->CallObjectMethod(accessToken, getTokenMethod_));
    checkException(env);

    jobject expirationTimeMillisLong = env->CallObjectMethod(accessToken, getExpirationTimeMilliSecondsMethod_);
    checkException(env);

    jlong expirationTimeMillis = env->CallLongMethod(expirationTimeMillisLong, longValueMethod_);

    return google::cloud::AccessToken{
        jStringToCString(env, token),
        std::chrono::system_clock::time_point(std::chrono::milliseconds(expirationTimeMillis))};
  }

 private:
  jobject providerObj_{nullptr};

  static inline JavaVM* vm_;
  static inline jclass longClass_;
  static inline jmethodID longValueMethod_{nullptr};
  static inline jclass providerClass_;
  static inline jmethodID constructor_{nullptr};
  static inline jmethodID initalizeMethod_{nullptr};
  static inline jmethodID getAccessTokenMethod_{nullptr};

  static inline jmethodID getTokenMethod_{nullptr};
  static inline jmethodID getExpirationTimeMilliSecondsMethod_{nullptr};
};

class DasGcsAccessTokenCredentials final : public gcs::oauth2::Credentials {
 public:
  DasGcsAccessTokenCredentials(const std::string& bucket)
      : impl_(std::make_shared<DasAwareAccessTokenProviderWrapper>(bucket)),
        cached_(std::make_shared<gc::oauth2_internal::CachedCredentials>(impl_)) {}

  google::cloud::StatusOr<std::string> AuthorizationHeader() override {
    return gc::oauth2_internal::AuthorizationHeaderJoined(*cached_);
  }

 private:
  std::shared_ptr<DasAwareAccessTokenProviderWrapper> impl_;
  std::shared_ptr<gc::oauth2_internal::CachedCredentials> cached_;
};

class DasGcsAccessTokenCredentialsProvider : public facebook::velox::filesystems::GcsOAuthCredentialsProvider {
 public:
  static constexpr const char* kDasGcsAccessTokenProviderName =
      "org.apache.hadoop.fs.gcs.wxd.DasAwareAccessTokenProvider";

  explicit DasGcsAccessTokenCredentialsProvider(
      const std::shared_ptr<facebook::velox::connector::hive::HiveConfig>& hiveConfig)
      : hiveConfig_(hiveConfig) {}

  std::shared_ptr<gcs::oauth2::Credentials> getCredentials(const std::string& bucket) override {
    return std::make_unique<DasGcsAccessTokenCredentials>(bucket);
  }

  static void registerFactory() {
    facebook::velox::filesystems::registerGcsOAuthCredentialsProvider(
        kDasGcsAccessTokenProviderName,
        [](const std::shared_ptr<facebook::velox::connector::hive::HiveConfig>& hiveConfig) {
          return std::make_shared<DasGcsAccessTokenCredentialsProvider>(hiveConfig);
        });
  }

 private:
  std::shared_ptr<facebook::velox::connector::hive::HiveConfig> hiveConfig_;
};
} // namespace gluten
