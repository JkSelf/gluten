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

#include <jni.h>
#include <memory>
#include <regex>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/auth/signer/AWSAuthSignerHelper.h>

#include "jni/JniCommon.h"

namespace gluten {

class JavaInputStreamBuf final : public std::streambuf {
 public:
  JavaInputStreamBuf(JavaVM* vm, JNIEnv* env, jobject inputStream) : vm_(vm), buffer_(BUFSIZ) {
    inputStream_ = env->NewGlobalRef(inputStream);
    auto inputStreamClass = env->GetObjectClass(inputStream);
    readMethod_ = getMethodIdOrError(env, inputStreamClass, "read", "([BII)I");
    env->DeleteLocalRef(inputStreamClass);
    setg(buffer_.data(), buffer_.data(), buffer_.data());
  }

  ~JavaInputStreamBuf() override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm_, &env);
    env->DeleteGlobalRef(inputStream_);
  }

 protected:
  int_type underflow() override {
    if (gptr() < egptr()) {
      return traits_type::to_int_type(*gptr());
    }

    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm_, &env);

    auto byteArray = env->NewByteArray(buffer_.size());
    const jint bytesRead =
        env->CallIntMethod(inputStream_, readMethod_, byteArray, 0, static_cast<jint>(buffer_.size()));
    checkException(env);

    if (bytesRead <= 0) {
      env->DeleteLocalRef(byteArray);
      return traits_type::eof();
    }

    env->GetByteArrayRegion(byteArray, 0, bytesRead, reinterpret_cast<jbyte*>(buffer_.data()));
    checkException(env);
    env->DeleteLocalRef(byteArray);

    setg(buffer_.data(), buffer_.data(), buffer_.data() + bytesRead);
    return traits_type::to_int_type(*gptr());
  }

 private:
  JavaVM* vm_;
  jobject inputStream_;
  jmethodID readMethod_;
  std::vector<char> buffer_;
};

class ContentBodyInputStreamAdapter final : public std::iostream {
 public:
  ContentBodyInputStreamAdapter(
      JavaVM* vm,
      JNIEnv* env,
      const std::shared_ptr<std::istream>& origin,
      jobject inputStream)
      : std::iostream(nullptr), origin_(origin), buffer_(vm, env, inputStream) {
    rdbuf(&buffer_);
  }

 private:
  std::shared_ptr<std::istream> origin_;
  JavaInputStreamBuf buffer_;
};

// Simulated HTTP request struct extracted from libcurl
struct CurlRequest {
  std::string uri;
  std::string method;
  std::vector<std::string> headerKeys;
  std::vector<std::string> headerValues;
  std::shared_ptr<std::istream> body{nullptr};
};

class DasS3CredentialsProvider final : public Aws::Auth::AWSCredentialsProvider {
 public:
  static void init(JavaVM* vm, JNIEnv* env) {
    vm_ = vm;

    class_ = createGlobalClassReferenceOrError(env, "Lorg/apache/spark/das/DasS3SignerWrapper;");
    constructor_ = getMethodIdOrError(env, class_, "<init>", "(Ljava/lang/String;)V");
    signMethod_ = getMethodIdOrError(
        env,
        class_,
        "sign",
        "(Ljava/lang/String;Ljava/lang/String;[Ljava/lang/String;[Ljava/lang/String;JLjava/lang/String;Ljava/lang/String;Ljava/lang/String;ZZI)Lorg/apache/spark/das/SignedRequest;");

    LOG(INFO) << "DASS3CredentialsProvider initialized successfully.";
  }

  static void tearDown(JNIEnv* env) {
    env->DeleteGlobalRef(class_);
  }

  DasS3CredentialsProvider(
      const Aws::String& awsAccessKeyId,
      const Aws::String& awsSecretAccessKey,
      const Aws::String& bucket)
      : bucket_(bucket),
        delegated_(std::make_shared<Aws::Auth::SimpleAWSCredentialsProvider>(awsAccessKeyId, awsSecretAccessKey)) {}

  ~DasS3CredentialsProvider() override {
    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm_, &env);
    if (signerInstance_ != nullptr) {
      env->DeleteGlobalRef(signerInstance_);
    }
  }

  Aws::Auth::AWSCredentials GetAWSCredentials() override {
    if (!delegated_) {
      throw std::runtime_error("Delegated credentials provider is not initialized.");
    }
    return delegated_->GetAWSCredentials();
  }

  bool CanSign() override {
    return true;
  }

  bool SignRequest(
      Aws::Http::HttpRequest& request,
      const char* region,
      const char* serviceName,
      bool signBody,
      std::chrono::milliseconds clockSkewOffset) override {
    initSigner();

    CurlRequest req;
    req.uri = request.GetURIString();
    req.method = Aws::Http::HttpMethodMapper::GetNameForHttpMethod(request.GetMethod());

    auto requestHeaders = request.GetHeaders();
    req.headerKeys.reserve(requestHeaders.size());
    req.headerValues.reserve(requestHeaders.size());
    for (auto& [key, value] : requestHeaders) {
      req.headerKeys.emplace_back(key.c_str());
      req.headerValues.emplace_back(value.c_str());
    }

    req.body = request.GetContentBody();

    JNIEnv* env = nullptr;
    attachCurrentThreadAsDaemonOrThrow(vm_, &env);

    jclass stringCls = env->FindClass("java/lang/String");
    jobjectArray jHeaderKeys = env->NewObjectArray(req.headerKeys.size(), stringCls, nullptr);
    jobjectArray jHeaderValues = env->NewObjectArray(req.headerValues.size(), stringCls, nullptr);

    for (jsize i = 0; i < static_cast<jsize>(req.headerKeys.size()); ++i) {
      jstring key = env->NewStringUTF(req.headerKeys[i].c_str());
      env->SetObjectArrayElement(jHeaderKeys, i, key);
      env->DeleteLocalRef(key);
    }

    for (jsize i = 0; i < static_cast<jsize>(req.headerValues.size()); ++i) {
      jstring value = env->NewStringUTF(req.headerValues[i].c_str());
      env->SetObjectArrayElement(jHeaderValues, i, value);
      env->DeleteLocalRef(value);
    }

    jstring jUrl = env->NewStringUTF(req.uri.c_str());
    jstring jMethod = env->NewStringUTF(req.method.c_str());
    jlong jBodyHandle = req.body != nullptr ? reinterpret_cast<jlong>(req.body.get()) : 0L;
    jstring jServiceName = env->NewStringUTF(serviceName);
    jstring jRegion = env->NewStringUTF(region);

    jstring jChecksum;
    jboolean unsignedTrailingPayload = false;

    if (request.GetRequestHash().second != nullptr && !request.GetRequestHash().first.empty()) {
      jChecksum = env->NewStringUTF(request.GetRequestHash().first.c_str());

      // From AwsAuthV4Signer.cpp:
      // If the request checksum, set the signer to use a unsigned trailing payload.
      // IBM COS doesn't support unsigned trailing payload: https://github.ibm.com/lakehouse/tracker/issues/32067
      if (request.GetContentBody() != nullptr && !isCos(request)) {
        unsignedTrailingPayload = true;

        // The payloadHash will be set to "STREAMING-UNSIGNED-PAYLOAD-TRAILER" and content body won't be read.
        jBodyHandle = 0L;
        Aws::String checksumHeaderValue = Aws::String("x-amz-checksum-") + request.GetRequestHash().first;
        request.DeleteHeader(checksumHeaderValue.c_str());
        request.SetHeaderValue(Aws::Http::AWS_TRAILER_HEADER, checksumHeaderValue);
        request.SetTransferEncoding(Aws::Http::CHUNKED_VALUE);
        request.HasContentEncoding()
            ? request.SetContentEncoding(Aws::String{Aws::Http::AWS_CHUNKED_VALUE} + "," + request.GetContentEncoding())
            : request.SetContentEncoding(Aws::Http::AWS_CHUNKED_VALUE);

        if (request.HasHeader(Aws::Http::CONTENT_LENGTH_HEADER)) {
          request.SetHeaderValue(
              Aws::Http::DECODED_CONTENT_LENGTH_HEADER, request.GetHeaderValue(Aws::Http::CONTENT_LENGTH_HEADER));
          request.DeleteHeader(Aws::Http::CONTENT_LENGTH_HEADER);
        }

        // Referenced from AWS SDK v2's AbstractAwsS3V4Signer.calculateContentHash():
        // x-amz-content-sha256 marked as STREAMING_UNSIGNED_PAYLOAD_TRAILER in interceptors
        // if Flexible checksum is set.
        static const char* STREAMING_UNSIGNED_PAYLOAD_TRAILER = "STREAMING-UNSIGNED-PAYLOAD-TRAILER";
        request.SetHeaderValue(Aws::Auth::AWSAuthHelper::X_AMZ_CONTENT_SHA256, STREAMING_UNSIGNED_PAYLOAD_TRAILER);
      }
    } else {
      jChecksum = env->NewStringUTF("");
    }
    jint clockSkew = std::chrono::duration_cast<std::chrono::seconds>(clockSkewOffset).count();

    jobject signedRequest = env->CallObjectMethod(
        signerInstance_,
        signMethod_,
        jUrl,
        jMethod,
        jHeaderKeys,
        jHeaderValues,
        jBodyHandle,
        jServiceName,
        jRegion,
        jChecksum,
        static_cast<jboolean>(signBody),
        unsignedTrailingPayload,
        clockSkew);
    checkException(env);

    jclass signedRequestClass = env->GetObjectClass(signedRequest);
    jmethodID headerKeysMethod = getMethodIdOrError(env, signedRequestClass, "headerKeys", "()[Ljava/lang/String;");
    jmethodID headerValuesMethod = getMethodIdOrError(env, signedRequestClass, "headerValues", "()[Ljava/lang/String;");

    auto signedHeaderKeysObj = static_cast<jobjectArray>(env->CallObjectMethod(signedRequest, headerKeysMethod));
    auto signedHeaderValuesObj = static_cast<jobjectArray>(env->CallObjectMethod(signedRequest, headerValuesMethod));
    checkException(env);

    const jsize signedHeaderCount = env->GetArrayLength(signedHeaderKeysObj);
    for (auto i = 0; i < signedHeaderCount; ++i) {
      auto keyObj = static_cast<jstring>(env->GetObjectArrayElement(signedHeaderKeysObj, i));
      auto valueObj = static_cast<jstring>(env->GetObjectArrayElement(signedHeaderValuesObj, i));

      const char* keyChars = env->GetStringUTFChars(keyObj, nullptr);
      const char* valueChars = env->GetStringUTFChars(valueObj, nullptr);

      request.SetHeaderValue(Aws::String(keyChars), Aws::String(valueChars));

      env->ReleaseStringUTFChars(keyObj, keyChars);
      env->ReleaseStringUTFChars(valueObj, valueChars);
    }

    // Ensure cleanup of JNI resources.
    env->DeleteLocalRef(jHeaderKeys);
    env->DeleteLocalRef(jHeaderValues);
    env->DeleteLocalRef(jUrl);
    env->DeleteLocalRef(jMethod);
    env->DeleteLocalRef(jServiceName);
    env->DeleteLocalRef(jRegion);
    env->DeleteLocalRef(jChecksum);

    return true;
  }

 private:
  Aws::String bucket_;
  std::shared_ptr<AWSCredentialsProvider> delegated_;

  std::once_flag signerInitFlag_;
  jobject signerInstance_{nullptr};

  void initSigner() {
    std::call_once(signerInitFlag_, [&]() {
      JNIEnv* env = nullptr;
      attachCurrentThreadAsDaemonOrThrow(vm_, &env);
      auto uri = "s3a://" + bucket_ + "/";
      auto instance = env->NewObject(class_, constructor_, env->NewStringUTF(uri.c_str()));

      checkException(env);
      GLUTEN_CHECK(instance, "Failed to create signer instance.");

      signerInstance_ = env->NewGlobalRef(instance);
      env->DeleteLocalRef(instance);
    });
  }

  static bool isCos(const Aws::Http::HttpRequest& request) {
    // https://cloud.ibm.com/docs/cloud-object-storage?topic=cloud-object-storage-endpoints
    // Pattern: s3.<region>.cloud-object-storage.(test.)?appdomain.cloud
    std::regex authorityPattern(R"(s3\.[a-z0-9-]+\.cloud-object-storage\.(test\.)?appdomain\.cloud)");
    const auto& authority = request.GetUri().GetAuthority();
    bool isCos = std::regex_match(authority, authorityPattern);
    LOG(INFO) << "Authority: " << authority << ", isCos: " << std::boolalpha << isCos;
    return isCos;
  }

  static inline JavaVM* vm_;
  static inline jclass class_;
  static inline jmethodID constructor_{nullptr};
  static inline jmethodID signMethod_{nullptr};
};
} // namespace gluten
