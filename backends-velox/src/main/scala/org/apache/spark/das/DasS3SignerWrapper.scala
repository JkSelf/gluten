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
package org.apache.spark.das

import org.apache.gluten.exception.GlutenException

import org.apache.spark.SparkEnv
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.internal.Logging

import org.apache.commons.lang3.ClassUtils
import org.apache.hadoop.conf.Configuration
import software.amazon.awssdk.auth.credentials.AwsCredentials
import software.amazon.awssdk.auth.signer.internal.AbstractAwsS3V4Signer
import software.amazon.awssdk.auth.signer.params.{AwsS3V4SignerParams, SignerChecksumParams}
import software.amazon.awssdk.core.checksums.Algorithm
import software.amazon.awssdk.http.{ContentStreamProvider, SdkHttpFullRequest, SdkHttpMethod}
import software.amazon.awssdk.regions.Region

import java.io.InputStream
import java.net.URI

import scala.collection.JavaConverters._

class DasS3SignerWrapper(uri: String) extends Logging {
  private lazy val hadoopConf = SparkHadoopUtil.get.newConfiguration(SparkEnv.get.conf)
  private lazy val awsCredentials = {
    val credentialClazz =
      ClassUtils.getClass("com.ibm.iae.s3.credentialprovider.WatsonxCredentialsProvider")

    credentialClazz
      .getConstructor(classOf[URI], classOf[Configuration])
      .newInstance(new URI(uri), hadoopConf)
      .asInstanceOf[software.amazon.awssdk.auth.credentials.AwsCredentialsProvider]
      .resolveCredentials()
  }

  private lazy val signer = {
    val signerClazz =
      ClassUtils.getClass("com.ibm.iae.s3.credentialprovider.WatsonxAWSV4Signer")
    signerClazz.getConstructor().newInstance().asInstanceOf[AbstractAwsS3V4Signer]
  }

  // scalastyle:off argcount
  def sign(
      url: String,
      method: String,
      headerKeys: Array[String],
      headerValues: Array[String],
      nativeStreamHandle: Long,
      signingName: String,
      signingRegion: String,
      checksum: String,
      signBody: Boolean,
      unsignedTrailingPayload: Boolean,
      clockSkew: Int): SignedRequest = {
    // FIXME: Do we need to check spark.gluten.velox.s3PayloadSigningPolicy=Never here
    // to disable payload signing?
    val payloadSigningEnabled = signBody && !unsignedTrailingPayload &&
      SparkEnv.get.conf.get("spark.gluten.velox.s3PayloadSigningPolicy", "Never") != "Never"

    val signerParams =
      constructSignerParams(
        payloadSigningEnabled,
        awsCredentials,
        signingName,
        signingRegion,
        checksum,
        clockSkew)

    val builder = SdkHttpFullRequest
      .builder()
      .uri(URI.create(url))
      .method(SdkHttpMethod.valueOf(method))

    headerKeys.zip(headerValues).foreach {
      case (k, v) =>
        builder.putHeader(k, v)
    }

    if (nativeStreamHandle != 0L) {
      // If not use unsigned trailing payload, and the request has body,
      // provide the content body for computing payload hash.
      builder.contentStreamProvider(new ContentStreamProvider {
        override def newStream(): InputStream = {
          new NativeIStreamInputStream(nativeStreamHandle)
        }
      })
    }

    val sdkRequest = builder.build()

    logInfo(
      s"Signing request with method: $method, url: $url" +
        s", payload signing enabled: $payloadSigningEnabled, checksum: [$checksum]" +
        ", headers:\n" +
        headerKeys.zip(headerValues).map(e => s"${e._1}: ${e._2}").mkString("\n"))

    val signedRequest = signer.sign(sdkRequest, signerParams)

    val headerEntries = signedRequest.headers().asScala.toSeq.flatMap {
      case (key, values) => values.asScala.map(value => key -> value)
    }

    logInfo("Signed request headers: " + headerEntries.map(e => s"${e._1}: ${e._2}").mkString(", "))

    SignedRequest(headerEntries.map(_._1).toArray, headerEntries.map(_._2).toArray)
  }
  // scalastyle:on argcount

  private class NativeIStreamInputStream(nativeHandle: Long) extends InputStream {
    require(nativeHandle != 0L, "nativeHandle must be non-zero")

    private var closed = false

    override def read(): Int = {
      val oneByte = new Array[Byte](1)
      val readBytes = DasS3SignerWrapper.this.read(nativeHandle, oneByte, 0, 1)
      if (readBytes <= 0) {
        -1
      } else {
        oneByte(0) & 0xff
      }
    }

    override def read(b: Array[Byte], off: Int, len: Int): Int = {
      if (b == null) {
        throw new NullPointerException("target buffer is null")
      }
      if (off < 0 || len < 0 || len > b.length - off) {
        throw new IndexOutOfBoundsException(
          s"Invalid offset/length: off=$off, len=$len, capacity=${b.length}")
      }
      if (len == 0) {
        return 0
      }
      if (closed) {
        throw new GlutenException("Native input stream already closed")
      }
      val bytes = DasS3SignerWrapper.this.read(nativeHandle, b, off, len)
      if (bytes == -1) {
        close()
      }
      bytes
    }

    override def close(): Unit = {
      if (!closed) {
        DasS3SignerWrapper.this.close(nativeHandle)
        closed = true
      }
    }
  }

  @native private[das] def read(nativeHandle: Long, b: Array[Byte], off: Int, len: Int): Int
  @native private[das] def close(nativeHandle: Long): Unit

  private def constructSignerParams(
      payloadSigningEnabled: Boolean,
      awsCredentials: AwsCredentials,
      signingName: String,
      signingRegion: String,
      checksum: String,
      clockSkew: Int): AwsS3V4SignerParams = {

    val builder = AwsS3V4SignerParams
      .builder()
      .enablePayloadSigning(payloadSigningEnabled)
      .enableChunkedEncoding(false)
      .awsCredentials(awsCredentials)
      .signingName(signingName)
      .signingRegion(Region.of(signingRegion))
      .timeOffset(clockSkew)

    if (checksum.nonEmpty) {
      builder
        .checksumParams(
          SignerChecksumParams
            .builder()
            .algorithm(Algorithm.fromValue(checksum))
            .checksumHeaderName("x-amz-checksum-" + checksum.toLowerCase())
            .isStreamingRequest(false) // isStreaminRequest is unused in AWS SDK v2.
            .build()
        )
    }

    builder.build()
  }
}
