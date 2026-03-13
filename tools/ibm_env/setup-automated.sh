#!/usr/bin/env bash

set -ex

# Usage information
if [ -z "$1" ]; then
  echo "Usage: $0 <spark-version> [ibm_spark_base_path]"
  echo "  spark-version: 3.4, 3.5, or 4.0"
  echo "  ibm_spark_base_path: Optional base path to IBM Spark installation"
  echo ""
  echo "Environment variables:"
  echo "  IBM_SPARK_BASE: Base path to IBM Spark installation (default: /ibm_spark or ibm/spark{version})"
  echo "  GLUTEN_ROOT: Root directory of Gluten project (default: current directory)"
  echo "  ARTIFACTORY_URL: Artifactory deployment URL"
  echo "  ARTIFACTORY_REPO_ID: Artifactory repository ID (must match server id in settings.xml)"
  echo "  MAVEN_SETTINGS: Optional path to settings.xml (default: ~/.m2/settings.xml)"
  exit 1
fi

SPARK_VERSION_MAJOR="$1"
GLUTEN_ROOT="${GLUTEN_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Artifactory deployment config
ARTIFACTORY_URL="${ARTIFACTORY_URL:-https://eu.artifactory.swg-devops.com:443/artifactory/hyc-ibmae-gluten-maven-local}"
ARTIFACTORY_REPO_ID="${ARTIFACTORY_REPO_ID:-my-local-releases}"
MAVEN_SETTINGS_ARG=""
if [ -n "$MAVEN_SETTINGS" ] && [ -f "$MAVEN_SETTINGS" ]; then
  MAVEN_SETTINGS_ARG="-s $MAVEN_SETTINGS"
fi

# Determine IBM Spark base path
if [ -n "$2" ]; then
  IBM_SPARK_BASE="$2"
elif [ -n "$IBM_SPARK_BASE" ]; then
  IBM_SPARK_BASE="$IBM_SPARK_BASE"
elif [ -d "ibm/spark${SPARK_VERSION_MAJOR}" ]; then
  IBM_SPARK_BASE="ibm/spark${SPARK_VERSION_MAJOR}"
elif [ -d "/ibm_spark" ]; then
  IBM_SPARK_BASE="/ibm_spark"
else
  echo "Error: Could not find IBM Spark installation."
  exit 1
fi

# mapping spark major version with full version, scala and pom files
case "$SPARK_VERSION_MAJOR" in
  "3.4")
    SPARK_FULL="3.4.4"
    SCALA_VERSION="2.12"
    POM_FILE="${GLUTEN_ROOT}/tools/ibm_env/spark344-parent-pom.xml"
    ;;
  "3.5")
    SPARK_FULL="3.5.4"
    SCALA_VERSION="2.12"
    POM_FILE="${GLUTEN_ROOT}/tools/ibm_env/spark354-parent-pom.xml"
    ;;
  "4.0")
    SPARK_FULL="4.0.0"
    SCALA_VERSION="2.13"
    POM_FILE="${GLUTEN_ROOT}/tools/ibm_env/spark400-parent-pom.xml"
    ;;
  *)
    echo "Unsupported Spark version: $SPARK_VERSION_MAJOR"
    echo "Supported versions are: 3.4, 3.5, 4.0"
    exit 1
    ;;
esac

#dirs
JARS_DIR="${IBM_SPARK_BASE}/jars"
EXTERNAL_JARS_DIR="${IBM_SPARK_BASE}/external-jars"
WXD_DIR="${IBM_SPARK_BASE}/wxd"
if [ ! -d "$WXD_DIR" ]; then
  PARENT_DIR=$(dirname "$IBM_SPARK_BASE")
  if [ -d "${PARENT_DIR}/wxd" ]; then
    WXD_DIR="${PARENT_DIR}/wxd"
  fi
fi

#find the most recent jar matching pattern
find_jar() {
  local pattern="$1"
  local search_dir="$2"
  
  if [ ! -d "$search_dir" ]; then
    echo "Warning: Directory $search_dir does not exist" >&2
    return 1
  fi
  
  #find files matching pattern, sort by modification time (newest first), take first
  local found=$(find "$search_dir" -maxdepth 1 -type f -name "$pattern" 2>/dev/null | sort | head -1)
  if [ -n "$found" ] && [ -f "$found" ]; then
    echo "$found"
    return 0
  fi
  
  # Try glob expansion if pattern contains wildcards
  if [[ "$pattern" == *"*"* ]]; then
    found=$(find "$search_dir" -maxdepth 1 -type f -name "$pattern" 2>/dev/null | head -1)
    if [ -n "$found" ] && [ -f "$found" ]; then
      echo "$found"
      return 0
    fi
  fi
  
  return 1
}

#deploy jar/pom file to artifactory
deploy_jar_simple() {
  local jar_file="$1"
  local group_id="${2:-}"
  local artifact_id="${3:-}"
  local version="${4:-}"
  local packaging="${5:-jar}"
  
  if [ ! -f "$jar_file" ]; then
    echo "Warning: File not found: $jar_file" >&2
    return 1
  fi
  
  local mvn_cmd="mvn org.apache.maven.plugins:maven-deploy-plugin:3.1.1:deploy-file -Dfile=${jar_file} -Dpackaging=${packaging} -Durl=${ARTIFACTORY_URL} -DrepositoryId=${ARTIFACTORY_REPO_ID}"
  
  if [ -n "$group_id" ]; then
    mvn_cmd="${mvn_cmd} -DgroupId=${group_id}"
  fi
  
  if [ -n "$artifact_id" ]; then
    mvn_cmd="${mvn_cmd} -DartifactId=${artifact_id}"
  fi
  
  if [ -n "$version" ]; then
    mvn_cmd="${mvn_cmd} -Dversion=${version}"
  fi
  
  echo "Deploying: $(basename "$jar_file")"

  #treat 409 (already exists in artifactory) as non failure.
  local mvn_output
  local mvn_status

  if [ -n "$MAVEN_SETTINGS_ARG" ]; then
    set +e
    mvn_output=$(eval "$mvn_cmd $MAVEN_SETTINGS_ARG" 2>&1)
    mvn_status=$?
    set -e
  else
    set +e
    mvn_output=$(eval "$mvn_cmd" 2>&1)
    mvn_status=$?
    set -e
  fi

  echo "$mvn_output"

  if [ "$mvn_status" -ne 0 ]; then
    if echo "$mvn_output" | grep -Eq 'status code: 409|status: 409|[^0-9]409([^0-9]|$)'; then
      echo "SKIIPPING deploy (already exists in artifactory): $(basename "$jar_file")"
      return 0
    fi
    echo "ERROR deploying $(basename "$jar_file")"
    return "$mvn_status"
  fi
}

#deploy jar with pattern matching
deploy_jar_pattern() {
  local pattern="$1"
  local search_dir="$2"
  local group_id="${3:-}"
  local artifact_id="${4:-}"
  local version="${5:-}"
  local packaging="${6:-jar}"
  
  local jar_file=$(find_jar "$pattern" "$search_dir")
  
  if [ -z "$jar_file" ]; then
    echo "Warning: Could not find jar matching pattern: $pattern in $search_dir" >&2
    return 1
  fi
  
  deploy_jar_simple "$jar_file" "$group_id" "$artifact_id" "$version" "$packaging"
}

echo "=========================================="
echo "Deploying IBM Spark artifacts to Artifactory"
echo "Spark Version: $SPARK_FULL (Scala $SCALA_VERSION)"
echo "IBM Spark Base: $IBM_SPARK_BASE"
echo "Artifactory URL: $ARTIFACTORY_URL"
echo "=========================================="
echo ""

#verify directories exist
if [ ! -d "$JARS_DIR" ] && [ ! -d "$EXTERNAL_JARS_DIR" ]; then
  echo "Error: Neither $JARS_DIR nor $EXTERNAL_JARS_DIR exist"
  echo "Please verify the IBM_SPARK_BASE path: $IBM_SPARK_BASE"
  exit 1
fi

#install parent POM
if [ ! -f "$POM_FILE" ]; then
  echo "Error: POM file not found: $POM_FILE" >&2
  exit 1
fi

echo "[1/6] Deploying parent POM..."
deploy_jar_simple "$POM_FILE" "org.apache.spark" "spark-parent_${SCALA_VERSION}" "${SPARK_FULL}" "pom"

echo ""
echo "[2/6] Deploying common dependencies..."
# Common external jars - auto-detect versions
deploy_jar_pattern "jackson-databind-*.jar" "$EXTERNAL_JARS_DIR"
deploy_jar_pattern "guava-*-jre.jar" "$EXTERNAL_JARS_DIR"
deploy_jar_pattern "commons-io-*.jar" "$EXTERNAL_JARS_DIR"

echo ""
echo "[3/6] Deploying Hadoop client jars..."
deploy_jar_pattern "hadoop-client-api-*.jar" "$EXTERNAL_JARS_DIR"
deploy_jar_pattern "hadoop-client-runtime-*.jar" "$EXTERNAL_JARS_DIR"

echo ""
echo "[4/6] Deploying Spark core jars..."
# Spark core jars
deploy_jar_pattern "spark-core_${SCALA_VERSION}-${SPARK_FULL}.jar" "$JARS_DIR"
deploy_jar_pattern "spark-kvstore_${SCALA_VERSION}-${SPARK_FULL}.jar" "$JARS_DIR"
deploy_jar_pattern "spark-launcher_${SCALA_VERSION}-${SPARK_FULL}.jar" "$JARS_DIR"

# Spark SQL and Catalyst - need explicit coordinates
# Spark 4.0 has these in jars/, while 3.4 and 3.5 have them in external-jars/
if [ "$SPARK_VERSION_MAJOR" == "4.0" ]; then
  deploy_jar_pattern "spark-sql_${SCALA_VERSION}-*.jar" "$JARS_DIR" \
    "org.apache.spark" "spark-sql_${SCALA_VERSION}" "${SPARK_FULL}"
  
  deploy_jar_pattern "spark-catalyst_${SCALA_VERSION}-*.jar" "$JARS_DIR" \
    "org.apache.spark" "spark-catalyst_${SCALA_VERSION}" "${SPARK_FULL}"
else
  deploy_jar_pattern "spark-sql_${SCALA_VERSION}-*.jar" "$EXTERNAL_JARS_DIR" \
    "org.apache.spark" "spark-sql_${SCALA_VERSION}" "${SPARK_FULL}"
  
  deploy_jar_pattern "spark-catalyst_${SCALA_VERSION}-*.jar" "$EXTERNAL_JARS_DIR" \
    "org.apache.spark" "spark-catalyst_${SCALA_VERSION}" "${SPARK_FULL}"
fi

# Network jars
if [ "$SPARK_VERSION_MAJOR" == "3.4" ]; then
  # Spark 3.4 has special version for network-common in external-jars
  deploy_jar_pattern "spark-network-common_${SCALA_VERSION}-*.jar" "$EXTERNAL_JARS_DIR" \
    "org.apache.spark" "spark-network-common_${SCALA_VERSION}" "${SPARK_FULL}"
  deploy_jar_pattern "spark-network-shuffle_${SCALA_VERSION}-*.jar" "$EXTERNAL_JARS_DIR"
elif [ "$SPARK_VERSION_MAJOR" == "4.0" ]; then
  # Spark 4.0 has network jars in jars/ directory with exact version
  deploy_jar_pattern "spark-network-common_${SCALA_VERSION}-${SPARK_FULL}.jar" "$JARS_DIR"
  deploy_jar_pattern "spark-network-shuffle_${SCALA_VERSION}-${SPARK_FULL}.jar" "$JARS_DIR"
elif [ "$SPARK_VERSION_MAJOR" == "3.5" ]; then
  # Spark 3.5 uses external-jars with exact version
  deploy_jar_pattern "spark-network-common_${SCALA_VERSION}-${SPARK_FULL}.jar" "$EXTERNAL_JARS_DIR"
  deploy_jar_pattern "spark-network-shuffle_${SCALA_VERSION}-${SPARK_FULL}.jar" "$EXTERNAL_JARS_DIR"
else
  echo "Unsupported Spark version: $SPARK_VERSION_MAJOR"
  echo "Supported versions are: 3.4, 3.5, 4.0"
  exit 1
fi
echo ""
echo "[5/6] Deploying Thrift and other dependencies..."
# Thrift jars with explicit coordinates
deploy_jar_pattern "libthrift-*.jar" "$EXTERNAL_JARS_DIR" \
  "org.apache.thrift" "libthrift" "0.12.0"

deploy_jar_pattern "libfb303-*.jar" "$JARS_DIR" \
  "org.apache.thrift" "libfb303" "0.9.3"

# Disaggregated shuffle
deploy_jar_pattern "spark-disaggregated-shuffle_${SCALA_VERSION}-${SPARK_FULL}_*.jar" "$EXTERNAL_JARS_DIR" \
  "com.ibm" "spark-disaggregated-shuffle_${SCALA_VERSION}" "${SPARK_FULL}_1.0.2"

echo ""
echo "[6/6] Deploying Delta and Hudi connectors..."
# Delta jars - version varies by Spark version
case "$SPARK_VERSION_MAJOR" in
  "3.4")
    deploy_jar_pattern "delta-core_${SCALA_VERSION}-*.jar" "$WXD_DIR" \
      "io.delta" "delta-core_${SCALA_VERSION}" "2.4.1"
    deploy_jar_pattern "delta-storage-*.jar" "$WXD_DIR" \
      "io.delta" "delta-storage" "2.4.1"
    ;;
  "3.5")
    deploy_jar_pattern "delta-spark_${SCALA_VERSION}-*.jar" "$WXD_DIR" \
      "io.delta" "delta-spark_${SCALA_VERSION}" "3.3.2"
    deploy_jar_pattern "delta-storage-*.jar" "$WXD_DIR" \
      "io.delta" "delta-storage" "3.3.2"
    ;;
  "4.0")
    deploy_jar_pattern "delta-spark_${SCALA_VERSION}-*.jar" "$WXD_DIR" \
      "io.delta" "delta-spark_${SCALA_VERSION}" "4.0.0"
    deploy_jar_pattern "delta-storage-*.jar" "$WXD_DIR" \
      "io.delta" "delta-storage" "4.0.0"
    ;;
esac

# Hudi jars - version varies by Spark version
case "$SPARK_VERSION_MAJOR" in
  "3.4")
    deploy_jar_pattern "hudi-spark3.4-bundle_${SCALA_VERSION}-*.jar" "$WXD_DIR" \
      "org.apache.hudi" "hudi-spark3.4-bundle_${SCALA_VERSION}" "0.14.1"
    ;;
  "3.5")
    deploy_jar_pattern "hudi-spark3.5-bundle_${SCALA_VERSION}-*.jar" "$WXD_DIR" \
      "org.apache.hudi" "hudi-spark3.5-bundle_${SCALA_VERSION}" "0.15.0"
    ;;
  "4.0")
    deploy_jar_pattern "hudi-spark4.0-bundle_${SCALA_VERSION}-*.jar" "$WXD_DIR" \
      "org.apache.hudi" "hudi-spark4.0-bundle_${SCALA_VERSION}" "1.1.0"
    ;;
esac

# deploy_jar_pattern "iceberg-spark-runtime-${SPARK_VERSION_MAJOR}_${SCALA_VERSION}-*.jar" "$EXTERNAL_JARS_DIR" \
#   "org.apache.iceberg" "iceberg-spark-runtime-${SPARK_VERSION_MAJOR}_${SCALA_VERSION}" "1.10.0"

echo ""
echo "=========================================="
echo "Deploy completed successfully for Spark $SPARK_FULL"
echo "=========================================="