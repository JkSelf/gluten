#!/usr/bin/env

set -ex

if [ -z "$1" ]; then
  echo "Usage: $0 <spark-version>"
  exit 1
fi


#spark3.4.4
if [ "$1" == "3.4" ]; then
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=./tools/ibm_env/spark344-parent-pom.xml -DgroupId=org.apache.spark -DartifactId=spark-parent_2.12 -Dversion=3.4.4 -Dpackaging=pom
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/jackson-databind-2.16.2.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/guava-32.1.1-jre.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/commons-io-2.18.0.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/external-jars/spark-network-common_2.12-3.4.5-ibm-*.jar`  -DgroupId=org.apache.spark -DartifactId=spark-network-common_2.12 -Dversion=3.4.4 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/external-jars/hadoop-client-api-3.4.0-ibm-*.jar`
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/external-jars/hadoop-client-runtime-3.4.0-ibm-*.jar`
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/external-jars/spark-sql_2.12-3.4.4-ibm-*.jar` -DgroupId=org.apache.spark -DartifactId=spark-sql_2.12 -Dversion=3.4.4 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/external-jars/spark-catalyst_2.12-3.4.4-ibm-*.jar` -DgroupId=org.apache.spark -DartifactId=spark-catalyst_2.12 -Dversion=3.4.4 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/jars/spark-core_2.12-3.4.4.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/jars/spark-kvstore_2.12-3.4.4.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/spark-network-shuffle_2.12-3.4.1.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/jars/spark-launcher_2.12-3.4.4.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/libthrift-0.18.0.jar -DgroupId=org.apache.thrift -DartifactId=libthrift -Dversion=0.12.0 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/jars/libfb303-0.9.3.jar -DgroupId=org.apache.thrift -DartifactId=libfb303 -Dversion=0.9.3 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/spark-disaggregated-shuffle_2.12-3.4.4_1.0.2.jar -DgroupId=com.ibm -DartifactId=spark-disaggregated-shuffle_2.12 -Dversion=3.4.4_1.0.2 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/wxd/delta-core_2.12-2.4.1-SNAPSHOT-ibm-*.jar` -DgroupId=io.delta -DartifactId=delta-core_2.12 -Dversion=2.4.1 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/wxd/delta-storage-2.4.1-SNAPSHOT-ibm-*.jar` -DgroupId=io.delta -DartifactId=delta-storage -Dversion=2.4.1 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/wxd/hudi-spark3.4-bundle_2.12-0.14.1-ibm-*.jar` -DgroupId=org.apache.hudi -DartifactId=hudi-spark3.4-bundle_2.12 -Dversion=0.14.1 -Dpackaging=jar
    # mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/iceberg-spark-runtime-3.4_2.12-1.10.0-ibm-1-20251121.jar -DgroupId=org.apache.iceberg -DartifactId=iceberg-spark-runtime-3.4_2.12 -Dversion=1.10.0 -Dpackaging=jar


elif [ "$1" == "3.5" ]; then

    #spark 3.5.4
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=./tools/ibm_env/spark354-parent-pom.xml -DgroupId=org.apache.spark -DartifactId=spark-parent_2.12 -Dversion=3.5.4 -Dpackaging=pom
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/jackson-databind-2.16.2.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/guava-32.1.1-jre.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/commons-io-2.18.0.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/external-jars/hadoop-client-api-3.4.0-ibm-*.jar`
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/external-jars/hadoop-client-runtime-3.4.0-ibm-*.jar`
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/jars/spark-core_2.12-3.5.4.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/external-jars/spark-sql_2.12-3.5.4-ibm-*.jar` -DgroupId=org.apache.spark -DartifactId=spark-sql_2.12 -Dversion=3.5.4 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/external-jars/spark-catalyst_2.12-3.5.4-ibm-*.jar` -DgroupId=org.apache.spark -DartifactId=spark-catalyst_2.12 -Dversion=3.5.4 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/jars/spark-kvstore_2.12-3.5.4.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/spark-network-common_2.12-3.5.4.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/spark-network-shuffle_2.12-3.5.4.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/jars/spark-launcher_2.12-3.5.4.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/libthrift-0.18.0.jar -DgroupId=org.apache.thrift -DartifactId=libthrift -Dversion=0.12.0 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/jars/libfb303-0.9.3.jar -DgroupId=org.apache.thrift -DartifactId=libfb303 -Dversion=0.9.3 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/spark-disaggregated-shuffle_2.12-3.5.4_1.0.2.jar -DgroupId=com.ibm -DartifactId=spark-disaggregated-shuffle_2.12 -Dversion=3.5.4_1.0.2 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/wxd/delta-spark_2.12-3.3.2-SNAPSHOT-ibm-*.jar` -DgroupId=io.delta -DartifactId=delta-spark_2.12 -Dversion=3.3.2 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/wxd/delta-storage-3.3.2-SNAPSHOT-ibm-*.jar` -DgroupId=io.delta -DartifactId=delta-storage -Dversion=3.3.2 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/wxd/hudi-spark3.5-bundle_2.12-0.15.0-ibm-*.jar` -DgroupId=org.apache.hudi -DartifactId=hudi-spark3.5-bundle_2.12 -Dversion=0.15.0 -Dpackaging=jar
    # mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/iceberg-spark-runtime-3.5_2.12-1.10.0-ibm-1-20251121.jar -DgroupId=org.apache.iceberg -DartifactId=iceberg-spark-runtime-3.5_2.12 -Dversion=1.10.0 -Dpackaging=jar


elif [ "$1" == "4.0" ]; then

    #spark 4.0.0
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=./tools/ibm_env/spark400-parent-pom.xml -DgroupId=org.apache.spark -DartifactId=spark-parent_2.12 -Dversion=4.0.0 -Dpackaging=pom
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/jackson-databind-2.16.2.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/guava-33.2.1-jre.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/commons-io-2.18.0.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/external-jars/hadoop-client-api-3.4.0-ibm-*.jar`
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/external-jars/hadoop-client-runtime-3.4.0-ibm-*.jar`
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/jars/spark-core_2.13-4.0.0.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/jars/spark-sql_2.13-4.0.0.jar -DgroupId=org.apache.spark -DartifactId=spark-sql_2.13 -Dversion=4.0.0 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/jars/spark-catalyst_2.13-4.0.0.jar -DgroupId=org.apache.spark -DartifactId=spark-catalyst_2.13 -Dversion=4.0.0 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/jars/spark-kvstore_2.13-4.0.0.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/jars/spark-network-common_2.13-4.0.0.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/jars/spark-network-shuffle_2.13-4.0.0.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/jars/spark-launcher_2.13-4.0.0.jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/libthrift-0.18.0.jar -DgroupId=org.apache.thrift -DartifactId=libthrift -Dversion=0.12.0 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/jars/libfb303-0.9.3.jar -DgroupId=org.apache.thrift -DartifactId=libfb303 -Dversion=0.9.3 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/spark-disaggregated-shuffle_2.13-4.0.0_1.0.2.jar -DgroupId=com.ibm -DartifactId=spark-disaggregated-shuffle_2.13 -Dversion=4.0.0_1.0.2 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/wxd/delta-spark_2.13-4.0.0.jar -DgroupId=io.delta -DartifactId=delta-spark_2.13 -Dversion=4.0.0 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/wxd/delta-storage-4.0.0.jar -DgroupId=io.delta -DartifactId=delta-storage -Dversion=4.0.0 -Dpackaging=jar
    mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=`ls /ibm_spark/wxd/hudi-spark4.0-bundle_2.13-1.1.0-ibm-*.jar` -DgroupId=org.apache.hudi -DartifactId=hudi-spark4.0-bundle_2.13 -Dversion=1.1.0 -Dpackaging=jar
    # mvn org.apache.maven.plugins:maven-install-plugin:3.1.4:install-file -Dfile=/ibm_spark/external-jars/iceberg-spark-runtime-4.0_2.13-1.10.0-ibm-1-20251121.jar -DgroupId=org.apache.iceberg -DartifactId=iceberg-spark-runtime-4.0_2.13 -Dversion=1.10.0 -Dpackaging=jar
else
  echo "Unsupported Spark version: $1"
  echo "Supported versions are: 3.4, 3.5, 4.0"
  exit 1
fi

