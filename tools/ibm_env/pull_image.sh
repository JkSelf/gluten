#!/usr/bin/env

IBM_HUMMINGBIRD_SPARK3_4="093347738777.dkr.ecr.us-east-1.amazonaws.com/devx/ibm-hummingbird-spark:3.4-latest"
IBM_HUMMINGBIRD_SPARK3_5="093347738777.dkr.ecr.us-east-1.amazonaws.com/devx/ibm-hummingbird-spark:3.5-latest"
IBM_HUMMINGBIRD_SPARK4_0="093347738777.dkr.ecr.us-east-1.amazonaws.com/devx/ibm-hummingbird-spark:4.0-latest"

docker pull ${IBM_HUMMINGBIRD_SPARK3_4} > /dev/null
docker pull ${IBM_HUMMINGBIRD_SPARK3_5} > /dev/null
docker pull ${IBM_HUMMINGBIRD_SPARK4_9} > /dev/null
docker create --name ibm-hummingbird-spark3_4 ${IBM_HUMMINGBIRD_SPARK3_4}
docker create --name ibm-hummingbird-spark3_5 ${IBM_HUMMINGBIRD_SPARK3_5}
docker create --name ibm-hummingbird-spark4_0 ${IBM_HUMMINGBIRD_SPARK4_0}

docker cp ibm-hummingbird-spark3_4:/opt/ibm/spark ibm/spark3.4
docker cp ibm-hummingbird-spark3_4:/opt/ibm/connectors/wxd ibm/spark3.4/wxd
docker cp ibm-hummingbird-spark3_5:/opt/ibm/spark ibm/spark3.5
docker cp ibm-hummingbird-spark3_5:/opt/ibm/connectors/wxd ibm/spark3.5/wxd
docker cp ibm-hummingbird-spark4_0:/opt/ibm/spark ibm/spark4.0
docker cp ibm-hummingbird-spark4_0:/opt/ibm/connectors/wxd ibm/spark4.0/wxd
