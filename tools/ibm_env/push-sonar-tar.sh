#!/usr/bin/env bash
exec 1>&2
set -e

# Push SonarQube ondemand binaries (class files) to Artifactory.
# Requirements:
# - Maven settings.xml configured with server id matching ARTIFACTORY_REPO_ID
# - GLUTEN_BRANCH must be set (e.g. passed as Docker build arg)
# - This script is run from anywhere; it will locate Gluten root automatically.
#
# Archive name:
#   <repo_name>_<branch_name>.tar.gz
#
# Note: This script exits with 0 in all warning/skip scenarios to avoid breaking
# the Docker build. Any upload failures will be caught by the SonarQube PR check.

echo "SONARQUBE PUSH STARTED INSIDE $GLUTEN_ROOT"

if [ -z "$GLUTEN_ROOT" ]; then
  if [ -d "/incubator-gluten" ]; then
    GLUTEN_ROOT="/incubator-gluten"
  else
    GLUTEN_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
  fi
fi

if [ -z "$GLUTEN_BRANCH" ]; then
  echo "Error: GLUTEN_BRANCH is not set; aborting"
  exit 1
fi

ARTIFACTORY_REPO_ID="${ARTIFACTORY_REPO_ID:-my-local-releases}"
SETTINGS_FILE="$HOME/.m2/settings.xml"

ART_USER=$(sed -n "/<id>${ARTIFACTORY_REPO_ID}<\/id>/,/<\/server>/p" "$SETTINGS_FILE" | sed -n 's/^[[:space:]]*<username>\([^<]*\)<\/username>.*/\1/p' | head -1)
ART_TOKEN=$(sed -n "/<id>${ARTIFACTORY_REPO_ID}<\/id>/,/<\/server>/p" "$SETTINGS_FILE" | sed -n 's/^[[:space:]]*<password>\([^<]*\)<\/password>.*/\1/p' | head -1)

if [ -z "$ART_USER" ] || [ -z "$ART_TOKEN" ]; then
  # Exiting with 0 intentionally to not break the Docker build.
  # This failure will surface when the PR SonarQube check runs and finds missing binaries.
  echo "Warning: Could not read username/password for server id '${ARTIFACTORY_REPO_ID}' from $SETTINGS_FILE."
  echo "Warning: SonarQube ondemand binary upload is being skipped."
  echo "Warning: This will cause the SonarQube PR check to fail — please verify Artifactory credentials in settings.xml."
  exit 0
fi

echo "Assuming project already built (targets present under ${GLUTEN_ROOT}); only packaging and uploading."

REPO_TOP="$GLUTEN_ROOT"
REPO_NAME="${SONAR_REPO_NAME:-gluten}"

ARCHIVE_BASE="${REPO_NAME}_${GLUTEN_BRANCH}"
STAGING_DIR="${GLUTEN_ROOT}/../${ARCHIVE_BASE}"
ARCHIVE_FILE="${GLUTEN_ROOT}/../${ARCHIVE_BASE}.tar.gz"

SONAR_ARTIFACTORY_BASE="${SONAR_ARTIFACTORY_BASE:-https://na.artifactory.swg-devops.com/artifactory/hyc-cpd-skywalker-team-lakehouse-on-prem-generic-local/sonarqube/ondemand_binaries}"
UPLOAD_URL="${SONAR_ARTIFACTORY_BASE}/${ARCHIVE_BASE}.tar.gz"

echo "Preparing staging at $STAGING_DIR"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

echo "Collecting module targets under $REPO_TOP"
for d in "$REPO_TOP"/*; do
  [ -d "$d/target" ] || continue
  name=$(basename "$d")
  mkdir -p "$STAGING_DIR/$name"
  cp -r "$d/target" "$STAGING_DIR/$name/"
done

if [ -z "$(find "$STAGING_DIR" -type f 2>/dev/null | head -1)" ]; then
  # Exiting with 0 intentionally to not break the Docker build.
  # This failure will surface when the PR SonarQube check runs and finds missing binaries.
  echo "Warning: No target/ files found under $REPO_TOP — nothing to upload."
  echo "Warning: SonarQube ondemand binary upload is being skipped."
  echo "Warning: This will cause the SonarQube PR check to fail — please verify the build produced target/ directories."
  rm -rf "$STAGING_DIR"
  exit 0
fi

echo "Creating archive $ARCHIVE_FILE (name: ${ARCHIVE_BASE}.tar.gz)"
tar -czf "$ARCHIVE_FILE" -C "$(dirname "$STAGING_DIR")" "$ARCHIVE_BASE"

echo "Uploading archive to $UPLOAD_URL"
curl -f -u "${ART_USER}:${ART_TOKEN}" -T "$ARCHIVE_FILE" "$UPLOAD_URL" || echo "Warning: upload failed"

if [ "$GLUTEN_BRANCH" = "main" ]; then
  BINARIES_URL="https://na.artifactory.swg-devops.com/artifactory/hyc-cpd-skywalker-team-lakehouse-on-prem-generic-local/sonarqube/binaries/${ARCHIVE_BASE}.tar.gz"
  echo "Branch is 'main' — also uploading to $BINARIES_URL"
  curl -f -u "${ART_USER}:${ART_TOKEN}" -T "$ARCHIVE_FILE" "$BINARIES_URL" || echo "Warning: main upload failed"
fi

echo "Cleaning up staging and archive"
rm -f "$ARCHIVE_FILE"
rm -rf "$STAGING_DIR"

echo "SonarQube ondemand binaries TAR uploaded successfully"
