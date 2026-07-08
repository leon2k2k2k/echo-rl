#!/usr/bin/env bash
set -euo pipefail

cd "${ECHO_REPO:-$(pwd)}"

# shellcheck source=/dev/null
source scripts/cluster_env.sh

base_image="${ECHO_DOCKER_BASE_IMAGE:-ubuntu:22.04}"
base_tar="${ECHO_DOCKER_BASE_IMAGE_TAR:-$ECHO_RUNTIME_ROOT/docker_cache/ubuntu_22_04.tar}"

mkdir -p "$(dirname "$base_tar")"

if ! docker image inspect "$base_image" >/dev/null 2>&1; then
  echo "Docker base image missing locally; pulling $base_image"
  docker pull "$base_image"
fi

echo "Saving $base_image to $base_tar"
docker save "$base_image" -o "$base_tar"
ls -lh "$base_tar"
