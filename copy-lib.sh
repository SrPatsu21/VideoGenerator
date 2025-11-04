#!/bin/bash
# copy-lib.sh
set -e

IMAGE="srpatsu21/dear-glfw-vulkan-compiler"
DEST="./lib-data"

echo "Copying /workspace/lib from image '$IMAGE' to $DEST..."

mkdir -p "$DEST"
docker run --rm -v "$(pwd)/lib-data":/copy-dest "$IMAGE" \
    bash -c "cp -r /workspace/lib/* /copy-dest/ || true"

echo "✅ Files copied to $DEST"
