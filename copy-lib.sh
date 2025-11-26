#!/bin/bash
# copy-lib.sh
set -e

IMAGE="srpatsu21/dear-glfw-vulkan-compiler"
DEST="./lib"

echo "Copying /workspace/lib from image '$IMAGE' to $DEST..."

mkdir -p "$DEST"
sudo docker run --rm -v "$(pwd)/$DEST":/copy-dest "$IMAGE" \
    sh -c "cp -r /workspace/lib/* /copy-dest/ || true"

echo "✅ Files copied to $DEST"
