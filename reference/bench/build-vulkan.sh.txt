#!/bin/bash
set -e

# Resolve Vulkan SDK path
VULKAN_ROOT="${HOME}/tools/vulkan"

if [ -n "$VULKAN_SDK" ]; then
  echo "Using VULKAN_SDK from environment: $VULKAN_SDK"
elif [ -f "$VULKAN_ROOT/vulkan-init.sh" ]; then
  source "$VULKAN_ROOT/vulkan-init.sh"
  version=$(vulkan_resolve_version)
  if [ -n "$version" ]; then
    vulkan_activate "$version"
    echo "Activated Vulkan SDK $version via vulkan-init.sh"
  else
    echo "No Vulkan SDK version could be resolved" >&2
    exit 1
  fi
else
  # Fallback: read .vulkan-active directly, or use latest known version
  if [ -f "$VULKAN_ROOT/.vulkan-active" ]; then
    version=$(cat "$VULKAN_ROOT/.vulkan-active")
  else
    version="1.4.350.1"
  fi
  VULKAN_SDK="$VULKAN_ROOT/$version/x86_64"
  export VULKAN_SDK
  echo "Using fallback Vulkan SDK: $VULKAN_SDK"
fi

# Source Vulkan environment (if not already done by vulkan_activate)
if [ -f "$VULKAN_SDK/setup-env.sh" ]; then
  source "$VULKAN_SDK/setup-env.sh"
fi

echo "=== Building GGML with Vulkan ==="
rm -rf ggml/build
cmake -S ggml -B ggml/build \
  -DGGML_VULKAN=ON \
  -DVulkan_DIR="$VULKAN_SDK/lib/cmake/Vulkan"
cmake --build ggml/build -j

echo ""
echo "=== Building qwen3-tts.cpp ==="
rm -rf build
cmake -S . -B build \
  -DGGML_VULKAN=ON \
  -DQWEN3_TTS_TIMING=ON
cmake --build build -j

echo ""
echo "=== Build complete ==="
echo "Run ./build/qwen3-tts-cli to test"
