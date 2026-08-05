#!/bin/sh

./build.sh
VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT ASAN_SYMBOLIZER_PATH=$(which llvm-symbolizer) ASAN_OPTIONS=symbolize=1 LSAN_OPTIONS=suppressions=lsan_suppressions.txt ./build/learn_vulkan
