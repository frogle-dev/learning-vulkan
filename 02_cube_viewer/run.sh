#!/bin/sh

./build.sh
VK_LAYER_VALIDATE_SYNC=1 ASAN_SYMBOLIZER_PATH=$(which llvm-symbolizer) ASAN_OPTIONS=symbolize=1 LSAN_OPTIONS=suppressions=LSAN.supp ./build/learn_vulkan
