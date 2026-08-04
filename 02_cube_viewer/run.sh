#!/bin/sh

./build.sh
ASAN_OPTIONS=symbolize=1 ./build/learn_vulkan
