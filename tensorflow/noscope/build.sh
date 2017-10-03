#!/bin/bash -e

#bazel build -c opt --copt=-mavx2 --config=cuda noscope
#bazel build -c opt --copt=-mavx2 --config=cuda noscope_stream_test
bazel build -c opt --copt=-mavx2 --config=cuda  stream_set_test --verbose_failures
#bazel build -c opt --copt=-mavx2 --config=cuda  stream_sets_test --verbose_failures
