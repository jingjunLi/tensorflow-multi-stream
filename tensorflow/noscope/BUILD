# pkg-config --cflags opencv
# pkg-config --libs opencv

load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
    "if_cuda_is_configured"
)

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cuda_library"
)

# tf_cuda_library(
cc_binary(
    name = "noscope",
    copts = ["-I/usr/local/include/opencv2", "-O3", "-fopenmp"] + if_cuda(["-DGOOGLE_CUDA=1"]),
    linkopts = [
        "-fopenmp",
        "-L/usr/local/lib",
        "-lopencv_core",
        "-lopencv_imgcodecs",
        "-lopencv_imgproc",
        "-lopencv_highgui",
        "-lopencv_videoio"
    ],
    srcs = glob(["*.h"]) + glob(["*.cc"]),
    deps = [
        "//tensorflow/core:tensorflow",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/noscope/darknet:yolo",
        "//tensorflow/core:cuda",
        "//tensorflow/core:gpu_lib",
        "@local_config_cuda//cuda:cuda_headers"
    ]
)
