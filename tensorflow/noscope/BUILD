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
    "tf_cuda_library",
)

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
    "tf_proto_library_cc",
)

CORE_SRCS = [
        "noscope_video.cc",
        "noscope_stream.cc",
        "stream_set.cc",
        "noscope_data.cc",
        "noscope_labeler.cc",
        "filters.cc",
        "mse.cc",
        "util/utils.cc",
        ]
CORE_HDRS = [
        "common.h",
        "stream_set.h",
        "noscope_stream.h",
        "noscope_video.h",
        "noscope_data.h",
        "noscope_labeler.h",
        "mse.h",
        "filters.h",
        "util/simple_queue.h",
        "util/utils.h",
        "MemoryTests.h",
#        "proto/noscope.pb.h",
        ]
# tf_cuda_library(
cc_binary(
    name = "noscope",
    copts = [
        "-I/usr/local/include", "-I/usr/local/include/opencv2", "-I/usr/local/include/opencv",
        "-O3", "-fopenmp", "-lprotobuf"
    ] + if_cuda(["-DGOOGLE_CUDA=1"]),
    linkopts = [
        "-fopenmp",
        "-L/usr/local/lib/",
        "-lopencv_core",
        "-lopencv_imgcodecs",
        "-lopencv_imgproc",
        "-lopencv_highgui",
        "-lopencv_videoio"
    ],
    srcs = CORE_HDRS + CORE_SRCS, #glob(["*.h"]) + glob(["*.cc"], exclude=["dumper.cc"]),
    deps = [
        "//tensorflow/core:tensorflow",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/noscope/darknet:yolo",
        "//tensorflow/core:cuda",
        "//tensorflow/core:gpu_lib",
        "@local_config_cuda//cuda:cuda_headers",
        "//tensorflow/core:noscope_proto_cc",
    ]
)

cc_binary(
    name = "dumper",
    copts = [
        "-I/usr/local/include", "-I/usr/local/include/opencv2", "-I/usr/local/include/opencv",
        "-O3", "-fopenmp"
    ],
    linkopts = [
        "-fopenmp",
        "-L/usr/local/lib/",
        "-lopencv_core",
        "-lopencv_imgcodecs",
        "-lopencv_imgproc",
        "-lopencv_highgui",
        "-lopencv_videoio"
    ],
    srcs = glob(["noscope_data.h"]) + glob(["noscope_data.cc", "dumper.cc"]),
    deps = [
        "//tensorflow/core:tensorflow",
        "//tensorflow/cc:scope",
    ]
)

cc_binary(
    name = "noscope_video_test",
    copts = [
        "-I/usr/local/include", "-I/usr/local/include/opencv2", "-I/usr/local/include/opencv",
        "-O3", "-fopenmp", "-lpthread", "-lprotobuf"
    ] + if_cuda(["-DGOOGLE_CUDA=1"]),
    linkopts = [
        "-fopenmp",
        "-L/usr/local/lib/",
        "-lopencv_core",
        "-lopencv_imgcodecs",
        "-lopencv_imgproc",
        "-lopencv_highgui",
        "-lopencv_videoio"
    ],
    srcs = CORE_HDRS + CORE_SRCS + ["test/noscope_video_test.cc"],
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

#cc_library(
#    name = "util",
#    srcs = glob(["util/util.*"]),
#)

cc_binary(
    name = "noscope_stream_test",
    copts = [
        "-I/usr/local/include", "-I/usr/local/include/opencv2", "-I/usr/local/include/opencv",
        "-O3", "-fopenmp", "-lpthread", "-lprotobuf"
    ] + if_cuda(["-DGOOGLE_CUDA=1"]),
    linkopts = [
        "-fopenmp",
        "-L/usr/local/lib/",
        "-lopencv_core",
        "-lopencv_imgcodecs",
        "-lopencv_imgproc",
        "-lopencv_highgui",
        "-lopencv_videoio"
    ],
    srcs = CORE_HDRS + CORE_SRCS + ["test/noscope_stream_test.cc"],
    deps = [
        "//tensorflow/core:noscope_proto_cc",
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

cc_binary(
    name = "stream_set_test",
    copts = [
        "-I/usr/local/include", "-I/usr/local/include/opencv2", "-I/usr/local/include/opencv",
        "-O3", "-fopenmp", "-lpthread", "-lprotobuf"
    ] + if_cuda(["-DGOOGLE_CUDA=1"]),
    linkopts = [
        "-fopenmp",
        "-L/usr/local/lib/",
        "-lopencv_core",
        "-lopencv_imgcodecs",
        "-lopencv_imgproc",
        "-lopencv_highgui",
        "-lopencv_videoio"
    ],
    srcs = CORE_HDRS + CORE_SRCS + ["test/stream_set_test.cc"],
    deps = [
        "//tensorflow/core:noscope_proto_cc",
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

cc_binary(
    name = "stream_sets_test",
    copts = [
        "-I/usr/local/include", "-I/usr/local/include/opencv2", "-I/usr/local/include/opencv",
        "-O3", "-fopenmp", "-lpthread", "-lprotobuf"
    ] + if_cuda(["-DGOOGLE_CUDA=1"]),
    linkopts = [
        "-fopenmp",
        "-L/usr/local/lib/",
        "-lopencv_core",
        "-lopencv_imgcodecs",
        "-lopencv_imgproc",
        "-lopencv_highgui",
        "-lopencv_videoio"
    ],
    srcs = CORE_HDRS + CORE_SRCS + ["test/stream_sets_test.cc"],
    deps = [
        "//tensorflow/core:noscope_proto_cc",
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
#tf_proto_library_cc(
#    name = "noscope_proto",
#    srcs = ["proto/noscope.proto"],
#    cc_api_version = 2,
#    protodeps = [":protos_all"],
#    visibility = ["//visibility:public"],
#)

#cc_binary(
#    name = "noscope_streams_test",
#    copts = [
#        "-I/usr/local/include", "-I/usr/local/include/opencv2", "-I/usr/local/include/opencv",
#        "-O3", "-fopenmp", "-lpthread"
#    ] + if_cuda(["-DGOOGLE_CUDA=1"]),
#    linkopts = [
#        "-fopenmp",
#        "-L/usr/local/lib/",
#        "-lopencv_core",
#        "-lopencv_imgcodecs",
#        "-lopencv_imgproc",
#        "-lopencv_highgui",
#        "-lopencv_videoio"
#    ],
#    #srcs = ["noscope_stream_test.cc"],
#    srcs = CORE_HDRS + CORE_SRCS + ["test/noscope_stream_test.cc"],
#    deps = [
#        "//tensorflow/core:tensorflow",
#        "//tensorflow/cc:cc_ops",
#        "//tensorflow/cc:ops",
#        "//tensorflow/cc:scope",
#        "//tensorflow/noscope/darknet:yolo",
#        "//tensorflow/core:cuda",
#        "//tensorflow/core:gpu_lib",
#        "@local_config_cuda//cuda:cuda_headers"
#    ]
#)
