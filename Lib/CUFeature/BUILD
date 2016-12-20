# Description:
#   Tensorflow C++ inference example for labeling images.

visibility = ["//visibility:public"],

#licenses(["notice"])  # Apache 2.0

#exports_files(["LICENSE"])


cc_library(
    name = "CUFeature",
    srcs = ["CUFeature.cpp"],
	hdrs = ["CUFeature.h"],
	deps = ["//tensorflow/HM16PXXA/Lib/TLibCommon:TLibCommon"],
	copts = ["-Itensorflow/HM16PXXA/Lib/"],
	visibility = ["//visibility:public"],
)
cc_inc_library(
    name = "CUFeature_lb",
    hdrs = glob(["*.h"]),
		visibility = ["//visibility:public"],
)


