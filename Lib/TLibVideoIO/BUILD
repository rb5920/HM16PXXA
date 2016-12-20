cc_library(
    name = "TLibVideoIO",
    srcs = ["TVideoIOYuv.cpp"],
    hdrs = ["TVideoIOYuv.h"],	
	copts = ["-Itensorflow/HM16PXXA/Lib/"],
	deps = ["//tensorflow/HM16PXXA/Lib/TLibCommon:TLibCommon_lb"],
    visibility = ["//visibility:public"],
)
cc_inc_library(
    name = "TLibVideoIO_lb",
    hdrs = ["TVideoIOYuv.h"],
	visibility = ["//visibility:public"],
)
