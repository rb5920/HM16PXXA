cc_library(
    name = "libmd5",
	srcs = ["libmd5.c"],
    hdrs = ["libmd5.h",
			"MD5.h"],
	copts = ["-Itensorflow/HM16PXXA/Lib"],
	visibility = ["//visibility:public"],
)
cc_inc_library(
    name = "libmd5_lb",
    hdrs = ["libmd5.h",
			"MD5.h"],
    deps = [":libmd5"],
	visibility = ["//visibility:public"],
)

