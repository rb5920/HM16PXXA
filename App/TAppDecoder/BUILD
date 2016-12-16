
cc_binary(
    name = "TAppDecoder",
    srcs = ["decmain.cpp"],
	copts = ["-Itensorflow/HM16P001A/Lib"],
	deps = [":TAppDecoder_lib"]
)
cc_library(
    name = "TAppDecoder_lib",
    srcs = ["TAppDecCfg.cpp",
			"TAppDecTop.cpp"],
    hdrs = ["TAppDecCfg.h",
			"TAppDecTop.h"],
	copts = ["-Itensorflow/HM16P001A/Lib"],
	deps = ["//tensorflow/HM16P001A/Lib/TLibDecoder:TLibDecoder",
			"//tensorflow/HM16P001A/Lib/TAppCommon:TAppCommon"],
)
