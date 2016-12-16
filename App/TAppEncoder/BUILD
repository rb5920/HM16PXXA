cc_binary(
    name = "TAppEncoder",
    srcs = ["encmain.cpp"],
	copts = ["-Itensorflow/HM16P001A/Lib"],
	deps = [":TAppEncoder_lib"]
)
cc_library(
    name = "TAppEncoder_lib",
    srcs = ["TAppEncCfg.cpp",
			"TAppEncTop.cpp"],
    hdrs = ["TAppEncCfg.h",
			"TAppEncTop.h"],
	copts = ["-Itensorflow/HM16P001A/Lib"],
	deps = ["//tensorflow/HM16P001A/Lib/TLibEncoder:TLibEncoder",
			"//tensorflow/HM16P001A/Lib/TAppCommon:TAppCommon",],
)
