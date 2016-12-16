all: build

BAZEL=$(shell which bazel)

.PHONY: buildHM16P001A
buildHM16P001A : 
		$(BAZEL) build //tensorflow/HM16P001A/Lib/NEURAL_CUP:NEURAL_CUP --copt="-O3"
		$(BAZEL) build //tensorflow/HM16P001A/Lib/TIME:TIMETEST --copt="-O3"
		$(BAZEL) build //tensorflow/HM16P001A/Lib/CUFeature:CUFeature --copt="-O3"
		$(BAZEL) build //tensorflow/HM16P001A/Lib/TLibVideoIO:TLibVideoIO --copt="-O3"
		$(BAZEL) build //tensorflow/HM16P001A/Lib/libmd5:libmd5 --copt="-O3"
		$(BAZEL) build //tensorflow/HM16P001A/Lib/TLibCommon:TLibCommon --copt="-O3"
		$(BAZEL) build //tensorflow/HM16P001A/Lib/TLibDecoder:TLibDecoder --copt="-O3"
		$(BAZEL) build //tensorflow/HM16P001A/Lib/TLibEncoder:TLibEncoder --copt="-O3"
		$(BAZEL) build //tensorflow/HM16P001A/Lib/TAppCommon:TAppCommon --copt="-O3"
		$(BAZEL) build //tensorflow/HM16P001A/App/TAppDecoder:TAppDecoder --copt="-O3"
		$(BAZEL) build //tensorflow/HM16P001A/App/TAppEncoder:TAppEncoder --copt="-O3"

.PHONY: clean
clean :
		$(BAZEL) clean

