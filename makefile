all: build

BAZEL=$(shell which bazel)

.PHONY: buildHM16PXXA
buildHM16PXXA : 
		$(BAZEL) build //tensorflow/HM16PXXA/Lib/NEURAL_CUP:NEURAL_CUP --copt="-O3"
		$(BAZEL) build //tensorflow/HM16PXXA/Lib/TIME:TIMETEST --copt="-O3"
		$(BAZEL) build //tensorflow/HM16PXXA/Lib/CUFeature:CUFeature --copt="-O3"
		$(BAZEL) build //tensorflow/HM16PXXA/Lib/TLibVideoIO:TLibVideoIO --copt="-O3"
		$(BAZEL) build //tensorflow/HM16PXXA/Lib/libmd5:libmd5 --copt="-O3"
		$(BAZEL) build //tensorflow/HM16PXXA/Lib/TLibCommon:TLibCommon --copt="-O3"
		$(BAZEL) build //tensorflow/HM16PXXA/Lib/TLibDecoder:TLibDecoder --copt="-O3"
		$(BAZEL) build //tensorflow/HM16PXXA/Lib/TLibEncoder:TLibEncoder --copt="-O3"
		$(BAZEL) build //tensorflow/HM16PXXA/Lib/TAppCommon:TAppCommon --copt="-O3"
		$(BAZEL) build //tensorflow/HM16PXXA/App/TAppDecoder:TAppDecoder --copt="-O3"
		$(BAZEL) build //tensorflow/HM16PXXA/App/TAppEncoder:TAppEncoder --copt="-O3"

.PHONY: clean
clean :
		$(BAZEL) clean

