#include <stdio.h>
#include <sys/time.h>
#include <iostream>
#include "TLibCommon/CommonDef.h"
#include "TLibCommon/TComDataCU.h"
#include "TLibCommon/TComPic.h"
using namespace std;
#define VAR_NORMAL_MODE 0
#define VAR_SUM_MODE 1
#define FEATURE_INTRAMODE 0
#define FEATURE_INTERMODE 1
#define FEATURE_PREDICTMODE 2
//class TComDataCU;

class CUFeature{
    private:
        Double SUMSIZE64;
        Double SUMVSIZE64;
        Double AVGSIZE64;
        Double VARSIZE64;
        Double VARAVGofROW64;
        Double VARAVGofCOL64;
        Double SUMSIZE32[4];
        Double SUMVSIZE32[4];
        Double AVGSIZE32[4];
        Double VARSIZE32[4];
        Double SUMSIZE16[4][4];
        Double SUMVSIZE16[4][4];
        Double AVGSIZE16[4][4];
        Double VARSIZE16[4][4];
        Double SUMSIZE8[4][4][4];
        Double SUMVSIZE8[4][4][4];
        Double AVGSIZE8[4][4][4];
        Double VARSIZE8[4][4][4];
        Double SUMSIZE4[256];
        Double SUMVSIZE4[256];
        Double AVGSIZE4[256];
        Double VARSIZE4[256];
	    Double DC_AVG;
	    Double AC_AVG;
        Double QP;
        Double VARDiffrenceofPreFrame;
        Double AVGDiffrenceofPreFrame;
        Double PREAVGSIZE64;
        Double PREVARSIZE64;
        bool INTERPRE_READY;
    public:
        void  SYSTEM_INIT(int);
        void  init();
        Double Average(Double* Data, Double data_num);
        void Variance(Double* Data, Double data_num, Double* Output1, Double* Output2, int mode);
        void ElevateFeature(TComDataCU* pcCU, UInt uiAbsPartIdx, UInt uiDepth);
        void PreElevateFeature(TComPic* pcPic, UInt CtuIndex);
        int getINTRADepth0(float* Output,UInt uiAbsPartIdx);
        int getINTRADepth1(float* Output,UInt uiAbsPartIdx);
        int getINTRADepth2(float* Output,UInt uiAbsPartIdx);
        int getINTERDepth0(float* Output,UInt uiAbsPartIdx);
        int getINTERDepth1(float* Output,UInt uiAbsPartIdx);
        int getINTERDepth2(float* Output,UInt uiAbsPartIdx);
        int getPMDepth0(float* Output,UInt uiAbsPartIdx);
        int getPMDepth1(float* Output,UInt uiAbsPartIdx);
        int getPMDepth2(float* Output,UInt uiAbsPartIdx);
        int getFeature(float* Output,int Depth,UInt uiAbsPartIdx,int Mode);
        void ElevateINTERDiff(TComDataCU* pcCU);
        Void MYpartialButterfly8(TCoeff *src, TCoeff *dst, Int shift, Int line);
        Void MYpartialButterfly16(TCoeff *src, TCoeff *dst, Int shift, Int line);
        Void MYpartialButterfly32(TCoeff *src, TCoeff *dst, Int shift, Int line);
        Void MYpartialButterfly4(TCoeff *src, TCoeff *dst, Int shift, Int line);
        Void MYxTrMxN(TCoeff *block, TCoeff *coeff, Int iWidth, Int iHeight);

};