#pragma once
#include <sys/time.h>
#include <iostream>
using namespace std;

enum TM_Name {
	GCFinit = 0,
	GCFElevateFeature = 1,
	GCFINTERDiff = 2,
	GCFgetFeature = 3,
	NEURAL_Depth0 = 4,
	NEURAL_Depth1 = 5,
	NEURAL_Depth2 = 6,
	RDCOST_Depth0=7,
	RDCOST_Depth1=8,
	RDCOST_Depth2=9,
	RDCOST_Depth3=10,
	RDCOST_Depth0INTER=11,
	RDCOST_Depth0INTRA=12,
	RDCOST_Depth1INTER=13,
	RDCOST_Depth1INTRA=14,
	RDCOST_Depth2INTER=15,
	RDCOST_Depth2INTRA=16,
	RDCOST_Depth3INTER=17,
	RDCOST_Depth3INTRA=18,
	RDCOST_Depth0INTERI=19,
	RDCOST_Depth1INTERI=20,
	RDCOST_Depth2INTERI=21,
	RDCOST_Depth3INTERI=22,
	RDCOST_Depth3SKIP=23,
	RDCOST_Depth2SKIP=24,
	RDCOST_Depth1SKIP=25,
	RDCOST_Depth0SKIP=26,
	RDC_INTRA_T2=27,
	RDC_INTRA_T3=28,
	RDC_INTRA_T4=29,
	RDCOST_TESTPCM = 30,
	estLUMA_tuRecurseCU=31,
	estLUMA_tuRecurseWithPU=32,
	estLUMA_tuRecurseWithPU_nextSection=33,
	estLUMA_predIntraAng=34,
	estLUMA_xRecurIntraCodingLumaQT=35,
	estLUMA_dPUCost_dBestPUCost=36,
	estLUMA_xRecurIntraCodingLumaQT2=37,
	estLUMA_dPUCost_dBestPUCost2=38,
	xRecurLuma_dPUCost_bCheckFull=39,
	xRecurLuma_dPUCost_bCheckSplit=40,
	NEURAL_Depth0INTER = 41,
	NEURAL_Depth1INTER = 42,
	NEURAL_Depth2INTER = 43,
	TM_Name_NUM
};

class TimeTest {
private:
	double sum_time[TM_Name_NUM] = {};
	//LARGE_INTEGER freq[TM_Name_NUM];
	struct timeval start[TM_Name_NUM], end[TM_Name_NUM];
	static const std::string TM_NAME_STR[TM_Name_NUM];
	FILE* ttfp;

public:

	TimeTest();
	~TimeTest();
	void start_fanc(int idx); 
	void end_fanc(int idx); 
	void cal(int idx);
	void sum_print(int idx);
	void end_all(int idx);
	void pri(int idx);
	void open();
	void closeup();
	void end_all_fpri(int idx);
	double getTime(int idx);
	

};

