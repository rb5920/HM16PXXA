#include "TimeTest.h"

extern FILE* gfpTIME;

TimeTest::TimeTest()
{
}


TimeTest::~TimeTest()
{
}

const std::string TimeTest::TM_NAME_STR[TM_Name_NUM] = {
	"GCFinit",
	"GCFElevateFeature",
	"GCFINTERDiff",
	"GCFgetFeature",
	"NEURAL_Depth0",
	"NEURAL_Depth1",
	"NEURAL_Depth2",
	"RDCOST_Depth0",
	"RDCOST_Depth1", 
	"RDCOST_Depth2",
	"RDCOST_Depth3",
	"RDCOST_Depth0INTER",
	"RDCOST_Depth0INTRA",
	"RDCOST_Depth1INTER",
	"RDCOST_Depth1INTRA",
	"RDCOST_Depth2INTER",
	"RDCOST_Depth2INTRA",
    "RDCOST_Depth3INTER",
	"RDCOST_Depth3INTRA",
	"RDCOST_Depth0INTERI",
	"RDCOST_Depth1INTERI",
	"RDCOST_Depth2INTERI",
	"RDCOST_Depth3INTERI",
	"RDCOST_Depth3SKIP",
	"RDCOST_Depth2SKIP",
	"RDCOST_Depth1SKIP",
	"RDCOST_Depth0SKIP",
	"RDC_INTRA_T2",
	"RDC_INTRA_T3",
	"RDC_INTRA_T4",
	"RDCOST_TESTPCM",
	"estLUMA_tuRecurseCU",
	"estLUMA_tuRecurseWithPU",
	"estLUMA_tuRecurseWithPU.nextSection",
	"estLUMA_predIntraAng",
	"estLUMA_xRecurIntraCodingLumaQT",
	"estLUMA_dPUCost_dBestPUCost",
	"estLUMA_xRecurIntraCodingLumaQT2",
	"estLUMA_dPUCost_dBestPUCost2",
	"xRecurLuma_dPUCost_bCheckFull",
	"xRecurLuma_dPUCost_bCheckSplit",
	"NEURAL_Depth0INTER",
	"NEURAL_Depth1INTER",
	"NEURAL_Depth2INTER",

};

void TimeTest::start_fanc(int idx) {
	gettimeofday(&start[idx],NULL);
}

void TimeTest::end_fanc(int idx) {
	gettimeofday(&end[idx],NULL);
}
void TimeTest::cal(int idx) {
	//sum_time[idx] += (double)(end[idx].QuadPart - start[idx].QuadPart);
	sum_time[idx] += (double)(1000000*(end[idx].tv_sec-start[idx].tv_sec)+end[idx].tv_usec-start[idx].tv_usec);
}

void TimeTest::sum_print(int idx) {

	printf("%s:::::%lf\n", TM_NAME_STR[idx].c_str(), sum_time[idx]);


}

double TimeTest::getTime(int idx)
{
	return sum_time[idx]/1000000;
}

void TimeTest::end_all(int idx) {
	gettimeofday(&end[idx],NULL);
	sum_time[idx] += (double)(1000000*(end[idx].tv_sec-start[idx].tv_sec)+end[idx].tv_usec-start[idx].tv_usec);
}

void TimeTest::end_all_fpri(int idx) {
	gettimeofday(&end[idx],NULL);
	sum_time[idx] += (double)(1000000*(end[idx].tv_sec-start[idx].tv_sec)+end[idx].tv_usec-start[idx].tv_usec);
	fprintf(ttfp,"%s,%lf\n", TM_NAME_STR[idx].c_str(), sum_time[idx]);
	}

void TimeTest::pri(int idx) {
	fprintf(gfpTIME,"%s,%lf,sec\n", TM_NAME_STR[idx].c_str(), sum_time[idx]/1000000);
}

void TimeTest::open() {
	ttfp = fopen("Timetest.csv", "ab");
}

void TimeTest::closeup() {
	fclose(ttfp);
}

