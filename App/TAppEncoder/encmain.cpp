/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2015, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     encmain.cpp
    \brief    Encoder application main
*/

#include <time.h>
#include <iostream>
#include "TAppEncTop.h"
#include "TAppCommon/program_options_lite.h"

//! \ingroup TAppEncoder
//! \{

#include "../Lib/TLibCommon/Debug.h"
FILE* gfpTIME;
#if NEURALNETWORK_DUMP_ENABLE
FILE* gfpCU0;
FILE* gfpCU1;
FILE* gfpCU2;
#endif
#if NEURALNETWORK_CU_INTRA_PREDICTION_ENABLE
TF_neural MINTRADepth0;
TF_neural MINTRADepth1;
TF_neural MINTRADepth2;
short *PredictedDepth;
#endif
#if NEURALNETWORK_CU_INTER_PREDICTION_ENABLE
TF_neural MINTERDepth0;
TF_neural MINTERDepth1;
TF_neural MINTERDepth2;
short *PredictedDepthINTER;
#endif
#if NEURALNETWORK_CU_PREDICTION_ENABLE || NEURALNETWORK_DUMP_ENABLE || NEURALNETWORK_PU_PREDICTION_ENABLE
#if NEURALNETWORK_DUMP_ENABLE
int CUFeature_cnt;
#endif
CUFeature GetCUFeature[1000];
#endif
#if NEURALNETWORK_TIMEEXECUTE_ENABLE
TimeTest timem;
int Count_Depth0=0;
int Count_Depth0_NEURAL=0;
int Count_Depth1=0;
int Count_Depth1_NEURAL=0;
int Count_Depth2=0;
int Count_Depth2_NEURAL=0;
int Count_Depth3=0;
int Count_Depth3_NEURAL=0;
#endif
// ====================================================================================================================
// Main function
// ====================================================================================================================

int main(int argc, char* argv[])
{
  TAppEncTop  cTAppEncTop;

  // print information
  fprintf( stdout, "\n" );
  fprintf( stdout, "HM software: Encoder Version [%s] (including RExt)", NV_VERSION );
  fprintf( stdout, NVM_ONOS );
  fprintf( stdout, NVM_COMPILEDBY );
  fprintf( stdout, NVM_BITS );
  fprintf( stdout, "\n\n" );

  // create application encoder class
  cTAppEncTop.create();
#if NEURALNETWORK_TIMEEXECUTE_ENABLE
  timem.open();
  timem.start_fanc(GCFinit);
#endif
#if NEURALNETWORK_CU_INTRA_PREDICTION_ENABLE
	MINTRADepth0.Create(0);
	MINTRADepth1.Create(1);
	MINTRADepth2.Create(2);
#endif
#if NEURALNETWORK_CU_INTER_PREDICTION_ENABLE
	MINTERDepth0.Create(3);
	MINTERDepth1.Create(4);
	MINTERDepth2.Create(5);
#endif
#if NEURALNETWORK_TIMEEXECUTE_ENABLE
  timem.end_all(GCFinit);
#endif
  // parse configuration
  try
  {
    if(!cTAppEncTop.parseCfg( argc, argv ))
    {
      cTAppEncTop.destroy();
#if ENVIRONMENT_VARIABLE_DEBUG_AND_TEST
      EnvVar::printEnvVar();
#endif
      return 1;
    }
  }
  catch (df::program_options_lite::ParseFailure &e)
  {
    std::cerr << "Error parsing option \""<< e.arg <<"\" with argument \""<< e.val <<"\"." << std::endl;
    return 1;
  }

#if PRINT_MACRO_VALUES
  printMacroSettings();
#endif

#if ENVIRONMENT_VARIABLE_DEBUG_AND_TEST
  EnvVar::printEnvVarInUse();
#endif

  // starting time
  Double dResult;
  clock_t lBefore = clock();

  // call encoding function
  cTAppEncTop.encode();

  // ending time
  dResult = (Double)(clock()-lBefore) / CLOCKS_PER_SEC;


	fprintf(gfpTIME,"%12.3f\n", dResult);
#if NEURALNETWORK_TIMEEXECUTE_ENABLE
  timem.pri(GCFinit);
  fprintf(gfpTIME,",,,");
  timem.pri(NEURAL_Depth0);
  fprintf(gfpTIME,",,,,INTRA,%lf,sec\n",timem.getTime(NEURAL_Depth0)-timem.getTime(NEURAL_Depth0INTER));
  fprintf(gfpTIME,",,,,INTER,%lf,sec\n",timem.getTime(NEURAL_Depth0INTER));
  fprintf(gfpTIME,",,,");
  timem.pri(NEURAL_Depth1);
  fprintf(gfpTIME,",,,,INTRA,%lf,sec\n",timem.getTime(NEURAL_Depth1)-timem.getTime(NEURAL_Depth1INTER));
  fprintf(gfpTIME,",,,,INTER,%lf,sec\n",timem.getTime(NEURAL_Depth1INTER));
  fprintf(gfpTIME,",,,");
  timem.pri(NEURAL_Depth2);
  fprintf(gfpTIME,",,,,INTRA,%lf,sec\n",timem.getTime(NEURAL_Depth2)-timem.getTime(NEURAL_Depth2INTER));
  fprintf(gfpTIME,",,,,INTER,%lf,sec\n",timem.getTime(NEURAL_Depth2INTER));
  timem.pri(GCFgetFeature);
  fprintf(gfpTIME,"Total,%lf,sec\n",timem.getTime(GCFinit)+timem.getTime(GCFgetFeature));
  fprintf(gfpTIME,"RDCOST_Depth0,%lf,sec,",timem.getTime(RDCOST_Depth0));
	fprintf(gfpTIME,"┈Count_Depth0,%d,Count_Depth0NEURAL,%d\n",Count_Depth0,Count_Depth0_NEURAL);
  fprintf(gfpTIME,",,,Depth0INTRA,%lf,sec\n",timem.getTime(RDCOST_Depth0INTRA));
  fprintf(gfpTIME,",,,Depth0INTER,%lf,sec\n",timem.getTime(RDCOST_Depth0INTER)+timem.getTime(RDCOST_Depth0INTERI));
  fprintf(gfpTIME,"RDCOST_Depth1,%lf,sec,",timem.getTime(RDCOST_Depth1));
	fprintf(gfpTIME,"┈Count_Depth1,%d,Count_Depth1NEURAL,%d\n",Count_Depth1,Count_Depth1_NEURAL);
  fprintf(gfpTIME,",,,Depth1INTRA,%lf,sec\n",timem.getTime(RDCOST_Depth1INTRA));
  fprintf(gfpTIME,",,,Depth1INTER,%lf,sec\n",timem.getTime(RDCOST_Depth1INTER)+timem.getTime(RDCOST_Depth1INTERI));
  fprintf(gfpTIME,"RDCOST_Depth2,%lf,sec,",timem.getTime(RDCOST_Depth2));
	fprintf(gfpTIME,"┈Count_Depth2,%d,Count_Depth2NEURAL,%d\n",Count_Depth2,Count_Depth2_NEURAL);
  fprintf(gfpTIME,",,,Depth2INTRA,%lf,sec\n",timem.getTime(RDCOST_Depth2INTRA));
  fprintf(gfpTIME,",,,Depth2INTER,%lf,sec\n",timem.getTime(RDCOST_Depth2INTER)+timem.getTime(RDCOST_Depth2INTERI));
  fprintf(gfpTIME,"RDCOST_Depth3,%lf,sec,",timem.getTime(RDCOST_Depth3));
	fprintf(gfpTIME,"┈Count_Depth3,%d\n",Count_Depth3);
  fprintf(gfpTIME,",,,Depth3INTRA,%lf,sec\n",timem.getTime(RDCOST_Depth3INTRA));
  fprintf(gfpTIME,",,,Depth3INTER,%lf,sec\n",timem.getTime(RDCOST_Depth3INTER)+timem.getTime(RDCOST_Depth3INTERI));
  fprintf(gfpTIME,"Others,%lf,sec\n",dResult-timem.getTime(GCFinit)-timem.getTime(GCFgetFeature)-timem.getTime(RDCOST_Depth0)-timem.getTime(RDCOST_Depth1)-timem.getTime(RDCOST_Depth2)-timem.getTime(RDCOST_Depth3));
  fprintf(gfpTIME,"Depth0SKIP,%lf\nDepth1SKIP,%lf\nDepth2SKIP,%lf\nDepth3SKIP,%lf\n",timem.getTime(RDCOST_Depth0SKIP),timem.getTime(RDCOST_Depth1SKIP),timem.getTime(RDCOST_Depth2SKIP),timem.getTime(RDCOST_Depth3SKIP));
  timem.closeup();
#endif
  printf("\n Total Time: %12.3f sec.\n", dResult);
#if NEURALNETWORK_CU_INTRA_PREDICTION_ENABLE
	MINTRADepth0.Destroy();
	MINTRADepth1.Destroy();
	MINTRADepth2.Destroy();
#endif
#if NEURALNETWORK_CU_INTER_PREDICTION_ENABLE
	MINTERDepth0.Destroy();
	MINTERDepth1.Destroy();
	MINTERDepth2.Destroy();
#endif
	fclose(gfpTIME);
#if NEURALNETWORK_DUMP_ENABLE
	fclose(gfpCU0);
	fclose(gfpCU1);
	fclose(gfpCU2);
#endif
  // destroy application encoder class
  cTAppEncTop.destroy();

  return 0;
}

//! \}
