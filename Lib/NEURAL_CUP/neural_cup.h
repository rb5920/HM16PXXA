
#include<fstream>
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/platform/env.h"
using namespace tensorflow;
using namespace std;

#define NN_INTRA_DEPTH0						0
#define NN_INTRA_DEPTH1						1
#define NN_INTRA_DEPTH2						2
#define NN_INTER_DEPTH0						3
#define NN_INTER_DEPTH1						4
#define NN_INTER_DEPTH2						5
#define NN_PREDICTIONMODE_DEPTH0			6
#define NN_PREDICTIONMODE_DEPTH1			7
#define NN_PREDICTIONMODE_DEPTH2			8

class TF_neural{
	private:
		Session* m_session;
		GraphDef m_graph_def;
		Status m_status;
		std::vector<tensorflow::Tensor> m_outputs;
	public:
		int Create(int);
		int Destroy();
		bool neural_cup(float* readinput,int number);
		void neural_cupPre(float** readinput,int number,int setnumber, int*Output);
};

