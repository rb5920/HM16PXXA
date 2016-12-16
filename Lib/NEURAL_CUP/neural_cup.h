
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

class TF_neural{
	private:
		/*static Session* session;
		static GraphDef graph_def;
		static Status status;
		static std::vector<std::pair<string, tensorflow::Tensor>> inputs;
		static std::vector<tensorflow::Tensor> outputs;
		static vector<Tensor>::iterator it;*/
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

