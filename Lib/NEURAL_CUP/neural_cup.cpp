
#include "neural_cup.h"
using namespace tensorflow;
using namespace std;


int TF_neural::Create(int ModelVersion){
	m_status = NewSession(SessionOptions(), &(m_session));
  if (!m_status.ok()) {
    std::cout << m_status.ToString() << "\n";
    return 0;
	}
  switch(ModelVersion)
  {
    case -1:
      m_status = ReadBinaryProto(Env::Default(), "models/rec_MIXT1N.pb", &(m_graph_def));
    break;
    case NN_INTRA_DEPTH0:
      m_status = ReadBinaryProto(Env::Default(), "models/rec_T24n.pb", &(m_graph_def));
    break;
    case NN_INTRA_DEPTH1:
      m_status = ReadBinaryProto(Env::Default(), "models/rec_T26n.pb", &(m_graph_def));
    break;
    case NN_INTRA_DEPTH2:
      m_status = ReadBinaryProto(Env::Default(), "models/rec_T56n.pb", &(m_graph_def));
    break;
    case NN_INTER_DEPTH0:
      m_status = ReadBinaryProto(Env::Default(), "models/rec_INTER_T09.pb", &(m_graph_def));
    break;
    case NN_INTER_DEPTH1:
      m_status = ReadBinaryProto(Env::Default(), "models/rec_INTER_T09D1.pb", &(m_graph_def));
    break;
    case NN_INTER_DEPTH2:
      m_status = ReadBinaryProto(Env::Default(), "models/rec_INTER_T09D2.pb", &(m_graph_def));
    break;
    case NN_PREDICTIONMODE_DEPTH0:
      m_status = ReadBinaryProto(Env::Default(), "models/rec_INTER_T09.pb", &(m_graph_def));
    break;
    case NN_PREDICTIONMODE_DEPTH1:
      m_status = ReadBinaryProto(Env::Default(), "models/rec_INTER_T09D1.pb", &(m_graph_def));
    break;
    case NN_PREDICTIONMODE_DEPTH2:
      m_status = ReadBinaryProto(Env::Default(), "models/rec_INTER_T09D2.pb", &(m_graph_def));
    break;
  }

  if (!m_status.ok()) {
    std::cout << m_status.ToString() << "\n";
    return 0;
	}
	m_status = m_session->Create(m_graph_def);
  if (!m_status.ok()) {
    std::cout << m_status.ToString() << "\n";
    return 0;
  }
	
	Tensor checkpoint_filepath(DT_STRING, TensorShape());


  switch(ModelVersion)
  {
    case -1:
      checkpoint_filepath.scalar<std::string>()() = "models/rec_MIXT1N";
    break;
    case NN_INTRA_DEPTH0:
      checkpoint_filepath.scalar<std::string>()() = "models/rec_T24";
    break;
    case NN_INTRA_DEPTH1:
      checkpoint_filepath.scalar<std::string>()() = "models/rec_T26";
    break;
    case NN_INTRA_DEPTH2:
      checkpoint_filepath.scalar<std::string>()() = "models/rec_T56";
    break;
    case NN_INTER_DEPTH0:
      checkpoint_filepath.scalar<std::string>()() = "models/rec_INTER_T09";
    break;
    case NN_INTER_DEPTH1:
      checkpoint_filepath.scalar<std::string>()() = "models/rec_INTER_T09D1";
    break;
    case NN_INTER_DEPTH2:
      checkpoint_filepath.scalar<std::string>()() = "models/rec_INTER_T09D2";
    break;
    case NN_PREDICTIONMODE_DEPTH0:
      checkpoint_filepath.scalar<std::string>()() = "models/rec_INTER_T09";
    break;
    case NN_PREDICTIONMODE_DEPTH1:
      checkpoint_filepath.scalar<std::string>()() = "models/rec_INTER_T09D1";
    break;
    case NN_PREDICTIONMODE_DEPTH2:
      checkpoint_filepath.scalar<std::string>()() = "models/rec_INTER_T09D2";
    break;
  }

  m_session->Run( {{ "save/Const", checkpoint_filepath },}, 
                       {}, {"save/restore_all"}, &(m_outputs));
	return 1;
}
int TF_neural::Destroy(){
	m_session->Close();
  return 0;
}
bool TF_neural::neural_cup(float readinput[],int number) {

	
	std::vector<std::pair<string, tensorflow::Tensor>> inputs;
	vector<Tensor>::iterator it;
	Tensor x(DT_FLOAT, TensorShape({1,number}));
  //auto input = x.tensor<float, 2>();
  for(int s=0;s<number;s++)
  {
    x.matrix<float>()(0,s) = *(readinput + s);
    //input(0,s) = *(readinput + s);
  }

  inputs = {{ "inputx", x },};

  m_session->Run(inputs, {"softmax_linear/logits"}, {}, &(m_outputs));

  it = m_outputs.begin();

  auto items = it->shaped<float, 2>({1, 2});

  if (items(0,0)>items(0,1))
  {
    return 0;
  }
  else
  {
    return 1;
  }

}
void TF_neural::neural_cupPre(float **readinput,int number,int setnumber,int *Output) {

	std::vector<std::pair<string, tensorflow::Tensor>> inputs;
	vector<Tensor>::iterator it;
	Tensor x(DT_FLOAT, TensorShape({setnumber,number}));
  //auto input = x.tensor<float, 2>();
  int s=0;
  for(int s=0;s<setnumber;s++)
  {
    for(int k=0;k<number;k++)
    {
      //printf("!!!!!!!!!!!!!!!!!!!!!!!!!!s=%d,k=%d,%f\n",s,k,readinput[s][k]);
      x.matrix<float>()(s,k) = readinput[s][k];
    }
  }
  inputs = {{ "inputx", x },};
  m_session->Run(inputs, {"softmax_linear/argmaxlogits"}, {}, &(m_outputs));
  it = m_outputs.begin();
  auto items = it->flat<int>();
  for(int s=0;s<setnumber;s++)
  {
    Output[s]=items(s);
  }
  
}
