#include "tensorflow/noscope/stream.h"

namespace noscope {

const cv::Size NoscopeStream::kDiffResol_(100, 100);
const cv::Size NoscopeStream::kDistResol_(50, 50);

noscope::filters::DifferenceFilter GetDiffFilter(const bool kUseBlocked,
                                              const bool kSkipDiffDetection) {
  noscope::filters::DifferenceFilter nothing{noscope::filters::DoNothing, "DoNothing"};
  noscope::filters::DifferenceFilter blocked{noscope::filters::BlockedMSE, "BlockedMSE"};
  noscope::filters::DifferenceFilter global{noscope::filters::GlobalMSE, "GlobalMSE"};

  if (kSkipDiffDetection) {
    return nothing;
  }
  if (kUseBlocked) {
    return blocked;
  } else {
    return global;
  }
}


using google::protobuf::io::FileInputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  //CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

template <typename Dtype>
void Debug_String(const std::string& str, const Dtype* ptr, int N) {
  std::cout << str << " ";
  int n = N/30;
  for (int i = 0; i < N;) {
    std::cout << ptr[i] << " "; 
    i += n;
  }
  std::cout << std::endl;
}

template void Debug_String<uint8_t>(const std::string& str, const uint8_t*  ptr, int N);
template void Debug_String<float>(const std::string& str, const float*  ptr, int N);

NoscopeStream::NoscopeStream(const std::string& param_file) :
    avg_(kDistResol_, CV_32FC3){
  StreamParameter param;
  ReadProtoFromTextFile(param_file.c_str(), &param);
  Init(param); 
}

NoscopeStream::NoscopeStream(const StreamParameter& param) :
    avg_(kDistResol_, CV_32FC3){
  Init(param);
}

void NoscopeStream::Init(const StreamParameter& in_param) {
  name_ = in_param.name();
  std::ifstream is(in_param.avg_fname());
  std::istream_iterator<float> start(is), end;
  std::vector<float> nums(start, end);
  if (nums.size() != kDistFrameSize_) {
    throw std::runtime_error("nums not right size");
  }
  memcpy(avg_.data, &nums[0], kDistFrameSize_ * sizeof(float));

  lower_diff_thresh_ = in_param.lower_thresh();
  upper_diff_thresh_ = in_param.upper_thresh();
  distill_thresh_lower_ = in_param.distill_thresh_lower();
  distill_thresh_upper_ = in_param.distill_thresh_upper();
  skip_diff_detection_ = in_param.skip_diff_detection();
  skip_small_cnn_ = in_param.skip_small_cnn();
  const_ref_ = in_param.const_ref();
  kRef_ = in_param.ref();
  graph_ = in_param.graph();

  diff_filt_ = noscope::GetDiffFilter(in_param.use_blocked(), in_param.skip_diff_detection());
  reader_ = new noscope::NoscopeVideo(in_param.video_param());
  vQueue_ = std::shared_ptr<noscope::SimpleQueue<noscope::Frame* > >(new noscope::SimpleQueue<noscope::Frame*>); 
  end_ = false;
  InitSession();
}

void NoscopeStream::SetUp(const size_t stream_id, std::shared_ptr<noscope::SimpleQueue<noscope::Frame*> > gQueue) {
  std::cout << "Stream :" << stream_id << "  " << name_ << " SetUp" << std::endl;
  stream_id_ = stream_id;
  gQueue_ = gQueue;
  reader_->SetUp(stream_id_, vQueue_);
}

void NoscopeStream::Start() {
  std::cout << "Stream :" << stream_id_ <<  " Start" << std::endl;
  reader_->Start();
  diff_thread_ = std::unique_ptr<std::thread>(
          new std::thread(&NoscopeStream::RunDifferenceFilter, this));

  small_cnn_thread_ = std::unique_ptr<std::thread>(
          new std::thread(&NoscopeStream::RunSmallCNN, this));
}

bool NoscopeStream::IsEnd() {
  return end_;
}

void NoscopeStream::InitSession() {
  tensorflow::SessionOptions opts;
  tensorflow::GraphDef graph_def;
  // YOLO needs some memory
  // 剩余内存分配给 YOLO 调大
  //opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.9);
  opts.config.mutable_gpu_options()->set_allow_growth(true);
  tensorflow::Status status = NewSession(opts, &session_);
  TF_CHECK_OK(status);

  status = tensorflow::ReadBinaryProto(
      tensorflow::Env::Default(),
      graph_, &graph_def);
  tensorflow::graph::SetDefaultDevice("/gpu:0", &graph_def);
  TF_CHECK_OK(status);

  status = session_->Create(graph_def);
  TF_CHECK_OK(status);
}

NoscopeStream::~NoscopeStream() {
  diff_thread_->join();
  small_cnn_thread_->join();
}

void NoscopeStream::RunDifferenceFilter() {
  noscope::Frame* frame_out;
  cv::Mat diff_frame(kDiffResol_, CV_8UC3);
  std::vector<uint8_t> diff_data(kDiffFrameSize_);

  vQueue_->Pop(&frame_out);
  cv::resize(frame_out->frame, diff_frame, kDiffResol_, 0, 0, cv::INTER_NEAREST);
  memcpy(&diff_data[0], diff_frame.data, kDiffFrameSize_);
  //Debug_String("DifferenceFilter diff_data: ", &diff_data[0], 7500);
  const uint8_t *kRefImage = &diff_data[0]; // (const uint8_t *)diff_frame.data;

  while (true) {
    //const uint8_t *ptr = diff_frame.ptr<uint8_t>(0);
    //Debug_String("DifferenceFilter diff_frame: ", ptr, 7500);
    float tmp = diff_filt_.fp((const uint8_t *)diff_frame.data, kRefImage);   
    frame_out->diff_confidence = tmp;
#if LOG
    std::cout << "Stream :" << frame_out->video_id << " frame id: " << frame_out->frame_id 
              <<  "   diff_confidence    " << tmp << std::endl;
#endif
    /*default kRefImage oNone */
    if (tmp < lower_diff_thresh_) {
      frame_out->label = oNone;
      frame_out->frame_status = kDiffFiltered;
      processed_.Push(frame_out);
    } else {
      frame_out->frame_status = kDiffUnfiltered;  
      small_input_.Push(frame_out);
    }
    if (!vQueue_->Pop(&frame_out)) {
      //processed_.NoMoreJobs();
      //std::cout << "small_input_ end!" << std::endl;
      small_input_.NoMoreJobs();
      return;
    }
    cv::resize(frame_out->frame, diff_frame, kDiffResol_, 0, 0, cv::INTER_NEAREST);
  }
}

void NoscopeStream::RunSmallCNN() {
  std::vector<noscope::Frame*> input_im;
  std::vector<cv::Mat> dist_frame_f(kMaxCNNImages_);
  std::vector<float> dist_data(kMaxCNNImages_ * kDistFrameSize_);
  noscope::Frame* frame_im;
  noscope::Frame* frame_o;
  cv::Mat dist_frame(kDistResol_, CV_8UC3);
  cv::Mat float_frame(kDistResol_, CV_32FC3);
  bool act_run = true;
  int kImagesToRun;
  
  using namespace tensorflow;
  const float* avg = (const float *) avg_.data;
  //Debug_String("Avg: ", avg, 7500);
  while (act_run) {
    for (int i = 0; i < kMaxCNNImages_; i++) {
      if (!small_input_.Pop(&frame_im)) {
        act_run = false;
        break;
      }
      cv::resize(frame_im->frame, dist_frame, kDistResol_, 0, 0, cv::INTER_NEAREST);
      //cv::resize(frame_im->frame, dist_frame, kDistResol_, 0, 0, cv::INTER_NEAREST);
      //dist_frame_f[i].create(kDistResol_, CV_32FC3);
      dist_frame.convertTo(dist_frame_f[i], CV_32FC3);

      //const uint8_t *ptr = dist_frame_f[i].ptr<uint8_t>(0);
      //std::cout << "Up Stream id: " << stream_id_  << " frame id: " << frame_im->frame_id;
      //Debug_Frame_constuint8("Test", ptr, 1500);
      
      //dist_frame_f[i] = float_frame;
      input_im.push_back(frame_im);
    }
    kImagesToRun = input_im.size();
#if 0
      std::cout << "SmallCNN Push_back : ";
    for (int i = 0; i < kImagesToRun; i++) {
      frame_im =input_im.back();
      input_im.pop_back();
      const uint8_t *ptr = frame_im->frame.ptr<uint8_t>(0);
      std::cout << "I: " << i << "frame id: " <<input_im.at(i)->frame_id;
      for (size_t j = 0; j < 7500 ;) {
        std::cout << ptr[j] << "  ";
        j += 250;
      }
      std::cout << std::endl;
    }
#endif
    Tensor input(DT_FLOAT,
                 TensorShape({kImagesToRun,
                              kDistResol_.height,
                              kDistResol_.width,
                              kNbChannels_}));
    auto input_mapped = input.tensor<float, 4>();
    float *tensor_start = &input_mapped(0, 0, 0, 0);
    for (int i = 0; i < kImagesToRun; i++) {
      float *output = tensor_start + i * kDistFrameSize_;

      //const uint8_t *ptr = dist_frame_f[i].ptr<uint8_t>(0);
      //std::cout << "Under Stream id: " << stream_id_ << " frame id: " << input_im[i]->frame_id;
      //DebugFrame_constuint8("", ptr, 7500);
      
      memcpy(&dist_data[i * kDistFrameSize_], dist_frame_f[i].data, kDistFrameSize_ * sizeof(float));
      //memcpy(&dist_data[i * kDistFrameSize_], float_frame.data, kDistFrameSize_ * sizeof(float));
      //float *input_f = (float*)(dist_frame_f[i].data);
      //DebugFrame_vectorfloat("dist_data: ", dist_data, 7500);
      for (size_t j = 0; j < kDistFrameSize_; j++)
        output[j] = dist_data[i * kDistFrameSize_ + j] / 255. - avg[j];
    }

    std::vector<tensorflow::Tensor> outputs;
    std::vector<std::pair<std::string, tensorflow::Tensor> > inputs = {
      {"input_img", input},
    };

    tensorflow::Status status = session_->Run(inputs, {"output_prob"}, {}, &outputs);

    {
      auto output_mapped = outputs[0].tensor<float, 2>();
      for (int i = 0; i < kImagesToRun; ++i) {
        Status s;
        frame_o = input_im.at(i); 
        frame_o->cnn_difference = output_mapped(i, 1);
#ifdef LOG
        std::cout << "Stream :" << frame_o->video_id << " frame id: " << frame_o->frame_id 
              <<  "   cnn_confidence    " << output_mapped(i, 1) << std::endl;
#endif
#if 0
      const uint8_t *ptr = frame_o->frame.ptr<uint8_t>(0);
      std::cout << "SmallCNN Push : I: " << i;
      for (size_t j = 0; j < 7500 ;) {
        std::cout << ptr[j] << "  ";
        j += 250;
      }
      std::cout << std::endl;
#endif
        if (output_mapped(i, 1) < distill_thresh_lower_) {
          frame_o->label = oNone;
          frame_o->frame_status = kDistillFiltered;
          processed_.Push(frame_o);
        } else if (output_mapped(i, 1) > distill_thresh_upper_) {
          frame_o->label = oHave;
          frame_o->frame_status = kDistillFiltered;
          processed_.Push(frame_o);
        } else {
          frame_o->frame_status = kDistillUnfiltered;
          gQueue_->Push(frame_o);
          //std::cout << "GQueue Push Stream : " << frame_o->video_id << std::endl;
        }
      }
    }

    input_im.clear();
    //dist_frame_f.clear();
  }
  end_ = true;
  gQueue_->Push(frame_o);
  //std::cout << "small_cnn end!" << std::endl;
}

} // namespace noscope
