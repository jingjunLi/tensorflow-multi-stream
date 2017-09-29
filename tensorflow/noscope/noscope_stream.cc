#include "tensorflow/noscope/noscope_stream.h"

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

NoscopeStream::NoscopeStream(const std::string& fname, const size_t kSkip, const size_t kNbFrames,
            const size_t kStart,
            tensorflow::Session *session,
            const std::string& avg_fname, std::shared_ptr<noscope::SimpleQueue<noscope::Frame*> > gQueue,
            const size_t stream_id, const bool kUseBlocked, const bool kSkipDiffDetection, 
            const float lower_thresh, const float upper_thresh, const bool const_ref, const size_t kRef,
            const float distill_thresh_lower, const float distill_thresh_upper) :
    lower_thresh_(lower_thresh),
    upper_thresh_(upper_thresh),
    const_ref_(const_ref),
    kRef_(kRef),
    stream_id_(stream_id),
    distill_thresh_lower_(distill_thresh_lower),
    distill_thresh_upper_(distill_thresh_upper),
    avg_(kDistResol_, CV_32FC3),
    session_(session),
    vQueue_(gQueue) {
  std::ifstream is(avg_fname);
  std::istream_iterator<float> start(is), end;
  std::vector<float> nums(start, end);
  if (nums.size() != kDistFrameSize_) {
    throw std::runtime_error("nums not right size");
  }
  // 将 帧 平均值 拷贝到 avg 的矩阵中 
  memcpy(avg_.data, &nums[0], kDistFrameSize_ * sizeof(float));

  diff_filt_ = noscope::GetDiffFilter(kUseBlocked, kSkipDiffDetection);
  reader_ = new noscope::NoscopeVideo(fname, kSkip, kNbFrames, kStart, gQueue, stream_id);

  diff_thread_ = std::unique_ptr<std::thread>(
          new std::thread(&NoscopeStream::RunDifferenceFilter, this));

  //small_cnn_thread_ = std::unique_ptr<std::thread>(
  //        new std::thread(&NoscopeStream::RunSmallCNN, this));
}

NoscopeStream::~NoscopeStream() {
    diff_thread_->join();
}

void NoscopeStream::RunDifferenceFilter() {
  noscope::Frame* frame_out;
  cv::Mat diff_frame(kDiffResol_, CV_8UC3);
  std::vector<uint8_t> diff_data(kDiffFrameSize_);

  vQueue_->Pop(&frame_out);
  cv::resize(frame_out->frame, diff_frame, kDiffResol_, 0, 0, cv::INTER_NEAREST);
  memcpy(&diff_data[0], diff_frame.data, kDiffFrameSize_);

  const uint8_t *kRefImage = &diff_data[0]; // (const uint8_t *)diff_frame.data;

  while (true) {
    float tmp = diff_filt_.fp((const uint8_t *)diff_frame.data, kRefImage);   
    frame_out->diff_confidence = tmp;
    std::cout << "frame id: " << frame_out->frame_id <<  "   diff_confidence    " << tmp << std::endl;

    if (tmp < lower_thresh_) {
      frame_out->label = oNone;
      frame_out->frame_status = kDiffFiltered;
      processed_.Push(frame_out);
    } else {
      frame_out->frame_status = kDiffUnfiltered;  
      small_input_.Push(frame_out);
    }
    if (!vQueue_->Pop(&frame_out)) {
      processed_.NoMoreJobs();
      small_input_.NoMoreJobs();
      return;
    }
    cv::resize(frame_out->frame, diff_frame, kDiffResol_, 0, 0, cv::INTER_NEAREST);
  }
}

//tensorflow::Tensor NoscopeStream::PopulateCNNFrames() {
//
//}

void NoscopeStream::RunSmallCNN() {
#if 0
  std::vector<noscope::Frame*> input_im;
  noscope::Frame* frame_im;
  cv::Mat dist_frame(kDistResol_, CV_8UC3);
  
  using namespace tensorflow;
  for (size_t i = 0; i < kMaxCNNImages_; i++) {
    if (!small_input_.Pop(&frame_im)) {
        break;
    }
    cv::resize(frame_im->frame, dist_frame, kDistResol_, 0, 0, cv::INTER_NEAREST);
    //frame_im->dist_frame_f.create(kDistResol_, CV_32FC3);
    dist_frame.convertTo(frame_im->dist_frame_f, CV_32FC3);
    input_im.push_back(frame_im);
  }
  const size_t kImagesToRun = input_im.size();
  const float* avg = (float *) avg_.data;
  Tensor input(DT_FLOAT,
               TensorShape({kImagesToRun,
                            kDistResol_.height,
                            kDistResol_.width,
                            kNbChannels_}));
  auto input_mapped = input.tensor<float, 4>();
  float *tensor_start = &input_mapped(0, 0, 0, 0);
  for (size_t i = 0; i < kImagesToRun; i++) {
    float *output = tensor_start + i * kDistFrameSize_;
    const float *input = input_im[i]->dist_frame_f.data;
    for (size_t j = 0; j < kDistFrameSize_; j++)
      output[j] = output[j] / 255. - avg[j];
  }
#endif
}

} // namespace noscope
