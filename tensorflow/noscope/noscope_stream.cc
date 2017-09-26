#include "tensorflow/noscope/noscope_stream.h"

namespace noscope {

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
            const size_t stream_id, const bool kUseBlocked, const bool kSkipDiffDetection) :
    avg_(NoscopeVideo::kDistResol_, CV_32FC3),
    session_(session),
    stream_id_(stream_id) {
  std::ifstream is(avg_fname);
  std::istream_iterator<float> start(is), end;
  std::vector<float> nums(start, end);
  if (nums.size() != NoscopeVideo::kDistFrameSize_) {
    throw std::runtime_error("nums not right size");
  }
  // 将 帧 平均值 拷贝到 avg 的矩阵中 
  memcpy(avg_.data, &nums[0], NoscopeVideo::kDistFrameSize_ * sizeof(float));

  diff_filt_ = noscope::GetDiffFilter(kUseBlocked, kSkipDiffDetection);
  //reader_ = new noscope::NoscopeVideo();

}

} // namespace noscope
