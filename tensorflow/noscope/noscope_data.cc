#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"

#include "tensorflow/noscope/noscope_data.h"

namespace noscope {

const cv::Size NoscopeData::kYOLOResol_(416, 416);
const cv::Size NoscopeData::kDiffResol_(100, 100);
const cv::Size NoscopeData::kDistResol_(50, 50);

const cv::Size NoscopeVideo::kDiffResol_(100, 100);
const cv::Size NoscopeVideo::kDistResol_(50, 50);

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

NoscopeData::NoscopeData(const std::string& fname,
                   const size_t kSkip, const size_t kNbFrames, const size_t kStart) :
    kNbFrames_(kNbFrames / kSkip),
    kSkip_(kSkip),
    yolo_data_(kYOLOFrameSize_ * kNbFrames_),
    diff_data_(kDiffFrameSize_ * kNbFrames_),
    dist_data_(kDistFrameSize_ * kNbFrames_) {
  cv::VideoCapture cap(fname);
  if (kStart > 0)
    cap.set(cv::CAP_PROP_POS_FRAMES, kStart - 1);

  cv::Mat frame;
  cv::Mat yolo_frame(NoscopeData::kYOLOResol_, CV_8UC3);
  cv::Mat diff_frame(NoscopeData::kDiffResol_, CV_8UC3);
  cv::Mat dist_frame(NoscopeData::kDistResol_, CV_8UC3);
  cv::Mat dist_frame_f(NoscopeData::kDistResol_, CV_32FC3);
  for (size_t i = 0; i < kNbFrames; i++) {
    cap >> frame;
    if (i % kSkip_ == 0) {
      const size_t ind = i / kSkip_;
      cv::resize(frame, yolo_frame, NoscopeData::kYOLOResol_, 0, 0, cv::INTER_NEAREST);
      cv::resize(frame, diff_frame, NoscopeData::kDiffResol_, 0, 0, cv::INTER_NEAREST);
      cv::resize(frame, dist_frame, NoscopeData::kDistResol_, 0, 0, cv::INTER_NEAREST);
      dist_frame.convertTo(dist_frame_f, CV_32FC3);

      if (!yolo_frame.isContinuous()) {
        throw std::runtime_error("yolo frame is not continuous");
      }
      if (!diff_frame.isContinuous()) {
        throw std::runtime_error("diff frame is not continuous");
      }
      if (!dist_frame.isContinuous()) {
        throw std::runtime_error("dist frame is not conintuous");
      }
      if (!dist_frame_f.isContinuous()) {
        throw std::runtime_error("dist frame f is not continuous");
      }

      memcpy(&yolo_data_[ind * kYOLOFrameSize_], yolo_frame.data, kYOLOFrameSize_);
      memcpy(&diff_data_[ind * kDiffFrameSize_], diff_frame.data, kDiffFrameSize_);
      memcpy(&dist_data_[ind * kDistFrameSize_], dist_frame_f.data, kDistFrameSize_ * sizeof(float));
    }
  }
}

static std::ifstream::pos_type filesize(const std::string& fname) {
  std::ifstream in(fname, std::ifstream::ate | std::ifstream::binary);
  return in.tellg();
}
NoscopeData::NoscopeData(const std::string& fname) :
    kNbFrames_(filesize(fname) / (kYOLOFrameSize_ + kDiffFrameSize_ + kDistFrameSize_* sizeof(float))),
    kSkip_(1),
    yolo_data_(kYOLOFrameSize_ * kNbFrames_),
    diff_data_(kDiffFrameSize_ * kNbFrames_),
    dist_data_(kDistFrameSize_ * kNbFrames_) {
  std::cerr << kNbFrames_ << "\n";
  std::ifstream in(fname, std::ifstream::binary);
  in.read((char *) &yolo_data_[0], yolo_data_.size());
  in.read((char *) &diff_data_[0], diff_data_.size());
  in.read((char *) &dist_data_[0], dist_data_.size() * sizeof(float));
}

void NoscopeData::DumpAll(const std::string& fname) {
  std::cerr << "Dumping " << kNbFrames_ << "\n";
  std::ofstream fout(fname, std::ios::binary | std::ios::out);
  fout.write((char *) &yolo_data_[0], yolo_data_.size());
  fout.write((char *) &diff_data_[0], diff_data_.size());
  fout.write((char *) &dist_data_[0], dist_data_.size() * sizeof(float));
}

NoscopeVideo::NoscopeVideo(const std::string& fname, const size_t kSkip, const size_t kNbFrames,
            const size_t kStart,
            std::shared_ptr<noscope::SimpleQueue<noscope::Frame *> > vQueue, const size_t id
            ) :
    kNbFrames_(kNbFrames),
    kSkip_(kSkip),
    kStart_(kStart),
    video_path_(fname),
    id_(id),
    cap_(fname),
    vQueue_(vQueue) {
  if (kStart_ > 0)
      cap_.set(cv::CAP_PROP_POS_FRAMES, kStart_ - 1);
  reading_thread_ = std::unique_ptr<std::thread>(
          new std::thread(&NoscopeVideo::Reading, this));

}

NoscopeVideo::~NoscopeVideo() {
    reading_thread_->join();
}

void NoscopeVideo::Reading() {
  cv::Mat frame; 
  noscope::Frame* frame_in;
  for (size_t i = 0; i < kNbFrames_; i++) {
    frame_in = new noscope::Frame();
    cap_ >> frame_in->frame;
    // 每 KSkip_ 帧 读取 1 帧 
    if (i % kSkip_ == 0) {
      //const size_t ind = i / kSkip_;
      // 将同一个frame 转换成 YOLO， difference 和 dist data 
      cv::resize(frame_in->frame, frame_in->diff_frame, NoscopeVideo::kDiffResol_, 0, 0, cv::INTER_NEAREST);
      // dist_frame_f ??? 用途 dist_frame 每个像素用 8 位无符号整形表示，转换成每个像素用32位浮点型
      //dist_frame.convertTo(dist_frame_f, CV_32FC3);

      if (!frame_in->diff_frame.isContinuous()) {
        throw std::runtime_error("diff frame is not continuous");
      }

      vQueue_->Push(frame_in);
      std::cout << "push " << i << std::endl;
    }
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
  memcpy(avg_.data, &nums[0], NoscopeData::kDistFrameSize_ * sizeof(float));

  diff_filt_ = noscope::GetDiffFilter(kUseBlocked, kSkipDiffDetection);
  //reader_ = new noscope::NoscopeVideo();

}

} // namespace noscope
