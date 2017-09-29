#include "tensorflow/noscope/noscope_video.h"

namespace noscope {

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
    cap_ >> frame;

    // 每 KSkip_ 帧 读取 1 帧 
    if (i % kSkip_ == 0) {
      frame_in = new noscope::Frame();
      frame_in->frame = frame;
      frame_in->frame_status = kUnprocessed; 
      frame_in->frame_id = i;
      frame_in->video_id = id_; 
      //const size_t ind = i / kSkip_;
      // 将同一个frame 转换成 YOLO， difference 和 dist data 
      // dist_frame_f ??? 用途 dist_frame 每个像素用 8 位无符号整形表示，转换成每个像素用32位浮点型
      //dist_frame.convertTo(dist_frame_f, CV_32FC3);

      //if (!frame_in->diff_frame.isContinuous()) {
      //  throw std::runtime_error("diff frame is not continuous");
      //}

      vQueue_->Push(frame_in);
      std::cout << "push " << i << std::endl;
    }
  }
  vQueue_->NoMoreJobs();
}

} // namespace noscope
