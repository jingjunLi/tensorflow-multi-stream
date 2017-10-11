#ifndef TENSORFLOW_NOSCOPE_VIDEO_H_
#define TENSORFLOW_NOSCOPE_VIDEO_H_

#include "tensorflow/noscope/common.h"

namespace noscope {

class NoscopeVideo {
  public:
    NoscopeVideo(const VideoParameter& param);
    void SetUp(const size_t stream_id, std::shared_ptr<noscope::SimpleQueue<noscope::Frame*> > fQueue);
    void Start();
    ~NoscopeVideo();
    void Reading();

    std::string video_path_;
    int id_; // stream id for tag 
    std::shared_ptr<noscope::SimpleQueue<noscope::Frame*> > fQueue_;
	  
  private:
/** the thread for reading video frames */
    std::unique_ptr<std::thread> reading_thread_;

    size_t kNbFrames_;
    size_t kSkip_;
    size_t kStart_;
    cv::VideoCapture cap_;
};

} // namespace noscope

#endif
