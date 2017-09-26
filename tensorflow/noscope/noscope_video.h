#ifndef TENSORFLOW_NOSCOPE_VIDEO_H_
#define TENSORFLOW_NOSCOPE_VIDEO_H_

#include "tensorflow/noscope/common.h"

namespace noscope {

class NoscopeVideo {
  public:
    NoscopeVideo(const std::string& fname, const size_t kSkip, const size_t kNbFrames,
            const size_t kStart,
            std::shared_ptr<noscope::SimpleQueue<noscope::Frame*> > vQueue,
            const size_t id
            );

    ~NoscopeVideo();
    void Reading();
    
	static const cv::Size kDiffResol_;
    static const cv::Size kDistResol_;
    constexpr static size_t kNbChannels_ = 3;

    constexpr static size_t kDiffFrameSize_ = 100 * 100 * kNbChannels_;
    constexpr static size_t kDistFrameSize_ = 50 * 50 * kNbChannels_;
	  
  private:
/** the thread for reading video frames */
    std::unique_ptr<std::thread> reading_thread_;

    const size_t kNbFrames_;
    const size_t kSkip_;
    const size_t kStart_;

    std::string video_path_;
    int id_;
    cv::VideoCapture cap_;

    std::shared_ptr<noscope::SimpleQueue<noscope::Frame*> > vQueue_;
};

} // namespace noscope

#endif
