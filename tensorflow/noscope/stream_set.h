#ifndef TENSORFLOW_NOSCOPE_STREAM_SET_H_
#define TENSORFLOW_NOSCOPE_STREAM_SET_H_

#include "tensorflow/noscope/common.h"
#include "tensorflow/noscope/darknet/src/yolo.h"
#include "tensorflow/noscope/stream.h"
#include <fcntl.h>

namespace noscope {

class StreamSet {
  public:
    StreamSet(const std::string& param_file);
    StreamSet(const StreamSetParameter& param);

    void Init(const StreamSetParameter& in_param);
    void SetUp();
    void Start();
    void RunYOLO();
    void YOLOTEST();
    ~StreamSet();

  private:
    std::string name_;
    yolo::YOLO *yolo_classifier_;

    std::shared_ptr<noscope::SimpleQueue<noscope::Frame*> > sQueue_; // gather frames for yolo
    std::vector<std::shared_ptr<NoscopeStream> > streams_;
    std::vector<std::string> stream_names_;

    static const cv::Size kYOLOResol_; 

    constexpr static size_t kNbChannels_ = 3;
    constexpr static size_t kYOLOFrameSize_ = 416 * 416 * kNbChannels_;   // small cnn
/** the thread for running YOLO */
    std::unique_ptr<std::thread> yolo_thread_;
    bool act_run_;
    int ends_;
    
};

} // namespace noscope

#endif
