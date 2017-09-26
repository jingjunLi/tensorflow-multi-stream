#ifndef TENSORFLOW_VUSE_VUSEDATA_H_
#define TENSORFLOW_VUSE_VUSEDATA_H_

#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "tensorflow/noscope/simple_queue.h"
#include "tensorflow/noscope/mse.h"
#include "tensorflow/noscope/filters.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include <fstream>
#include <thread>

namespace noscope {

class NoscopeData {
 public:
  static const cv::Size kYOLOResol_;
  static const cv::Size kDiffResol_;
  static const cv::Size kDistResol_;

  constexpr static size_t kNbChannels_ = 3;
  constexpr static size_t kYOLOFrameSize_ = 416 * 416 * kNbChannels_;
  constexpr static size_t kDiffFrameSize_ = 100 * 100 * kNbChannels_;
  constexpr static size_t kDistFrameSize_ = 50 * 50 * kNbChannels_;

  const size_t kNbFrames_;
  const size_t kSkip_;

  std::vector<uint8_t> yolo_data_;
  std::vector<uint8_t> diff_data_;
  std::vector<float> dist_data_;

  NoscopeData(const std::string& fname, const size_t kSkip, const size_t kNbFrames, const size_t kStart);

  NoscopeData(const std::string& fname);

  void DumpAll(const std::string& fname);
};

enum Status {
  kUnprocessed,
  kSkipped,
  kDiffFiltered,
  kDiffUnfiltered,
  kDistillFiltered,
  kDistillUnfiltered,
  kYoloLabeled
};

enum Object {
  oNone,
  oPerson,
  oCar,
  oBus,
  oBoat
};

struct Frame {
  cv::Mat frame;
  cv::Mat yolo_frame;
  cv::Mat diff_frame;
  cv::Mat dist_frame_f;
  Status frame_status;
  Object label;
  int frame_id;
  int video_id;
  float diff_confidence;
  float cnn_difference;
  float yolo_confidence; 
};

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

class NoscopeStream {
  public:
    NoscopeStream(const std::string& fname, const size_t kSkip, const size_t kNbFrames, const size_t kStart,
            	tensorflow::Session *session,
                const std::string& avg_fname, std::shared_ptr<noscope::SimpleQueue<noscope::Frame *> > gQueue,
			    const size_t stream_id, const bool kUseBlocked, const bool kSkipDiffDetection	
                );
    void RunDifferenceFilter(const float lower_thresh, const float upper_thresh,
                           const bool const_ref, const size_t kRef);

    void PopulateCNNFrames();

    void RunSmallCNN(const float lower_thresh, const float upper_thresh);

    float lower_thresh_;
    float upper_thresh_;
    float const_ref_;
    float kRef_;
    const size_t stream_id_;

  private:
    noscope::filters::DifferenceFilter diff_filt_;
	std::unique_ptr<noscope::NoscopeVideo> reader_;
	
	cv::Mat avg_;
	tensorflow::Session *session_;

    std::unique_ptr<noscope::SimpleQueue<noscope::Frame*> > vQueue_;

/** the thread for running differenceFilter */
    std::unique_ptr<std::thread> diff_thread_;

};



} // namespace noscope

#endif
