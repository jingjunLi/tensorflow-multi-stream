#ifndef TENSORFLOW_NOSCOPE_STREAM_H_
#define TENSORFLOW_NOSCOPE_STREAM_H_

#include "tensorflow/noscope/common.h"
#include "tensorflow/noscope/noscope_video.h"

namespace noscope {

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
