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
			    const size_t stream_id, const bool kUseBlocked, const bool kSkipDiffDetection,
                const float lower_thresh, const float upper_thresh, const bool const_ref, const size_t kRef,
                const float distill_thresh_lower, const float distill_thresh_upper
                );
    ~NoscopeStream();
    void RunDifferenceFilter();
    void PopulateCNNFrames();

    void RunSmallCNN();

    const float lower_thresh_;
    const float upper_thresh_;
    const float const_ref_;
    const float kRef_;
    const size_t stream_id_;
    const float distill_thresh_lower_;
    const float distill_thresh_upper_;

    constexpr static size_t kNbChannels_ = 3;

    constexpr static size_t kDiffFrameSize_ = 100 * 100 * kNbChannels_;
    constexpr static size_t kDistFrameSize_ = 50 * 50 * kNbChannels_;
    static const cv::Size kDiffResol_;
    static const cv::Size kDistResol_;

  private:
    noscope::filters::DifferenceFilter diff_filt_;
	noscope::NoscopeVideo* reader_;
	
	cv::Mat avg_;
	tensorflow::Session *session_;

    std::shared_ptr<noscope::SimpleQueue<noscope::Frame*> > vQueue_;

/** the thread for running differenceFilter */
    std::unique_ptr<std::thread> diff_thread_;
/** the thread for running Small CNN */
    std::unique_ptr<std::thread> small_cnn_thread_;

    noscope::SimpleQueue<noscope::Frame*> processed_;
    noscope::SimpleQueue<noscope::Frame*> small_input_;
    
    constexpr static size_t kMaxCNNImages_ = 64;
};

} // namespace noscope

#endif
