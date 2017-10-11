#ifndef TENSORFLOW_NOSCOPE_STREAM_H_
#define TENSORFLOW_NOSCOPE_STREAM_H_

#include "tensorflow/noscope/common.h"
#include "tensorflow/noscope/video.h"
#include <fcntl.h>

namespace noscope {

using google::protobuf::Message;
bool ReadProtoFromTextFile(const char* filename, Message* proto); 

template <typename Dtype>
void Debug_String(const std::string& str, const Dtype* ptr, int N);

class NoscopeStream {
  public:
    NoscopeStream(const std::string& param_file);
    NoscopeStream(const StreamParameter& param);
    void Init(const StreamParameter& in_param);
    void SetUp(const size_t stream_id, std::shared_ptr<noscope::SimpleQueue<noscope::Frame*> > gQueue);
	void InitSession();
    void Start();
    void RunDifferenceFilter();
    bool IsEnd();

    void RunSmallCNN();
    ~NoscopeStream();

    constexpr static size_t kNbChannels_ = 3;

    constexpr static size_t kDiffFrameSize_ = 100 * 100 * kNbChannels_; // difference filters 
    constexpr static size_t kDistFrameSize_ = 50 * 50 * kNbChannels_;   // small cnn
    static const cv::Size kDiffResol_;
    static const cv::Size kDistResol_;

    size_t stream_id_;
    std::shared_ptr<noscope::SimpleQueue<noscope::Frame*> > gQueue_; // gather frames for yolo
	noscope::NoscopeVideo* reader_;

  private:
    std::string graph_;
    std::string name_;
    float lower_diff_thresh_;      // differenceFilter thresh
    float upper_diff_thresh_;
    float distill_thresh_lower_;   // small cnn thresh
    float distill_thresh_upper_; 
    bool skip_diff_detection_;
    bool skip_small_cnn_;
    bool const_ref_;    // unused 
    size_t kRef_;       // unused 
    noscope::filters::DifferenceFilter diff_filt_;
	
	cv::Mat avg_;
	tensorflow::Session *session_;

    std::shared_ptr<noscope::SimpleQueue<noscope::Frame*> > vQueue_; // reading video frames

/** the thread for running differenceFilter */
    std::unique_ptr<std::thread> diff_thread_;
/** the thread for running Small CNN */
    std::unique_ptr<std::thread> small_cnn_thread_;

    noscope::SimpleQueue<noscope::Frame*> processed_;
    noscope::SimpleQueue<noscope::Frame*> small_input_;
    
    constexpr static size_t kMaxCNNImages_ = 64; // small cnn batchsize
    bool end_;
};

} // namespace noscope

#endif
