#include "tensorflow/noscope/stream_set.h"

namespace noscope {

const cv::Size StreamSet::kYOLOResol_(416, 416);

StreamSet::StreamSet(const std::string& param_file) {
  StreamSetParameter param;
  ReadProtoFromTextFile(param_file.c_str() , &param);
  Init(param); 
}

StreamSet::StreamSet(const StreamSetParameter& param) {
  Init(param);
}

StreamSet::~StreamSet() {
  yolo_thread_->join();
}

void StreamSet::Init(const StreamSetParameter& param) {
  name_ = param.name();
  sQueue_ = std::shared_ptr<noscope::SimpleQueue<noscope::Frame* > >(new noscope::SimpleQueue<noscope::Frame*>); 
  //streams_.resize(param.stream_size());
  //std::cout << "Streams Init , nums: " << streams_.size() << "params : " << param.stream_size() << std::endl;
  for (int stream_id = 0; stream_id < param.stream_size(); ++stream_id) {
    const StreamParameter& stream_param = param.stream(stream_id);
    auto noscope_stream = std::shared_ptr<NoscopeStream>(new NoscopeStream(stream_param)); 
    streams_.push_back(noscope_stream);
    noscope_stream->SetUp(stream_id, sQueue_);
    stream_names_.push_back(stream_param.name());
  }

  yolo_classifier_ = new yolo::YOLO(param.yolo_cfg(), param.yolo_weights(), param.yolo_class());
  act_run_ = true;
  ends_ = 0;

}

void StreamSet::SetUp() {

}

void StreamSet::Start() {
  std::cout << "Streams Start , nums: " << streams_.size() << std::endl;
  for (size_t stream_id = 0; stream_id < streams_.size(); ++stream_id) {
    //streams_[stream_id].get()->Start();
    streams_[stream_id]->Start();
  }

  yolo_thread_ = std::unique_ptr<std::thread>(
          new std::thread(&StreamSet::RunYOLO, this));
}

static image ipl_to_image(IplImage* src) {
  unsigned char *data = (unsigned char *)src->imageData;
  // float *data = (float *) src->imageData;
  int h = src->height;
  int w = src->width;
  int c = src->nChannels;
  int step = src->widthStep;// / sizeof(float);
  image out = make_image(w, h, c);
  int count = 0;

  for (int k = 0; k < c; ++k) {
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        out.data[count++] = data[i*step + j*c + (2 - k)] / 255.;
        // out.data[count++] = data[i*step + j*c + k] / 255.;
      }
    }
  }
  return out;
}

void StreamSet::RunYOLO() {
  if (act_run_) {
    noscope::Frame* frame_out;
    cv::Mat yolo_frame(kYOLOResol_, CV_8UC3);
    std::vector<uint8_t> yolo_data(kYOLOFrameSize_);
    float yolo_confidence;
    while(sQueue_->Pop(&frame_out)) {
      cv::resize(frame_out->frame, yolo_frame, kYOLOResol_, 0, 0, cv::INTER_NEAREST);
     // memcpy(&yolo_data[0], yolo_frame.data, kYOLOFrameSize_);
     // 
     // cv::Mat cpp_frame(kYOLOResol_, CV_8UC3, 
     //                   const_cast<uint8_t *>(&yolo_data[0]));
      IplImage frame = yolo_frame;
      image yolo_f = ipl_to_image(&frame);
      yolo_confidence = yolo_classifier_->LabelFrame(yolo_f);
      frame_out->yolo_confidence = yolo_confidence;
      free_image(yolo_f);
      frame_out->frame_status = kYoloLabeled; 
      Object cls = (yolo_confidence > 0) ? oHave : oNone;
      frame_out->label = cls;
      std::cout << "Stream :" << frame_out->video_id << " frame_id: " << frame_out->frame_id 
          << " yolo_confidence: " << yolo_confidence << std::endl;
      ends_ = 0;
      for (int i = 0; i < streams_.size(); ++i) {
        if (streams_[i]->IsEnd())
          ends_++;
      }
      if (ends_== streams_.size() )
        sQueue_->NoMoreJobs();
    }
  }
}

} // namespace noscope
