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
    StreamParameter stream_param = param.stream(stream_id);
    auto noscope_stream = std::shared_ptr<NoscopeStream>(new NoscopeStream(stream_param)); 
    noscope_stream->SetUp(stream_id, sQueue_);
    streams_.push_back(noscope_stream);
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

image ipl_to_image_set(IplImage* src) {
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
  bool nojobs = false;
  if (act_run_) {
    noscope::Frame* frame_out;
    cv::Mat yolo_frame(kYOLOResol_, CV_8UC3);
    std::vector<uint8_t> yolo_data(kYOLOFrameSize_);
    float yolo_confidence;
    while(sQueue_->Pop(&frame_out)) {
#if 0
      std::cout << "YOLO Pop: ";
      const uint8_t *ptr = frame_out->frame.ptr<uint8_t>(0);
      for (size_t j = 0; j < 7500 ;) {
        std::cout << ptr[j] << "  ";
        j += 250;
      }
      std::cout << std::endl;
#endif
      cv::resize(frame_out->frame, yolo_frame, kYOLOResol_, 0, 0, cv::INTER_NEAREST);
      IplImage frame = yolo_frame;
      image yolo_f = ipl_to_image_set(&frame);
      yolo_confidence = yolo_classifier_->LabelFrame(yolo_f);
      frame_out->yolo_confidence = yolo_confidence;
      free_image(yolo_f);
      frame_out->frame_status = kYoloLabeled; 
      Object cls = (yolo_confidence > 0) ? oHave : oNone;
      frame_out->label = cls;
      std::cout << "Stream :" << frame_out->video_id << " frame_id: " << frame_out->frame_id 
          << " yolo_confidence: " << yolo_confidence << std::endl;
      if (!nojobs) {
        int sn = 0;
        for (int i = 0; i < streams_.size(); ++i) {
          if (streams_[i]->IsEnd()) {
            //std::cout << "Stream: " << i << " end!" << std::endl;
            sn++;
          }
        }
        if (sn == streams_.size()) {
          nojobs = true;
          sQueue_->NoMoreJobs();
        }
      }
      //ends_ = 0;
      //for (int i = 0; i < streams_.size(); ++i) {
      //  if (streams_[i]->IsEnd())
      //    ends_++;
      //}
      //if (ends_ == streams_.size() ) {
      //  std::cout << "End!" << std::endl;
      //  sQueue_->NoMoreJobs();
      //}
    }
  }
}

void StreamSet::YOLOTEST() {
  cv::VideoCapture cap;
  cv::Mat frame;
  noscope::Frame* frame_out;
  cv::Mat yolo_frame(kYOLOResol_, CV_8UC3);
  std::vector<uint8_t> yolo_data(kYOLOFrameSize_);
  float yolo_confidence;
  cap.open("/home/li/opensource/stanford-futuredata/data/videos/jackson-town-square.mp4");
  for (int i = 0; i < 100; i++) {
    cap >> frame;
      cv::resize(frame, yolo_frame, kYOLOResol_, 0, 0, cv::INTER_NEAREST);
     // memcpy(&yolo_data[0], yolo_frame.data, kYOLOFrameSize_);
     // 
     // cv::Mat cpp_frame(kYOLOResol_, CV_8UC3, 
     //                   const_cast<uint8_t *>(&yolo_data[0]));
      IplImage frame_i = yolo_frame;
      image yolo_f = ipl_to_image_set(&frame_i);
      yolo_confidence = yolo_classifier_->LabelFrame(yolo_f);
      free_image(yolo_f);
      std::cout << "i : " << i << "  confidence: " << yolo_confidence << std::endl ;
  }
}

} // namespace noscope
