#ifndef TENSORFLOW_NOSCOPE_COMMON_H_
#define TENSORFLOW_NOSCOPE_COMMON_H_

#include <stdlib.h>
#include <x86intrin.h>

#include <string>
#include <fstream>
#include <thread>

#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"

#include "tensorflow/noscope/util/simple_queue.h"
#include "tensorflow/noscope/mse.h"
#include "tensorflow/noscope/filters.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"

// ok C++
namespace noscope {

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
  Status frame_status;
  Object label;
  int frame_id;
  int video_id;
  float diff_confidence;
  float cnn_difference;
  float yolo_confidence; 
};

} // namespace noscope

#endif  // TENSORFLOW_NOSCOPE_COMMON_H_
