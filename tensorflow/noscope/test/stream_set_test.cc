#include <sys/mman.h>
#include <chrono>
#include <ctime>
#include <random>
#include <algorithm>
#include <iterator>
#include <memory>

#include "tensorflow/core/util/command_line_flags.h"

#include "tensorflow/noscope/MemoryTests.h"
//#include "tensorflow/noscope/noscope_labeler.h"
#include "tensorflow/noscope/noscope_data.h"
#include "tensorflow/noscope/darknet/src/yolo.h"
#include "tensorflow/noscope/noscope_data.h"
#include "tensorflow/noscope/stream_set.h"

#define LOG 1
using tensorflow::Flag;

int main(int argc, char* argv[]) {
  //auto sQueue = std::shared_ptr<noscope::SimpleQueue<noscope::Frame* > >(new noscope::SimpleQueue<noscope::Frame*>); 
  //noscope::StreamSet stream_set("/home/li/opensource/stanford-futuredata/tensorflow-noscope/tensorflow/noscope/proto/stream_set.prototxt"); 
  std::string prototxt(argv[1]);
  noscope::StreamSet stream_set(prototxt); 
  stream_set.Start();

  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  return 0;
}
