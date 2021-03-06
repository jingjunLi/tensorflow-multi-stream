#include <sys/mman.h>

#include <chrono>
#include <ctime>
#include <random>
#include <algorithm>
#include <iterator>
#include <memory>

#include "tensorflow/core/util/command_line_flags.h"

#include "tensorflow/noscope/MemoryTests.h"
#include "tensorflow/noscope/noscope_labeler.h"
#include "tensorflow/noscope/noscope_data.h"
#include "tensorflow/noscope/darknet/src/yolo.h"
#include "tensorflow/noscope/noscope_data.h"
#include "tensorflow/noscope/noscope_video.h"

int main(int argc, char* argv[]) {
#if 0
  std::vector<std::shared_ptr<noscope::SimpleQueue<noscope::Frame*> > > gQueues;
  std::shared_ptr<noscope::SimpleQueue<noscope::Frame* > > fQueue(new noscope::SimpleQueue<noscope::Frame*>);
  noscope::NoscopeVideo reader("/home/li/opensource/stanford-futuredata/data/videos/jackson-town-square.mp4", 3, 100, 5, fQueue, 0);
#endif

  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  return 0;
}
