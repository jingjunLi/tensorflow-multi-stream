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
#include "tensorflow/noscope/noscope_stream.h"

using tensorflow::Flag;

static bool file_exists(const std::string& name) {
  std::ifstream f(name.c_str());
  return f.good();
}

static void SpeedTests() {
  const size_t kImgSize = 100 * 100 * 3;
  const size_t kDelay = 10;
  const size_t kFrames = 100000;
  const size_t kNumThreads = 32;
  std::vector<uint8_t> speed_tests(kFrames * kImgSize);
  {
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(kNumThreads)
    for (size_t i = kDelay; i < kFrames; i++) {
      noscope::filters::GlobalMSE(&speed_tests[i * kImgSize], &speed_tests[(i - kDelay) * kImgSize]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "BlockedMSE: " << diff.count() << " s" << std::endl;
  }
  {
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(kNumThreads)
    for (size_t i = kDelay; i < kFrames; i++) {
      noscope::filters::BlockedMSE(&speed_tests[i * kImgSize], &speed_tests[100 * kImgSize]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "GlobalMSE: " << diff.count() << " s" << std::endl;
  }
}

static tensorflow::Session* InitSession(const std::string& graph_fname) {
  tensorflow::Session *session;
  tensorflow::SessionOptions opts;
  tensorflow::GraphDef graph_def;
  // YOLO needs some memory
  // opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.8);
   opts.config.mutable_gpu_options()->set_allow_growth(true);
  tensorflow::Status status = NewSession(opts, &session);
  TF_CHECK_OK(status);

  status = tensorflow::ReadBinaryProto(
      tensorflow::Env::Default(),
      graph_fname, &graph_def);
  tensorflow::graph::SetDefaultDevice("/gpu:0", &graph_def);
  TF_CHECK_OK(status);

  status = session->Create(graph_def);
  TF_CHECK_OK(status);

  return session;
}

static noscope::NoscopeData* LoadVideo(const std::string& video, const std::string& dumped_videos,
                                 const int kSkip, const int kNbFrames, const int kStartFrom) {
  auto start = std::chrono::high_resolution_clock::now();
  noscope::NoscopeData *data = NULL;
  if (dumped_videos == "/dev/null") {
    data = new noscope::NoscopeData(video, kSkip, kNbFrames, kStartFrom);
  } else {
    if (file_exists(dumped_videos)) {
      std::cerr << "Loading dumped video\n";
      data = new noscope::NoscopeData(dumped_videos);
    } else {
      std::cerr << "Dumping video\n";
      data = new noscope::NoscopeData(video, kSkip, kNbFrames, kStartFrom);
      data->DumpAll(dumped_videos);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Loaded video\n";
  std::chrono::duration<double> diff = end - start;
  std::cout << "Time to load (and resize) video: " << diff.count() << " s" << std::endl;
  return data;
}


int main(int argc, char* argv[]) {

  std::string video_name("/home/li/opensource/stanford-futuredata/data/videos/jackson-town-square.mp4"); 
  const size_t kSkip = 3;
  const size_t kNbFrames = 100;
  const size_t kStartFrom = 5;
  std::string graph("/home/li/opensource/stanford-futuredata/data/cnn-models/jackson-town-square_convnet_128_32_0.pb");
  std::string avg_name("/home/li/opensource/stanford-futuredata/data/cnn-avg/jackson-town-square.txt");
  std::shared_ptr<noscope::SimpleQueue<noscope::Frame* > > fQueue(new noscope::SimpleQueue<noscope::Frame*>);
  const size_t stream_id = 0;
  const bool kUseBlocked = 0;
  const bool kSkipDiffDetection = 0;
  const float lower_thresh = 0.0;
  const float upper_thresh = 10000000;
  const bool const_ref = 0;
  const size_t kRef = 0;
  const float distill_thresh_lower = 0.197155194194;
  const float distill_thresh_upper = 0.622490004004;
  
  tensorflow::Session *session = InitSession(graph);
  
  noscope::NoscopeStream stream_test(video_name, kSkip, kNbFrames, kStartFrom, 
          session, avg_name, fQueue, stream_id, kUseBlocked, kSkipDiffDetection, lower_thresh,
          upper_thresh, const_ref, kRef, distill_thresh_lower, distill_thresh_upper);

  std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  //std::vector<std::unique_ptr<noscope::NoscopeStream > > nStreams;
  //for (int i = 0; i < video_num; ++i) {
      //auto gQueue = std::shared_ptr<noscope::SimpleQueue<noscope::Frame*> >(new noscope::SimpleQueue<noscope::Frame *>);
      //std::shared_ptr<noscope::SimpleQueue<noscope::Frame* > > gQueue(new noscope::SimpleQueue<noscope::Frame* >());
      //gQueues.push_back(gQueue);

      //std::unique_ptr<noscope::NoscopeStream > nStream(new noscope::SimpleQueue<noscope::NoscopeStream>());
      //nStreams.push_back(nStream);
  //}
#if 0  
  noscope::NoscopeData *data = LoadVideo(video, dumped_videos, kSkip, kNbFrames, kStartFrom);
  noscope::filters::DifferenceFilter df = GetDiffFilter(kUseBlocked, kSkipDiffDetection);

  noscope::NoscopeLabeler labeler = noscope::NoscopeLabeler(
      session,
      yolo_classifier,
      df,
      avg_fname,
      *data);

  std::cerr << "Loaded NoscopeLabeler\n";

  auto start = std::chrono::high_resolution_clock::now();
  labeler.RunDifferenceFilter(diff_thresh, 10000000, kUseBlocked, kRefImage);
  auto diff_end = std::chrono::high_resolution_clock::now();
  if (!kSkipSmallCNN) {
    labeler.PopulateCNNFrames();
    labeler.RunSmallCNN(distill_thresh_lower, distill_thresh_upper);
  }
  auto dist_end = std::chrono::high_resolution_clock::now();
  labeler.RunYOLO(true);
  auto yolo_end = std::chrono::high_resolution_clock::now();
  std::vector<double> runtimes(4);
  {
    std::chrono::duration<double> diff = yolo_end - start;
    std::cout << "Total time: " << diff.count() << " s" << std::endl;

    diff = diff_end - start;
    runtimes[0] = diff.count();
    diff = dist_end - start;
    runtimes[1] = diff.count();
    diff = yolo_end - start;
    runtimes[2] = diff.count();
    runtimes[3] = diff.count();
  }
  runtimes[2] -= runtimes[1];
  runtimes[1] -= runtimes[0];
  labeler.DumpConfidences(confidence_csv,
                          graph,
                          kSkip,
                          kSkipSmallCNN,
                          diff_thresh,
                          distill_thresh_lower,
                          distill_thresh_upper,
                          runtimes);
#endif
  return 0;
}
