
  std::string video_name("/home/li/opensource/stanford-futuredata/data/videos/jackson-town-square.mp4"); 
  const size_t kSkip = 300;
  const size_t kNbFrames = 10000;
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

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  //std::vector<std::unique_ptr<noscope::NoscopeStream > > nStreams;
  //for (int i = 0; i < video_num; ++i) {
      //auto gQueue = std::shared_ptr<noscope::SimpleQueue<noscope::Frame*> >(new noscope::SimpleQueue<noscope::Frame *>);
      //std::shared_ptr<noscope::SimpleQueue<noscope::Frame* > > gQueue(new noscope::SimpleQueue<noscope::Frame* >());
      //gQueues.push_back(gQueue);

      //std::unique_ptr<noscope::NoscopeStream > nStream(new noscope::SimpleQueue<noscope::NoscopeStream>());
      //nStreams.push_back(nStream);
  //}
