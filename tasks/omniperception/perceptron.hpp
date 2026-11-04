#ifndef OMNIPERCEPTION__PERCEPTRON_HPP
#define OMNIPERCEPTION__PERCEPTRON_HPP

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "decider.hpp"
#include "detection.hpp"
#include "io/hikrobot/hikrobot.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/thread_safe_queue.hpp"

namespace omniperception
{

class Perceptron
{
public:
  explicit Perceptron(const std::string & config_path);

  ~Perceptron();

  std::vector<DetectionResult> get_detection_queue();

private:
  struct OmniCamera
  {
    std::string name;
    std::unique_ptr<io::HikRobot> camera;
    std::shared_ptr<auto_aim::YOLO> yolo;
  };

  void parallel_infer(OmniCamera & cam);

  std::vector<OmniCamera> cameras_;
  std::vector<std::thread> threads_;
  tools::ThreadSafeQueue<DetectionResult, true> detection_queue_;
  Decider decider_;
  bool stop_flag_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
};

}  // namespace omniperception
#endif
