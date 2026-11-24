#include "perceptron.hpp"

#include <chrono>
#include <thread>

#include "tools/logger.hpp"
#include "tools/yaml.hpp"

namespace omniperception
{
Perceptron::Perceptron(const std::string & config_path)
: detection_queue_(10), decider_(config_path), stop_flag_(false)
{
  const auto yaml = tools::load(config_path);
  const double exposure_ms = tools::read<double>(yaml, "omni_exposure_ms", 1.8);
  const double gain = tools::read<double>(yaml, "omni_gain", 14.0);
  const auto vid_pid = tools::read<std::string>(yaml, "omni_vid_pid", std::string("2bdf:0001"));

  if (!yaml["omni_cameras"] || !yaml["omni_cameras"].IsSequence()) {
    tools::logger()->warn("[Omni] no hikrobot camera configured");
  } else {
    for (const auto & node : yaml["omni_cameras"]) {
      const auto name = node["name"].as<std::string>();
      const auto serial = node["serial"] ? node["serial"].as<std::string>() : std::string();
      if (serial.empty()) {
        tools::logger()->warn("[Omni] camera {} serial is empty, skip", name);
        continue;
      }

      const double cam_exposure = node["exposure_ms"] ? node["exposure_ms"].as<double>() : exposure_ms;
      const double cam_gain = node["gain"] ? node["gain"].as<double>() : gain;

      OmniCamera cam;
      cam.name = name;
      cam.camera = std::make_unique<io::HikRobot>(cam_exposure, cam_gain, vid_pid, serial);
      cam.yolo = std::make_shared<auto_aim::YOLO>(config_path, false);
      cameras_.push_back(std::move(cam));
    }
  }

  for (size_t i = 0; i < cameras_.size(); ++i) {
    threads_.emplace_back([this, i] { parallel_infer(cameras_[i]); });
  }

  tools::logger()->info("[Omni] Perceptron initialized with {} hikrobot camera(s).", cameras_.size());
}

Perceptron::~Perceptron()
{
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_flag_ = true;
  }
  condition_.notify_all();

  for (auto & t : threads_) {
    if (t.joinable()) t.join();
  }
  tools::logger()->info("Perceptron destructed.");
}

std::vector<DetectionResult> Perceptron::get_detection_queue()
{
  std::vector<DetectionResult> result;
  DetectionResult temp;

  while (!detection_queue_.empty()) {
    detection_queue_.pop(temp);
    result.push_back(std::move(temp));
  }

  return result;
}

cv::Mat Perceptron::latest_image() const
{
  std::lock_guard<std::mutex> lock(image_mutex_);
  return latest_image_.clone();
}

void Perceptron::parallel_infer(OmniCamera & cam)
{
  try {
    while (true) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_flag_) return;
      }

      cv::Mat img;
      std::chrono::steady_clock::time_point timestamp;
      if (!cam.camera->try_read(img, timestamp, 30) || img.empty()) continue;
      {
        std::lock_guard<std::mutex> lock(image_mutex_);
        latest_image_ = img.clone();
      }

      auto armors = cam.yolo->detect(img);
      if (armors.empty()) continue;

      const auto delta_angle = decider_.delta_angle(armors, cam.name);
      DetectionResult dr;
      dr.armors = std::move(armors);
      dr.timestamp = timestamp;
      dr.delta_yaw = delta_angle[0] / 57.3;
      dr.delta_pitch = delta_angle[1] / 57.3;
      dr.camera = cam.name;
      detection_queue_.push(dr);
    }
  } catch (const std::exception & e) {
    tools::logger()->error("Exception in parallel_infer[{}]: {}", cam.name, e.what());
  }
}

}  // namespace omniperception
