#ifndef OMNIPERCEPTION__DETECTION_HPP
#define OMNIPERCEPTION__DETECTION_HPP

#include <chrono>
#include <list>
#include <string>

#include "tasks/auto_aim/armor.hpp"

namespace omniperception
{
// 一个识别结果可能包含多个 armor，需要排序和过滤。
struct DetectionResult
{
  std::list<auto_aim::Armor> armors;
  std::chrono::steady_clock::time_point timestamp;
  double delta_yaw;    // rad，相对云台正前方
  double delta_pitch;  // rad
  std::string camera;

  DetectionResult & operator=(const DetectionResult & other)
  {
    if (this != &other) {
      armors = other.armors;
      timestamp = other.timestamp;
      delta_yaw = other.delta_yaw;
      delta_pitch = other.delta_pitch;
      camera = other.camera;
    }
    return *this;
  }
};
}  // namespace omniperception

#endif
