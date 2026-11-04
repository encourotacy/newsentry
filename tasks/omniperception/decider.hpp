#ifndef OMNIPERCEPTION__DECIDER_HPP
#define OMNIPERCEPTION__DECIDER_HPP

#include <Eigen/Dense>
#include <cstdint>
#include <list>
#include <opencv2/core.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "detection.hpp"
#include "io/command.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/target.hpp"

namespace omniperception
{
class Decider
{
public:
  explicit Decider(const std::string & config_path);

  io::Command decide(const std::vector<DetectionResult> & detection_queue);

  // 返回值单位是 degree。先解出装甲板在相机前的位置，再用 R、t 变到车体中心，
  // 角度从车体中心指向装甲板。
  Eigen::Vector2d delta_angle(
    const std::list<auto_aim::Armor> & armors, const std::string & camera);

  bool armor_filter(std::list<auto_aim::Armor> & armors);

  void set_priority(std::list<auto_aim::Armor> & armors);
  // 对队列中的每一个 DetectionResult 进行过滤，同时将 DetectionResult 排序
  void sort(std::vector<DetectionResult> & detection_queue);

  Eigen::Vector4d get_target_info(
    const std::list<auto_aim::Armor> & armors, const std::list<auto_aim::Target> & targets);

  // 导航忽略名单。bit0 是 1 号英雄，bit5 是 6 号哨兵，bit7 是 8 号基地。
  void set_ignore_ids(uint8_t ignore_ids);

  void get_auto_aim_target(
    std::list<auto_aim::Armor> & armors, const std::vector<int8_t> & auto_aim_target);

private:
  struct CameraGeometry
  {
    cv::Matx33d camera_matrix;
    cv::Mat distort_coeffs;
    Eigen::Matrix3d R_camera2body;
    Eigen::Vector3d t_camera2body;
  };

  int mode_;
  auto_aim::Color enemy_color_;
  std::unordered_map<std::string, CameraGeometry> cameras_;
  uint8_t ignore_mask_ = 0;

  // 定义ArmorName到ArmorPriority的映射类型
  using PriorityMap = std::unordered_map<auto_aim::ArmorName, auto_aim::ArmorPriority>;

  const PriorityMap mode1 = {
    {auto_aim::ArmorName::ONE, auto_aim::ArmorPriority::SECOND},
    {auto_aim::ArmorName::TWO, auto_aim::ArmorPriority::FORTH},
    {auto_aim::ArmorName::THREE, auto_aim::ArmorPriority::FIRST},
    {auto_aim::ArmorName::FOUR, auto_aim::ArmorPriority::FIRST},
    {auto_aim::ArmorName::FIVE, auto_aim::ArmorPriority::THIRD},
    {auto_aim::ArmorName::SENTRY, auto_aim::ArmorPriority::THIRD},
    {auto_aim::ArmorName::OUTPOST, auto_aim::ArmorPriority::FIFTH},
    {auto_aim::ArmorName::BASE, auto_aim::ArmorPriority::FIFTH},
    {auto_aim::ArmorName::NOT_ARMOR, auto_aim::ArmorPriority::FIFTH}};

  const PriorityMap mode2 = {
    {auto_aim::ArmorName::TWO, auto_aim::ArmorPriority::FIRST},
    {auto_aim::ArmorName::ONE, auto_aim::ArmorPriority::SECOND},
    {auto_aim::ArmorName::THREE, auto_aim::ArmorPriority::SECOND},
    {auto_aim::ArmorName::FOUR, auto_aim::ArmorPriority::SECOND},
    {auto_aim::ArmorName::FIVE, auto_aim::ArmorPriority::SECOND},
    {auto_aim::ArmorName::SENTRY, auto_aim::ArmorPriority::THIRD},
    {auto_aim::ArmorName::OUTPOST, auto_aim::ArmorPriority::THIRD},
    {auto_aim::ArmorName::BASE, auto_aim::ArmorPriority::THIRD},
    {auto_aim::ArmorName::NOT_ARMOR, auto_aim::ArmorPriority::THIRD}};
};

enum PriorityMode
{
  MODE_ONE = 1,
  MODE_TWO
};

}  // namespace omniperception

#endif
