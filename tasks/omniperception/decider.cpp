#include "decider.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include "tools/logger.hpp"
#include "tools/yaml.hpp"

namespace omniperception
{
namespace
{
int robot_id(auto_aim::ArmorName name)
{
  switch (name) {
    case auto_aim::ArmorName::ONE: return 1;
    case auto_aim::ArmorName::TWO: return 2;
    case auto_aim::ArmorName::THREE: return 3;
    case auto_aim::ArmorName::FOUR: return 4;
    case auto_aim::ArmorName::FIVE: return 5;
    case auto_aim::ArmorName::SENTRY: return 6;
    case auto_aim::ArmorName::OUTPOST: return 7;
    case auto_aim::ArmorName::BASE: return 8;
    default: return 0;
  }
}

auto_aim::ArmorName armor_from_robot_id(int id)
{
  switch (id) {
    case 1: return auto_aim::ArmorName::ONE;
    case 2: return auto_aim::ArmorName::TWO;
    case 3: return auto_aim::ArmorName::THREE;
    case 4: return auto_aim::ArmorName::FOUR;
    case 5: return auto_aim::ArmorName::FIVE;
    case 6: return auto_aim::ArmorName::SENTRY;
    case 7: return auto_aim::ArmorName::OUTPOST;
    case 8: return auto_aim::ArmorName::BASE;
    default: return auto_aim::ArmorName::NOT_ARMOR;
  }
}

constexpr double LIGHTBAR_LENGTH = 56e-3;
constexpr double BIG_ARMOR_WIDTH = 230e-3;
constexpr double SMALL_ARMOR_WIDTH = 135e-3;
constexpr double BASE_ARMOR_WIDTH = 129e-3;

std::vector<cv::Point3f> armor_object_points(auto_aim::ArmorName name)
{
  const double width = (name == auto_aim::ArmorName::ONE)      ? BIG_ARMOR_WIDTH
                       : (name == auto_aim::ArmorName::BASE ||
                          name == auto_aim::ArmorName::OUTPOST) ? BASE_ARMOR_WIDTH
                                                                : SMALL_ARMOR_WIDTH;
  const float half_width = static_cast<float>(width / 2);
  const float half_height = static_cast<float>(LIGHTBAR_LENGTH / 2);
  return {
    {0, half_width, half_height},
    {0, -half_width, half_height},
    {0, -half_width, -half_height},
    {0, half_width, -half_height}};
}

}  // namespace

Decider::Decider(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  enemy_color_ = (tools::read<std::string>(yaml, "enemy_color") == "red") ? auto_aim::Color::RED
                                                                           : auto_aim::Color::BLUE;
  mode_ = tools::read<int>(yaml, "omni_mode", static_cast<int>(MODE_ONE));

  if (!yaml["omni_cameras"] || !yaml["omni_cameras"].IsSequence()) {
    tools::logger()->warn("[Omni] omni_cameras is missing");
    return;
  }

  for (const auto & cam : yaml["omni_cameras"]) {
    const auto name = cam["name"].as<std::string>();
    const auto matrix_data = cam["camera_matrix"].as<std::vector<double>>();
    const auto distort_data = cam["distort_coeffs"].as<std::vector<double>>();
    const auto rotation_data = cam["R_omnicamera2body"].as<std::vector<double>>();
    const auto translation_data = cam["t_omnicamera2body"].as<std::vector<double>>();
    if (
      matrix_data.size() != 9 || distort_data.size() != 5 || rotation_data.size() != 9 ||
      translation_data.size() != 3) {
      tools::logger()->warn("[Omni] camera {} calibration size is wrong, skip", name);
      continue;
    }

    CameraGeometry geometry;
    geometry.camera_matrix = cv::Matx33d(matrix_data.data());
    geometry.distort_coeffs = cv::Mat(distort_data).clone();
    geometry.R_camera2body =
      Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(rotation_data.data());
    geometry.t_camera2body =
      Eigen::Vector3d(translation_data[0], translation_data[1], translation_data[2]);
    cameras_[name] = geometry;

    const Eigen::Vector3d optical_axis = geometry.R_camera2body.col(2);
    const double yaw_deg = std::atan2(optical_axis.y(), optical_axis.x()) * 57.3;
    tools::logger()->info(
      "[Omni] camera {} fx {:.1f} fy {:.1f}, optical axis yaw {:.1f} deg, t [{:.3f}, {:.3f}, {:.3f}] m",
      name, geometry.camera_matrix(0, 0), geometry.camera_matrix(1, 1), yaw_deg,
      geometry.t_camera2body.x(), geometry.t_camera2body.y(), geometry.t_camera2body.z());
  }
}

io::Command Decider::decide(const std::vector<DetectionResult> & detection_queue)
{
  if (detection_queue.empty()) {
    return io::Command{false, false, 0, 0};
  }

  DetectionResult dr = detection_queue.front();
  if (dr.armors.empty()) return io::Command{false, false, 0, 0};
  tools::logger()->info(
    "omniperceptron find {} from {}, delta yaw is {:.4f}",
    auto_aim::ARMOR_NAMES[dr.armors.front().name_], dr.camera, dr.delta_yaw * 57.3);

  return io::Command{true, false, dr.delta_yaw, dr.delta_pitch};
}

Eigen::Vector2d Decider::delta_angle(
  const std::list<auto_aim::Armor> & armors, const std::string & camera)
{
  Eigen::Vector2d delta_angle = Eigen::Vector2d::Zero();
  if (armors.empty()) return delta_angle;

  const auto found = cameras_.find(camera);
  if (found == cameras_.end()) {
    tools::logger()->warn("[Omni] camera \"{}\" has no geometry", camera);
    return delta_angle;
  }

  const auto & geometry = found->second;
  const auto & armor = armors.front();

  Eigen::Vector3d point_body = Eigen::Vector3d::Zero();
  bool located = false;
  if (armor.points_.size() == 4) {
    cv::Mat rvec;
    cv::Mat tvec;
    const bool ok = cv::solvePnP(
      armor_object_points(armor.name_), armor.points_, geometry.camera_matrix,
      geometry.distort_coeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE);
    if (ok) {
      Eigen::Vector3d armor_in_camera;
      cv::cv2eigen(tvec, armor_in_camera);
      if (armor_in_camera.z() > 0.05) {
        point_body = geometry.R_camera2body * armor_in_camera + geometry.t_camera2body;
        located = true;
      }
    }
  }
  if (!located) {
    std::vector<cv::Point2f> normalized;
    cv::undistortPoints(
      std::vector<cv::Point2f>{armor.center_}, normalized, geometry.camera_matrix,
      geometry.distort_coeffs);
    point_body = geometry.R_camera2body * Eigen::Vector3d(normalized[0].x, normalized[0].y, 1.0);
  }

  delta_angle[0] = std::atan2(point_body.y(), point_body.x()) * 57.3;
  // 车体 z 向上为正。云台俯仰向下为正，所以取反，和主相机火控的符号一致。
  delta_angle[1] =
    -std::atan2(point_body.z(), std::hypot(point_body.x(), point_body.y())) * 57.3;
  return delta_angle;
}

bool Decider::armor_filter(std::list<auto_aim::Armor> & armors)
{
  if (armors.empty()) return true;

  armors.remove_if([&](const auto_aim::Armor & a) { return a.color_ != enemy_color_; });

  // 导航 /request_auto_aim_ignore：对应位为 1 的机器人不打
  armors.remove_if([&](const auto_aim::Armor & a) {
    const int id = robot_id(a.name_);
    if (id <= 0 || id > 8) return false;
    return (ignore_mask_ & static_cast<uint8_t>(1u << (id - 1))) != 0;
  });

  return armors.empty();
}

void Decider::set_priority(std::list<auto_aim::Armor> & armors)
{
  if (armors.empty()) return;

  const PriorityMap & priority_map = (mode_ == MODE_ONE) ? mode1 : mode2;
  for (auto & armor : armors) {
    armor.priority_ = priority_map.at(armor.name_);
  }
}

void Decider::sort(std::vector<DetectionResult> & detection_queue)
{
  if (detection_queue.empty()) return;

  // 对每个 DetectionResult 调用 armor_filter 和 set_priority
  for (auto & dr : detection_queue) {
    armor_filter(dr.armors);
    set_priority(dr.armors);

    // 对每个 DetectionResult 中的 armors 进行排序
    dr.armors.sort(
      [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority_ < b.priority_; });
  }

  detection_queue.erase(
    std::remove_if(
      detection_queue.begin(), detection_queue.end(),
      [](const DetectionResult & dr) { return dr.armors.empty(); }),
    detection_queue.end());

  std::sort(
    detection_queue.begin(), detection_queue.end(),
    [](const DetectionResult & a, const DetectionResult & b) {
      return a.armors.front().priority_ < b.armors.front().priority_;
    });
}

Eigen::Vector4d Decider::get_target_info(
  const std::list<auto_aim::Armor> & armors, const std::list<auto_aim::Target> & targets)
{
  if (armors.empty() || targets.empty()) return Eigen::Vector4d::Zero();

  auto target = targets.front();

  for (const auto & armor : armors) {
    if (armor.name_ != target.name) continue;
    return Eigen::Vector4d{
      armor.xyz_in_gimbal_[0], armor.xyz_in_gimbal_[1], 1,
      static_cast<double>(robot_id(armor.name_))};
  }

  return Eigen::Vector4d::Zero();
}

void Decider::set_ignore_ids(uint8_t ignore_ids) { ignore_mask_ = ignore_ids; }

void Decider::get_auto_aim_target(
  std::list<auto_aim::Armor> & armors, const std::vector<int8_t> & auto_aim_target)
{
  if (auto_aim_target.empty()) return;

  std::vector<auto_aim::ArmorName> auto_aim_targets;

  for (const auto & target : auto_aim_target) {
    const auto name = armor_from_robot_id(target);
    if (name == auto_aim::ArmorName::NOT_ARMOR) {
      tools::logger()->warn("Received invalid auto_aim target value: {}", int(target));
      continue;
    }
    auto_aim_targets.push_back(name);
    tools::logger()->info("nav send auto_aim target is {}", auto_aim::ARMOR_NAMES[name]);
  }

  if (auto_aim_targets.empty()) return;

  armors.remove_if([&](const auto_aim::Armor & a) {
    return std::find(auto_aim_targets.begin(), auto_aim_targets.end(), a.name_) ==
           auto_aim_targets.end();
  });
}

}  // namespace omniperception
