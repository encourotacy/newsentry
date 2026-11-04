#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdlib>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "tools/yaml.hpp"

const std::string keys =
    "{help h usage ? |                                        | 输出命令行参数说明}"
    "{config-path c  | configs/calibration_omni.yaml          | yaml配置文件路径 }"
    "{@input-folder  | assets/chessboard_calib_omni_extrinsic | 已知位姿的图像   }";

namespace {

std::vector<cv::Point3f> corners_in_world(const cv::Size & pattern_size, float square_size) {
    std::vector<cv::Point3f> corners_3d;
    for (int i = 0; i < pattern_size.height; i++) {
        for (int j = 0; j < pattern_size.width; j++) {
            corners_3d.push_back({j * square_size, i * square_size, 0});
        }
    }
    return corners_3d;
}

// 板面朝后视相机、横平竖直时的板坐标系到车体坐标系。
// 板的 x 沿内角点列方向，y 沿行方向向下，z 穿进板里。
Eigen::Matrix3d board_to_body(double yaw_deg) {
    Eigen::Matrix3d canonical;
    canonical << 0, 0, -1, 1, 0, 0, 0, -1, 0;
    const double yaw = yaw_deg / 57.3;
    Eigen::Matrix3d about_z;
    about_z << std::cos(yaw), -std::sin(yaw), 0, std::sin(yaw), std::cos(yaw), 0, 0, 0, 1;
    return about_z * canonical;
}

// 光轴水平、绕车体竖直轴转到 look_yaw 的相机旋转。pitch 和 roll 固定为 0。
Eigen::Matrix3d yaw_only_camera_rotation(double look_yaw) {
    Eigen::Matrix3d forward;
    forward << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    Eigen::Matrix3d about_z;
    about_z << std::cos(look_yaw), -std::sin(look_yaw), 0, std::sin(look_yaw), std::cos(look_yaw), 0,
        0, 0, 1;
    return about_z * forward;
}

bool detect(
    const cv::Mat & img, const cv::Size & pattern_size, double preview_scale,
    std::vector<cv::Point2f> & corners) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat compressed;
    cv::resize(gray, compressed, cv::Size(), preview_scale, preview_scale, cv::INTER_AREA);
    const int flags =
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
    std::vector<cv::Point2f> found_corners;
    if (!cv::findChessboardCorners(compressed, pattern_size, found_corners, flags)) return false;
    for (auto & corner : found_corners) {
        corner.x /= preview_scale;
        corner.y /= preview_scale;
    }
    cv::cornerSubPix(
        gray, found_corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01));
    corners = found_corners;
    return true;
}

}  // namespace

int main(int argc, char * argv[]) {
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }

    const auto input_folder = cli.get<std::string>(0);
    const auto config_path = cli.get<std::string>("config-path");
    const auto yaml = tools::load(config_path);
    const cv::Size pattern_size(tools::read<int>(yaml, "pattern_cols"), tools::read<int>(yaml, "pattern_rows"));
    const auto square_size_mm = static_cast<float>(tools::read<double>(yaml, "square_size_mm"));
    const double preview_scale = tools::read<double>(yaml, "preview_scale", 0.5);
    const auto origin = tools::read<std::vector<double>>(yaml, "board_origin_m");
    const double board_yaw_deg = tools::read<double>(yaml, "board_yaw_deg");
    const auto matrix_data = tools::read<std::vector<double>>(yaml, "camera_matrix");
    const auto distort_data = tools::read<std::vector<double>>(yaml, "distort_coeffs");
    if (origin.size() != 3 || matrix_data.size() != 9 || distort_data.size() != 5) {
        fmt::print("错误: board_origin_m 需要 3 个数，camera_matrix 需要 9 个数，distort_coeffs 需要 5 个数\n");
        return 1;
    }

    cv::Matx33d camera_matrix(matrix_data.data());
    cv::Mat distort(distort_data);
    const auto object_points = corners_in_world(pattern_size, square_size_mm);
    const Eigen::Matrix3d rotation_board2body = board_to_body(board_yaw_deg);
    const Eigen::Vector3d translation_board2body(origin[0], origin[1], origin[2]);

    fmt::print("\n=== 后视相机外参标定 ===\n");
    fmt::print(
        "标定板原点在车体坐标系: [{:.3f}, {:.3f}, {:.3f}] m，额外 yaw {:.2f} deg\n", origin[0],
        origin[1], origin[2], board_yaw_deg);
    fmt::print("不读取云台或 IMU。pitch、roll 固定为 0，只保留光轴的 yaw。\n\n");

    std::vector<Eigen::Vector3d> translations;
    std::vector<double> look_yaws;
    std::vector<double> axis_pitches;
    for (int i = 1; true; i++) {
        const auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
        const auto img = cv::imread(img_path);
        if (img.empty()) break;

        std::vector<cv::Point2f> corners;
        if (!detect(img, pattern_size, preview_scale, corners)) {
            fmt::print("[failure] {} 未找到棋盘格\n", img_path);
            continue;
        }

        cv::Mat rvec, tvec;
        cv::solvePnP(
            object_points, corners, camera_matrix, distort, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
        cv::Mat rotation_cv;
        cv::Rodrigues(rvec, rotation_cv);
        Eigen::Matrix3d rotation_board2camera;
        Eigen::Vector3d translation_board2camera;
        cv::cv2eigen(rotation_cv, rotation_board2camera);
        cv::cv2eigen(tvec, translation_board2camera);
        translation_board2camera /= 1e3;

        const Eigen::Matrix3d rotation_camera2body =
            rotation_board2body * rotation_board2camera.transpose();
        const Eigen::Vector3d translation_camera2body =
            translation_board2body - rotation_camera2body * translation_board2camera;
        const Eigen::Vector3d optical_axis = rotation_camera2body.col(2);
        look_yaws.push_back(std::atan2(optical_axis.y(), optical_axis.x()));
        axis_pitches.push_back(std::atan2(optical_axis.z(), std::hypot(optical_axis.x(), optical_axis.y())));
        translations.push_back(translation_camera2body);
        fmt::print(
            "[success] {}  t=[{:.3f}, {:.3f}, {:.3f}] m  光轴 yaw {:.2f} deg  pitch {:.2f} deg\n",
            img_path, translation_camera2body.x(), translation_camera2body.y(),
            translation_camera2body.z(), look_yaws.back() * 57.3, axis_pitches.back() * 57.3);
    }

    if (translations.empty()) {
        fmt::print("错误: 没有可用图像。先把板放到已知位姿，再用 capture_chessboard_omni 保存到该目录\n");
        return 1;
    }

    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    double yaw_sin = 0;
    double yaw_cos = 0;
    double pitch_sum = 0;
    for (size_t i = 0; i < translations.size(); i++) {
        translation += translations[i];
        yaw_sin += std::sin(look_yaws[i]);
        yaw_cos += std::cos(look_yaws[i]);
        pitch_sum += axis_pitches[i];
    }
    translation /= static_cast<double>(translations.size());
    const double look_yaw = std::atan2(yaw_sin, yaw_cos);
    const double mean_pitch = pitch_sum / static_cast<double>(axis_pitches.size());
    const Eigen::Matrix3d rotation = yaw_only_camera_rotation(look_yaw);

    std::vector<double> translation_data(translation.data(), translation.data() + 3);
    // Eigen 默认列优先，yaml 里的旋转矩阵按行写出，和主相机的 R_camera2gimbal 一致。
    std::vector<double> rotation_row_major = {
        rotation(0, 0), rotation(0, 1), rotation(0, 2), rotation(1, 0), rotation(1, 1),
        rotation(1, 2), rotation(2, 0), rotation(2, 1), rotation(2, 2)};

    YAML::Emitter result;
    result << YAML::BeginMap;
    result << YAML::Comment(fmt::format(
        "光轴 yaw {:.2f} deg，原始 pitch {:.2f} deg 已固定为 0", look_yaw * 57.3, mean_pitch * 57.3));
    result << YAML::Key << "R_omnicamera2body" << YAML::Value << YAML::Flow << rotation_row_major;
    result << YAML::Key << "t_omnicamera2body" << YAML::Value << YAML::Flow << translation_data;
    result << YAML::Newline << YAML::EndMap;

    fmt::print("\n==================== 后视相机外参 ====================\n");
    fmt::print("{}\n", result.c_str());
    fmt::print("====================================================\n");
    fmt::print("t 的单位是 m，顺序为前后、左右、上下。左右分量里已经包含相机偏离车体中心的距离。\n");
    if (std::abs(mean_pitch) * 57.3 > 2.0) {
        fmt::print("注意: 解出的光轴 pitch 为 {:.2f} deg，输出的 R 仍把 pitch 固定成了 0。\n", mean_pitch * 57.3);
    }
    return 0;
}
