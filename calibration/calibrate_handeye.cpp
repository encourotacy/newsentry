#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <cstdlib>
#include <fstream>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
    "{help h usage ? |                                  | 输出命令行参数说明}"
    "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
    "{@input-folder  | assets/chessboard_calib          | 输入文件夹路径   }";

namespace {

Eigen::Quaterniond read_q(const std::string & q_path) {
    std::ifstream q_file(q_path);
    double w, x, y, z;
    q_file >> w >> x >> y >> z;
    return {w, x, y, z};
}

std::vector<cv::Point3f> corners_in_world(
    const cv::Size & pattern_size, const float square_size) {
    // 假设标定板在 z=0 的平面上，左上角第一个角点的坐标为 (0, 0, 0)
    // 输出所有角点在这个坐标系下的实际坐标。
    std::vector<cv::Point3f> corners_3d;

    for (int i = 0; i < pattern_size.height; i++) {
        for (int j = 0; j < pattern_size.width; j++) {
            corners_3d.push_back({j * square_size, i * square_size, 0});
        }
    }

    return corners_3d;
}

void load(const std::string & input_folder, const std::string & config_path, 
    std::vector<double> & R_gimbal2imubody_data, std::vector<cv::Mat> & R_gimbal2world_list,
    std::vector<cv::Mat> & t_gimbal2world_list, std::vector<cv::Mat> & rvecs,
    std::vector<cv::Mat> & tvecs) {
    /* ===== 读取参数 ===== */
    auto yaml = YAML::LoadFile(config_path);
    auto pattern_cols = yaml["pattern_cols"].as<int>();
    auto pattern_rows = yaml["pattern_rows"].as<int>();
    auto square_size_mm = yaml["square_size_mm"].as<double>();
    auto preview_scale = yaml["preview_scale"].as<double>(0.5);
    R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
    auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
    auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
    
    if (preview_scale > 1.0 || preview_scale <= 0.0) {
        fmt::print("错误：preview_scale 必须在 (0, 1]，当前值为 {}", preview_scale);
        exit(1);
    }

    cv::Size pattern_size(pattern_cols, pattern_rows);
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_gimbal2imubody(R_gimbal2imubody_data.data());
    cv::Matx33d camera_matrix(camera_matrix_data.data());
    cv::Mat distort_coeffs(distort_coeffs_data);

    fmt::print("标定板参数: {}x{} 内角点, 方格大小 {}mm\n", pattern_cols, pattern_rows, square_size_mm);
    fmt::print("开始加载标定数据...\n\n");
    
    /* ===== 加载标定数据 ===== */
    int success_count = 0;
    for (int i = 1; true; i++) {
        std::vector<cv::Point2f> corners;
        auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
        auto q_path = fmt::format("{}/{}.txt", input_folder, i);
        auto img = cv::imread(img_path);
        if (img.empty()) break;

        // 读取四元数，计算旋转矩阵和平移向量
        Eigen::Quaterniond q = read_q(q_path);
        Eigen::Matrix3d R_imubody2world = q.toRotationMatrix();
        Eigen::Matrix3d R_gimbal2world =
            R_gimbal2imubody.transpose() * R_imubody2world * R_gimbal2imubody;
        Eigen::Vector3d ypr = tools::eulers(R_gimbal2world, 2, 1, 0) * 57.3;
        cv::Mat t_gimbal2world = (cv::Mat_<double>(3, 1) << 0, 0, 0);       
            // 程序中以云台旋转中心为原点

        // 压缩图片
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::Mat compressed_gray;
        cv::resize(gray, compressed_gray, cv::Size(), preview_scale, preview_scale, cv::INTER_AREA);
        
        // 检测棋盘格点
        const int flags =
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
        std::vector<cv::Point2f> search_corners;
        bool found = cv::findChessboardCorners(compressed_gray, pattern_size, search_corners, flags);

        if (found) {
            corners = search_corners;
            for (auto & corner : corners) {
                corner.x /= preview_scale;
                corner.y /= preview_scale;
            }
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01));
        }

        // 画图和显示
        auto preview = img.clone();
        tools::draw_text(preview, fmt::format("yaw   {:.2f}", ypr[0]), {40, 40}, {0, 0, 255});
        tools::draw_text(preview, fmt::format("pitch {:.2f}", ypr[1]), {40, 80}, {0, 0, 255});
        tools::draw_text(preview, fmt::format("roll  {:.2f}", ypr[2]), {40, 120}, {0, 0, 255});

        cv::drawChessboardCorners(preview, pattern_size, corners, found);
        cv::resize(preview, preview, {}, 0.5, 0.5);
        cv::imshow("Press any key to continue", preview);
        cv::waitKey(0);

        fmt::print(
            "[{}] {} - {}\n", found ? "success" : "failure", img_path,
            found ? "已添加到标定数据" : "跳过");
        if (!found) continue;
        success_count++;

        // PnP解算以求出每张图片的旋转、平移向量
        cv::Mat R_gimbal2world_cv;
        cv::eigen2cv(R_gimbal2world, R_gimbal2world_cv);
        cv::Mat rvec, tvec;
        auto corners_3d = corners_in_world(pattern_size, square_size_mm);

        cv::solvePnP(
            corners_3d,             // 1. 世界坐标系中的3D点（单位：米）
            corners,                // 2. 对应的图像像素坐标（单位：像素）
            camera_matrix,          // 3. 相机内参矩阵 (3x3)：[fx, 0, cx; 0, fy, cy; 0, 0, 1]
            distort_coeffs,         // 4. 畸变系数 (k1, k2, p1, p2, k3)
            rvec,                   // 5. 输出的旋转向量 (3x1)
            tvec,                   // 6. 输出的平移向量 (3x1)，单位 mm
            false,                  // 7. useExtrinsicGuess：是否使用上一帧的rvec/tvec作为初始值，false表示不用
            cv::SOLVEPNP_ITERATIVE  // 8. 求解方法：non-linear Levenberg-Marquardt minimization scheme 
            );
        
        // 写入数据
        R_gimbal2world_list.emplace_back(R_gimbal2world_cv);
        t_gimbal2world_list.emplace_back(t_gimbal2world);
        rvecs.emplace_back(rvec);
        tvecs.emplace_back(tvec);
    }
    fmt::print("\n成功加载 {} 张有效图像\n", success_count);
    if (success_count < 10) {
        fmt::print("警告: 图像数量较少（建议至少15张），可能影响标定精度\n");
    }
}

void print_yaml(
    const std::vector<double> & R_gimbal2imubody_data, const cv::Mat & R_camera2gimbal,
    const cv::Mat & t_camera2gimbal, const Eigen::Vector3d & ypr) {
    
    YAML::Emitter result;
    std::vector<double> R_camera2gimbal_data(
        R_camera2gimbal.begin<double>(), R_camera2gimbal.end<double>());
    std::vector<double> t_camera2gimbal_data(
        t_camera2gimbal.begin<double>(), t_camera2gimbal.end<double>());

    result << YAML::BeginMap;
    result << YAML::Key << "R_gimbal2imubody";
    result << YAML::Value << YAML::Flow << R_gimbal2imubody_data;
    result << YAML::Newline;
    result << YAML::Newline;
    result << YAML::Comment(fmt::format(
        "相机同理想情况的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree", ypr[0], ypr[1], ypr[2]));
    result << YAML::Key << "R_camera2gimbal";
    result << YAML::Value << YAML::Flow << R_camera2gimbal_data;
    result << YAML::Key << "t_camera2gimbal";
    result << YAML::Value << YAML::Flow << t_camera2gimbal_data;
    result << YAML::Newline;
    result << YAML::EndMap;

    fmt::print("\n==================== 标定结果 ====================\n");
    fmt::print("{}\n", result.c_str());
    fmt::print("==================================================\n\n");
    fmt::print("请将上述结果复制到你的机器人配置文件中（如 configs/standard4.yaml）\n");
}

} // namespace 

int main(int argc, char * argv[]) {
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }
    auto input_folder = cli.get<std::string>(0);
    auto config_path = cli.get<std::string>("config-path");

    fmt::print("\n=== 棋盘格手眼标定程序 ===\n\n");

    std::vector<double> R_gimbal2imubody_data;
    std::vector<cv::Mat> R_gimbal2world_list, t_gimbal2world_list;
    std::vector<cv::Mat> rvecs, tvecs;
    load(input_folder, config_path, R_gimbal2imubody_data, 
        R_gimbal2world_list, t_gimbal2world_list,
        rvecs, tvecs);
    
    if (R_gimbal2world_list.empty()) {
        fmt::print("错误: 没有有效的标定数据\n");
        return -1;
    }

    fmt::print("\n开始手眼标定计算...\n");

    cv::Mat R_camera2gimbal, t_camera2gimbal;
    cv::calibrateHandEye(
        R_gimbal2world_list,  // 云台相对世界坐标的旋转
        t_gimbal2world_list,  // 云台相对世界坐标的平移
        rvecs,                // 标定板相对相机的旋转
        tvecs,                // 标定板相对相机的平移
        R_camera2gimbal,      // 输出：相机相对云台的旋转
        t_camera2gimbal       // 输出：相机相对云台的平移
    );
    t_camera2gimbal /= 1e3; // 算出的单位是 mm，需要转化为 m

    fmt::print("标定计算完成！\n");

    Eigen::Matrix3d R_camera2gimbal_eigen;
    cv::cv2eigen(R_camera2gimbal, R_camera2gimbal_eigen);
    Eigen::Matrix3d R_gimbal2ideal{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}};
    Eigen::Matrix3d R_camera2ideal = R_gimbal2ideal * R_camera2gimbal_eigen;
    Eigen::Vector3d ypr = tools::eulers(R_camera2ideal, 1, 0, 2) * 57.3;

    print_yaml(R_gimbal2imubody_data, R_camera2gimbal, t_camera2gimbal, ypr);

    fmt::print("提示: 相机偏角表示相机安装的理想程度\n");
    fmt::print("      如果偏角较大（>5度），建议调整相机安装位置\n");
    return 0;
}