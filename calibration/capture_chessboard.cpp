#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/ros2/ros2_gimbal.hpp"
#include "tools/math_tools.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/yaml.hpp"

const std::string keys = 
    "{help h usage ?  |                                     | 输出命令行参数说明}"
    "{@config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
    "{o output-folder | assets/chessboard_calib             | 输出文件夹路径   }";

namespace {

void write_q(const std::string & q_path, const Eigen::Quaterniond & q) {
    std::ofstream q_file(q_path);
    Eigen::Vector4d xyzw = q.coeffs();
    q_file << fmt::format("{} {} {} {}", xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
}

bool detect_chessboard(const cv::Mat & img, const cv::Size & pattern_size,
    std::vector<cv::Point2f> & corners, double preview_scale) {
    // 转换和压缩图片
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat compressed_gray;
    cv::resize(gray, compressed_gray, cv::Size(), preview_scale, preview_scale, cv::INTER_AREA);
        // 这里表示将 gray 按 preview_scale 缩放后存入 compressed_gray。
        // 方法 cv::INTER_AREA 在缩小图像时表现更好。
        // 详见 https://docs.opencv.org/5.0/main_modules/imgproc_transform.html#resize
    
    // 检测棋盘格点
    const int flags =
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
        // flag 是一串二进制，每一位表示某个功能是否打开，所以上面使用或来组合多个功能
        // CALIB_CB_ADAPTIVE_THRESH: 使用自适应阈值处理将图像转换为黑白，而不是使用固定阈值（根据图像平均亮度计算得出）
        // CALIB_CB_NORMALIZE_IMAGE: 在应用固定或自适应阈值处理之前，使用 equalizeHist 对图像伽马值进行归一化
        // CALIB_CB_FAST_CHECK: 对图像执行快速检查以查找棋盘角，若未找到则跳过调用。
            // 这可以在未观察到棋盘的退化条件下大幅加快函数调用速度。
        // 详见https://docs.opencv.org/5.0/main_modules/objdetect_qrcode.html#calib-cb-adaptive-thresh
    std::vector<cv::Point2f> search_corners;
    bool found = cv::findChessboardCorners(compressed_gray, pattern_size, search_corners, flags);
        // 函数会判断输入图像是否存在棋盘图案，并定位棋盘内部的角点。
        // 这里的 search_corners 是一种防御性编程，防止在 corners 中只写了一半就返回了。
    if (!found) {
        corners.clear();
        return false;
    }
    
    // 还原角点位置到原图像
    corners = search_corners;
    for (auto & corner : corners) {
        corner.x /= preview_scale;
        corner.y /= preview_scale;
    }

    // 细化角点位置
    cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01));
        // 该函数使用迭代算法细化角点位置，直到达到指定的精度或达到最大迭代次数。
        // cv::Size(11, 11)是搜索范围，cv::Size(-1, -1)是死区，-1 表示不设置死区
        // TermCriteria 是迭代终止条件，这里表示最多迭代 30 次、或者精度达到 0.01 时停止迭代
    return true;
}

} // namespace

int main(int argc,char* argv[]) {
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }
    const std::string config_path = cli.get<std::string>(0);
    const std::string output_folder = cli.get<std::string>("o");
    std::filesystem::create_directories(output_folder); // 新建输出文件夹
    fmt::print("\n=== 相机标定拍摄程序 ===\n\n");

    /* ===== 读取参数，创建变量 ===== */
    auto yaml = tools::load(config_path);
    const int pattern_cols = tools::read<int>(yaml, "pattern_cols"); // 标定板内部角点的列数
    const int pattern_rows = tools::read<int>(yaml, "pattern_rows"); // 标定板内部角点的行数
    const double square_size_mm = tools::read<double>(yaml, "square_size_mm"); // 标定板方格边长
    const double preview_scale = tools::read<double>(yaml, "preview_scale", 0.5); // 图像缩放比例，缩小图像以加快处理速度
    const int detect_interval = tools::read<int>(yaml, "detect_interval", 2); // 检测的间隔帧数
    io::ROS2Gimbal gimbal(config_path); // 订阅导航发布的 /serial/gimbal_status
    io::Camera camera(config_path);
    const cv::Size pattern_size(pattern_cols, pattern_rows);

    if (preview_scale > 1.0 || preview_scale <= 0.0) {
        tools::logger()->error("preview_scale 必须在 (0, 1]，当前值为 {}", preview_scale);
        return -1;
    }

    tools::logger()->info(
    "标定板参数: {}x{} 内角点, 方格大小 {}mm", pattern_cols, pattern_rows, square_size_mm);

    /* ===== 中间变量 ===== */
    cv::Mat image;
    std::chrono::steady_clock::time_point timestamp;
    int save_count = 0;
    int frame_id = 0;
    bool found = false;
    std::vector<cv::Point2f> corners; // 角点的位置

    /* ===== 主程序 ===== */
    while (true) {
        camera.read(image, timestamp);
        frame_id++;

        // 读取导航转发的云台四元数
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
        const bool imu_ready = gimbal.has_imu();
        if (imu_ready) { q = gimbal.q_at(timestamp); }

        // 检测图像
        if (frame_id % std::max(1, detect_interval) == 0 || corners.empty()) {
            found = detect_chessboard(image, pattern_size, corners, preview_scale);
        }

        // 画图
        cv::Mat preview = image.clone();
        cv::drawChessboardCorners(preview, pattern_size, corners, found);
        Eigen::Vector3d zyx = tools::eulers(q, 2, 1, 0) * 57.3;
        tools::draw_text(preview, fmt::format("Z {:.2f}", zyx[0]), {40, 40}, {0, 0, 255});
        tools::draw_text(preview, fmt::format("Y {:.2f}", zyx[1]), {40, 80}, {0, 0, 255});
        tools::draw_text(preview, fmt::format("X {:.2f}", zyx[2]), {40, 120}, {0, 0, 255});
        tools::draw_text(
            preview, imu_ready ? "Gimbal ready" : "Gimbal waiting", {40, 160},
            imu_ready ? cv::Scalar(0, 180, 0) : cv::Scalar(0, 0, 255));
        tools::draw_text(
            preview, found ? "Chessboard found" : "Chessboard lost", {40, 200},
            found ? cv::Scalar(0, 180, 0) : cv::Scalar(0, 0, 255));
        tools::draw_text(preview, fmt::format("Saved {}", save_count), {40, 240}, {255, 128, 0});
        tools::draw_text(
            preview, fmt::format("Detect every {} frame", std::max(1, detect_interval)), {40, 280},
            {255, 128, 0});
        cv::resize(preview, preview, {}, preview_scale, preview_scale, cv::INTER_AREA);

        cv::imshow("Press s to save, q to quit", preview);

        // 检测按键
        const int key = cv::waitKey(1);
        if (key == 'q') break;
        if (key != 's') continue;

        if (!found) {
            tools::logger()->warn("未检测到棋盘格，忽略本次保存");
            continue;
        }
        if (!imu_ready) {
            tools::logger()->warn("IMU 尚未就绪，忽略本次保存");
            continue;
        }

        ++save_count;
        const std::string img_path = fmt::format("{}/{}.jpg", output_folder, save_count);
        const std::string q_path = fmt::format("{}/{}.txt", output_folder, save_count);
        cv::imwrite(img_path, image);
        write_q(q_path, q);
        tools::logger()->info("[{}] Saved in {}", save_count, output_folder);
    }

    tools::logger()->warn("注意四元数输出顺序为wxyz");
    return 0;
}