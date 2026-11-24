#include <fmt/core.h>

#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/yaml.hpp"

const std::string keys =
    "{help h usage ? |                              | 输出命令行参数说明}"
    "{@config-path   | configs/calibration_omni.yaml | yaml配置文件路径 }"
    "{o output-folder | assets/chessboard_calib_omni | 输出文件夹路径   }";

namespace {

bool detect_chessboard(
    const cv::Mat & img, const cv::Size & pattern_size, std::vector<cv::Point2f> & corners,
    double preview_scale) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat compressed_gray;
    cv::resize(gray, compressed_gray, cv::Size(), preview_scale, preview_scale, cv::INTER_AREA);

    const int flags =
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
    std::vector<cv::Point2f> search_corners;
    if (!cv::findChessboardCorners(compressed_gray, pattern_size, search_corners, flags)) {
        corners.clear();
        return false;
    }

    corners = search_corners;
    for (auto & corner : corners) {
        corner.x /= preview_scale;
        corner.y /= preview_scale;
    }
    cv::cornerSubPix(
        gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01));
    return true;
}

}  // namespace

int main(int argc, char * argv[]) {
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }

    const std::string config_path = cli.get<std::string>(0);
    const std::string output_folder = cli.get<std::string>("o");
    std::filesystem::create_directories(output_folder);

    auto yaml = tools::load(config_path);
    const cv::Size pattern_size(tools::read<int>(yaml, "pattern_cols"), tools::read<int>(yaml, "pattern_rows"));
    const double preview_scale = tools::read<double>(yaml, "preview_scale", 0.5);
    const int detect_interval = std::max(1, tools::read<int>(yaml, "detect_interval", 2));
    if (preview_scale > 1.0 || preview_scale <= 0.0) {
        tools::logger()->error("preview_scale 必须在 (0, 1]，当前值为 {}", preview_scale);
        return 1;
    }

    io::Camera camera(config_path);
    tools::logger()->info(
        "[OmniCalib] 后视相机采图，不连接云台。标定板 {}x{}，输出 {}", pattern_size.width,
        pattern_size.height, output_folder);

    cv::Mat image;
    std::chrono::steady_clock::time_point timestamp;
    std::vector<cv::Point2f> corners;
    int save_count = 0;
    int frame_id = 0;
    bool found = false;

    while (true) {
        camera.read(image, timestamp);
        // 全向镜头装反，标定前把画面转正。保存下去的照片已经是转正后的，计算内参时不要再转。
        cv::Mat upright;
        cv::rotate(image, upright, cv::ROTATE_180);
        image = upright;
        frame_id++;
        if (frame_id % detect_interval == 0 || corners.empty()) {
            found = detect_chessboard(image, pattern_size, corners, preview_scale);
        }

        cv::Mat preview = image.clone();
        cv::drawChessboardCorners(preview, pattern_size, corners, found);
        tools::draw_text(
            preview, found ? "Chessboard found" : "Chessboard lost", {40, 40},
            found ? cv::Scalar(0, 180, 0) : cv::Scalar(0, 0, 255));
        tools::draw_text(preview, fmt::format("Saved {}", save_count), {40, 80}, {255, 128, 0});
        tools::draw_text(preview, "s: save   q: quit", {40, 120}, {255, 255, 255});
        cv::resize(preview, preview, {}, preview_scale, preview_scale, cv::INTER_AREA);
        cv::imshow("Omni chessboard", preview);

        const int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q') break;
        if (key != 's' && key != 'S') continue;
        if (!found) {
            tools::logger()->warn("未检测到棋盘格，忽略本次保存");
            continue;
        }

        ++save_count;
        const std::string img_path = fmt::format("{}/{}.jpg", output_folder, save_count);
        cv::imwrite(img_path, image);
        tools::logger()->info("[{}] 已保存 {}", save_count, img_path);
    }
    return 0;
}
