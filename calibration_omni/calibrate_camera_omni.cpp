#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <cstdlib>
#include <opencv2/opencv.hpp>

#include "tools/yaml.hpp"

const std::string keys =
    "{help h usage ? |                                  | 输出命令行参数说明}"
    "{config-path c  | configs/calibration_omni.yaml    | yaml配置文件路径 }"
    "{@input-folder  | assets/chessboard_calib_omni     | 输入文件夹路径   }";

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

void load(
    const std::string & input_folder, const std::string & config_path, cv::Size & img_size,
    std::vector<std::vector<cv::Point3f>> & obj_points,
    std::vector<std::vector<cv::Point2f>> & img_points) {
    auto yaml = tools::load(config_path);
    const cv::Size pattern_size(tools::read<int>(yaml, "pattern_cols"), tools::read<int>(yaml, "pattern_rows"));
    const auto square_size_mm = static_cast<float>(tools::read<double>(yaml, "square_size_mm"));
    const double preview_scale = tools::read<double>(yaml, "preview_scale", 0.5);
    if (preview_scale > 1.0 || preview_scale <= 0.0) {
        fmt::print("错误：preview_scale 必须在 (0, 1]，当前值为 {}\n", preview_scale);
        exit(1);
    }

    int success_count = 0;
    for (int i = 1; true; i++) {
        const auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
        auto img = cv::imread(img_path);
        if (img.empty()) break;
        img_size = img.size();

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::Mat compressed_gray;
        cv::resize(gray, compressed_gray, cv::Size(), preview_scale, preview_scale, cv::INTER_AREA);

        const int flags =
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
        std::vector<cv::Point2f> search_corners;
        const bool found =
            cv::findChessboardCorners(compressed_gray, pattern_size, search_corners, flags);
        std::vector<cv::Point2f> corners;
        if (found) {
            corners = search_corners;
            for (auto & corner : corners) {
                corner.x /= preview_scale;
                corner.y /= preview_scale;
            }
            cv::cornerSubPix(
                gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01));
        }

        auto preview = img.clone();
        cv::drawChessboardCorners(preview, pattern_size, corners, found);
        cv::resize(preview, preview, {}, preview_scale, preview_scale);
        cv::imshow("Press any key to continue", preview);
        cv::waitKey(0);
        fmt::print(
            "[{}] {} - {}\n", found ? "success" : "failure", img_path,
            found ? "已添加到标定数据" : "跳过");
        if (!found) continue;
        success_count++;
        img_points.push_back(corners);
        obj_points.push_back(corners_in_world(pattern_size, square_size_mm));
    }
    fmt::print("\n成功加载 {} 张有效图像\n", success_count);
    if (success_count < 10) {
        fmt::print("警告: 图像数量较少（建议至少15张），可能影响标定精度\n");
    }
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
    fmt::print("\n=== 后视相机内参标定 ===\n\n");

    cv::Size img_size;
    std::vector<std::vector<cv::Point3f>> obj_points;
    std::vector<std::vector<cv::Point2f>> img_points;
    load(input_folder, config_path, img_size, obj_points, img_points);
    if (obj_points.empty()) {
        fmt::print("错误: 没有有效的标定数据\n");
        return 1;
    }

    cv::Mat camera_matrix, distort_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    const auto criteria =
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, DBL_EPSILON);
    cv::calibrateCamera(
        obj_points, img_points, img_size, camera_matrix, distort_coeffs, rvecs, tvecs,
        cv::CALIB_FIX_K3, criteria);

    double error_sum = 0;
    size_t total_points = 0;
    for (size_t i = 0; i < obj_points.size(); i++) {
        std::vector<cv::Point2f> reprojected;
        cv::projectPoints(
            obj_points[i], rvecs[i], tvecs[i], camera_matrix, distort_coeffs, reprojected);
        total_points += reprojected.size();
        for (size_t j = 0; j < reprojected.size(); j++) {
            error_sum += cv::norm(img_points[i][j] - reprojected[j]);
        }
    }
    const double error = error_sum / static_cast<double>(total_points);

    std::vector<double> matrix_data(camera_matrix.begin<double>(), camera_matrix.end<double>());
    std::vector<double> distort_data(distort_coeffs.begin<double>(), distort_coeffs.end<double>());
    YAML::Emitter result;
    result << YAML::BeginMap;
    result << YAML::Comment(fmt::format("重投影误差: {:.4f}px", error));
    result << YAML::Key << "camera_matrix" << YAML::Value << YAML::Flow << matrix_data;
    result << YAML::Key << "distort_coeffs" << YAML::Value << YAML::Flow << distort_data;
    result << YAML::Newline << YAML::EndMap;

    fmt::print("\n==================== 后视相机内参 ====================\n");
    fmt::print("{}\n", result.c_str());
    fmt::print("====================================================\n");
    fmt::print("请把 camera_matrix 和 distort_coeffs 写入 configs/calibration_omni.yaml\n");
    fmt::print("然后把标定板放到已知位姿，再运行 calibrate_extrinsic_omni\n");
    return 0;
}
