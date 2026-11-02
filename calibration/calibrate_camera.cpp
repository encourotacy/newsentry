#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <cstdlib>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "tools/yaml.hpp"

const std::string keys =
    "{help h usage ? |                                  | 输出命令行参数说明}"
    "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
    "{@input-folder  | assets/chessboard_calib          | 输入文件夹路径   }";

namespace {

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

void load(const std::string & input_folder, const std::string & config_path, cv::Size & img_size,
    std::vector<std::vector<cv::Point3f>> & obj_points,
    std::vector<std::vector<cv::Point2f>> & img_points){
    /* ===== 读取参数 ===== */ 
    auto yaml = tools::load(config_path);
    const int pattern_cols = tools::read<int>(yaml, "pattern_cols"); // 标定板内部角点的列数
    const int pattern_rows = tools::read<int>(yaml, "pattern_rows"); // 标定板内部角点的行数
    const double square_size_mm = tools::read<double>(yaml, "square_size_mm"); // 标定板方格边长
    const double preview_scale = tools::read<double>(yaml, "preview_scale", 0.5); // 图像缩放比例，缩小图像以加快处理速度
    cv::Size pattern_size(pattern_cols, pattern_rows);
    if (preview_scale > 1.0 || preview_scale <= 0.0) {
        fmt::print("错误：preview_scale 必须在 (0, 1]，当前值为 {}", preview_scale);
        exit(1);
    }

    fmt::print("标定板参数: {}x{} 内角点, 方格大小 {}mm\n", pattern_cols, pattern_rows, square_size_mm);
    fmt::print("开始加载标定数据...\n\n");

    /* ===== 加载标定数据 ===== */
    int success_count = 0;
    for (int i = 1; true; i++) {
        std::vector<cv::Point2f> corners;
        auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
        auto img = cv::imread(img_path);
        if (img.empty()) break;
        img_size = img.size();

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
        cv::drawChessboardCorners(preview, pattern_size, corners, found);
        cv::resize(preview, preview, {}, preview_scale, preview_scale);
        cv::imshow("Press any key to continue", preview);
        cv::waitKey(0);
        fmt::print(
            "[{}] {} - {}\n", found ? "success" : "failure", img_path,
            found ? "已添加到标定数据" : "跳过");
        if (!found) continue;
        success_count++;
        img_points.emplace_back(corners);
        obj_points.emplace_back(corners_in_world(pattern_size, square_size_mm));
    }
    fmt::print("\n成功加载 {} 张有效图像\n", success_count);
    if (success_count < 10) {
        fmt::print("警告: 图像数量较少（建议至少15张），可能影响标定精度\n");
    }
}

void print_result(const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs, double error){
    YAML::Emitter result;
    std::vector<double> camera_matrix_data(
        camera_matrix.begin<double>(), camera_matrix.end<double>());
    std::vector<double> distort_coeffs_data(
        distort_coeffs.begin<double>(), distort_coeffs.end<double>());

    result << YAML::BeginMap;
    result << YAML::Comment(fmt::format("重投影误差: {:.4f}px", error));
    result << YAML::Key << "camera_matrix";
    result << YAML::Value << YAML::Flow << camera_matrix_data;
    result << YAML::Key << "distort_coeffs";
    result << YAML::Value << YAML::Flow << distort_coeffs_data;
    result << YAML::Newline;
    result << YAML::EndMap;

    fmt::print("\n==================== 相机内参标定结果 ====================\n");
    fmt::print("{}\n", result.c_str());
    fmt::print("=========================================================\n\n");
    fmt::print("请将上述结果复制到 configs/calibration_chessboard.yaml 中\n");
    fmt::print("然后进行手眼标定（运行 calibrate_chessboard_handeye）\n");
}

void evaluate_calibration(const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs,
    double error, const cv::Size & img_size) {

    fmt::print("\n==================== 标定结果评估 ====================\n");

    const double fx = camera_matrix.at<double>(0, 0);
    const double fy = camera_matrix.at<double>(1, 1);
    const double cx = camera_matrix.at<double>(0, 2);
    const double cy = camera_matrix.at<double>(1, 2);

    // calibrateCamera 输出的畸变系数可能是 1x5 或 5x1，统一成 1x5 方便读取。
    cv::Mat dist = distort_coeffs.reshape(1, 1);
    const double k1 = dist.at<double>(0, 0);
    const double k2 = dist.at<double>(0, 1);
    const double p1 = dist.at<double>(0, 2);
    const double p2 = dist.at<double>(0, 3);
    const double k3 = dist.at<double>(0, 4);

    bool all_ok = true;

    // 1. 重投影误差
    if (error < 0.5) {
        fmt::print("[OK]   重投影误差 {:.4f} px < 0.5 px，优秀\n", error);
    } else if (error < 1.0) {
        fmt::print("[OK]   重投影误差 {:.4f} px < 1.0 px，良好\n", error);
    } else {
        fmt::print("[FAIL] 重投影误差 {:.4f} px >= 1.0 px，建议检查角点或重新采集\n", error);
        all_ok = false;
    }

    // 2. 切向畸变 p1, p2 应当很小
    const double p_limit = 1e-3;
    if (std::abs(p1) < p_limit && std::abs(p2) < p_limit) {
        fmt::print("[OK]   切向畸变 p1={:.6e}, p2={:.6e}，均 < 1e-3\n", p1, p2);

        if (std::abs(p1) > 1e-4 || std::abs(p2) > 1e-4) {
            fmt::print("[WARN] 切向畸变偏大（>1e-4），建议检查标定板平整度和角点检测\n");
        }
    } else {
        fmt::print("[FAIL] 切向畸变过大：p1={:.6e}, p2={:.6e}，应 < 1e-3\n", p1, p2);
        all_ok = false;
    }

    // 3. fx 与 fy 应当接近
    const double f_diff = std::abs(fx - fy);
    if (f_diff < 30.0) {
        fmt::print("[OK]   |fx-fy|={:.2f} < 30，传感器像素接近正方形\n", f_diff);
    } else if (f_diff < 100.0) {
        fmt::print("[WARN] |fx-fy|={:.2f}，在 30~100 之间，需检查图像质量/角点检测\n", f_diff);
    } else {
        fmt::print("[FAIL] |fx-fy|={:.2f} > 100，内参明显不合理\n", f_diff);
        all_ok = false;
    }

    // 4. 光学中心应接近图像中心
    const double dx = cx - img_size.width / 2.0;
    const double dy = cy - img_size.height / 2.0;

    if (std::abs(dx) <= 30.0 && std::abs(dy) <= 30.0) {
        fmt::print("[OK]   光学中心 ({:.2f},{:.2f})，相对图像中心偏移 dx={:.2f}, dy={:.2f}\n",
            cx, cy, dx, dy);
    } else {
        fmt::print("[FAIL] 光学中心偏移过大：dx={:.2f}, dy={:.2f}，应接近分辨率一半\n", dx, dy);
        all_ok = false;
    }

    fmt::print("---------------------------------------------------------\n");

    if (all_ok) {
        fmt::print("综合结论：标定结果基本可用。\n");
    } else {
        fmt::print("综合结论：标定结果存在不合理项，建议重新采集标定图像并重新标定。\n");
    }

    fmt::print("=========================================================\n");
}

} // namespace

int main(int argc,char* argv[]) {
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }
    auto input_folder = cli.get<std::string>(0);
    auto config_path = cli.get<std::string>("config-path");
    
    /* ===== 读取参数和加载数据 ===== */
    fmt::print("\n=== 棋盘格相机内参标定程序 ===\n\n");
    cv::Size img_size;
    std::vector<std::vector<cv::Point3f>> obj_points;
    std::vector<std::vector<cv::Point2f>> img_points;
    load(input_folder, config_path, img_size, obj_points, img_points);

    if (obj_points.empty()) {
        fmt::print("错误: 没有有效的标定数据\n");
        exit(1);
    }

    /* ===== 相机标定计算 ===== */ 
    fmt::print("\n开始相机标定计算...\n");

    cv::Mat camera_matrix, distort_coeffs; // 相机内参矩阵和畸变系数
    std::vector<cv::Mat> rvecs, tvecs; // 每张照片的旋转向量和平移向量
    auto criteria =
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, DBL_EPSILON);
        // 迭代停止条件，这里的意思是迭代 10 0次，在精度达到 DBL_EPSILON(约2.22e-16) 时提前停止
    cv::calibrateCamera(
        obj_points, img_points, img_size, camera_matrix, distort_coeffs, rvecs, tvecs,
        cv::CALIB_FIX_K3, criteria);
        // 使用张正友标定法标定
        // cv::CALIB_FIX_K3 表示固定畸变系数 k3，避免过拟合

    fmt::print("标定计算完成！\n");

    /* ===== 重投影并计算误差 ===== */
    double error_sum = 0;
    size_t total_points = 0;
    for (size_t i = 0; i < obj_points.size(); i++) {
        std::vector<cv::Point2f> reprojected_points;
        cv::projectPoints(
        obj_points[i], rvecs[i], tvecs[i], camera_matrix, distort_coeffs, reprojected_points);

        total_points += reprojected_points.size();
        for (size_t j = 0; j < reprojected_points.size(); j++)
            error_sum += cv::norm(img_points[i][j] - reprojected_points[j]);
    }
    auto error = error_sum / total_points;

    /* ===== 输出参数结果 ===== */
    print_result(camera_matrix, distort_coeffs, error);
    fmt::print("提示: 重投影误差 < 0.5px 为优秀，< 1.0px 为良好\n");
    fmt::print("      如果误差较大，建议重新采集数据或增加图像数量\n");
    evaluate_calibration(camera_matrix, distort_coeffs, error, img_size);
    return 0;
}