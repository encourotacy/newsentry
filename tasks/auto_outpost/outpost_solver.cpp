#include "outpost_solver.hpp"

#include "tools/yaml.hpp"

namespace auto_outpost {
//=== 初始化 ===
Solver::Solver(const std::string& config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity()) {
    auto yaml = tools::load(config_path);

    auto R_gimbal2imubody_data = tools::read<std::vector<double>>(yaml, "R_gimbal2imubody");
    auto R_camera2gimbal_data = tools::read<std::vector<double>>(yaml, "R_camera2gimbal");
    auto t_camera2gimbal_data = tools::read<std::vector<double>>(yaml, "t_camera2gimbal");
    R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
    R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
    t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

    auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
    auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
    Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
    cv::eigen2cv(camera_matrix, camera_matrix_);
    cv::eigen2cv(distort_coeffs, distort_coeffs_);
}

void Solver::set_R_gimbal2world(const Eigen::Quaterniond& q) {
    Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
    R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

}  // namespace auto_outpost
