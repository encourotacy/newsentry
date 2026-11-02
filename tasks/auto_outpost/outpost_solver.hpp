#ifndef AUTO_OUTPOST__OUTPOST_SOLVER_HPP
#define AUTO_OUTPOST__OUTPOST_SOLVER_HPP

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <string>

namespace auto_outpost {
/*  */
class Solver {
   public:
    explicit Solver(const std::string& config_path);
    void set_R_gimbal2world(const Eigen::Quaterniond& q);

   private:
    // 相机参数
    cv::Mat camera_matrix_;
    cv::Mat distort_coeffs_;
    Eigen::Matrix3d R_gimbal2imubody_;
    Eigen::Matrix3d R_camera2gimbal_;
    Eigen::Vector3d t_camera2gimbal_;
    Eigen::Matrix3d R_gimbal2world_;
};

}  // namespace auto_outpost

#endif  // AUTO_OUTPOST__OUTPOST_SOLVER_HPP
