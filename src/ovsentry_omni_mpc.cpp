#include <fmt/core.h>

#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

#include "io/camera.hpp"
#include "io/command.hpp"
#include "io/ros2/ros2_gimbal.hpp"
#include "io/ros2/subscribe.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/recorder.hpp"

const std::string cli_keys =
    "{help h usage ? |                     | 输出命令行参数说明}"
    "{@config-path   | configs/sentry.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char* argv[]) {
    cv::CommandLineParser cli(argc, argv, cli_keys);
    auto config_path = cli.get<std::string>(0);
    if (cli.has("help") || config_path.empty()) {
        cli.printMessage();
        return 0;
    }
    /* === 创建各种类 === */
    tools::logger()->info("[OVSentry Omni MPC] auto aim = mpc");
    tools::Exiter exiter;      // 退出检测器，检测到Ctrl+C则会修改一个标记以表示需要退出
    tools::Recorder recorder;  // 内录，记录画面与四元数以便之后回放

    io::ROS2Gimbal gimbal(config_path);  // 云台类，接收和发送数据
    io::ArmorIgnoreSubscriber armor_ignore(config_path);
    io::StateSubscriber gimbal_mode(config_path);
    io::Camera camera(config_path);      // 相机，读取相机画面

    auto_aim::YOLO yolo(config_path);
    auto_aim::Solver aim_solver(config_path);
    auto_aim::Tracker tracker(config_path, aim_solver);
    auto_aim::Planner planner(config_path);
    omniperception::Decider decider(config_path);
    omniperception::Perceptron perceptron(config_path);

    /* === 主程序 === */
    tools::logger()->info("[OVSentry Omni MPC] started");

    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    io::MPCCommand command{};
    int frame_count = 0;
    auto last_mode = io::ROS2Mode::auto_aim;

    while (!exiter.exit()) {
        //--- 读取相机画面 ---
        try {
            camera.read(img, timestamp);
            if (img.empty()) {
                tools::logger()->warn("[OVSentry Omni MPC] 读取到空图像，跳过此帧");
                continue;
            }
        } catch (const std::exception& e) {
            tools::logger()->error("[OVSentry Omni MPC] 读取摄像机图像失败: {}", e.what());
            continue;
        }
        frame_count++;

        //--- 读取 IMU 四元数 ---
        const auto q = gimbal.q_at(timestamp);
        aim_solver.set_R_gimbal2world(q);
        // outpost_solver.set_R_gimbal2world(q);

        // recorder.record(img, q, timestamp);  // 如果需要取消内录，将这行注释掉即可

        //--- 解析模式 ---
        const auto mode = gimbal_mode.mode();
        if (mode != last_mode) {
            tools::logger()->info("[OVSentry Omni MPC] switch mode={}", gimbal_mode.mode_name());
            last_mode = mode;
        }

        if (mode == io::ROS2Mode::auto_aim) {
            //--- 自瞄模式，计算自瞄指令 ---
            auto armors = yolo.detect(img, frame_count);
            tracker.filter(armors, cv::Point2f(img.cols * 0.5f, img.rows * 0.5f), armor_ignore.ignore_ids());

            auto targets = tracker.track(armors, timestamp);

            if (tracker.state() == "lost") {
                decider.set_ignore_ids(armor_ignore.ignore_ids());
                auto detection_queue = perceptron.get_detection_queue();
                decider.sort(detection_queue);
                const io::Command omni = decider.decide(detection_queue);
                command = {};
                command.control = omni.control;
                command.shoot = false;
                if (omni.control) {
                    command.pitch = 0.26;
                    // 全向 yaw 的 0° 是车尾。小云台朝前，再加 180°。
                    const double aim_yaw =
                        tools::limit_rad(gimbal.state().big_yaw + omni.yaw + CV_PI);
                    command.yaw = aim_yaw;
                    command.big_yaw = aim_yaw;
                    command.small_yaw = aim_yaw;
                }
            } else {
                std::optional<auto_aim::Target> target_for_plan =
                    targets.empty() ? std::nullopt : std::make_optional(targets.front());
                command = planner.plan(target_for_plan, gimbal.bullet_speed());
                if (command.control && target_for_plan.has_value()) {
                    const auto ekf_x = target_for_plan->ekf_x();
                    const double center_yaw = std::atan2(ekf_x[2], ekf_x[0]);
                    command.small_yaw = command.yaw;
                    command.big_yaw = gimbal.state().big_yaw + tools::limit_rad(center_yaw - gimbal.state().big_yaw);
                    // 小 yaw 指向装甲板的中心，大 yaw 指向敌方车体的旋转中心
                }
            }
            command.rad2degree();

            gimbal.send(command);
        } else if (mode == io::ROS2Mode::outpost) {
            gimbal.send(false, false, 0.0, 0.0, 0.0, 
                        0.0, 0.0, 0.0, 0.0, 0.0, 
                        0.0, 0.0, 0.0);
        } else {
            gimbal.send(false, false, 0.0, 0.0, 0.0, 
                        0.0, 0.0, 0.0, 0.0, 0.0, 
                        0.0, 0.0, 0.0);
        }
    }

    gimbal.send(false, false, 0.0, 0.0, 0.0, 
                        0.0, 0.0, 0.0, 0.0, 0.0, 
                        0.0, 0.0, 0.0);

    return 0;
}
