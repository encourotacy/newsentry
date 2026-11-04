#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <chrono>
#include <string>

#include "armor.hpp"
#include "solver.hpp"
#include "target.hpp"

namespace auto_aim {

class Tracker {
   public:
    Tracker(const std::string& config_path, Solver& solver);

    std::string state() const { return state_; };

    // ignore_ids 是导航忽略名单的位掩码，bit0 为 1 号，bit7 为 8 号。0 表示都不忽略。
    void filter(std::list<Armor>& armors, cv::Point2f img_center,
                size_t ignore_ids = static_cast<size_t>(0));

    std::list<Target> track(std::list<Armor>& armors, std::chrono::steady_clock::time_point t);

   private:
    Solver& solver_;
    Color enemy_color_;

    std::chrono::steady_clock::time_point last_timestamp_;
    std::string state_;  // 当前跟踪状态
    int detect_count_;   // 观察到目标的帧数
    int lost_count_;     // 丢失目标的帧数
    Target target_;      // 目标

    void state_machine(bool found);
    bool set_target(std::list<Armor>& armors, std::chrono::steady_clock::time_point t);
    bool update_target(std::list<Armor>& armors, std::chrono::steady_clock::time_point t);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP