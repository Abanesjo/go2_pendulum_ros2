#include <array>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include <Eigen/Dense>

#include "onnxruntime_cxx_api.h"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "go2_interfaces/srv/set_goal.hpp"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int NUM_LEG = 12;
static constexpr int OBS_DIM = 56;
static constexpr int ACTION_DIM = 12;

// Policy order for leg joints
static const std::array<std::string, NUM_LEG> LEG_JOINT_NAMES = {
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
};

static const std::string PENDULUM_JOINT1_NAME = "pendulum_joint1";
static const std::string PENDULUM_JOINT2_NAME = "pendulum_joint2";

// Hard joint limits (rad) in policy order
static constexpr std::array<float, NUM_LEG> HARD_LIMIT_LOW = {
    -1.0472f, -1.0472f, -1.0472f, -1.0472f,   // hips
    -1.5708f, -1.5708f, -0.5236f, -0.5236f,    // thighs (front, rear)
    -2.7227f, -2.7227f, -2.7227f, -2.7227f,    // calfs
};
static constexpr std::array<float, NUM_LEG> HARD_LIMIT_HIGH = {
     1.0472f,  1.0472f,  1.0472f,  1.0472f,
     3.4907f,  3.4907f,  4.5379f,  4.5379f,
    -0.83776f,-0.83776f,-0.83776f,-0.83776f,
};

static inline float wrap_to_pi(float a) {
    a = std::fmod(a + M_PI, 2.0 * M_PI);
    if (a < 0.0) a += 2.0 * M_PI;
    return a - M_PI;
}

// ---------------------------------------------------------------------------
// ONNX Policy wrapper (identical to g1_rl_deploy)
// ---------------------------------------------------------------------------
class OnnxPolicy {
public:
    OnnxPolicy(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "go2_policy") {
        session_options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(),
                                                  session_options_);
        for (size_t i = 0; i < session_->GetInputCount(); ++i) {
            auto type_info = session_->GetInputTypeInfo(i);
            input_shapes_.push_back(
                type_info.GetTensorTypeAndShapeInfo().GetShape());
            auto name = session_->GetInputNameAllocated(i, allocator_);
            input_name_strs_.push_back(name.get());
            size_t size = 1;
            for (auto dim : input_shapes_.back()) size *= dim;
            input_sizes_.push_back(size);
        }
        for (auto& s : input_name_strs_) input_names_.push_back(s.c_str());
        auto out_type = session_->GetOutputTypeInfo(0);
        output_shape_ = out_type.GetTensorTypeAndShapeInfo().GetShape();
        auto out_name = session_->GetOutputNameAllocated(0, allocator_);
        output_name_str_ = out_name.get();
        output_name_ = output_name_str_.c_str();
    }

    std::vector<float> infer(std::vector<float>& obs) {
        auto memory_info =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::vector<Ort::Value> input_tensors;
        for (size_t i = 0; i < input_names_.size(); ++i) {
            auto tensor = Ort::Value::CreateTensor<float>(
                memory_info, obs.data(), input_sizes_[i],
                input_shapes_[i].data(), input_shapes_[i].size());
            input_tensors.push_back(std::move(tensor));
        }
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), input_tensors.data(), input_tensors.size(),
            &output_name_, 1);
        auto* floatarr =
            output_tensors.front().GetTensorMutableData<float>();
        return std::vector<float>(floatarr, floatarr + output_shape_[1]);
    }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<std::string> input_name_strs_;
    std::vector<const char*> input_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<int64_t> input_sizes_;
    std::string output_name_str_;
    const char* output_name_;
    std::vector<int64_t> output_shape_;
};

// ---------------------------------------------------------------------------
// ROS2 Node — Go2 RL Policy Deploy
// ---------------------------------------------------------------------------
class Go2RLControllerNode : public rclcpp::Node {
public:
    Go2RLControllerNode()
        : Node("rl_controller_node"), time_(0.0), gait_index_(0.0f),
          running_policy_(false), state_received_(false),
          has_imu_(false), has_prev_base_(false) {

        // Declare parameters
        this->declare_parameter<std::string>("model_path", "");
        this->declare_parameter<double>("control_dt", 0.02);
        this->declare_parameter<double>("publish_rate", 500.0);
        this->declare_parameter<double>("standup_duration", 3.0);
        this->declare_parameter<double>("action_scale", 0.25);
        this->declare_parameter<double>("lpf_cutoff_hz", 8.0);
        this->declare_parameter<double>("soft_limit_factor", 0.9);
        this->declare_parameter<double>("action_bound_margin", 1.0);
        this->declare_parameter<std::vector<double>>("default_joint_pos",
            std::vector<double>{
                0.1, -0.1, 0.1, -0.1,      // FL/FR/RL/RR hip
                0.8, 0.8, 1.0, 1.0,         // FL/FR/RL/RR thigh
                -1.5, -1.5, -1.5, -1.5});   // FL/FR/RL/RR calf

        // Read parameters
        std::string model_path = this->get_parameter("model_path").as_string();
        control_dt_ = this->get_parameter("control_dt").as_double();
        double publish_rate = this->get_parameter("publish_rate").as_double();
        standup_duration_ = this->get_parameter("standup_duration").as_double();
        action_scale_ = static_cast<float>(
            this->get_parameter("action_scale").as_double());
        float lpf_cutoff = static_cast<float>(
            this->get_parameter("lpf_cutoff_hz").as_double());
        soft_limit_factor_ = static_cast<float>(
            this->get_parameter("soft_limit_factor").as_double());
        action_bound_margin_ = static_cast<float>(
            this->get_parameter("action_bound_margin").as_double());
        auto default_pos_d =
            this->get_parameter("default_joint_pos").as_double_array();
        for (int i = 0; i < NUM_LEG; ++i)
            default_pos_[i] = static_cast<float>(default_pos_d[i]);

        // Compute LPF alpha
        lpf_alpha_ = std::exp(
            -2.0f * static_cast<float>(M_PI) * lpf_cutoff *
            static_cast<float>(control_dt_));

        // Compute soft joint limits and action bounds
        ComputeLimits();

        // Load policy
        RCLCPP_INFO(this->get_logger(), "Loading policy: %s", model_path.c_str());
        policy_ = std::make_unique<OnnxPolicy>(model_path);

        // Init state
        last_action_.resize(ACTION_DIM, 0.0f);
        filtered_action_.resize(ACTION_DIM, 0.0f);
        desired_positions_.assign(default_pos_.begin(), default_pos_.end());

        auto qos = rclcpp::SensorDataQoS().keep_last(1);

        // Subscribers
        joint_states_sub_ = this->create_subscription<
            sensor_msgs::msg::JointState>(
            "/joint_states", qos,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
                JointStatesCallback(msg);
            });

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu", qos,
            [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                ImuCallback(msg);
            });

        base_pose_sub_ = this->create_subscription<
            geometry_msgs::msg::PoseStamped>(
            "/pose/base_link", qos,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                BasePoseCallback(msg);
            });

        goal_sub_ = this->create_subscription<
            geometry_msgs::msg::PoseStamped>(
            "/goal", qos,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                GoalCallback(msg);
            });

        // Publishers
        joint_cmd_pub_ = this->create_publisher<
            sensor_msgs::msg::JointState>("/joint_commands", qos);
        policy_active_pub_ = this->create_publisher<
            std_msgs::msg::Bool>("/policy_active", qos);

        // Services
        set_goal_srv_ = this->create_service<go2_interfaces::srv::SetGoal>(
            "/set_goal",
            [this](const go2_interfaces::srv::SetGoal::Request::SharedPtr req,
                   go2_interfaces::srv::SetGoal::Response::SharedPtr res) {
                SetGoalService(req, res);
            });

        toggle_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "/toggle_policy_mode",
            [this](const std_srvs::srv::Trigger::Request::SharedPtr,
                   std_srvs::srv::Trigger::Response::SharedPtr res) {
                TogglePolicyMode(res);
            });

        // Timers
        control_timer_ = this->create_wall_timer(
            std::chrono::microseconds(
                static_cast<int>(control_dt_ * 1e6)),
            [this] { Control(); });

        publish_timer_ = this->create_wall_timer(
            std::chrono::microseconds(
                static_cast<int>(1e6 / publish_rate)),
            [this] { PublishCommand(); });

        RCLCPP_INFO(this->get_logger(),
            "Go2 RL controller ready (control=%.0fHz, publish=%.0fHz)",
            1.0 / control_dt_, publish_rate);
    }

private:
    // ----- Limits computation -----
    void ComputeLimits() {
        for (int i = 0; i < NUM_LEG; ++i) {
            float center = 0.5f * (HARD_LIMIT_LOW[i] + HARD_LIMIT_HIGH[i]);
            float half = 0.5f * (HARD_LIMIT_HIGH[i] - HARD_LIMIT_LOW[i])
                         * soft_limit_factor_;
            soft_limit_low_[i] = center - half;
            soft_limit_high_[i] = center + half;

            float a_low = (soft_limit_low_[i] - default_pos_[i]) / action_scale_;
            float a_high = (soft_limit_high_[i] - default_pos_[i]) / action_scale_;
            float a_center = 0.5f * (a_low + a_high);
            float a_half = 0.5f * (a_high - a_low) * action_bound_margin_;
            action_low_[i] = a_center - a_half;
            action_high_[i] = a_center + a_half;
        }
    }

    // ----- Callbacks -----
    void JointStatesCallback(
        const sensor_msgs::msg::JointState::SharedPtr msg) {
        for (size_t j = 0; j < msg->name.size(); ++j) {
            for (int i = 0; i < NUM_LEG; ++i) {
                if (msg->name[j] == LEG_JOINT_NAMES[i]) {
                    if (j < msg->position.size())
                        motor_q_[i] = static_cast<float>(msg->position[j]);
                    if (j < msg->velocity.size())
                        motor_dq_[i] = static_cast<float>(msg->velocity[j]);
                    break;
                }
            }
            if (msg->name[j] == PENDULUM_JOINT1_NAME) {
                if (j < msg->position.size())
                    pendulum_pos_[0] = static_cast<float>(msg->position[j]);
                if (j < msg->velocity.size())
                    pendulum_vel_[0] = static_cast<float>(msg->velocity[j]);
            } else if (msg->name[j] == PENDULUM_JOINT2_NAME) {
                if (j < msg->position.size())
                    pendulum_pos_[1] = static_cast<float>(msg->position[j]);
                if (j < msg->velocity.size())
                    pendulum_vel_[1] = static_cast<float>(msg->velocity[j]);
            }
        }
        state_received_ = true;
    }

    void ImuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        imu_quat_[0] = static_cast<float>(msg->orientation.w);
        imu_quat_[1] = static_cast<float>(msg->orientation.x);
        imu_quat_[2] = static_cast<float>(msg->orientation.y);
        imu_quat_[3] = static_cast<float>(msg->orientation.z);
        imu_gyro_[0] = static_cast<float>(msg->angular_velocity.x);
        imu_gyro_[1] = static_cast<float>(msg->angular_velocity.y);
        imu_gyro_[2] = static_cast<float>(msg->angular_velocity.z);
        has_imu_ = true;
    }

    void BasePoseCallback(
        const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        double now_s = this->now().seconds();

        Eigen::Vector3f pos(
            static_cast<float>(msg->pose.position.x),
            static_cast<float>(msg->pose.position.y),
            static_cast<float>(msg->pose.position.z));

        Eigen::Quaternionf quat(
            static_cast<float>(msg->pose.orientation.w),
            static_cast<float>(msg->pose.orientation.x),
            static_cast<float>(msg->pose.orientation.y),
            static_cast<float>(msg->pose.orientation.z));
        quat.normalize();

        // Numerical differentiation for base linear velocity
        if (has_prev_base_) {
            float dt = static_cast<float>(now_s - prev_base_time_);
            if (dt > 1e-6f) {
                Eigen::Vector3f vel_w = (pos - prev_base_pos_) / dt;
                Eigen::Vector3f vel_b = quat.conjugate() * vel_w;
                base_lin_vel_b_[0] = vel_b.x();
                base_lin_vel_b_[1] = vel_b.y();
                base_lin_vel_b_[2] = vel_b.z();
            }
        }

        base_pos_ = pos;
        base_quat_ = quat;
        prev_base_pos_ = pos;
        prev_base_time_ = now_s;
        has_prev_base_ = true;

        // Extract yaw for goal tracking
        float siny = 2.0f * (quat.w() * quat.z() + quat.x() * quat.y());
        float cosy = 1.0f - 2.0f * (quat.y() * quat.y() + quat.z() * quat.z());
        base_yaw_ = std::atan2(siny, cosy);
    }

    void GoalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        target_x_ = static_cast<float>(msg->pose.position.x);
        target_y_ = static_cast<float>(msg->pose.position.y);
        auto& q = msg->pose.orientation;
        float siny = 2.0f * (static_cast<float>(q.w) * static_cast<float>(q.z)
                            + static_cast<float>(q.x) * static_cast<float>(q.y));
        float cosy = 1.0f - 2.0f * (static_cast<float>(q.y) * static_cast<float>(q.y)
                                    + static_cast<float>(q.z) * static_cast<float>(q.z));
        target_yaw_ = std::atan2(siny, cosy);
    }

    void SetGoalService(
        const go2_interfaces::srv::SetGoal::Request::SharedPtr req,
        go2_interfaces::srv::SetGoal::Response::SharedPtr res) {
        target_x_ = static_cast<float>(req->goal.pose.position.x);
        target_y_ = static_cast<float>(req->goal.pose.position.y);
        auto& q = req->goal.pose.orientation;
        float siny = 2.0f * (static_cast<float>(q.w) * static_cast<float>(q.z)
                            + static_cast<float>(q.x) * static_cast<float>(q.y));
        float cosy = 1.0f - 2.0f * (static_cast<float>(q.y) * static_cast<float>(q.y)
                                    + static_cast<float>(q.z) * static_cast<float>(q.z));
        target_yaw_ = std::atan2(siny, cosy);
        res->success = true;
        res->message = "Goal updated";
    }

    void TogglePolicyMode(
        std_srvs::srv::Trigger::Response::SharedPtr res) {
        if (running_policy_) {
            running_policy_ = false;
            time_ = 0.0;
            ResetPolicyState();
            res->success = true;
            res->message = "mode=stand";
            RCLCPP_INFO(this->get_logger(), "Mode switched to stand");
        } else {
            running_policy_ = false;
            time_ = 0.0;  // restart standup
            ResetPolicyState();
            res->success = true;
            res->message = "mode=policy (standup first)";
            RCLCPP_INFO(this->get_logger(),
                "Mode switched to policy (standing up for %.1fs first)",
                standup_duration_);
        }
    }

    void ResetPolicyState() {
        std::fill(last_action_.begin(), last_action_.end(), 0.0f);
        std::fill(filtered_action_.begin(), filtered_action_.end(), 0.0f);
        gait_index_ = 0.0f;
        std::fill(std::begin(clock_inputs_), std::end(clock_inputs_), 0.0f);
    }

    // ----- Main control loop (50 Hz) -----
    void Control() {
        if (!state_received_) return;

        sensor_msgs::msg::JointState cmd;
        cmd.header.stamp = this->now();

        time_ += control_dt_;

        if (time_ < standup_duration_) {
            // Standup: interpolate to default positions
            float ratio = std::clamp(
                static_cast<float>(time_ / standup_duration_), 0.0f, 1.0f);
            for (int i = 0; i < NUM_LEG; ++i) {
                desired_positions_[i] =
                    (1.0f - ratio) * motor_q_[i] + ratio * default_pos_[i];
            }
            if (!running_policy_) {
                static bool printed = false;
                if (!printed) {
                    RCLCPP_INFO(this->get_logger(),
                        "Phase 1: Standing up (%.0fs)...", standup_duration_);
                    printed = true;
                }
            }
        } else {
            if (!running_policy_) {
                running_policy_ = true;
                RCLCPP_INFO(this->get_logger(), "Phase 2: Policy active");
            }

            if (!has_imu_ || !has_prev_base_) {
                RCLCPP_WARN_THROTTLE(this->get_logger(),
                    *this->get_clock(), 2000,
                    "Waiting for /imu and /pose/base_link...");
                return;
            }

            // Build 56-dim observation
            std::vector<float> obs;
            obs.reserve(OBS_DIM);

            // [0:3] base_lin_vel_b (differentiated from /pose/base_link)
            for (int i = 0; i < 3; ++i)
                obs.push_back(base_lin_vel_b_[i]);

            // [3:6] base_ang_vel_b (from IMU gyroscope)
            for (int i = 0; i < 3; ++i)
                obs.push_back(imu_gyro_[i]);

            // [6:9] projected_gravity_b
            Eigen::Quaternionf q(
                imu_quat_[0], imu_quat_[1], imu_quat_[2], imu_quat_[3]);
            Eigen::Vector3f gravity_b =
                q.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
            obs.push_back(gravity_b.x());
            obs.push_back(gravity_b.y());
            obs.push_back(gravity_b.z());

            // [9:12] goal_error_b
            float dx_w = target_x_ - base_pos_.x();
            float dy_w = target_y_ - base_pos_.y();
            float c = std::cos(base_yaw_);
            float s = std::sin(base_yaw_);
            obs.push_back(c * dx_w + s * dy_w);       // err_x_b
            obs.push_back(-s * dx_w + c * dy_w);      // err_y_b
            obs.push_back(wrap_to_pi(target_yaw_ - base_yaw_));

            // [12:24] leg_joint_pos_rel
            for (int i = 0; i < NUM_LEG; ++i)
                obs.push_back(motor_q_[i] - default_pos_[i]);

            // [24:36] leg_joint_vel
            for (int i = 0; i < NUM_LEG; ++i)
                obs.push_back(motor_dq_[i]);

            // [36:38] pendulum_pos
            obs.push_back(pendulum_pos_[0]);
            obs.push_back(pendulum_pos_[1]);

            // [38:40] pendulum_vel
            obs.push_back(pendulum_vel_[0]);
            obs.push_back(pendulum_vel_[1]);

            // [40:52] prev_action
            for (int i = 0; i < ACTION_DIM; ++i)
                obs.push_back(last_action_[i]);

            // [52:56] clock_inputs
            for (int i = 0; i < 4; ++i)
                obs.push_back(clock_inputs_[i]);

            if (static_cast<int>(obs.size()) != OBS_DIM) {
                RCLCPP_ERROR(this->get_logger(),
                    "Obs dim mismatch: %zu != %d",
                    obs.size(), OBS_DIM);
                return;
            }

            // Infer
            auto raw_action = policy_->infer(obs);

            // Action pipeline: clamp -> LPF -> clamp -> scale -> hard clamp
            for (int i = 0; i < ACTION_DIM; ++i) {
                float bounded = std::clamp(
                    raw_action[i], action_low_[i], action_high_[i]);
                float filtered = lpf_alpha_ * filtered_action_[i]
                                + (1.0f - lpf_alpha_) * bounded;
                filtered = std::clamp(
                    filtered, action_low_[i], action_high_[i]);
                filtered_action_[i] = filtered;
                last_action_[i] = filtered;

                float desired = default_pos_[i]
                              + action_scale_ * filtered;
                desired_positions_[i] = std::clamp(
                    desired, soft_limit_low_[i], soft_limit_high_[i]);
            }

            // Update gait clock
            UpdateClockInputs();

            static int print_counter = 0;
            if (++print_counter % 250 == 0) {
                RCLCPP_INFO(this->get_logger(),
                    "t=%.1f pendulum=[%.3f,%.3f]",
                    time_, pendulum_pos_[0], pendulum_pos_[1]);
            }
        }
    }

    // ----- 500 Hz publish -----
    void PublishCommand() {
        sensor_msgs::msg::JointState cmd;
        cmd.header.stamp = this->now();
        cmd.name.assign(LEG_JOINT_NAMES.begin(), LEG_JOINT_NAMES.end());
        cmd.position.resize(NUM_LEG);
        for (int i = 0; i < NUM_LEG; ++i)
            cmd.position[i] = desired_positions_[i];
        joint_cmd_pub_->publish(cmd);

        // Publish phase indicator for bridge gain switching
        std_msgs::msg::Bool phase_msg;
        phase_msg.data = running_policy_;
        policy_active_pub_->publish(phase_msg);
    }

    // ----- Gait clock (same as Isaac training) -----
    void UpdateClockInputs() {
        constexpr float freq = 3.0f;
        constexpr float phase = 0.5f;
        constexpr float offset = 0.0f;
        constexpr float bound = 0.0f;
        constexpr float duration = 0.5f;

        gait_index_ = std::fmod(
            gait_index_ + static_cast<float>(control_dt_) * freq, 1.0f);

        float foot_indices[4] = {
            gait_index_ + phase + offset + bound,
            gait_index_ + offset,
            gait_index_ + bound,
            gait_index_ + phase,
        };

        for (int f = 0; f < 4; ++f) {
            float r = std::fmod(foot_indices[f], 1.0f);
            if (r < 0.0f) r += 1.0f;
            float remapped;
            if (r < duration)
                remapped = r * (0.5f / duration);
            else
                remapped = 0.5f + (r - duration) * (0.5f / (1.0f - duration));
            clock_inputs_[f] = std::sin(
                2.0f * static_cast<float>(M_PI) * remapped);
        }
    }

    // ----- ROS2 -----
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
        joint_states_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
        base_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
        goal_sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr
        joint_cmd_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr policy_active_pub_;
    rclcpp::Service<go2_interfaces::srv::SetGoal>::SharedPtr set_goal_srv_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr toggle_srv_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp::TimerBase::SharedPtr publish_timer_;

    // ----- Policy -----
    std::unique_ptr<OnnxPolicy> policy_;
    std::vector<float> last_action_;
    std::vector<float> filtered_action_;
    std::vector<float> desired_positions_;

    // ----- Parameters -----
    std::array<float, NUM_LEG> default_pos_{};
    double control_dt_, standup_duration_;
    float action_scale_, lpf_alpha_;
    float soft_limit_factor_, action_bound_margin_;

    // ----- Computed limits -----
    std::array<float, NUM_LEG> soft_limit_low_{};
    std::array<float, NUM_LEG> soft_limit_high_{};
    std::array<float, NUM_LEG> action_low_{};
    std::array<float, NUM_LEG> action_high_{};

    // ----- Sensor state -----
    std::array<float, NUM_LEG> motor_q_{};
    std::array<float, NUM_LEG> motor_dq_{};
    std::array<float, 4> imu_quat_{};
    std::array<float, 3> imu_gyro_{};

    // Base pose + differentiation
    Eigen::Vector3f base_pos_{Eigen::Vector3f::Zero()};
    Eigen::Vector3f prev_base_pos_{Eigen::Vector3f::Zero()};
    Eigen::Quaternionf base_quat_{Eigen::Quaternionf::Identity()};
    float base_lin_vel_b_[3] = {};
    float base_yaw_ = 0.0f;
    double prev_base_time_ = 0.0;

    // Pendulum (read from /joint_states, estimated by go2_bridge)
    float pendulum_pos_[2] = {};
    float pendulum_vel_[2] = {};

    // Goal
    float target_x_ = 0.0f, target_y_ = 0.0f, target_yaw_ = 0.0f;

    // Mode
    float gait_index_;
    float clock_inputs_[4] = {};
    double time_;
    bool running_policy_;
    bool state_received_;
    bool has_imu_;
    bool has_prev_base_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Go2RLControllerNode>());
    rclcpp::shutdown();
    return 0;
}
