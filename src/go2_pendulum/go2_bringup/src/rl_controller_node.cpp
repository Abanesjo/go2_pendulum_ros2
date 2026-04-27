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

static inline float wrap_to_pi(float a) {
    a = std::fmod(a + M_PI, 2.0 * M_PI);
    if (a < 0.0) a += 2.0 * M_PI;
    return a - M_PI;
}

enum class ControllerState {
    SITTING,
    STANDING_UP,
    STANDING,
    SITTING_DOWN,
    POLICY,
    DAMPED,
};

static const char* StateName(ControllerState state) {
    switch (state) {
        case ControllerState::SITTING:
            return "sitting";
        case ControllerState::STANDING_UP:
            return "standing_up";
        case ControllerState::STANDING:
            return "standing";
        case ControllerState::SITTING_DOWN:
            return "sitting_down";
        case ControllerState::POLICY:
            return "policy";
        case ControllerState::DAMPED:
            return "damped";
    }
    return "unknown";
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
          startup_delay_elapsed_(0.0),
          running_policy_(false), state_received_(false),
          has_imu_(false), has_goal_target_(false), has_base_pose_(false),
          has_prev_base_for_obs_(false), emergency_damp_(false),
          standup_initialized_(false), command_ready_(false),
          standup_goal_latched_(false),
          logged_first_joint_state_(false),
          logged_first_command_publish_(false),
          state_(ControllerState::SITTING), transition_elapsed_(0.0),
          transition_duration_(0.0) {

        // Declare parameters
        this->declare_parameter<std::string>("model_path", "");
        this->declare_parameter<double>("control_dt", 0.02);
        this->declare_parameter<double>("publish_rate", 500.0);
        this->declare_parameter<double>("standup_duration", 3.0);
        this->declare_parameter<double>("sit_duration", 3.0);
        this->declare_parameter<double>("action_scale", 0.25);
        this->declare_parameter<bool>("zero_pendulum", false);
        this->declare_parameter<std::vector<double>>("default_joint_pos",
            std::vector<double>{
                0.1, -0.1, 0.1, -0.1,      // FL/FR/RL/RR hip
                0.8, 0.8, 1.0, 1.0,         // FL/FR/RL/RR thigh
                -1.5, -1.5, -1.5, -1.5});   // FL/FR/RL/RR calf
        this->declare_parameter<std::vector<double>>("sit_joint_pos",
            std::vector<double>{
                0.0, 0.0, 0.0, 0.0,         // FL/FR/RL/RR hip
                1.3, 1.3, 1.3, 1.3,         // FL/FR/RL/RR thigh
                -2.5, -2.5, -2.5, -2.5});   // FL/FR/RL/RR calf

        // Read parameters
        std::string model_path = this->get_parameter("model_path").as_string();
        control_dt_ = this->get_parameter("control_dt").as_double();
        double publish_rate = this->get_parameter("publish_rate").as_double();
        standup_duration_ = this->get_parameter("standup_duration").as_double();
        sit_duration_ = this->get_parameter("sit_duration").as_double();
        action_scale_ = static_cast<float>(
            this->get_parameter("action_scale").as_double());
        zero_pendulum_ = this->get_parameter("zero_pendulum").as_bool();
        auto default_pos_d =
            this->get_parameter("default_joint_pos").as_double_array();
        for (int i = 0; i < NUM_LEG; ++i)
            default_pos_[i] = static_cast<float>(default_pos_d[i]);
        auto sit_pos_d =
            this->get_parameter("sit_joint_pos").as_double_array();
        for (int i = 0; i < NUM_LEG; ++i)
            sit_pos_[i] = static_cast<float>(sit_pos_d[i]);

        // Load policy
        RCLCPP_INFO(this->get_logger(), "Loading policy: %s", model_path.c_str());
        policy_ = std::make_unique<OnnxPolicy>(model_path);

        // Init state
        last_action_.resize(ACTION_DIM, 0.0f);
        desired_positions_.resize(NUM_LEG, 0.0f);

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

        policy_toggle_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/policy_toggle_request", qos,
            [this](const std_msgs::msg::Bool::SharedPtr msg) {
                PolicyToggleCallback(msg);
            });

        stand_request_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/stand_request", qos,
            [this](const std_msgs::msg::Bool::SharedPtr msg) {
                StandRequestCallback(msg);
            });

        sit_request_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/sit_request", qos,
            [this](const std_msgs::msg::Bool::SharedPtr msg) {
                SitRequestCallback(msg);
            });

        emergency_damp_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/emergency_damp", qos,
            [this](const std_msgs::msg::Bool::SharedPtr msg) {
                EmergencyDampCallback(msg);
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
    // ----- Callbacks -----
    void JointStatesCallback(
        const sensor_msgs::msg::JointState::SharedPtr msg) {
        std::array<bool, NUM_LEG> leg_position_seen{};
        for (size_t j = 0; j < msg->name.size(); ++j) {
            for (int i = 0; i < NUM_LEG; ++i) {
                if (msg->name[j] == LEG_JOINT_NAMES[i]) {
                    if (j < msg->position.size()) {
                        motor_q_[i] = static_cast<float>(msg->position[j]);
                        leg_position_seen[i] = true;
                    }
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
        state_received_ = std::all_of(
            leg_position_seen.begin(), leg_position_seen.end(),
            [](bool seen) { return seen; });

        if (state_received_ && !logged_first_joint_state_) {
            logged_first_joint_state_ = true;
            RCLCPP_INFO(this->get_logger(),
                "Received first complete /joint_states (FL_hip=%.3f)",
                motor_q_[0]);
        }
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

        base_pos_ = pos;
        base_quat_ = quat;
        has_base_pose_ = true;

        // Extract yaw for goal tracking
        float siny = 2.0f * (quat.w() * quat.z() + quat.x() * quat.y());
        float cosy = 1.0f - 2.0f * (quat.y() * quat.y() + quat.z() * quat.z());
        base_yaw_ = std::atan2(siny, cosy);
    }

    void GoalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        if (emergency_damp_) return;
        if (!CanAcceptExternalGoal()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(),
                *this->get_clock(), 2000,
                "Ignoring /goal until policy is active");
            return;
        }

        target_x_ = static_cast<float>(msg->pose.position.x);
        target_y_ = static_cast<float>(msg->pose.position.y);
        auto& q = msg->pose.orientation;
        float siny = 2.0f * (static_cast<float>(q.w) * static_cast<float>(q.z)
                            + static_cast<float>(q.x) * static_cast<float>(q.y));
        float cosy = 1.0f - 2.0f * (static_cast<float>(q.y) * static_cast<float>(q.y)
                                    + static_cast<float>(q.z) * static_cast<float>(q.z));
        target_yaw_ = std::atan2(siny, cosy);
        has_goal_target_ = true;
        RCLCPP_INFO(this->get_logger(),
            "Goal updated while policy is active");
    }

    void SetGoalService(
        const go2_interfaces::srv::SetGoal::Request::SharedPtr req,
        go2_interfaces::srv::SetGoal::Response::SharedPtr res) {
        if (emergency_damp_) {
            res->success = false;
            res->message = "Emergency damping latched; restart launch to continue";
            return;
        }
        if (!CanAcceptExternalGoal()) {
            res->success = false;
            res->message = "Ignoring goal until policy is active";
            return;
        }

        target_x_ = static_cast<float>(req->goal.pose.position.x);
        target_y_ = static_cast<float>(req->goal.pose.position.y);
        auto& q = req->goal.pose.orientation;
        float siny = 2.0f * (static_cast<float>(q.w) * static_cast<float>(q.z)
                            + static_cast<float>(q.x) * static_cast<float>(q.y));
        float cosy = 1.0f - 2.0f * (static_cast<float>(q.y) * static_cast<float>(q.y)
                                    + static_cast<float>(q.z) * static_cast<float>(q.z));
        target_yaw_ = std::atan2(siny, cosy);
        has_goal_target_ = true;
        res->success = true;
        res->message = "Goal updated while policy is active";
    }

    void PolicyToggleCallback(const std_msgs::msg::Bool::SharedPtr msg) {
        if (!msg->data) return;
        HandlePolicyToggle("joy button[10]");
    }

    void StandRequestCallback(const std_msgs::msg::Bool::SharedPtr msg) {
        if (!msg->data) return;
        HandleStandRequest("joy button[11]");
    }

    void SitRequestCallback(const std_msgs::msg::Bool::SharedPtr msg) {
        if (!msg->data) return;
        HandleSitRequest("joy button[12]");
    }

    void EmergencyDampCallback(const std_msgs::msg::Bool::SharedPtr msg) {
        if (!msg->data || emergency_damp_) return;

        emergency_damp_ = true;
        state_ = ControllerState::DAMPED;
        running_policy_ = false;
        ResetPolicyState();
        SetDefaultDesiredPositions();
        command_ready_ = true;
        RCLCPP_FATAL(this->get_logger(),
            "Emergency damping latched; publishing default targets with policy inactive");
    }

    void TogglePolicyMode(
        std_srvs::srv::Trigger::Response::SharedPtr res) {
        if (emergency_damp_) {
            res->success = false;
            res->message = "Emergency damping latched; restart launch to continue";
            return;
        }

        const bool accepted = HandlePolicyToggle("toggle service");
        res->success = accepted;
        res->message = accepted
            ? std::string("state=") + StateName(state_)
            : std::string("policy toggle ignored in state=") + StateName(state_);
    }

    void SetDefaultDesiredPositions() {
        for (int i = 0; i < NUM_LEG; ++i)
            desired_positions_[i] = default_pos_[i];
    }

    void SetSitDesiredPositions() {
        for (int i = 0; i < NUM_LEG; ++i)
            desired_positions_[i] = sit_pos_[i];
    }

    bool BeginPoseTransition(
        ControllerState transition_state,
        const std::array<float, NUM_LEG>& target,
        double duration,
        const char* source) {
        if (!command_ready_) {
            RCLCPP_WARN(this->get_logger(),
                "Ignoring %s until startup has latched measured joint positions",
                source);
            return false;
        }

        running_policy_ = false;
        ResetPolicyState();
        if (transition_state == ControllerState::STANDING_UP) {
            has_goal_target_ = false;
            standup_goal_latched_ = false;
        }
        for (int i = 0; i < NUM_LEG; ++i) {
            transition_start_pos_[i] = desired_positions_[i];
            transition_target_pos_[i] = target[i];
        }
        transition_elapsed_ = 0.0;
        transition_duration_ = std::max(duration, control_dt_);
        state_ = transition_state;

        RCLCPP_INFO(this->get_logger(),
            "%s accepted; state=%s duration=%.2fs",
            source, StateName(state_), transition_duration_);
        return true;
    }

    void UpdatePoseTransition(ControllerState complete_state) {
        transition_elapsed_ += control_dt_;
        const float ratio = std::clamp(
            static_cast<float>(transition_elapsed_ / transition_duration_),
            0.0f, 1.0f);
        for (int i = 0; i < NUM_LEG; ++i) {
            desired_positions_[i] =
                (1.0f - ratio) * transition_start_pos_[i]
                + ratio * transition_target_pos_[i];
        }

        if (ratio >= 1.0f) {
            state_ = complete_state;
            if (state_ == ControllerState::STANDING) {
                SetDefaultDesiredPositions();
                LatchGoalFromCurrentPose(true, "standup complete");
            } else if (state_ == ControllerState::SITTING) {
                SetSitDesiredPositions();
            }
            RCLCPP_INFO(this->get_logger(),
                "Pose transition complete; state=%s", StateName(state_));
        }
    }

    bool HandlePolicyToggle(const char* source) {
        if (emergency_damp_) return false;

        if (state_ == ControllerState::POLICY) {
            running_policy_ = false;
            ResetPolicyState();
            SetDefaultDesiredPositions();
            state_ = ControllerState::STANDING;
            LatchGoalFromCurrentPose(true, "policy stopped");
            RCLCPP_INFO(this->get_logger(),
                "%s accepted; policy stopped, state=standing", source);
            return true;
        }

        if (state_ != ControllerState::STANDING) {
            RCLCPP_WARN(this->get_logger(),
                "%s ignored; policy can only start from standing "
                "(current state=%s)",
                source, StateName(state_));
            return false;
        }

        if (!standup_goal_latched_
            && !LatchGoalFromCurrentPose(true, "policy toggle")) {
            RCLCPP_WARN(this->get_logger(),
                "%s ignored; waiting for /pose/base_link to set default goal",
                source);
            return false;
        }

        running_policy_ = true;
        ResetPolicyState();
        SetDefaultDesiredPositions();
        state_ = ControllerState::POLICY;
        RCLCPP_INFO(this->get_logger(),
            "%s accepted; policy will run when sensors are ready", source);
        return true;
    }

    bool HandleStandRequest(const char* source) {
        if (emergency_damp_) return false;

        if (state_ == ControllerState::SITTING
            || state_ == ControllerState::SITTING_DOWN) {
            return BeginPoseTransition(
                ControllerState::STANDING_UP,
                default_pos_,
                standup_duration_,
                source);
        }

        RCLCPP_WARN(this->get_logger(),
            "%s ignored; stand is valid from sitting/sitting_down "
            "(current state=%s)",
            source, StateName(state_));
        return false;
    }

    bool CanAcceptExternalGoal() const {
        return standup_goal_latched_ && state_ == ControllerState::POLICY;
    }

    bool LatchGoalFromCurrentPose(bool force, const char* source) {
        if (!force && has_goal_target_) return true;
        if (!has_base_pose_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(),
                *this->get_clock(), 2000,
                "%s: waiting for /pose/base_link to latch current-pose goal",
                source);
            return false;
        }

        target_x_ = base_pos_.x();
        target_y_ = base_pos_.y();
        target_yaw_ = base_yaw_;
        has_goal_target_ = true;
        standup_goal_latched_ = true;
        RCLCPP_INFO(this->get_logger(),
            "%s: goal latched at current pose x=%.3f y=%.3f yaw=%.3f",
            source, target_x_, target_y_, target_yaw_);
        return true;
    }

    bool HandleSitRequest(const char* source) {
        if (emergency_damp_) return false;

        if (state_ == ControllerState::STANDING
            || state_ == ControllerState::STANDING_UP) {
            return BeginPoseTransition(
                ControllerState::SITTING_DOWN,
                sit_pos_,
                sit_duration_,
                source);
        }

        RCLCPP_WARN(this->get_logger(),
            "%s ignored; sit is valid from standing/standing_up "
            "(current state=%s)",
            source, StateName(state_));
        return false;
    }

    void ResetPolicyState() {
        std::fill(last_action_.begin(), last_action_.end(), 0.0f);
        gait_index_ = 0.0f;
        std::fill(std::begin(clock_inputs_), std::end(clock_inputs_), 0.0f);
        std::fill(std::begin(base_lin_vel_b_), std::end(base_lin_vel_b_), 0.0f);
        prev_control_base_pos_ = base_pos_;
        has_prev_base_for_obs_ = has_base_pose_;
    }

    void UpdateBaseVelocityEstimate() {
        if (!has_base_pose_) return;

        if (has_prev_base_for_obs_) {
            const float dt = static_cast<float>(control_dt_);
            Eigen::Vector3f vel_w = (base_pos_ - prev_control_base_pos_) / dt;
            Eigen::Vector3f vel_b = base_quat_.conjugate() * vel_w;
            base_lin_vel_b_[0] = vel_b.x();
            base_lin_vel_b_[1] = vel_b.y();
            base_lin_vel_b_[2] = vel_b.z();
        } else {
            std::fill(std::begin(base_lin_vel_b_), std::end(base_lin_vel_b_), 0.0f);
        }

        prev_control_base_pos_ = base_pos_;
        has_prev_base_for_obs_ = true;
    }

    // ----- Main control loop (50 Hz) -----
    void Control() {
        if (!state_received_) return;

        if (emergency_damp_) {
            running_policy_ = false;
            state_ = ControllerState::DAMPED;
            SetDefaultDesiredPositions();
            command_ready_ = true;
            return;
        }

        if (!standup_initialized_) {
            startup_delay_elapsed_ += control_dt_;
            if (startup_delay_elapsed_ < STARTUP_DELAY_SEC) {
                RCLCPP_INFO_THROTTLE(this->get_logger(),
                    *this->get_clock(), 2000,
                    "Waiting %.1fs before standup initialization",
                    STARTUP_DELAY_SEC);
                return;
            }

            standup_start_pos_ = motor_q_;
            for (int i = 0; i < NUM_LEG; ++i)
                desired_positions_[i] = standup_start_pos_[i];

            standup_initialized_ = true;
            command_ready_ = true;
            time_ = 0.0;
            state_ = ControllerState::SITTING;
            RCLCPP_INFO(this->get_logger(),
                "Startup initialized; holding measured sitting pose");
            return;
        }

        time_ += control_dt_;
        UpdateBaseVelocityEstimate();

        switch (state_) {
            case ControllerState::SITTING:
                running_policy_ = false;
                return;

            case ControllerState::STANDING_UP:
                UpdatePoseTransition(ControllerState::STANDING);
                return;

            case ControllerState::STANDING:
                running_policy_ = false;
                SetDefaultDesiredPositions();
                if (!standup_goal_latched_) {
                    LatchGoalFromCurrentPose(true, "standup complete");
                }
                return;

            case ControllerState::SITTING_DOWN:
                UpdatePoseTransition(ControllerState::SITTING);
                return;

            case ControllerState::POLICY:
                running_policy_ = true;
                SetDefaultDesiredPositions();
                break;

            case ControllerState::DAMPED:
                running_policy_ = false;
                SetDefaultDesiredPositions();
                return;
        }

        if (!has_imu_ || !has_base_pose_) {
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
            obs.push_back(zero_pendulum_ ? 0.0f : pendulum_pos_[0]);
            obs.push_back(zero_pendulum_ ? 0.0f : pendulum_pos_[1]);

            // [38:40] pendulum_vel
            obs.push_back(zero_pendulum_ ? 0.0f : pendulum_vel_[0]);
            obs.push_back(zero_pendulum_ ? 0.0f : pendulum_vel_[1]);

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

            // Map the raw policy output directly into joint position targets.
            for (int i = 0; i < ACTION_DIM; ++i) {
                last_action_[i] = raw_action[i];
                desired_positions_[i] =
                    default_pos_[i] + action_scale_ * raw_action[i];
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

    // ----- Joint command publish -----
    void PublishCommand() {
        // Avoid sending a first-frame jump to default pose before standup
        // has latched the measured joint state as its start pose.
        if (!command_ready_) {
            std_msgs::msg::Bool phase_msg;
            phase_msg.data = false;
            policy_active_pub_->publish(phase_msg);
            return;
        }

        sensor_msgs::msg::JointState cmd;
        cmd.header.stamp = this->now();
        cmd.name.assign(LEG_JOINT_NAMES.begin(), LEG_JOINT_NAMES.end());
        cmd.position.resize(NUM_LEG);
        for (int i = 0; i < NUM_LEG; ++i)
            cmd.position[i] = desired_positions_[i];
        joint_cmd_pub_->publish(cmd);

        if (!logged_first_command_publish_) {
            logged_first_command_publish_ = true;
            RCLCPP_INFO(this->get_logger(),
                "Published first /joint_commands (FL_hip=%.3f)",
                desired_positions_[0]);
        }

        // Publish phase indicator for bridge gain switching
        std_msgs::msg::Bool phase_msg;
        phase_msg.data =
            running_policy_ && state_ == ControllerState::POLICY;
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
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr policy_toggle_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr stand_request_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sit_request_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr emergency_damp_sub_;
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
    std::vector<float> desired_positions_;

    // ----- Parameters -----
    static constexpr double STARTUP_DELAY_SEC = 1.0;
    std::array<float, NUM_LEG> default_pos_{};
    std::array<float, NUM_LEG> sit_pos_{};
    std::array<float, NUM_LEG> standup_start_pos_{};
    std::array<float, NUM_LEG> transition_start_pos_{};
    std::array<float, NUM_LEG> transition_target_pos_{};
    double control_dt_, standup_duration_, sit_duration_;
    float action_scale_;

    // ----- Sensor state -----
    std::array<float, NUM_LEG> motor_q_{};
    std::array<float, NUM_LEG> motor_dq_{};
    std::array<float, 4> imu_quat_{};
    std::array<float, 3> imu_gyro_{};

    // Base pose + differentiation
    Eigen::Vector3f base_pos_{Eigen::Vector3f::Zero()};
    Eigen::Vector3f prev_control_base_pos_{Eigen::Vector3f::Zero()};
    Eigen::Quaternionf base_quat_{Eigen::Quaternionf::Identity()};
    float base_lin_vel_b_[3] = {};
    float base_yaw_ = 0.0f;

    // Pendulum (read from /joint_states, estimated by go2_bridge)
    float pendulum_pos_[2] = {};
    float pendulum_vel_[2] = {};

    // Pendulum override
    bool zero_pendulum_ = false;

    // Goal
    float target_x_ = 0.0f, target_y_ = 0.0f, target_yaw_ = 0.0f;

    // Mode
    float gait_index_;
    float clock_inputs_[4] = {};
    double time_;
    double startup_delay_elapsed_;
    bool running_policy_;
    bool state_received_;
    bool has_imu_;
    bool has_goal_target_;
    bool has_base_pose_;
    bool has_prev_base_for_obs_;
    bool emergency_damp_;
    bool standup_initialized_;
    bool command_ready_;
    bool standup_goal_latched_;
    bool logged_first_joint_state_;
    bool logged_first_command_publish_;
    ControllerState state_;
    double transition_elapsed_;
    double transition_duration_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Go2RLControllerNode>());
    rclcpp::shutdown();
    return 0;
}
