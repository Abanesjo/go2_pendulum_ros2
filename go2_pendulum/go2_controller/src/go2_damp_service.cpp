#include <functional>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

#include "go2_controller/srv/go2_damp.hpp"

using unitree::robot::ChannelFactory;
using unitree::robot::go2::SportClient;

class Go2DampService final : public rclcpp::Node
{
public:
  Go2DampService()
  : rclcpp::Node("go2_damp_service")
  {
    const std::string iface = this->declare_parameter<std::string>("network_interface", "enp130s0");
    ChannelFactory::Instance()->Init(0, iface);

    sport_client_.SetTimeout(10.0f);
    sport_client_.Init();

    service_ = this->create_service<go2_controller::srv::Go2Damp>(
      "go2_damp",
      std::bind(
        &Go2DampService::HandleRequest,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Go2 damp service ready on interface '%s'", iface.c_str());
  }

private:
  void HandleRequest(
    const std::shared_ptr<go2_controller::srv::Go2Damp::Request>,
    std::shared_ptr<go2_controller::srv::Go2Damp::Response> response)
  {
    RCLCPP_WARN(this->get_logger(), "Damp mode will power down the motors. The robot will collapse if standing.");

    const int32_t ret = sport_client_.Damp();
    if (ret == 0) {
      response->success = true;
      response->message = "Damp mode activated successfully.";
      return;
    }

    response->success = false;
    response->message = "Failed to activate damp mode. Error code: " + std::to_string(ret);
  }

  SportClient sport_client_;
  rclcpp::Service<go2_controller::srv::Go2Damp>::SharedPtr service_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Go2DampService>());
  rclcpp::shutdown();
  return 0;
}
