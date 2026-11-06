// Run a YOLO model exported with embedded NMS on a camera topic, without Python.
//
// The input matches the ultralytics predictor: the image scaled to fit, padded evenly with 114,
// RGB, 0 to 1. The output rows are x0, y0, x1, y1, score, class in the padded image.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <vector>

#include "opencv2/imgproc.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

namespace
{

constexpr float kPad = 114.0F;
constexpr int kColumns = 6;

std::map<int, std::string> class_names(const Ort::Session & session)
{
  Ort::AllocatorWithDefaultOptions allocator;
  const auto text = session.GetModelMetadata().LookupCustomMetadataMapAllocated("names", allocator);
  std::map<int, std::string> names;
  if (!text) {
    return names;
  }
  const std::string value(text.get());
  const std::regex entry(R"((\d+): '([^']*)')");
  for (auto it = std::sregex_iterator(value.begin(), value.end(), entry);
    it != std::sregex_iterator(); ++it)
  {
    names[std::stoi((*it)[1])] = (*it)[2];
  }
  return names;
}

}  // namespace

class YoloOnnxNode : public rclcpp::Node
{
public:
  YoloOnnxNode()
  : Node("yolo_onnx"),
    env_(ORT_LOGGING_LEVEL_WARNING, "yolo_onnx"),
    session_(env_, declare_parameter<std::string>("model").c_str(), Ort::SessionOptions())
  {
    const auto shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    height_ = static_cast<int>(shape.at(2));
    width_ = static_cast<int>(shape.at(3));
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_.GetInputNameAllocated(0, allocator).get();
    output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
    names_ = class_names(session_);
    input_.resize(static_cast<std::size_t>(3 * height_ * width_));
    output_ = create_publisher<vision_msgs::msg::Detection2DArray>("~/detections", 10);
    subscription_ = create_subscription<sensor_msgs::msg::Image>(
      "/camera/image", rclcpp::QoS(1).best_effort(),
      [this](const sensor_msgs::msg::Image::ConstSharedPtr image) {on_image(*image);});
  }

private:
  void on_image(const sensor_msgs::msg::Image & image)
  {
    if (image.encoding != "rgb8" && image.encoding != "bgr8") {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000, "unsupported encoding %s", image.encoding.c_str());
      return;
    }
    const double scale = std::min(
      static_cast<double>(height_) / image.height, static_cast<double>(width_) / image.width);
    const int resized_height = static_cast<int>(std::round(image.height * scale));
    const int resized_width = static_cast<int>(std::round(image.width * scale));
    const double pad_top = (height_ - resized_height) / 2.0;
    const double pad_left = (width_ - resized_width) / 2.0;
    const int top = static_cast<int>(std::round(pad_top - 0.1));
    const int left = static_cast<int>(std::round(pad_left - 0.1));
    const cv::Mat source(
      static_cast<int>(image.height), static_cast<int>(image.width), CV_8UC3,
      const_cast<std::uint8_t *>(image.data.data()), image.step);
    cv::resize(source, resized_, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);
    fill_input(image.encoding == "bgr8", top, left);

    const auto memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    const std::array<std::int64_t, 4> shape{1, 3, height_, width_};
    auto tensor = Ort::Value::CreateTensor<float>(
      memory, input_.data(), input_.size(), shape.data(), shape.size());
    const char * inputs[] = {input_name_.c_str()};
    const char * outputs[] = {output_name_.c_str()};
    const auto result = session_.Run(
      Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);

    const float * rows = result[0].GetTensorData<float>();
    const auto count = result[0].GetTensorTypeAndShapeInfo().GetShape().at(1);
    vision_msgs::msg::Detection2DArray message;
    message.header = image.header;
    for (std::int64_t row = 0; row < count; ++row) {
      const float * r = rows + row * kColumns;
      if (r[4] <= 0.0F) {
        continue;
      }
      message.detections.push_back(detection(r, scale, left, top, image));
    }
    output_->publish(message);
  }

  void fill_input(bool bgr, int top, int left)
  {
    std::fill(input_.begin(), input_.end(), kPad / 255.0F);
    const std::size_t plane = static_cast<std::size_t>(height_ * width_);
    for (int y = 0; y < resized_.rows; ++y) {
      const auto * pixels = resized_.ptr<std::uint8_t>(y);
      for (int x = 0; x < resized_.cols; ++x) {
        const std::size_t at = static_cast<std::size_t>((y + top) * width_ + x + left);
        input_[at] = pixels[3 * x + (bgr ? 2 : 0)] / 255.0F;
        input_[plane + at] = pixels[3 * x + 1] / 255.0F;
        input_[2 * plane + at] = pixels[3 * x + (bgr ? 0 : 2)] / 255.0F;
      }
    }
  }

  vision_msgs::msg::Detection2D detection(
    const float * r, double scale, int left, int top, const sensor_msgs::msg::Image & image) const
  {
    const auto limit = [](double value, double high) {return std::clamp(value, 0.0, high);};
    const double x0 = limit((r[0] - left) / scale, image.width);
    const double y0 = limit((r[1] - top) / scale, image.height);
    const double x1 = limit((r[2] - left) / scale, image.width);
    const double y1 = limit((r[3] - top) / scale, image.height);
    vision_msgs::msg::Detection2D detection;
    detection.bbox.center.position.x = (x0 + x1) / 2;
    detection.bbox.center.position.y = (y0 + y1) / 2;
    detection.bbox.size_x = x1 - x0;
    detection.bbox.size_y = y1 - y0;
    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    const int id = static_cast<int>(r[5]);
    const auto name = names_.find(id);
    hypothesis.hypothesis.class_id = name == names_.end() ? std::to_string(id) : name->second;
    hypothesis.hypothesis.score = r[4];
    detection.results.push_back(hypothesis);
    return detection;
  }

  Ort::Env env_;
  Ort::Session session_;
  int height_{0};
  int width_{0};
  std::string input_name_;
  std::string output_name_;
  std::map<int, std::string> names_;
  std::vector<float> input_;
  cv::Mat resized_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr output_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<YoloOnnxNode>());
  rclcpp::shutdown();
  return 0;
}
