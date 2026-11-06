cmake_minimum_required(VERSION 3.20)
project(yolo_onnx_cpp)

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

# The directory of an ONNX Runtime release archive.
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime directory")
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "Set ONNXRUNTIME_ROOT to an ONNX Runtime release directory")
endif()

add_executable(yolo_onnx_node src/yolo_onnx_node.cpp)
target_compile_features(yolo_onnx_node PRIVATE cxx_std_17)
target_compile_options(yolo_onnx_node PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(yolo_onnx_node PRIVATE ${ONNXRUNTIME_ROOT}/include)
target_link_libraries(yolo_onnx_node
  ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so opencv_core opencv_imgproc rclcpp::rclcpp
  ${sensor_msgs_TARGETS} ${vision_msgs_TARGETS})
set_target_properties(yolo_onnx_node PROPERTIES BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib
  INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)

install(TARGETS yolo_onnx_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
ament_package()
