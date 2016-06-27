cmake_minimum_required(VERSION 2.8)
project(save_video)

set(OpenCV_DIR ${CMAKE_CURRENT_LIST_DIR}/../../opencv/build)
find_package(OpenCV REQUIRED )

add_executable(detect detect.cpp)
target_link_libraries(detect ${OpenCV_LIBS})

set_target_properties(detect PROPERTIES COMPILE_FLAGS "-std=c++0x")

