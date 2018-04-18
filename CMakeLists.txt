cmake_minimum_required(VERSION 2.8)

project(untitled)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(OpenCV 3.3.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
