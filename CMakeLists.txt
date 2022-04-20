cmake_minimum_required(VERSION 3.2)
project(kalmanFilterTest)

include_directories(include)
include_directories("/usr/include/eigen3")

add_library(kalman_include include/kalmanFilter.cpp)

add_executable(kalmanFilterTest src/kalmanTest.cpp)
target_link_libraries(kalmanFilterTest kalman_include)