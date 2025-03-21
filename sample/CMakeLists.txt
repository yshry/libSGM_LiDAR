cmake_minimum_required(VERSION 3.18)

project(samples LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# required packages
find_package(OpenCV REQUIRED)

set(SRCS_COMMON sample_common.cpp sample_common.h)

# sample executable
add_executable(spread_sgm_image spread_sgm_image.cpp ${SRCS_COMMON})
target_include_directories(spread_sgm_image PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(spread_sgm_image sgm ${OpenCV_LIBS})
