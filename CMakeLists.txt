cmake_minimum_required(VERSION 3.18)

option(ENABLE_SAMPLES       "Build samples" ON)
option(LIBSGM_SHARED        "Build a shared library" ON)
option(BUILD_OPENCV_WRAPPER "Make library compatible with cv::Mat and cv::cuda::GpuMat of OpenCV" OFF)
set(CUDA_ARCHS "52;61;72;75;86" CACHE STRING "List of architectures to generate device code for")

project(libSGM_LiDAR)

configure_file(
${PROJECT_SOURCE_DIR}/include/libsgm_config.h.in
${PROJECT_SOURCE_DIR}/include/libsgm_config.h
)

add_subdirectory(src)

if(ENABLE_SAMPLES)
	add_subdirectory(sample)
endif()
