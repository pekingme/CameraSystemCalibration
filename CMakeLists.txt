cmake_minimum_required(VERSION 2.6)
project(camerasystemcalibration)

set(CALIBRATION_LIB_TYPE SHARED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")

find_package(OpenCV REQUIRED)
find_package(FFmpeg REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
)

set(CALIBRATION_HEADERS
camera_system_calibration.h
camera.h
video_synchronizer.h
video_clip.h
utils.h
)

add_library(${PROJECT_NAME} ${CALIBRATION_LIB_TYPE}
camera_system_calibration.cpp
camera.cpp
video_synchronizer.cpp
video_clip.cpp
utils.cpp
${CALIBRATION_HEADERS}
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${FFMPEG_LIBRARIES}
ceres
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/examples)
add_executable(calibrate_example calibrate_example.cpp)
target_link_libraries(calibrate_example ${PROJECT_NAME})