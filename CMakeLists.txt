cmake_minimum_required(VERSION 2.8)

project(image_classifier)

add_definitions(-std=c++11)
add_definitions("-Wall")

find_package(OpenCV REQUIRED)

if(True)
    find_package(OpenMP)
    if(NOT TARGET OpenMP::OpenMP_CXX AND (OpenMP_CXX_FOUND OR OPENMP_FOUND))
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

include_directories(${OpenCV_DIR}/include
                    ncnn_root_dir/build/install/include/ncnn)    # change ncnn_root_dir to yours
link_directories(${OpenCV_DIR}/lib/
                    ncnn_root_dir/build/install/lib)             # change ncnn_root_dir to yours

set(OpenCV_LIBS opencv_core 
                opencv_highgui 
                opencv_imgproc 
                opencv_imgcodecs 
                opencv_videoio)

add_executable(image_classifier image_classifier.cpp)
target_link_libraries(image_classifier ncnn ${OpenCV_LIBS})
