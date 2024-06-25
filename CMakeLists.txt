cmake_minimum_required(VERSION 3.20.0)
project(AMWF_color_correction)

set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )
find_package( OpenMP )
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(color_correction main.cpp utils.h utils.cpp)

target_link_libraries( color_correction ${OpenCV_LIBS} )