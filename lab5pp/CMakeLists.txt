cmake_minimum_required(VERSION 3.16)
project(lab5_mandelbrot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)
add_executable(mandelbrot main.cpp)
target_link_libraries(mandelbrot PRIVATE Threads::Threads)
