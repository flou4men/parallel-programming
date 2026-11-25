cmake_minimum_required(VERSION 3.16)
project(lab2_openmp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Без явно указанного типа сборки CMake берёт пустой, компилятор включает /Od (-O0),
# и времена вырастают в разы — измерения становятся бессмысленными.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenMP REQUIRED)

add_executable(matrix main.cpp)
target_link_libraries(matrix PRIVATE OpenMP::OpenMP_CXX)
