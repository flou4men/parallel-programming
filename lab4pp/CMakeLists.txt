cmake_minimum_required(VERSION 3.20)
project(lab4_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Без явно заданного типа сборки компилятор включает -O0, и времена ядра вырастают в разы.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Архитектура GPU: 89 — Ada Lovelace (RTX 40xx). Для другой карты задайте своё значение,
# например 86 (RTX 30xx) или 120 (RTX 50xx).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

add_executable(matrix_cuda main.cu)
target_compile_options(matrix_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math=false>)
