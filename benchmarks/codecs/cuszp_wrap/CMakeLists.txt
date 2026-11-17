# SPDX-License-Identifier: Apache-2.0
# Builds cuSZp (third_party/cuSZp, unmodified) on Windows.
# The original CMakeLists gives the same name to its static and shared targets, which Ninja refuses with MSVC, and its
# command-line tool includes <unistd.h>: an empty substitute header is provided in shim/.
cmake_minimum_required(VERSION 3.21)
project(cuszp_wrap LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)
file(GLOB CUSZP_SOURCES ${CUSZP_SRC}/src/*.cu)
add_library(cuSZp STATIC ${CUSZP_SOURCES})
target_include_directories(cuSZp PUBLIC ${CUSZP_SRC}/include ${CUSZP_SRC}/include/cuSZp ${CMAKE_CURRENT_SOURCE_DIR}/shim)
target_link_libraries(cuSZp PUBLIC CUDA::cudart)
add_executable(cuSZp_cli ${CUSZP_SRC}/examples/cuSZp.cpp)
target_link_libraries(cuSZp_cli PRIVATE cuSZp)
# glibc defines "uint"; MSVC does not.
target_compile_definitions(cuSZp PUBLIC "uint=unsigned int")
