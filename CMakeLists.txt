# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)
project(psitreillis VERSION 0.2.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# By default only the library, its command-line tool and the examples are built. The measurement instruments of tools/
# (turbulence solver, measurement front end, tracer experiment, visual comparator, reference reader) are built on
# request: -DPSIT_BUILD_TOOLS=ON.
option(PSIT_BUILD_EXAMPLES "Build the examples of examples/" ON)
option(PSIT_BUILD_TOOLS "Build the measurement instruments of tools/" OFF)

find_package(CUDAToolkit REQUIRED)

# Target GPU architecture. "native": that of the GPU of the build machine, which must have one. On a machine without
# a GPU (continuous integration), give an architecture number: -DPSIT_CUDA_ARCHITECTURES=86.
set(PSIT_CUDA_ARCHITECTURES native CACHE STRING "CUDA architecture(s) targeted by the kernels")

# ---- Library: the public interface (include/psitreillis.h) -----------------------------------------------------------
# What another program calls: open a .psi file, decode a frame to a grid, sample the velocity at arbitrary points,
# compress a sequence. psit is its command-line front end (info, compress, decompress, sample, convert).
add_library(psitreillis SHARED src/psitreillis.cu)
target_include_directories(psitreillis PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(psitreillis PRIVATE CUDA::cufft CUDA::cudart)
set_target_properties(psitreillis PROPERTIES CUDA_ARCHITECTURES "${PSIT_CUDA_ARCHITECTURES}" VERSION ${PROJECT_VERSION} SOVERSION 0
                      CXX_VISIBILITY_PRESET hidden CUDA_VISIBILITY_PRESET hidden)
if(MSVC)
  # Thrust (particle sort) requires the conforming preprocessor of MSVC.
  target_compile_options(psitreillis PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=/Zc:preprocessor>)
endif()

add_executable(psit src/psit.cpp)
target_link_libraries(psit PRIVATE psitreillis)

if(PSIT_BUILD_EXAMPLES)
  add_executable(example_roundtrip examples/roundtrip.c)
  target_link_libraries(example_roundtrip PRIVATE psitreillis)
  if(UNIX)
    target_link_libraries(example_roundtrip PRIVATE m)   # cos, sin: the mathematics library is linked separately on Linux
  endif()
endif()

# ---- Measurement instruments (tools/) --------------------------------------------------------------------------------
if(PSIT_BUILD_TOOLS)
  # Turbulence solver: pseudo-spectral direct simulation (cuFFT). Produces the .seq sequences.
  add_executable(dns tools/dns.cu)
  target_include_directories(dns PRIVATE src)
  target_link_libraries(dns PRIVATE CUDA::cufft)
  set_target_properties(dns PROPERTIES CUDA_ARCHITECTURES "${PSIT_CUDA_ARCHITECTURES}")

  # Measurement front end of the codec: pack (raw frames -> .psi file) and play (reads the .psi file alone, decodes on
  # the GPU, advects particles), with the timers and the checks that the benches read.
  # -lineinfo: source/assembly correlation in the profilers, at no run-time cost.
  add_executable(psi_stream tools/psi_stream.cu)
  target_include_directories(psi_stream PRIVATE src)
  target_link_libraries(psi_stream PRIVATE CUDA::cufft)
  set_target_properties(psi_stream PROPERTIES CUDA_ARCHITECTURES "${PSIT_CUDA_ARCHITECTURES}")
  target_compile_options(psi_stream PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  if(MSVC)
    target_compile_options(psi_stream PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=/Zc:preprocessor>)
  endif()

  # Reference reader on CPU, written from docs/FORMAT.md alone: C++17, neither CUDA nor any external library.
  find_package(Threads REQUIRED)
  add_executable(psi_ref tools/reference_reader.cpp)
  target_link_libraries(psi_ref PRIVATE Threads::Threads)

  # zfp with its CUDA backend (cuZFP): the compressed reference of the tracer experiment and of the visual comparator.
  # The official CUDA build of zfp is wired for GCC; its sources are therefore compiled as they are, without modifying
  # third_party/zfp, with the native CUDA support of CMake.
  set(ZFP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/zfp)
  if(EXISTS ${ZFP_ROOT}/src/cuda_zfp/cuZFP.cu)
    set(ZFP_SOURCES ${ZFP_ROOT}/src/zfp.c ${ZFP_ROOT}/src/bitstream.c)
    foreach(dim 1 2 3 4)
      foreach(type f d i l)
        list(APPEND ZFP_SOURCES ${ZFP_ROOT}/src/encode${dim}${type}.c ${ZFP_ROOT}/src/decode${dim}${type}.c)
      endforeach()
    endforeach()
    add_library(zfp_cuda STATIC ${ZFP_SOURCES} ${ZFP_ROOT}/src/cuda_zfp/cuZFP.cu)
    target_include_directories(zfp_cuda PUBLIC ${ZFP_ROOT}/include)
    target_compile_definitions(zfp_cuda PUBLIC ZFP_WITH_CUDA PRIVATE _CRT_SECURE_NO_WARNINGS)
    set_target_properties(zfp_cuda PROPERTIES CUDA_ARCHITECTURES "${PSIT_CUDA_ARCHITECTURES}")

    # Tracer experiment: tracers advected in one frozen frame, seen through each reconstruction path (Psi-treillis,
    # tricubic, trilinear texture, cuZFP, or a grid decoded by another compressor).
    add_executable(tracers tools/tracers.cu)
    target_include_directories(tracers PRIVATE src tools)
    target_link_libraries(tracers PRIVATE CUDA::cufft zfp_cuda)
    target_compile_definitions(tracers PRIVATE PSIT_WITH_ZFP)
    set_target_properties(tracers PROPERTIES CUDA_ARCHITECTURES "${PSIT_CUDA_ARCHITECTURES}")
    if(MSVC)
      target_compile_options(tracers PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=/Zc:preprocessor>)
    endif()

    # Visual comparator in split screen: Psi-treillis, cuZFP + texture and raw frames, side by side.
    # Win32 window and GDI, no external dependency.
    if(WIN32)
      add_executable(psi_view tools/viewer/psi_view.cu tools/viewer/view_window.cpp)
      target_include_directories(psi_view PRIVATE src tools tools/viewer)
      target_link_libraries(psi_view PRIVATE CUDA::cufft zfp_cuda user32 gdi32)
      target_compile_definitions(psi_view PRIVATE PSIT_WITH_ZFP)
      set_target_properties(psi_view PROPERTIES CUDA_ARCHITECTURES "${PSIT_CUDA_ARCHITECTURES}")
      if(MSVC)
        target_compile_options(psi_view PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=/Zc:preprocessor>)
      endif()
    endif()
  else()
    message(STATUS "third_party/zfp is absent: the tracer experiment and the visual comparator are not built")
  endif()
endif()

# ---- Installation: cmake --install <build> --prefix <folder> ---------------------------------------------------------
# Another CMake project then does find_package(psitreillis) and target_link_libraries(... psitreillis::psitreillis).
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS psitreillis psit EXPORT psitreillisTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/psitreillis.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT psitreillisTargets NAMESPACE psitreillis:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/psitreillis)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/psitreillisConfigVersion.cmake
                                 VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/psitreillisConfig.cmake
     "include(\"\${CMAKE_CURRENT_LIST_DIR}/psitreillisTargets.cmake\")\n")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/psitreillisConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/psitreillisConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/psitreillis)
