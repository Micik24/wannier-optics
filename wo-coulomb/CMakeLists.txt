cmake_minimum_required(VERSION 3.16)

project (wo-coulomb LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-Wall -O3")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS False)

option(WO_COULOMB_BUILD_GPU "Build GPU backend binary" OFF)
option(WO_COULOMB_BUILD_BENCH "Build wo-coulomb benchmark binary" ON)

set(WO_COULOMB_CPU_TARGET ${PROJECT_NAME}.x)
set(WO_COULOMB_GPU_TARGET ${PROJECT_NAME}-gpu.x)

set(WO_COULOMB_COMMON_SOURCES
    src/main.cpp
    src/backend/backend_factory.cpp
    src/backend/cpu/fft_executor_fftw.cpp
)

add_executable(${WO_COULOMB_CPU_TARGET}
    ${WO_COULOMB_COMMON_SOURCES}
    src/backend/cpu/backend_cpu.cpp
    src/backend/cpu/density_metrics_cpu.cpp
)
target_compile_definitions(${WO_COULOMB_CPU_TARGET} PRIVATE WO_BACKEND_CPU=1)

# parallelization
find_package(OpenMP 4.0 REQUIRED)
find_package(MPI REQUIRED)

# libaries and targets
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW IMPORTED_TARGET REQUIRED fftw3)
include_directories(PkgConfig::FFTW)
link_libraries(PkgConfig::FFTW)

target_include_directories(${WO_COULOMB_CPU_TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    PkgConfig::FFTW
)
target_link_libraries(${WO_COULOMB_CPU_TARGET} PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::FFTW)

if (WO_COULOMB_BUILD_GPU)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAToolkit REQUIRED)
    add_executable(${WO_COULOMB_GPU_TARGET}
        ${WO_COULOMB_COMMON_SOURCES}
        src/backend/gpu/backend_gpu.cpp
        src/backend/gpu/fft_executor_cufft.cpp
        src/backend/gpu/density_metrics_gpu.cpp
        src/backend/gpu/density_metrics_kernels.cu
    )
    target_compile_definitions(${WO_COULOMB_GPU_TARGET} PRIVATE WO_BACKEND_GPU=1)
    target_include_directories(${WO_COULOMB_GPU_TARGET} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external
        PkgConfig::FFTW
    )
    target_link_libraries(${WO_COULOMB_GPU_TARGET} PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::FFTW CUDA::cudart CUDA::cufft)
endif()

if (WO_COULOMB_BUILD_BENCH)
    add_executable(wo-coulomb-bench.x
        bench/bench_coulomb.cpp
        src/backend/cpu/fft_executor_fftw.cpp
        src/backend/cpu/density_metrics_cpu.cpp
    )
    target_include_directories(wo-coulomb-bench.x PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external
        PkgConfig::FFTW
    )
    target_link_libraries(wo-coulomb-bench.x PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::FFTW)

    add_executable(wo-coulomb-bench-lfe-density.x
        bench/bench_lfe_density.cpp
        src/backend/cpu/density_metrics_cpu.cpp
    )
    target_include_directories(wo-coulomb-bench-lfe-density.x PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/external
        PkgConfig::FFTW
    )
    target_link_libraries(wo-coulomb-bench-lfe-density.x PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::FFTW)

    if (WO_COULOMB_BUILD_GPU)
        add_executable(wo-coulomb-bench-lfe-density-gpu.x
            bench/bench_lfe_density.cpp
            src/backend/gpu/density_metrics_gpu.cpp
            src/backend/gpu/density_metrics_kernels.cu
        )
        target_include_directories(wo-coulomb-bench-lfe-density-gpu.x PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/external
            PkgConfig::FFTW
        )
        target_link_libraries(wo-coulomb-bench-lfe-density-gpu.x PUBLIC
            MPI::MPI_CXX
            OpenMP::OpenMP_CXX
            PkgConfig::FFTW
            CUDA::cudart
        )
    endif()
endif()

# tell the program its git commit id
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions("-DGIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")

# Try to find GoogleTest
find_package(GTest QUIET)

# Check if GoogleTest was found
if (GTest_FOUND)
    message(STATUS "GoogleTest found. Enabling tests.")
    enable_testing()
    include(GoogleTest)
    add_subdirectory(tests)
else()
    message(WARNING "GoogleTest not found. Tests will not be built.")
endif()


install(TARGETS ${WO_COULOMB_CPU_TARGET} RUNTIME DESTINATION bin)
if (WO_COULOMB_BUILD_GPU)
    install(TARGETS ${WO_COULOMB_GPU_TARGET} RUNTIME DESTINATION bin)
endif()
