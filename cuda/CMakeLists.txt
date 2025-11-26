cmake_minimum_required(VERSION 3.10)
project(GpuMatmulProject LANGUAGES CXX CUDA)

# Enable C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenMP for CPU multi-threading
find_package(OpenMP REQUIRED)

# Define executable and source files
add_executable(gpu_matmul_app
    src/main.cu
    src/matmul_cpu.cpp
    src/matmul_gpu.cu
)

# Link necessary libraries (OpenMP and CUDA standard libraries)
target_link_libraries(gpu_matmul_app PRIVATE OpenMP::OpenMP_CXX)

# Set CUDA properties
# Use "all" to compile for all architectures available on your system, 
# or specify a minimum capability like "80" for Ampere
set_property(TARGET gpu_matmul_app PROPERTY CUDA_ARCHITECTURES "all")

# Include the source directory for headers
target_include_directories(gpu_matmul_app PRIVATE src)