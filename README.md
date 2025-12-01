# GPU Runtime Optimization for Matrix Multiplication (GEMM)

This project analyzes and implements progressively optimized versions of matrix multiplication across CPU (single-threaded, multi-threaded) and GPU (CUDA) architectures. The goal is to study how memory hierarchy, thread geometry, concurrency, and transfer strategies influence performance on modern NVIDIA GPUs.

## 🚀 Project Overview

Matrix multiplication is a core operation in scientific computing and machine learning. This project benchmarks seven implementations of GEMM:

* V0 – CPU single-threaded
* V1 – CPU multi-threaded (OpenMP)
* V2 – Naive CUDA kernel
* V3 – CUDA with coalesced global memory access
* V4 – CUDA with shared memory tiling
* V5 – CUDA with asynchronous streams
* V6 – CUDA with pinned (page-locked) memory + async streams

Each version builds upon the previous one to measure the effectiveness of key GPU runtime strategies.

## 📊 Key Results

* GPU acceleration (V2) provides an ~80× speedup over multi-threaded CPU (V1).
* Shared memory tiling (V4) delivers the most meaningful kernel-level performance improvement.
* Pinned memory + async transfers (V6) achieves the fastest overall runtime, especially for mid-sized matrices (N ≈ 1024–4096).
* Streams provide limited benefit when kernels are already fast; overhead can outweigh gains.

## 🧪 Features and Contributions

* Implemented and benchmarked seven CPU/GPU versions of GEMM.
* Integrated shared memory tiling, coalesced access, streams, and pinned memory.
* Built a profiling workflow using NVIDIA tools to examine occupancy, memory stalls, and transfer timelines.
* Produced detailed analysis of how each optimization impacts throughput.

## How to build and run
```
mkdir build
cd build
cmake ..
make -j
./gpu_matmul_app
```

## 📝 Conclusion

Modern performance gains come from architecture-aware optimizations—not new CPU generations. This project demonstrates how understanding GPU memory hierarchy, thread behavior, and transfer mechanisms enables order-of-magnitude improvements in throughput.

## 🤝 Contributors

* Bhargavi Kurukunda - bhargavi_kurukunda@ucsb.edu
* Vikas Kalagi - vikaskalagi@ucsb.edu
