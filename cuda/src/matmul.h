#pragma once

#include <string>
#include <iostream>

// Type definition for consistency
using DType = float;

// ----------------------------------------------------------------------
// CUDA-Specific Definitions (Only include if compiling CUDA code)
// The __CUDACC__ macro is automatically defined by the NVCC compiler.
// ----------------------------------------------------------------------
#ifdef __CUDACC__
    #include <cuda_runtime.h>

    // Utility for CUDA error checking (only useful in CUDA files)
    #define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
    {
       if (code != cudaSuccess) 
       {
          fprintf(stderr,"CUDA Error: %s %s:%d\n", cudaGetErrorString(code), file, line);
          if (abort) exit(code);
       }
    }
#endif
// ----------------------------------------------------------------------

// --- CPU Implementations ---
// These are standard C++ and don't require CUDA headers.
void matMulCPU_Single(const DType* A, const DType* B, DType* C, int N);
void matMulCPU_Multi(const DType* A, const DType* B, DType* C, int N);

// --- GPU Implementations (CUDA API) ---
// These are called from C++ host code, so their signatures must be visible.
// The actual implementation is in matmul_gpu.cu
void matMulGPU_Naive(const DType* A, const DType* B, DType* C, int N);
void matMulGPU_Coalesced(const DType* A, const DType* B, DType* C, int N);
void matMulGPU_Shared(const DType* A, const DType* B, DType* C, int N);
void matMulGPU_AsyncStreams(const DType* A, const DType* B, DType* C, int N);
void matMulGPU_PinnedAsync( DType* A,  DType* B, DType* C, int N);