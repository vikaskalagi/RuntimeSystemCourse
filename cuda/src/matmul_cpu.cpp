#include "matmul.h"
#include <omp.h>
#include <algorithm> // For std::min

// --- CPU V1: Single-Threaded (Baseline) ---
void matMulCPU_Single(const DType* A, const DType* B, DType* C, int N) {
    // Standard triple-nested loop for C = A * B
    for (int i = 0; i < N; ++i) { // Row of A
        for (int j = 0; j < N; ++j) { // Column of B
            DType sum = 0.0f;
            for (int k = 0; k < N; ++k) { // Dot product
                // Accessing elements: A[i, k] and B[k, j]
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// --- CPU V2: Multi-Threaded (OpenMP) ---
void matMulCPU_Multi(const DType* A, const DType* B, DType* C, int N) {
    // Using OpenMP to parallelize the outer loops
    // The 'i' and 'j' loops are independent and can be parallelized.
    
    // Use collapse(2) to parallelize both outer loops for better workload balance
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) { // Row of A
        for (int j = 0; j < N; ++j) { // Column of B
            DType sum = 0.0f;
            for (int k = 0; k < N; ++k) { // Dot product
                // Accessing elements: A[i, k] and B[k, j]
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}