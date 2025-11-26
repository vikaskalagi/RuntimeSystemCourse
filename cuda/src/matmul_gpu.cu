#include "matmul.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>
#include <vector>

#define TILE_SIZE 32 
#define NUM_STREAMS 4 

// --- Kernel V2/V3: Naive (1 Thread per C Element, 2D Grid) ---
// Note: This kernel has severe B matrix striding issues (poor coalescing).
__global__ void matMulNaiveKernel(const DType* A, const DType* B, DType* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        DType sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


// --- Kernel V4/V5: Shared Memory Tiling (Fastest Kernel) ---
// This kernel is the optimal architectural implementation.
__global__ void matMulSharedKernel(const DType* A, const DType* B, DType* C, int N) {
    __shared__ DType As[TILE_SIZE][TILE_SIZE];
    __shared__ DType Bs[TILE_SIZE][TILE_SIZE];

    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    DType Cvalue = 0.0f; 

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        int tile_col = t * TILE_SIZE;

        int A_row_idx = block_row + ty;
        int A_col_idx = tile_col + tx;
        int B_row_idx = tile_col + ty;
        int B_col_idx = block_col + tx;

        // Load the tiles into Shared Memory (Coalesced access to A and B)
        if (A_row_idx < N && A_col_idx < N) {
            As[ty][tx] = A[A_row_idx * N + A_col_idx];
        } else { As[ty][tx] = 0.0f; }

        if (B_row_idx < N && B_col_idx < N) {
            Bs[ty][tx] = B[B_row_idx * N + B_col_idx];
        } else { Bs[ty][tx] = 0.0f; }

        __syncthreads();

        // Compute using Shared Memory (Fastest phase)
        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write the final result back to Global Memory
    int C_row = block_row + ty;
    int C_col = block_col + tx;
    if (C_row < N && C_col < N) {
        C[C_row * N + C_col] = Cvalue;
    }
}


// -----------------------------------------------------------------------------------
// --- Host Wrapper Functions ---
// -----------------------------------------------------------------------------------

// V2: CUDA Naive (Baseline GPU Version)
void matMulGPU_Naive(const DType* A, const DType* B, DType* C, int N) {
    DType *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(DType);

    cudaCheckError(cudaMalloc((void **)&d_A, size));
    cudaCheckError(cudaMalloc((void **)&d_B, size));
    cudaCheckError(cudaMalloc((void **)&d_C, size));

    cudaCheckError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    // Launch configuration: Standard TILE_SIZE x TILE_SIZE (32x32)
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    matMulNaiveKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaFree(d_A));
    cudaCheckError(cudaFree(d_B));
    cudaCheckError(cudaFree(d_C));
}


// V3: CUDA Coalesced (Optimized Block Geometry for Coalescing)
void matMulGPU_Coalesced(const DType* A, const DType* B, DType* C, int N) {
    DType *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(DType);

    cudaCheckError(cudaMalloc((void **)&d_A, size));
    cudaCheckError(cudaMalloc((void **)&d_B, size));
    cudaCheckError(cudaMalloc((void **)&d_C, size));
    cudaCheckError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    // --- CRITICAL V3 FIX: Optimized Block Geometry ---
    // Use a wide block (high X, low Y) to ensure better coalescing 
    // on the C matrix write (C[row*N + col]) where threads are spaced 
    // closely in the column direction. Total threads per block is 256.
    const int THREADS_X = 64; 
    const int THREADS_Y = 4; // (128 * 2 = 256 threads)

    dim3 threadsPerBlock(THREADS_X, THREADS_Y);
    dim3 numBlocks((N + THREADS_X - 1) / THREADS_X, (N + THREADS_Y - 1) / THREADS_Y);

    matMulNaiveKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N); 

    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaFree(d_A));
    cudaCheckError(cudaFree(d_B));
    cudaCheckError(cudaFree(d_C));
}


// V4: CUDA Shared Memory Tiled (Architectural Opt)
void matMulGPU_Shared(const DType* A, const DType* B, DType* C, int N) {
    DType *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(DType);

    cudaCheckError(cudaMalloc((void **)&d_A, size));
    cudaCheckError(cudaMalloc((void **)&d_B, size));
    cudaCheckError(cudaMalloc((void **)&d_C, size));
    cudaCheckError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    matMulSharedKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaFree(d_A));
    cudaCheckError(cudaFree(d_B));
    cudaCheckError(cudaFree(d_C));
}


// V5: CUDA Async Streams (Concurrency Opt)
void matMulGPU_AsyncStreams(const DType* A, const DType* B, DType* C, int N) {
    DType *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(DType);

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) cudaCheckError(cudaStreamCreate(&streams[i]));

    cudaCheckError(cudaMalloc((void **)&d_A, size));
    cudaCheckError(cudaMalloc((void **)&d_B, size));
    cudaCheckError(cudaMalloc((void **)&d_C, size));

    cudaCheckError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    int chunk_rows = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        int start_row = i * chunk_rows;
        int current_rows = std::min(chunk_rows, N - start_row);
        
        if (current_rows <= 0) break;

        size_t chunk_size = current_rows * N * sizeof(DType);
        
        const DType* h_A_chunk = A + start_row * N;
        DType* h_C_chunk = C + start_row * N;

        DType* d_A_chunk = d_A + start_row * N;
        DType* d_C_chunk = d_C + start_row * N;

        dim3 numBlocks( (N + TILE_SIZE - 1) / TILE_SIZE, (current_rows + TILE_SIZE - 1) / TILE_SIZE );
        
        // ASYNC H2D Copy
        cudaCheckError(cudaMemcpyAsync(d_A_chunk, h_A_chunk, chunk_size, cudaMemcpyHostToDevice, streams[i]));
        
        // KERNEL Launch
        matMulSharedKernel<<<numBlocks, threadsPerBlock, 0, streams[i]>>>(d_A_chunk, d_B, d_C_chunk, N);
        
        // ASYNC D2H Copy
        cudaCheckError(cudaMemcpyAsync(h_C_chunk, d_C_chunk, chunk_size, cudaMemcpyDeviceToHost, streams[i]));
    }
    
    // Synchronize and Cleanup...
    for (int i = 0; i < NUM_STREAMS; ++i) cudaCheckError(cudaStreamSynchronize(streams[i]));
    for (int i = 0; i < NUM_STREAMS; ++i) cudaCheckError(cudaStreamDestroy(streams[i]));
    cudaCheckError(cudaFree(d_A));
    cudaCheckError(cudaFree(d_B));
    cudaCheckError(cudaFree(d_C));
}


// V6: CUDA Pinned Memory + Async Streams (Concurrency & Transfer Opt)
void matMulGPU_PinnedAsync( DType* A,  DType* B, DType* C, int N) {
    // Reverting V6 to the previous functional streaming version
    DType *d_A, *d_B, *d_C;
     DType *h_A_pinned = A;
     DType *h_B_pinned = B;
     DType *h_C_pinned = C;
    size_t size = N * N * sizeof(DType);
    
    // cudaCheckError(cudaHostAlloc((void**)&h_A_pinned, size, cudaHostAllocMapped));
    // cudaCheckError(cudaHostAlloc((void**)&h_C_pinned, size, cudaHostAllocMapped));

    // memcpy(h_A_pinned, A, size);
    
    cudaCheckError(cudaMalloc((void **)&d_A, size));
    cudaCheckError(cudaMalloc((void **)&d_B, size));
    cudaCheckError(cudaMalloc((void **)&d_C, size));

    // printf("N = %d,  size bytes = %zu\n", N, size);
    // printf("d_B = %p, B = %p, size = %zu\n", (void*)d_B, (void*)B, size);

    cudaCheckError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) cudaCheckError(cudaStreamCreate(&streams[i]));
    
    int chunk_rows = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

    for (int i = 0; i < NUM_STREAMS; ++i) {
        int start_row = i * chunk_rows;
        int current_rows = std::min(chunk_rows, N - start_row);
        
        if (current_rows <= 0) break;

        size_t chunk_size = current_rows * N * sizeof(DType);

        const DType* h_A_chunk = h_A_pinned + start_row * N;
        DType* h_C_chunk = h_C_pinned + start_row * N;

        DType* d_A_chunk = d_A + start_row * N;
        DType* d_C_chunk = d_C + start_row * N;

        dim3 numBlocks( (N + TILE_SIZE - 1) / TILE_SIZE, (current_rows + TILE_SIZE - 1) / TILE_SIZE );
        
        cudaCheckError(cudaMemcpyAsync(d_A_chunk, h_A_chunk, chunk_size, cudaMemcpyHostToDevice, streams[i]));
        
        matMulSharedKernel<<<numBlocks, threadsPerBlock, 0, streams[i]>>>(d_A_chunk, d_B, d_C_chunk, N);
        
        cudaCheckError(cudaMemcpyAsync(h_C_chunk, d_C_chunk, chunk_size, cudaMemcpyDeviceToHost, streams[i]));
    }
    
    for (int i = 0; i < NUM_STREAMS; ++i) cudaCheckError(cudaStreamSynchronize(streams[i]));
    
    // memcpy(C, h_C_pinned, size); 

    for (int i = 0; i < NUM_STREAMS; ++i) cudaCheckError(cudaStreamDestroy(streams[i]));
    cudaCheckError(cudaFree(d_A));
    cudaCheckError(cudaFree(d_B));
    cudaCheckError(cudaFree(d_C));
    // cudaCheckError(cudaFreeHost(h_A_pinned));
    // cudaCheckError(cudaFreeHost(h_C_pinned));
}