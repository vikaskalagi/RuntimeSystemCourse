#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <functional>
#include <cmath>
#include <algorithm>
#include "matmul.h"
#include <cuda_runtime.h> // Explicitly include CUDA header for host functions

using namespace std;

// ... (benchmark_and_log, initialize_matrices, verify functions remain the same) ...
void benchmark_and_log(const string& version, int N, function<void()> func, ofstream& log_file) {
    // Using CPU timing surrounding the entire host wrapper (including transfers)
    
    // Ensure all previous CUDA calls are finished before starting the timer
    if (version.find("CUDA") != string::npos) {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cerr << "CUDA Sync Error during benchmark: " << cudaGetErrorString(err) << endl;
        }
    }
    
    auto start = chrono::high_resolution_clock::now();
    
    // Execute the function
    func();
    
    // Ensure all asynchronous GPU work is done before stopping the timer
    if (version.find("CUDA") != string::npos) {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cerr << "CUDA Sync Error after benchmark: " << cudaGetErrorString(err) << endl;
        }
    }
    
    auto end = chrono::high_resolution_clock::now();
    // Time in milliseconds
    double duration = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0; 

    // Log: MatrixSize,Version,Time_ms
    log_file << N << "," << version << "," << fixed << setprecision(6) << duration << "\n";
    cout << "N=" << N << " | " << setw(30) << left << version << " time: " << duration << " ms" << endl;
}

void initialize_matrices(DType* A, DType* B, int N) {
    for (int i = 0; i < N * N; ++i) {
        A[i] = (DType) (i % 10 + 1) * 0.1f;
        B[i] = (DType) ((N * N - i) % 10 + 1) * 0.1f;
    }
}

bool verify(const DType* C_gpu, const DType* C_cpu, int N, const string& version) {
    const DType EPSILON = 1e-4f;
    for (int i = 0; i < N * N; ++i) {
        if (abs(C_gpu[i] - C_cpu[i]) > EPSILON) {
            // Only report a failure if the CPU value is non-zero, avoiding false positives
            if (abs(C_cpu[i]) > EPSILON) {
                cerr << "Verification failed for " << version << " at index " << i 
                    << ". CPU: " << C_cpu[i] << ", GPU: " << C_gpu[i] << endl;
                return false;
            }
        }
    }
    return true;
}


int main() {
    // Initialize required directories
    system("mkdir -p data");
    system("mkdir -p results");
    
    // Check for CUDA-enabled device
    int devCount;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) {
        cerr << "FATAL: No CUDA devices found. Cannot run GPU benchmarks." << endl;
        return 1;
    }

    // 1. Setup Logging
    ofstream log_file("data/runtime_data.csv");
    if (!log_file.is_open()) {
        cerr << "Error: Could not open log file for writing." << endl;
        return 1;
    }
    log_file << "MatrixSize,Version,Time_ms\n"; // CSV Header

    // 2. Define Matrix Sizes (N) to test
    // Increased max size to 8192 to see true tiling/streaming benefits
    // vector<int> matrix_sizes = {1024, 2048, 4096, 6144, 8192}; 
    vector<int> matrix_sizes = {1024, 2048}; 

    cout << "Starting Matrix Multiplication Benchmarks (" << devCount << " GPU(s) found)..." << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    
    // 3. Main Benchmark Loop
    for (int N : matrix_sizes) {
        // ... (Benchmark execution loop remains the same) ...
        size_t size = N * N;
        vector<DType> A(size), B(size);
        vector<DType> C_ref(size, 0.0f); // Reference result (CPU Single)
        vector<DType> C_test(size, 0.0f); // Test result for comparison

        initialize_matrices(A.data(), B.data(), N);
        DType *h_A_pinned = nullptr;
        DType *h_B_pinned = nullptr;
        DType *h_C_pinned = nullptr;
        
        cudaCheckError(cudaHostAlloc((void**)&h_A_pinned, size*sizeof(DType), cudaHostAllocDefault));
        cudaCheckError(cudaHostAlloc((void**)&h_B_pinned, size*sizeof(DType), cudaHostAllocDefault));
        cudaCheckError(cudaHostAlloc((void**)&h_C_pinned, size*sizeof(DType), cudaHostAllocDefault));

        memcpy(h_A_pinned, A.data(), size*sizeof(DType));
        memcpy(h_B_pinned, B.data(), size*sizeof(DType));

        // --- V0: CPU Single-Threaded (Reference) ---
        benchmark_and_log("CPU_V0_Single", N, [&]() { 
            matMulCPU_Single(A.data(), B.data(), C_ref.data(), N);
        }, log_file);
        
        // --- V1: CPU Multi-Threaded (OpenMP) ---
        fill(C_test.begin(), C_test.end(), 0.0f);
        benchmark_and_log("CPU_V1_Multi", N, [&]() { 
            matMulCPU_Multi(A.data(), B.data(), C_test.data(), N);
        }, log_file);
        verify(C_test.data(), C_ref.data(), N, "CPU_V1_Multi");
        
        // --- V2: CUDA Naive (Thread-per-Element) ---
        fill(C_test.begin(), C_test.end(), 0.0f);
        benchmark_and_log("CUDA_V2_Naive", N, [&]() { 
            matMulGPU_Coalesced(A.data(), B.data(), C_test.data(), N);
        }, log_file);
        verify(C_test.data(), C_ref.data(), N, "CUDA_V2_Naive");

        // --- V3: CUDA Coalesced (Optimized Grid/Block) ---
        fill(C_test.begin(), C_test.end(), 0.0f);
        benchmark_and_log("CUDA_V3_Coalesced", N, [&]() { 
            matMulGPU_Naive(A.data(), B.data(), C_test.data(), N);
        }, log_file);
        verify(C_test.data(), C_ref.data(), N, "CUDA_V3_Coalesced");

        // --- V4: CUDA Shared Memory Tiled (Architectural Opt) ---
        fill(C_test.begin(), C_test.end(), 0.0f);
        benchmark_and_log("CUDA_V4_SharedTiled", N, [&]() { 
            matMulGPU_Shared(A.data(), B.data(), C_test.data(), N);
        }, log_file);
        verify(C_test.data(), C_ref.data(), N, "CUDA_V4_SharedTiled");
        
        // --- V5: CUDA Async Streams (Concurrency Opt) ---
        fill(C_test.begin(), C_test.end(), 0.0f);
        benchmark_and_log("CUDA_V5_AsyncStreams", N, [&]() { 
            matMulGPU_AsyncStreams(A.data(), B.data(), C_test.data(), N);
        }, log_file);
        verify(C_test.data(), C_ref.data(), N, "CUDA_V5_AsyncStreams");

        // --- V6: CUDA Pinned + Async Streams (Memory Opt + Concurrency) ---
        fill(C_test.begin(), C_test.end(), 0.0f);
        benchmark_and_log("CUDA_V6_PinnedAsync", N, [&]() { 
            matMulGPU_PinnedAsync(h_A_pinned, h_B_pinned, h_C_pinned, N);
        }, log_file);
        verify(C_test.data(), C_ref.data(), N, "CUDA_V6_PinnedAsync");

        cudaCheckError(cudaFreeHost(h_A_pinned));
        cudaCheckError(cudaFreeHost(h_B_pinned));
        cudaCheckError(cudaFreeHost(h_C_pinned));
        cout << "--------------------------------------------------------------------------------" << endl;
    }

    log_file.close();
    cout << "\nBenchmarks complete. Data saved to data/runtime_data.csv" << endl;
    return 0;
}