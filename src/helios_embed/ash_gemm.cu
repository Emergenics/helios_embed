// --- START OF FILE src/helios_embed/ash_gemm.cu ---
#include "ash_gemm.h"
#include <cuda_runtime.h>

#define TILE_DIM_M 128
#define TILE_DIM_N 128
#define THREADS_PER_BLOCK 128
#define K_PANEL_WORDS 8

__global__ void tiled_ash_gemm_kernel(
    const unsigned long long* __restrict__ A, 
    const unsigned long long* __restrict__ B_T, 
    int* __restrict__ C, 
    int M, int N, int K, int K_packed) 
{
    int tm = blockIdx.y; int tn = blockIdx.x;
    
    __shared__ unsigned long long a_p[TILE_DIM_M * K_PANEL_WORDS];
    __shared__ unsigned long long b_p[TILE_DIM_N * K_PANEL_WORDS];
    
    int tid = threadIdx.x; 
    int row_in_tile = tid;
    int current_row = tm * TILE_DIM_M + row_in_tile;
    
    int accumulator[TILE_DIM_N] = {0};

    for (int kb = 0; kb < K_packed; kb += K_PANEL_WORDS) {
        // Cooperatively load panels into shared memory
        for (int ko = 0; ko < K_PANEL_WORDS; ++ko) {
            if (row_in_tile < TILE_DIM_M && current_row < M && (kb + ko) < K_packed) {
                a_p[row_in_tile * K_PANEL_WORDS + ko] = A[current_row * K_packed + kb + ko];
            }
            if (row_in_tile < TILE_DIM_N && (tn * TILE_DIM_N + row_in_tile) < N && (kb + ko) < K_packed) {
                b_p[row_in_tile * K_PANEL_WORDS + ko] = B_T[(tn * TILE_DIM_N + row_in_tile) * K_packed + kb + ko];
            }
        }
        __syncthreads();

        // Compute tile dot products from shared memory
        if (current_row < M) {
            for (int nit = 0; nit < TILE_DIM_N; ++nit) {
                int current_col = tn * TILE_DIM_N + nit;
                if (current_col < N) {
                    int agreement_count = 0;
                    for (int kip = 0; kip < K_PANEL_WORDS; ++kip) {
                        agreement_count += __popcll(~(a_p[row_in_tile * K_PANEL_WORDS + kip] ^ b_p[nit * K_PANEL_WORDS + kip]));
                    }
                    accumulator[nit] += agreement_count;
                }
            }
        }
        __syncthreads();
    }
    
    // Write results to global memory
    if (current_row < M) {
        for (int nit = 0; nit < TILE_DIM_N; ++nit) {
            int current_col = tn * TILE_DIM_N + nit;
            if (current_col < N) {
                C[current_row * N + current_col] = 2 * accumulator[nit] - K;
            }
        }
    }
}

torch::Tensor gemm_ash_algebra_cuda(
    const torch::Tensor& A_packed,
    const torch::Tensor& B_packed_T,
    int K)
{
    TORCH_CHECK(A_packed.is_cuda() && B_packed_T.is_cuda(), "Inputs must be CUDA tensors.");
    TORCH_CHECK(A_packed.dim() == 2 && B_packed_T.dim() == 2, "Inputs must be 2D.");
    TORCH_CHECK(A_packed.dtype() == torch::kInt64 && B_packed_T.dtype() == torch::kInt64, "Inputs must be int64 (uint64 packed).");

    const int M = A_packed.size(0);
    const int K_packed = A_packed.size(1);
    const int N = B_packed_T.size(0);
    
    TORCH_CHECK(B_packed_T.size(1) == K_packed, "Inner packed dimensions must match.");

    auto C_out = torch::empty({M, N}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));

    dim3 threads(THREADS_PER_BLOCK, 1, 1);
    dim3 blocks( (N + TILE_DIM_N - 1) / TILE_DIM_N, (M + TILE_DIM_M - 1) / TILE_DIM_M, 1 );

    tiled_ash_gemm_kernel<<<blocks, threads>>>(
        reinterpret_cast<const unsigned long long*>(A_packed.data_ptr<int64_t>()),
        reinterpret_cast<const unsigned long long*>(B_packed_T.data_ptr<int64_t>()),
        C_out.data_ptr<int>(),
        M, N, K, K_packed);
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Ash GEMM Kernel Launch Error: ") + cudaGetErrorString(err));
    }

    return C_out;
}
// --- END OF FILE src/helios_embed/ash_gemm.cu ---