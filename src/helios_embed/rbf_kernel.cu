// --- START OF FILE src/helios_embed/rbf_kernel.cu ---
#include "rbf_kernel.h"
#include <cuda_runtime.h>

// Tile dimension for shared memory. Must be a power of 2, 16 or 32 are good choices.
#define TILE_DIM 16

// This kernel computes one tile of the output K_nm matrix.
__global__ void rbf_kernel_fused(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    float* __restrict__ K_out,
    int N, int m, int D, float neg_gamma) // Pass negative gamma to avoid negation in the loop
{
    // Shared memory for one tile of X and one tile of Y
    __shared__ float X_tile[TILE_DIM][TILE_DIM + 1]; // +1 for padding to avoid bank conflicts
    __shared__ float Y_tile[TILE_DIM][TILE_DIM + 1];

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices determine the top-left corner of the tile in the output matrix
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Global row and column indices for the output matrix
    int global_row = block_row * TILE_DIM + ty;
    int global_col = block_col * TILE_DIM + tx;

    float sum = 0.0f;

    // Loop over the D dimension in tiles
    for (int t = 0; t < (D + TILE_DIM - 1) / TILE_DIM; ++t) {
        // --- Phase 1: Cooperatively load tiles into shared memory ---
        int X_load_col = t * TILE_DIM + tx;
        int Y_load_col = t * TILE_DIM + tx;
        
        // Coalesced read for X tile
        if (global_row < N && X_load_col < D) {
            X_tile[ty][tx] = X[global_row * D + X_load_col];
        } else {
            X_tile[ty][tx] = 0.0f;
        }

        // Coalesced read for Y tile
        if (global_col < m && Y_load_col < D) {
            Y_tile[ty][tx] = Y[global_col * D + Y_load_col];
        } else {
            Y_tile[ty][tx] = 0.0f;
        }
        __syncthreads();

        // --- Phase 2: Compute squared Euclidean distance for one tile dimension ---
        // Each thread computes its own sum += (x-y)^2 over the D dimension tile
        if (global_row < N && global_col < m) {
            for (int k = 0; k < TILE_DIM; ++k) {
                if ((t * TILE_DIM + k) < D) {
                    float diff = X_tile[ty][k] - Y_tile[tx][k]; // Transposed access to Y_tile
                    sum += diff * diff;
                }
            }
        }
        __syncthreads();
    }

    // --- Phase 3: Apply exponential and write out ---
    if (global_row < N && global_col < m) {
        K_out[global_row * m + global_col] = expf(neg_gamma * sum);
    }
}


torch::Tensor rbf_kernel_fused_cuda(
    const torch::Tensor& X,
    const torch::Tensor& Y,
    float gamma)
{
    TORCH_CHECK(X.is_cuda() && Y.is_cuda(), "Inputs must be CUDA tensors.");
    TORCH_CHECK(X.dim() == 2 && Y.dim() == 2, "Inputs must be 2D.");
    TORCH_CHECK(X.size(1) == Y.size(1), "Inner dimensions must match.");

    const int N = X.size(0);
    const int m = Y.size(0);
    const int D = X.size(1);

    auto K_out = torch::empty({N, m}, X.options());

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((m + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    rbf_kernel_fused<<<blocks, threads>>>(
        X.contiguous().data_ptr<float>(),
        Y.contiguous().data_ptr<float>(),
        K_out.data_ptr<float>(),
        N, m, D, -gamma); // Pass negative gamma
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Fused RBF Kernel Launch Error: ") + cudaGetErrorString(err));
    }
    
    return K_out;
}
// --- END OF FILE src/helios_embed/rbf_kernel.cu ---