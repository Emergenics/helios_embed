// --- START OF FILE nystrom_engine/incremental_nystrom_engine.h ---
#pragma once
#include <torch/extension.h>
#include "nystrom_engine.h" 
class IncrementalNystromEngine {
public:
    IncrementalNystromEngine(torch::Tensor landmarks, float gamma,float ridge);
    torch::Tensor build(const torch::Tensor& X);
    torch::Tensor update(const torch::Tensor& X_new, const torch::Tensor& Phi_old);

private:
    torch::Tensor landmarks_;
    float gamma_;
    float ridge_;
    torch::Tensor K_mm_inv_sqrt_; // The expensive, cached part
    void initialize_engine();
};
// --- END OF FILE nystrom_engine/incremental_nystrom_engine.h ---