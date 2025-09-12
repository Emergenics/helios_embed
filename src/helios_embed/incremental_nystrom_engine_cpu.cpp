// --- START OF FILE src/helios_embed/incremental_nystrom_engine_cpu.cpp ---
#include "incremental_nystrom_engine.h"
#include <stdexcept>

// CPU-only stub implementations for the stateful engine class.

IncrementalNystromEngine::IncrementalNystromEngine(
    torch::Tensor landmarks,
    float gamma,
    float ridge)
{
    TORCH_CHECK(false, "Helios.Embed was compiled in CPU-only mode. CUDA functionality is not available.");
}

torch::Tensor IncrementalNystromEngine::build(const torch::Tensor& X) {
    TORCH_CHECK(false, "Helios.Embed was compiled in CPU-only mode. CUDA functionality is not available.");
    return torch::Tensor();
}

torch::Tensor IncrementalNystromEngine::update(const torch::Tensor& X_new_in, const torch::Tensor& Phi_old_in) {
    TORCH_CHECK(false, "Helios.Embed was compiled in CPU-only mode. CUDA functionality is not available.");
    return torch::Tensor();
}
// --- END OF FILE src/helios_embed/incremental_nystrom_engine_cpu.cpp ---