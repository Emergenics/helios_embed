// --- START OF FILE src/helios_embed/helios_embed_pybind.cpp (Version 1.2.1 - Final) ---
#include <torch/extension.h>
#include "nystrom_engine.h"
#include "incremental_nystrom_engine.h"
#include "rbf_kernel.h"
#include "ash_gemm.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Helios.Embed: Nystrom Engine with Ash GEMM Accelerator";

    py::class_<IncrementalNystromEngine>(m, "IncrementalNystromEngine")
        .def(py::init<torch::Tensor, float, float>(), py::arg("landmarks"), py::arg("gamma"), py::arg("ridge"))
        .def("build", &IncrementalNystromEngine::build)
        .def("update", &IncrementalNystromEngine::update);

    m.def("compute_rkhs_embedding", &compute_rkhs_embedding_nystrom, "Stateless Nystrom feature embedding (RBF Kernel).", py::arg("X"), py::arg("landmarks"), py::arg("gamma"), py::arg("ridge"));
    
    m.def("rbf_kernel_fused_cuda", &rbf_kernel_fused_cuda, "Custom fused CUDA kernel for RBF matrix computation.", py::arg("X"), py::arg("Y"), py::arg("gamma"));

    m.def("gemm_ash_algebra_cuda", &gemm_ash_algebra_cuda, "Performs a GEMM for bipolar (±1) matrices using the 'Difference-as-Compute' (XNOR+POPCOUNT) paradigm.", py::arg("A_packed"), py::arg("B_packed_T"), py::arg("K"));
}
// --- END OF FILE src/helios_embed/helios_embed_pybind.cpp (Version 1.2.1 - Final) ---