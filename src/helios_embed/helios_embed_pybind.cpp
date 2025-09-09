// --- START OF FILE src/helios_embed/helios_embed_pybind.cpp (Version 1.2.0) ---
#include <torch/extension.h>
#include "nystrom_engine.h"
#include "incremental_nystrom_engine.h"
#include "hybrid_kernel.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Helios.Embed: Nystrom Engine with Hybrid Kernel Accelerator";

    // --- Original Bindings (Unchanged) ---
    py::class_<IncrementalNystromEngine>(m, "IncrementalNystromEngine")
        .def(py::init<torch::Tensor, float, float>(), py::arg("landmarks"), py::arg("gamma"), py::arg("ridge"))
        .def("build", &IncrementalNystromEngine::build)
        .def("update", &IncrementalNystromEngine::update);

    m.def("compute_rkhs_embedding", &compute_rkhs_embedding_nystrom, "Stateless Nystrom feature embedding (ATen-backed).", py::arg("X"), py::arg("landmarks"), py::arg("gamma"), py::arg("ridge"));
          
    // --- New Binding for the Hybrid Kernel ---
    m.def("rbf_kernel_hybrid_cuda", &rbf_kernel_hybrid_cuda, 
          "Computes the RBF kernel matrix using a hybrid cuBLAS + fused kernel approach.",
          py::arg("X"), py::arg("Y"), py::arg("gamma"));
}
// --- END OF FILE src/helios_embed/helios_embed_pybind.cpp (Version 1.2.0) ---