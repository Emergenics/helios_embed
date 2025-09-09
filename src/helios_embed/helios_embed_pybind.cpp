// --- START OF FILE helios_embed/helios_embed_pybind.cpp (CORRECTED) ---
#include <torch/extension.h>
#include "nystrom_engine.h"
#include "incremental_nystrom_engine.h"

namespace py = pybind11;

// The module name here MUST match the 'name' in the CUDAExtension in setup.py
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Helios.Embed: The Nystrom Feature Engine (Core C++/CUDA Backend)";

    py::class_<IncrementalNystromEngine>(m, "IncrementalNystromEngine")
        .def(py::init<torch::Tensor, float, float>())
        .def("build", &IncrementalNystromEngine::build)
        .def("update", &IncrementalNystromEngine::update);

    m.def("compute_rkhs_embedding", &compute_rkhs_embedding_nystrom, 
          "Stateless Nystrom feature embedding (RBF Kernel).");
}
// --- END OF FILE helios_embed/helios_embed_pybind.cpp (CORRECTED) ---