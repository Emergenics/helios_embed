// --- START OF FILE src/helios_embed/helios_embed_pybind.cpp (Version 2.1.0 - FINAL) ---
#include <torch/extension.h>
#include "nystrom_engine.h"
#include "incremental_nystrom_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Helios.Embed: The Final, Validated Nystrom Feature Engine";

    py::class_<IncrementalNystromEngine>(m, "IncrementalNystromEngine")
        .def(py::init<torch::Tensor, float, float>(), py::arg("landmarks"), py::arg("gamma"), py::arg("ridge"))
        .def("build", &IncrementalNystromEngine::build)
        .def("update", &IncrementalNystromEngine::update);

    m.def("compute_rkhs_embedding", &compute_rkhs_embedding_nystrom, 
          "The single, production-ready, stateless Nystrom feature embedding function.", 
          py::arg("X"), py::arg("landmarks"), py::arg("gamma"), py::arg("ridge"));
}
// --- END OF FILE src/helios_embed/helios_embed_pybind.cpp (Version 2.1.0 - FINAL) ---