// --- START OF FILE src/helios_embed/helios_embed_pybind.cpp (FINAL v2.4.0 with Build Info) ---
#include <torch/extension.h>
#include "nystrom_engine.h"
#include "incremental_nystrom_engine.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Helios.Embed: The Final, Validated Nystrom Feature Engine";

    py::class_<IncrementalNystromEngine>(m, "IncrementalNystromEngine")
        .def(py::init<torch::Tensor, float, float>(), py::arg("landmarks"), py::arg("gamma"), py::arg("ridge"))
        .def("build", &IncrementalNystromEngine::build)
        .def("update", &IncrementalNystromEngine::update);

    m.def("compute_rkhs_embedding", &compute_rkhs_embedding, 
          "The single, production-ready, stateless Nystrom feature embedding function.", 
          py::arg("X"), py::arg("landmarks"), py::arg("gamma"), py::arg("ridge"));
          
    // --- THIS IS THE NEW, CRITICAL ADDITION ---
    m.def("get_build_info", [](){
        py::dict d;
        d["torch_version"] = HELIOS_BUILD_TORCH_VERSION;
        d["cuda_version"]  = HELIOS_BUILD_CUDA_VERSION;
        d["arch_list"]     = HELIOS_BUILD_ARCH_LIST;
        d["cxx11_abi"]     = HELIOS_BUILD_CXX11_ABI;
        return d;
    }, "Returns a dictionary of build-time metadata for compatibility checks.");
    // --- END OF NEW ADDITION ---
}
// --- END OF FILE src/helios_embed/helios_embed_pybind.cpp (FINAL v2.4.0 with Build Info) ---