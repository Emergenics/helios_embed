// --- START OF FILE src/helios_embed/helios_embed_pybind.cpp (FINAL v2.5.0 with
// CPU Guards) ---
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

// --- THIS IS THE CRITICAL FIX: The HELIOS_CPU_BUILD Guard ---
// The setup.py script defines this macro ONLY during a CPU-only build.
// This prevents the compiler from trying to include or bind CUDA-only code.
#ifndef HELIOS_CPU_BUILD
#include "incremental_nystrom_engine.h"
#include "nystrom_engine.h"
#endif
// --- END OF CRITICAL FIX ---

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Helios.Embed: The Final, Validated Nystrom Feature Engine";

  // The get_build_info function is always available, for both CPU and CUDA
  // builds.
  m.def(
      "get_build_info",
      []() {
        py::dict d;
        d["torch_version"] = HELIOS_BUILD_TORCH_VERSION;
        d["cuda_version"] = HELIOS_BUILD_CUDA_VERSION;
// The arch_list macro might not be defined in a CPU build, so we guard it.
#ifdef HELIOS_BUILD_ARCH_LIST
        d["arch_list"] = HELIOS_BUILD_ARCH_LIST;
#else
        d["arch_list"] = "";
#endif
        d["cxx11_abi"] = HELIOS_BUILD_CXX11_ABI;
        return d;
      },
      "Returns a dictionary of build-time metadata for compatibility checks.");

// --- Guard the CUDA-only function and class bindings ---
#ifndef HELIOS_CPU_BUILD
  py::class_<IncrementalNystromEngine>(m, "IncrementalNystromEngine")
      .def(py::init<torch::Tensor, float, float>(), py::arg("landmarks"),
           py::arg("gamma"), py::arg("ridge"))
      .def("build", &IncrementalNystromEngine::build)
      .def("update", &IncrementalNystromEngine::update);

  m.def("compute_rkhs_embedding", &compute_rkhs_embedding,
        "The single, production-ready, stateless Nystrom feature embedding "
        "function.",
        py::arg("X"), py::arg("landmarks"), py::arg("gamma"), py::arg("ridge"));
#endif
}
// --- END OF FILE src/helios_embed/helios_embed_pybind.cpp (FINAL v2.5.0 with
// CPU Guards) ---