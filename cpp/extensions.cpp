#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "engine.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using namespace std;


vector<at::Tensor> infer(mobilenet::Engine &engine, at::Tensor data) {
    CHECK_INPUT(data);

    int batch = data.size(0);
    auto scores = at::zeros({batch, 2}, data.options());

    vector<void *> buffers;
    for (auto buffer : {data, scores}) {
        buffers.push_back(buffer.data<float>());
    }

    engine.infer(buffers, batch);

    return {scores};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<mobilenet::Engine>(m, "Engine")
        .def(pybind11::init<const char *, size_t, const vector<int>&,
            float, bool>())
        // .def("save", &mobilenet::Engine::save)
        // .def("infer", &mobilenet::Engine::infer)
        // .def_property_readonly("input_size", &sample_onnx::Engine::getInputSize)
        .def_static("load", [](const string &path) {
            return new mobilenet::Engine(path);
        })
        .def("__call__", [](mobilenet::Engine &engine, at::Tensor data) {
            return infer(engine, data);
        });
}