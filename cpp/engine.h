#pragma once

#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

using namespace std;
using namespace nvinfer1;

namespace mobilenet {

class Engine {
public:
    Engine(const string &engine_path, bool verbose=true);
    Engine(const char *onnx_model, size_t onnx_size, const vector<int> &dynamic_batch_opts,
        float score_thresh, bool verbose, size_t workspace_size=(1ULL << 30));

    ~Engine();

    void save(const string &path);
    void infer(vector<void *> &buffers, int batch);

    // Get (h, w) size of the fixed input
    vector<int> getInputSize();

    // Get max allowed batch size
    int getMaxBatchSize();

private:
    IRuntime *_runtime = nullptr;
    ICudaEngine *_engine = nullptr;
    IExecutionContext *_context = nullptr;
    cudaStream_t _stream = nullptr;

    void _load(const string &path);
    void _prepare();

};
}