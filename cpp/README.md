# Deepstream
requirements:
- pytorch: 1.8.1+cu102
- tensorrt: 7.0.0

## Env
```bash
- x86
docker run --name ds-classify --runtime nvidia -dit -v /mnt/sda1/fisheye_classify:/workspace hub.cxview.ai/deepstream-people:0.1-base-x86

- aarch
docker run --name ds-classify --runtime nvidia -dit -v /home/mdt/hoangnt/greeting_fisheye:/workspace hub.cxview.ai/deepstream-people:0.1.1-base-aarch
```
## Build

```bash
mkdir build && cd build
cmake -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" -DPLATFORM_TEGRA=ON ..
make -j4
```

## Export TRT
```bash
./export model.onnx model.plan
```

## Benchmark trt
```bash
cd /usr/src/tensorrt/bin
./trtexec --loadEngine=/workspace/weights/model.plan --explicitBatch --shapes=input_0:1x3x224x224
```
###### FPS = (1 / Mean GPU Compute) x 1000

# Experiments
| model | Dataset     |  epoch    |   Best Accuracy (ours testset)	 |   Params   |   GMac  | Jetson Nano  TRT(224, 224) | Stage |
|-------|--------|--------|--------|--------|--------|--------|--------|
| mnasnet_b1 | old | 20  | 73.62% | 4.38 M  | 0.32 | 16.68 ms; 59,95 FPS | Production |
| rexnet_100 | customer_staff_20220816 | 30  | 98.56%  | 4.8 M  | 0.40 | 22.69 ms; 44,07 FPS | Staging |