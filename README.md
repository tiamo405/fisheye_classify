# MLflow
```bash
mlflow server --host 192.168.100.89 --backend-store-uri experiments --default-artifact-root ./artifacts
mlflow ui --host 192.168.100.89
```

# Dashboard
```bash
http://192.168.100.175:5000
```

# Training
```python
python train.py --arch rexnet_100 --data_dir datasets/customer_staff_20220816 --exp_name pmc_rexnet_100 --exp_id 3 --epoch 50
```

# Export onnx
```python
python3 scripts/export_onnx.py --runID 86391f9f84db465b92a70c653290d171 --arch rexnet_100
```

# Test onnx
```bash
scripts/test_onnx.ipynb
```