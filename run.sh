# CUDA_VISIBLE_DEVICES=1 python train.py --arch rexnet_100 --data_dir /mnt/nvme0n1/phuongnam/fisheye_classify/data --exp_name pmc_rexnet_100 --exp_id 1 --epoch 20

# CUDA_VISIBLE_DEVICES=1 python3 scripts/export_onnx.py --runID cdb293e6fd234cf9a0f9e78514150b7d
# model onnx luu o id/model/data/last.onnx

CUDA_VISIBLE_DEVICES=1 python3 scripts/test_onnx.py 