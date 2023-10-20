import argparse
import io
import os
import sys
import onnxruntime as rt
import timm
import torch
import torch.nn as nn

pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(pwd, ".."))

from models import PMC


def parse_opt():
	parser = argparse.ArgumentParser()
	parser.add_argument('--runID', type=str, default='eb1759cb0763421f891ba22b665fd620', help='')
	parser.add_argument('--arch', type=str, default='rexnet_100', help='')
	parser.add_argument('--features', type=int, default=1280, help='')
	parser.add_argument('--pretrained', type=bool, default=False, help='')
	parser.add_argument('--num_cls', type=int, default=3, help='')
	parser.add_argument('--shape', type=list, default=[224, 224], help='')

	return parser.parse_args()


opt = parse_opt()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PMC(arch=opt.arch, params={"features": opt.features, "cls": opt.num_cls}, training=False)

# weight_path = "mlruns/2aba0ce12a2647fdb0f9f9ca6a9ddaab/artifacts/model/data/last.ckpt"
weight_path = f"mlruns/{opt.runID}/artifacts/model/data/last.ckpt"

checkpoint = torch.load(weight_path)
model.load_state_dict(checkpoint["state_dict"])
model = model.to(device)
model.eval()

input_names = ['input']
output_names = ['output']
onnx_bytes = io.BytesIO()
zero_input = torch.zeros([1, 3] + opt.shape, dtype=torch.float32)
zero_input = zero_input.to(device)
dynamic_axes = {input_names[0]: {0: 'batch'}}
onnx_model = weight_path.replace("ckpt", "onnx")

for _, name in enumerate(output_names):
	dynamic_axes[name] = dynamic_axes[input_names[0]]
extra_args = {'opset_version': 10, 'verbose': False,
			  'input_names': input_names, 'output_names': output_names,
			  'dynamic_axes': dynamic_axes}
torch.onnx.export(model, zero_input, onnx_bytes, **extra_args)
with open(onnx_model, 'wb') as out:
	out.write(onnx_bytes.getvalue())

print(f"[ONNX] Exported at {onnx_model}")

sess = rt.InferenceSession(onnx_model)
print("====INPUT====")
for i in sess.get_inputs():
	print(f"Name: {i.name}, Shape: {i.shape}, Dtype: {i.type}")
print("====OUTPUT====")
for i in sess.get_outputs():
	print(f"Name: {i.name}, Shape: {i.shape}, Dtype: {i.type}")
