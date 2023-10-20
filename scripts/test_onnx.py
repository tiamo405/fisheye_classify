import onnxruntime as rt
import argparse
from imutils import paths
import cv2
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

runID = "cdb293e6fd234cf9a0f9e78514150b7d"
image_shape = [224, 224]
threshold = 0.85
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

test_transforms = transforms.Compose(
	[
		transforms.Resize([224, 224]),
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
	]
)

onnx_model = f"mlruns/{runID}/artifacts/model/data/last.onnx"
# onnx_model = "mlruns/66a5e35cfbde4006bda9e3325e4a4ae8/artifacts/model/data/last.onnx"
sess = rt.InferenceSession(onnx_model)
print("====INPUT====")
for i in sess.get_inputs():
	print(f"Name: {i.name}, Shape: {i.shape}, Dtype: {i.type}")
print("====OUTPUT====")
for i in sess.get_outputs():
	print(f"Name: {i.name}, Shape: {i.shape}, Dtype: {i.type}")

imagePaths = sorted(list(paths.list_images("/mnt/nvme0n1/locpv/anh_duoc_si")))

correct = 0
staff_correct = 0
customer_correct = 0
total_sample = len(imagePaths)
total_staff = 0
total_customer = 0
# dict_label={
#       "black": 0,
#       "customer": 1,
#       "white": 2
# }
dict_label = {
      0:"black",
      1:"customer",
      2:"white"
}
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
labels = []
for i in range(total_sample):
    # true_name_label = imagePaths[i].split("/")[-2]
    # true_label = dict_label[true_name_label]

    ori = cv2.imread(imagePaths[i])
    img = Image.fromarray(cv2.cvtColor(ori, cv2.COLOR_BGR2RGB))
    img = test_transforms(img)
    if img.ndimension() == 3:
        img = torch.unsqueeze(img, 0)

    pred = sess.run(None, {"input": img.numpy()})[0][0].tolist()

    label_pred = dict_label[np.argmax(pred)]
    labels.append(label_pred)
    # if label_pred == true_label:
    #       correct += 1
    
    # print(f"true_name_label:{true_name_label},  pred: {np.argmax(pred)}, score: {pred[np.argmax(pred)]}")
for image_path, label in zip(imagePaths, labels):  
    destination_filename = f"{label}_{os.path.basename(image_path)}"
    os.makedirs(os.path.join('pred', label), exist_ok= True)
    destination_path = os.path.join('pred', label,destination_filename)
    image = cv2.imread(image_path)
    image = cv2.putText(image, label, (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[np.argmax(pred)], 1)
    cv2.imwrite(filename= destination_path, img= image)

# print(f"acc: {correct/total_sample}")

