import numpy as np
import cv2
import torch

import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from pyolo.models.experimental import attempt_load
from pyolo.utils.general import check_img_size, polygon_non_max_suppression, polygon_scale_coords
from pyolo.utils.torch_utils import select_device, time_synchronized
from pyolo.utils.datasets import letterbox
from pyolo.utils.crop_rotate import crop_polygon


device = select_device('0')
half = True
imgsz = 1024
classify = False # Polygon does not support second-stage classifier
conf_thres=0.85  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=100  # maximum detections per image
classes=None
agnostic_nms=False
augment=False
is_fit = False
weights = "../weights/polygon_best_1024.pt"

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = transforms.Compose(
	[
		transforms.Resize([224, 224]),
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
	]
)


def clf_preprocess(image):
	img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	img = data_transforms(img)
	if img.ndimension() == 3:
		img = torch.unsqueeze(img, 0)

	return img

def dt_preprocess(img, img_size=1024, stride=32):
	# Padded resize
	img = letterbox(img, img_size, stride=stride)[0]

	# Convert
	img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
	img = np.ascontiguousarray(img)
	img = torch.from_numpy(img)
	img = img.half() if half else img.float()  # uint8 to fp16/32
	img /= 255.0  # 0 - 255 to 0.0 - 1.0
	if img.ndimension() == 3:
		img = img.unsqueeze(0)

	return img