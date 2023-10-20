import torch
import numpy as np
import timeit
import sys
import os
import uuid
import imutils
from imutils.video import FileVideoStream, VideoStream

pwd = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(pwd, ".."))

from yolov5.models.yolo import Model
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.helper import *
from utils import *

from mobilenet._C import Engine

model_path = "/workspace/weights/gender_classify_mobilenet.pth"
engine = Engine.load(model_path.replace("pth", "plan"))
print("Load engine success...")

# Load model
cfg = '/workspace/yolov5/models/yolov5s.yaml'
weights = '/workspace/weights/yolov5.pt'
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model(cfg=cfg,ch=3, nc=1, anchors=None)
ckpt = torch.load(weights, map_location=device)  # load checkpoint
model.load_state_dict(ckpt["model"].state_dict())
model = model.eval().to(device)

model_classify = models.mobilenet_v2(pretrained=False)
model_classify.classifier[1] = nn.Linear(model_classify.last_channel, 2)
model_classify.load_state_dict(torch.load(model_path))
model_classify.eval()
model_classify.to(device)

# vs = FileVideoStream("/workspace/video/nautilus.mp4").start()
vs = VideoStream("rtsp://chuaboc-ai:torano123@so2chuaboc.cameraddns.net:1554/Streaming/Channels/1101").start()
ROI = [552, 190, 1429, 700] # cua
# ROI = [29, 273, 1582, 978] # quay
shape = (1920, 1080)
frame_count = 0
interval = 100

while True:
	image = vs.read()
	if image is None:
		break

	image = cv2.resize(image, shape)
	
	if frame_count % interval == 0:
		inp = preprocess(image.copy(), input_size, auto=False, scale=False)
		inp = torch.from_numpy(inp).to(device)

		with torch.no_grad():
			pred = model(inp, augment=False, visualize=False)[0]

		pred = non_max_suppression(pred, score_thresh, nms_thresh, classes, agnostic_nms, max_det=top_n)

		for i, det in enumerate(pred):  # detections per image
			if len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(inp.shape[2:], det[:, :4], image.shape[:2]).round()
				for *xyxy, conf, cls in reversed(det):
					x1, y1, x2, y2 = [int(bb.cpu().numpy()) for bb in xyxy]
					cx = (x1 + x2) // 2
					cy = (y1 + y2) // 2

					if (ROI[0] < cx < ROI[2]) and (ROI[1] < cy < ROI[3]):
						# cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
						im_crop = image.copy()[y1:y2, x1:x2]
					
						inp = clf_preprocess(im_crop)
						inp = inp.to(device)
						data = engine(inp)[0]
						# with torch.no_grad():
						# 	data = model_classify(inp)
						score = torch.nn.functional.softmax(data, dim=1)[0] # score = [male, female]
						print(score)

						if im_crop is not None:
							# im_crop = cv2.resize(im_crop, (128, 256))
							if score[0] > score[1]:
								cv2.imwrite(f"/workspace/cpp/data/male/{uuid.uuid1()}.jpg", im_crop)
								cv2.imshow("male", im_crop)
								key = cv2.waitKey(1) & 0xff
								if key == ord('q'):
									break
							else:
								cv2.imwrite(f"/workspace/cpp/data/female/{uuid.uuid1()}.jpg", im_crop)
								cv2.imshow("female", im_crop)
								key = cv2.waitKey(1) & 0xff
								if key == ord('q'):
									break
	frame_count+=1

	cv2.imshow("image", imutils.resize(image, width=800))
	key = cv2.waitKey(1) & 0xff
	if key == ord('q'):
		break

cv2.destroyAllWindows()