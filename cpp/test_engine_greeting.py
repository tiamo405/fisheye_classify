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

from utils import *
from mobilenet._C import Engine

model_path = "/workspace/weights/greeting_classify_mobilenet_softmax.pth"
# engine = Engine.load(model_path.replace("pth", "plan"))
# print("Load engine success...")

# Load model
model_detect = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model_detect.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check image size
names = model_detect.module.names if hasattr(model_detect, 'module') else model_detect.names  # get class names
if half:
	model_detect.half()  # to FP16

model_classify = models.mobilenet_v2(pretrained=False)
model_classify.classifier[1] = nn.Linear(model_classify.last_channel, 2)
model_classify.load_state_dict(torch.load(model_path))
model_classify.eval()
model_classify.to(device)

vs = FileVideoStream("/workspace/video/pharmacity_greeting_cut.mp4").start()
interval = 2
frame_count = 0

while True:
	image = vs.read()
	if image is None:
		break

	img = dt_preprocess(image.copy()).to(device)
	pred = model_detect(img, augment=augment)[0]

	# Apply polygon NMS
	dets = polygon_non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
	t2 = time_synchronized()

	# Process detections
	for i, det in enumerate(dets):  # detections per image
		if det is not None and len(det):
			det[:, :8] = polygon_scale_coords(img.shape[2:], det[:, :8], image.shape).round()
			det = det.cpu().numpy()
			for *xyxyxyxy, conf, cls in det:
				xyxyxyxy = [int(i) for i in xyxyxyxy]
				pts = np.array([xyxyxyxy[i:i+2] for i in range(0, len(xyxyxyxy), 2)], np.int32)
				pts = pts.reshape((-1, 1, 2))
				
				im_crop = None

				if is_fit:
					im_crop = crop_polygon(image.copy(), pts)
					# cv2.polylines(image, [pts], True, (0, 0, 255), 3)
				else:
					p1 = pts[0][0]
					p2 = pts[1][0]
					p3 = pts[2][0]
					p4 = pts[3][0]
					xmin = np.array([p1[0], p2[0], p3[0], p4[0]]).min()
					ymin = np.array([p1[1], p2[1], p3[1], p4[1]]).min()
					xmax = np.array([p1[0], p2[0], p3[0], p4[0]]).max()
					ymax = np.array([p1[1], p2[1], p3[1], p4[1]]).max()

					im_crop = image.copy()[ymin:ymax, xmin:xmax]
					# cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)

				# if frame_count % interval == 0:
				# 	cv2.imwrite(f"data/{frame_count}.jpg", im_crop)
				# im_crop = imutils.rotate_bound(im_crop, -90)
				inp = clf_preprocess(im_crop)
				inp = inp.to(device)
				# data = engine(inp)[0]
				with torch.no_grad():
					data = model_classify(inp)
				# score = torch.nn.functional.softmax(data, dim=1)[0]
				score = data[0]
				print(score)
				label = "no greeting"
				is_greeting = score[1] > 0.8
				if is_greeting:
					label = "greeting"
					cv2.polylines(image, [pts], True, (0, 255, 0), 3)
					cv2.putText(image, f"{label} - {score[1]:.2f}", tuple(pts[0][0]),
								cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
					# cv2.imwrite("greeting.jpg", im_crop)
					# print(score, label)

					if im_crop is not None:
						cv2.imshow("crop", im_crop)
						key = cv2.waitKey(1) & 0xff
						if key == ord('q'):
							break

	frame_count+=1

	cv2.imshow("image", imutils.resize(image, width=1080))
	key = cv2.waitKey(1) & 0xff
	if key == ord('q'):
		break

cv2.destroyAllWindows()