import numpy as np
import cv2
import timm
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision.transforms.functional as tvf
import imutils
import timeit
from imutils import paths
from matplotlib import pyplot as plt
from collections import OrderedDict

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = transforms.Compose(
	[
		transforms.Resize([224, 224]),
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
	]
)

# model_classify = models.mobilenet_v2(pretrained=False)
model_classify = models.mobilenet_v3_large(pretrained=False)
# model_classify = models.mnasnet1_0(pretrained=False)
# model_classify = models.efficientnet_b0(pretrained=False)
# model_classify = timm.create_model('efficientnet_b1_pruned', pretrained=False)
# model_classify.classifier = nn.Linear(1280, 2) #efficientnet_b1_pruned
# model_classify.classifier[1] = nn.Linear(1280, 2)
model_classify.classifier[3] = nn.Linear(1280, 2) #mobilenet_v3_large
checkpoint = torch.load("/mnt/sda1/greeting_fisheye/weights/staff_classify_mobilenetv3.pth")
for key in list(checkpoint.keys()):
    if "classifier.0" in key:
        checkpoint[key.replace("classifier.0", "classifier")] = checkpoint[key]
        del checkpoint[key]
model_classify.load_state_dict(checkpoint)
model_classify.eval()

imagePaths = sorted(list(paths.list_images("/mnt/sda1/greeting_fisheye/test")))
for img_path in imagePaths:
    cv2_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    cv2_img = imutils.rotate_bound(cv2_img, -90)
    img = Image.fromarray(cv2_img)
    img = data_transforms(img)
    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        t0 = timeit.default_timer()
        output = model_classify(img)
        t1 = timeit.default_timer()
        print(t1-t0)
    score = torch.nn.functional.softmax(output, dim=1)[0]
    label = None
    if score[0] > 0.8:
        label = "KH"
    else:
        label = "NV"
    print(score)
    plt.title(label)
    plt.imshow(cv2_img)
    plt.show()