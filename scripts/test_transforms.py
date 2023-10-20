import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from torchvision import transforms as T
from PIL import Image
from pathlib import Path



plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(1)

def searching_all_files(directory: Path):   
    file_list = [] # A list for storing files existing in directories

    for x in directory.iterdir():
        if x.is_file():

           file_list.append(x)
        else:

           file_list.append(searching_all_files(directory/x))

    return file_list


def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.imshow(img)
        plt.show()


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = T.Compose(
	[
		T.Resize([224, 224]),
		T.RandomVerticalFlip(),
		T.RandomHorizontalFlip(),
		# T.RandomAffine(degrees=(0, 180), translate=(0.1, 0.3), scale=(0.5, 0.75)),
		T.ToTensor(),
		# T.Normalize(mean, std),
	]
)

data_dir = "/mnt/sda1/Datasets/classification/pharma3/train/staff"

list_files = searching_all_files(Path(f"{data_dir}"))

random.shuffle(list_files)

img_paths = random.choices(list_files, k=100)
for img_path in img_paths:
    cv2_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

    img = Image.fromarray(cv2_img)
    transformed_img = data_transforms(img)
    print(transformed_img)
    show([transformed_img])
