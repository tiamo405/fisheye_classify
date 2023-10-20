from torchvision import datasets, models, transforms
from torchsampler import ImbalancedDatasetSampler
from torchmetrics.functional import accuracy, f1_score, precision_recall_curve
from torch.optim import lr_scheduler
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping, ModelPruning
from PIL import Image
from mlflow.tracking import MlflowClient
from adamp import AdamP
import torch.nn.functional as F
import torch.nn as nn
import torch
import pytorch_lightning as pl
import numpy as np
import mlflow.pytorch
import mlflow
from pathlib import Path
import uuid
import os
import io
import argparse
import timm
from datetime import datetime

import matplotlib.pyplot as plt

plt.switch_backend('agg')


# Support: ['get_time', 'l2_norm', 'make_weights_for_balanced_classes', 'get_val_pair', 'get_val_data', 'separate_irse_bn_paras', 'separate_resnet_bn_paras', 'warm_up_lr', 'schedule_lr', 'de_preprocess', 'hflip_batch', 'ccrop_batch', 'gen_plot', 'perform_val', 'buffer_val', 'AverageMeter', 'accuracy']

torch.manual_seed(17)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def get_time():
	return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def l2_norm(input, axis=1):
	norm = torch.norm(input, 2, axis, True)
	output = torch.div(input, norm)

	return output


def make_weights_for_balanced_classes(images, nclasses):
	'''
			Make a vector of weights for each image in the dataset, based
			on class frequency. The returned vector of weights can be used
			to create a WeightedRandomSampler for a DataLoader to have
			class balancing when sampling for a training batch.
					images - torchvisionDataset.imgs
					nclasses - len(torchvisionDataset.classes)
			https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
	'''
	count = [0] * nclasses
	for item in images:
		count[item[1]] += 1  # item is (img-data, label-id)
	weight_per_class = [0.] * nclasses
	N = float(sum(count))  # total number of images
	for i in range(nclasses):
		weight_per_class[i] = N / float(count[i])
	weight = [0] * len(images)
	for idx, val in enumerate(images):
		weight[idx] = weight_per_class[val[1]]

	return weight


# def get_val_pair(path, name):
#     carray = bcolz.carray(rootdir = os.path.join(path, name), mode = 'r')
#     issame = np.load('{}/{}_list.npy'.format(path, name))

#     return carray, issame


# def get_val_data(data_path):
# 	lfw, lfw_issame = get_val_pair(data_path, 'lfw')
# 	cfp_ff, cfp_ff_issame = get_val_pair(data_path, 'cfp_ff')
# 	cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
# 	agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
# 	calfw, calfw_issame = get_val_pair(data_path, 'calfw')
# 	cplfw, cplfw_issame = get_val_pair(data_path, 'cplfw')
# 	vgg2_fp, vgg2_fp_issame = get_val_pair(data_path, 'vgg2_fp')

# 	return lfw, cfp_ff, cfp_fp, agedb_30, calfw, cplfw, vgg2_fp, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_30_issame, calfw_issame, cplfw_issame, vgg2_fp_issame


def separate_irse_bn_paras(modules):
	if not isinstance(modules, list):
		modules = [*modules.modules()]
	paras_only_bn = []
	paras_wo_bn = []
	for layer in modules:
		if 'model' in str(layer.__class__):
			continue
		if 'container' in str(layer.__class__):
			continue
		else:
			if 'batchnorm' in str(layer.__class__):
				paras_only_bn.extend([*layer.parameters()])
			else:
				paras_wo_bn.extend([*layer.parameters()])

	return paras_only_bn, paras_wo_bn


def separate_resnet_bn_paras(modules):
	all_parameters = modules.parameters()
	paras_only_bn = []

	for pname, p in modules.named_parameters():
		if pname.find('bn') >= 0:
			paras_only_bn.append(p)

	paras_only_bn_id = list(map(id, paras_only_bn))
	paras_wo_bn = list(
		filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))

	return paras_only_bn, paras_wo_bn


def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
	for params in optimizer.param_groups:
		params['lr'] = batch * init_lr / num_batch_warm_up

	# print(optimizer)


def schedule_lr(optimizer):
	for params in optimizer.param_groups:
		params['lr'] /= 10.

	print(optimizer)


def de_preprocess(tensor):

	return tensor * 0.5 + 0.5


hflip = transforms.Compose([
	de_preprocess,
	transforms.ToPILImage(),
	transforms.functional.hflip,
	transforms.ToTensor(),
	transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def hflip_batch(imgs_tensor):
	hfliped_imgs = torch.empty_like(imgs_tensor)
	for i, img_ten in enumerate(imgs_tensor):
		hfliped_imgs[i] = hflip(img_ten)

	return hfliped_imgs


ccrop = transforms.Compose([
	de_preprocess,
	transforms.ToPILImage(),
	transforms.Resize([128, 128]),  # smaller side resized
	transforms.CenterCrop([112, 112]),
	transforms.ToTensor(),
	transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def ccrop_batch(imgs_tensor):
	ccropped_imgs = torch.empty_like(imgs_tensor)
	for i, img_ten in enumerate(imgs_tensor):
		ccropped_imgs[i] = ccrop(img_ten)

	return ccropped_imgs


def gen_plot(fpr, tpr):
	"""Create a pyplot plot and save to buffer."""
	plt.figure()
	plt.xlabel("FPR", fontsize=14)
	plt.ylabel("TPR", fontsize=14)
	plt.title("ROC Curve", fontsize=14)
	plot = plt.plot(fpr, tpr, linewidth=2)
	buf = io.BytesIO()
	plt.savefig(buf, format='jpeg')
	buf.seek(0)
	plt.close()

	return buf


def buffer_val(writer, db_name, acc, best_threshold, roc_curve_tensor, epoch):
	writer.add_scalar('{}_Accuracy'.format(db_name), acc, epoch)
	writer.add_scalar('{}_Best_Threshold'.format(
		db_name), best_threshold, epoch)
	writer.add_image('{}_ROC_Curve'.format(db_name), roc_curve_tensor, epoch)


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
