from utils import *


class PMC(pl.LightningModule):
	def __init__(self, arch, params, training=True):
		super().__init__()
		self.params = params
		if arch == "mnasnet_b1":
			self.model = timm.create_model('mnasnet_b1')
			self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.params["features"], self.params["cls"]), nn.Softmax(1))
		elif arch == "mnasnet_a1":
			self.model = timm.create_model('mnasnet_a1')
			self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.params["features"], self.params["cls"]), nn.Softmax(1))
		elif arch == "mixnet_s":
			self.model = timm.create_model('mixnet_s', pretrained=training)
			self.model.classifier = nn.Sequential(nn.Linear(self.params["features"], self.params["cls"]), nn.Softmax(1))
		elif arch == "mixnet_m":
			self.model = timm.create_model('mixnet_m', pretrained=training)
			self.model.classifier = nn.Sequential(nn.Linear(self.params["features"], self.params["cls"]), nn.Softmax(1))
		elif arch == "efficientnet_b0":
			self.model = timm.create_model('efficientnet_b0', pretrained=training)
			self.model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.params["features"], self.params["cls"]), nn.Softmax(1))
		elif arch == "rexnet_100":
			self.model = timm.create_model('rexnet_100', pretrained=training)
			self.model.head.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.params["features"], self.params["cls"]), nn.Softmax(1))

		if training:
			print("[Training] Freezing Backbone...")
			for name, param in self.model.named_parameters():
				if "classifier" or "head" in name:
					continue
				param.requires_grad = False
		self.model = self.model.to(device)
		print(self.model)
	
	def forward(self, x):
		self.model.eval()
		with torch.no_grad():
			pred = self.model(x)
		return pred

	def training_step(self, batch, batch_nb):
		x, y = batch
		logits = self.model(x)
		loss = F.cross_entropy(logits, y)
		pred = logits.argmax(dim=1)
		acc = accuracy(pred, y)
		# Use the current of PyTorch logger
		self.log("train_loss", loss, on_epoch=True)
		self.log("train_acc", acc, on_epoch=True)
		return loss
	
	def validation_step(self, batch, batch_nb):
		x, y = batch
		logits = self.model(x)
		loss = F.cross_entropy(logits, y)
		pred = logits.argmax(dim=1)
		acc = accuracy(pred, y)
		# precision, recall = precision_recall(pred, y, average='weighted', num_classes=self.params["cls"])
		precision, recall, thresholds = precision_recall_curve(pred, y, pos_label=1)
		f1 = f1_score(pred, y, average='weighted', num_classes=self.params["cls"])
		# ix = np.argmax(f1.cpu().detach().numpy())
		# best_threshold = thresholds[ix]

		self.log("val_loss", loss, on_epoch=True)
		self.log("val_acc", acc, on_epoch=True)
		self.log("val_precision", precision, on_epoch=True)
		self.log("val_recall", recall, on_epoch=True)
		self.log("val_f1", f1, on_epoch=True)
		# self.log("val_best_threshold", best_threshold, on_epoch=True)

		return loss

	def test_step(self, batch, batch_nb):
		loss, acc = self._shared_eval_step(batch, batch_nb)
		self.log("test_acc", acc, on_epoch=True)
		return acc
	
	def _shared_eval_step(self, batch, batch_nb):
		x, y = batch
		logits = self.model(x)
		loss = F.cross_entropy(logits, y)
		pred = logits.argmax(dim=1)
		acc = accuracy(pred, y)
		return loss, acc

	def configure_optimizers(self):
		# optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params["learning_rate"],
										# weight_decay=self.params["weight_decay"], amsgrad=True)
		# optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params["learning_rate"], momentum=0.9)
		optimizer = AdamP(self.model.parameters(), lr=self.params["learning_rate"], betas=(0.9, 0.999),
							weight_decay=self.params["weight_decay"], nesterov=True)

		scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=self.params["gamma"], verbose=False)
		motinor = {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "metric_to_track"}

		# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
		# motinor = {"optimizer": optimizer, "lr_scheduler": scheduler, "interval": "epoch", "monitor": 'val_recall'}

		return motinor