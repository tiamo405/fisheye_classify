from utils import *
from models import PMC

def parse_opt():
	parser = argparse.ArgumentParser()
	parser.add_argument('--arch', type=str, default='mnasnet_b1', help='')
	parser.add_argument('--features', type=int, default=1280, help='')
	parser.add_argument('--data_dir', type=str,
						default='/mnt/nvme0n1/phuongnam/fisheye_classify/data', help='')
	parser.add_argument('--epoch', type=int, default=1, help='')
	parser.add_argument('--batch_size', type=int, default=8, help='')
	parser.add_argument('--tag', type=str, default='fisheye', help='')
	parser.add_argument('--artifact', type=str, default='classify', help='')
	parser.add_argument('--exp_name', type=str, default='pmc_mnasnet', help='')
	parser.add_argument('--exp_id', type=int, default=1, help='')

	return parser.parse_args()


opt = parse_opt()


train_transforms = transforms.Compose(
	[
		transforms.Resize([224, 224]),
		transforms.RandomVerticalFlip(p=0.5),
		transforms.RandomHorizontalFlip(p=0.5),
		# transforms.RandomRotation(degrees=(-180, 180)),
		transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)),
		transforms.RandomAdjustSharpness(sharpness_factor=2),
		transforms.RandomAutocontrast(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	]
)

val_transforms = transforms.Compose(
	[
		transforms.Resize([224, 224]),
		transforms.RandomVerticalFlip(p=0.5),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)),
		transforms.RandomAdjustSharpness(sharpness_factor=2),
		transforms.RandomAutocontrast(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	]
)

test_transforms = transforms.Compose(
	[
		transforms.Resize([224, 224]),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	]
)

image_datasets = {}

for phase in ["train", "val", "test"]:
	if phase == "train":
		image_datasets[phase] = datasets.ImageFolder(f"{opt.data_dir}/{phase}", train_transforms)
	elif phase == "val":
		image_datasets[phase] = datasets.ImageFolder(f"{opt.data_dir}/{phase}", val_transforms)
	elif phase == "test":
		image_datasets[phase] = datasets.ImageFolder(f"{opt.data_dir}/{phase}", test_transforms)

num_labels = len(image_datasets["train"].classes)
params = {"num_epochs": opt.epoch, "learning_rate": 0.001,
		  "weight_decay": 1e-4, "gamma": 0.1, "cls": num_labels, "features": opt.features}

# create a weighted random sampler to process imbalanced data
# weights = make_weights_for_balanced_classes(image_datasets["train"].imgs, num_labels)
# weights = torch.DoubleTensor(weights)
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
sampler = ImbalancedDatasetSampler(image_datasets["train"])

dataloaders = {
	i: torch.utils.data.DataLoader(
		image_datasets[i], batch_size=opt.batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True, sampler=sampler if i == "train" else None
	)
	for i in ["train", "val", "test"]
}


def print_auto_logged_info(r):
	tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
	artifacts = [f.path for f in MlflowClient(
	).list_artifacts(r.info.run_id, "model")]
	print("run_id: {}".format(r.info.run_id))
	print("artifacts: {}".format(artifacts))
	print("params: {}".format(r.data.params))
	print("metrics: {}".format(r.data.metrics))
	print("tags: {}".format(tags))


if __name__ == '__main__':
	# torch.multiprocessing.freeze_support()

	model = PMC(arch=opt.arch, params=params, training=True)
	# print(model)

	# Auto log all MLflow entities
	mlflow.pytorch.autolog()

	experiment_id = str(opt.exp_id)

	try:
		experiment_id = mlflow.create_experiment(opt.exp_name,
												 artifact_location=Path.cwd().joinpath("mlruns").as_uri(),
												 tags={"version": "v1", "priority": "P1"})
	except Exception as e:
		print(f"ERROR: {e}")
	experiment = mlflow.get_experiment(experiment_id)
	print("Name: {}".format(experiment.name))
	print("Experiment_id: {}".format(experiment.experiment_id))

	# Train the model
	with mlflow.start_run(run_name=uuid.uuid4(), experiment_id=experiment_id) as run:
		# Initialize a trainer
		log_dir = f"./mlruns/{run.info.run_id}/artifacts/model/data"
		checkpoint_callback = ModelCheckpoint(
			dirpath=log_dir, save_weights_only=False, save_last=True, monitor='val_acc', mode='max')
		early_stopping = EarlyStopping('val_loss')
		model_prune = ModelPruning(
            pruning_fn="l1_unstructured",
            amount=0.01,
            use_global_unstructured=True
        )
		tqdmp_callback = TQDMProgressBar(refresh_rate=10)
		trainer = pl.Trainer(accelerator='gpu', max_epochs=opt.epoch,
							 enable_progress_bar=True, callbacks=[checkpoint_callback, tqdmp_callback])

		# Phase train / val
		trainer.fit(model, dataloaders["train"], dataloaders["val"])

		# Phase test: automatically auto-loads the best weights from the previous run
		trainer.test(dataloaders=dataloaders["test"], ckpt_path="best")

		print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
