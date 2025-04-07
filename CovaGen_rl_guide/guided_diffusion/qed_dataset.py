
import pickle

import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class QED_Dataset(Dataset):
	def __init__(self, file):
		# self.data = pd.read_csv(csv_file,dtype=np.float32)
		# self.labels = self.data["type"].values
		# self.data = self.data.drop("type", axis=1).values
		with open(file,"rb") as f:
			self.data = pickle.load(f)
		self.ts = []
		self.labels = []
		for key,items in self.data.items():
			self.ts.append(key)
			self.labels.append(items)

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		sample = self.ts[idx]
		label = torch.tensor(self.labels[idx], dtype=torch.long)
		return sample, label

class QED_reg_Dataset(Dataset):
	def __init__(self, file):
		# self.data = pd.read_csv(csv_file,dtype=np.float32)
		# self.labels = self.data["type"].values
		# self.data = self.data.drop("type", axis=1).values
		with open(file,"rb") as f:
			self.data = pickle.load(f)
		self.ts = []
		self.labels = []
		for key,items in self.data.items():
			self.ts.append(key)
			self.labels.append(items)

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		sample = self.ts[idx]
		label = torch.tensor(self.labels[idx], dtype=torch.float)
		return sample, label


def load_data(
	*,
	data_dir,
	batch_size,
	reg=False,
	class_cond=False,
	deterministic=False,
	random_crop=False,
	random_flip=True,
	state='train'
):
	"""
	For a dataset, create a generator over (images, kwargs) pairs.

	Each images is an NCHW float tensor, and the kwargs dict contains zero or
	more keys, each of which map to a batched Tensor of their own.
	The kwargs dict can be used for class labels, in which case the key is "y"
	and the values are integer tensors of class labels.

	:param data_dir: a dataset directory.
	:param batch_size: the batch size of each returned pair.
	:param image_size: the size to which images are resized.
	:param class_cond: if True, include a "y" key in returned dicts for class
					   label. If classes are not available and this is true, an
					   exception will be raised.
	:param deterministic: if True, yield results in a deterministic order.
	:param random_crop: if True, randomly crop the images for augmentation.
	:param random_flip: if True, randomly flip the images for augmentation.
	"""
	if not data_dir:
		raise ValueError("unspecified data directory")
	file_path = data_dir
	classes = [0,1]
	# if class_cond:
	#     # Assume classes are the first part of the filename,
	#     # before an underscore.
	#     class_names = [bf.basename(path).split("_")[0] for path in all_files]
	#     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
	#     classes = [sorted_classes[x] for x in class_names]
	if reg:
		datas = QED_reg_Dataset(
			file=file_path
		)
	else:
		datas = QED_Dataset(
			file=file_path
		)
	train_data, val_data = torch.utils.data.random_split(datas, lengths=[60000, len(datas)-60000])
	if deterministic:
		train_loader = DataLoader(
			train_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
		)
		val_loader = DataLoader(
			val_data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
		)
	else:
		train_loader = DataLoader(
			train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
		)
		val_loader = DataLoader(
			val_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
		)
	if state == 'train':
		while True:
			yield from train_loader
	elif state == 'val':
		while True:
			yield from val_loader

if __name__=="__main__":
	datas = load_data(data_dir='/dataset/guided-diffusion-main/scripts/alldata_dev.pkl',batch_size=2)
	batch, extra = next(datas)
	print("done")