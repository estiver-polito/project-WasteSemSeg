from .base_modules import SeparableConvBnRelu,ConvBNReLU
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
	def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
		super(Block, self).__init__()

		if out_filters != in_filters or strides != 1:
			self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
			self.skipbn = nn.BatchNorm2d(out_filters)
		else:
			self.skip = None

		self.relu = nn.ReLU(inplace=True)
		rep = []

		filters = in_filters
		if grow_first:
			rep.append(self.relu)
			rep.append(SeparableConvBnRelu(in_filters, out_filters, 3, stride=1, padding=1))
			rep.append(nn.BatchNorm2d(out_filters))
			filters = out_filters

		for i in range(reps - 1):
			rep.append(self.relu)
			rep.append(SeparableConvBnRelu(filters, filters, 3, stride=1, padding=1))
			rep.append(nn.BatchNorm2d(filters))

		if not grow_first:
			rep.append(self.relu)
			rep.append(SeparableConvBnRelu(in_filters, out_filters, 3, stride=1, padding=1))
			rep.append(nn.BatchNorm2d(out_filters))

		if not start_with_relu:
			rep = rep[1:]
		else:
			rep[0] = nn.ReLU(inplace=False)

		if strides != 1:
			rep.append(nn.MaxPool2d(3, strides, 1))
		self.rep = nn.Sequential(*rep)

	def forward(self, inp):
		x = self.rep(inp)

		if self.skip is not None:
			skip = self.skip(inp)
			skip = self.skipbn(skip)
		else:
			skip = inp

		x += skip
		return x


class Xception(nn.Module):
	"""
	Xception optimized for the ImageNet dataset, as specified in
	https://arxiv.org/pdf/1610.02357.pdf
	"""

	def __init__(self, num_classes=5):
		""" Constructor
		Args:
			num_classes: number of classes
		"""
		super(Xception, self).__init__()
		self.num_classes = num_classes

		self.conv1_xception39 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=0, bias=False)
		self.maxpool_xception39 = nn.MaxPool2d(kernel_size=3, stride=2)

		# P3
		self.block1_xception39 = Block(in_filters=8, out_filters=16, reps=1, strides=2, start_with_relu=True, grow_first=True)
		self.block2_xception39 = Block(in_filters=16, out_filters=16, reps=3, strides=1, start_with_relu=True, grow_first=True)

		# P4
		self.block3_xception39 = Block(in_filters=16, out_filters=32, reps=1, strides=2, start_with_relu=True, grow_first=True)
		self.block4_xception39 = Block(in_filters=32, out_filters=32, reps=7, strides=1, start_with_relu=True, grow_first=True)

		# P5
		self.block5_xception39 = Block(in_filters=32, out_filters=64, reps=1, strides=2, start_with_relu=True, grow_first=True)
		self.block6_xception39 = Block(in_filters=64, out_filters=64, reps=3, strides=1, start_with_relu=True, grow_first=True)

		self.fc_xception39 = nn.Linear(in_features=64, out_features=self.num_classes)



	def forward(self, input):
		
		y = self.conv1_xception39(input)
		y = self.maxpool_xception39(y)
		y = self.block1_xception39(y)
		y = self.block2_xception39(y)
		y = self.block3_xception39(y)
		y = self.block4_xception39(y)
		y = self.block5_xception39(y)
		y = self.block6_xception39(y)
		
		y = F.adaptive_avg_pool2d(y, (1, 1))
		y = y.view(y.size(0), -1)
		
		y = self.fc_xception39(y)
		return y


def xception39(num_classes=5):
	import torch
	model = Xception(num_classes=num_classes)

	return model