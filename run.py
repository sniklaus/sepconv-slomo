#!/usr/bin/env python

import torch
import torch.utils.serialization

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys

try:
	from sepconv import sepconv # the custom separable convolution layer
except:
	sys.path.insert(0, './sepconv'); import sepconv # you should consider upgrading python
# end

##########################################################

assert(int(torch.__version__.replace('.', '')) >= 40) # requires at least pytorch version 0.4.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.cuda.device(1) # change this if you have a multiple graphics cards and you want to utilize them

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'lf'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './out.png'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use, l1 or lf, please see our paper for more details
	if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument # path to the first frame
	if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument # path to the second frame
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		def Basic(intInput, intOutput):
			return torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False)
			)
		# end

		def Subnet():
			return torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=64, out_channels=51, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False),
				torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
				torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
			)
		# end

		self.moduleConv1 = Basic(6, 32)
		self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

		self.moduleConv2 = Basic(32, 64)
		self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

		self.moduleConv3 = Basic(64, 128)
		self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

		self.moduleConv4 = Basic(128, 256)
		self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

		self.moduleConv5 = Basic(256, 512)
		self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

		self.moduleDeconv5 = Basic(512, 512)
		self.moduleUpsample5 = torch.nn.Sequential(
			torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleDeconv4 = Basic(512, 256)
		self.moduleUpsample4 = torch.nn.Sequential(
			torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleDeconv3 = Basic(256, 128)
		self.moduleUpsample3 = torch.nn.Sequential(
			torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleDeconv2 = Basic(128, 64)
		self.moduleUpsample2 = torch.nn.Sequential(
			torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleVertical1 = Subnet()
		self.moduleVertical2 = Subnet()
		self.moduleHorizontal1 = Subnet()
		self.moduleHorizontal2 = Subnet()

		self.modulePad = torch.nn.ReplicationPad2d([ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ])

		self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))
	# end

	def forward(self, tensorFirst, tensorSecond):
		tensorJoin = torch.cat([ tensorFirst, tensorSecond ], 1)

		tensorConv1 = self.moduleConv1(tensorJoin)
		tensorPool1 = self.modulePool1(tensorConv1)

		tensorConv2 = self.moduleConv2(tensorPool1)
		tensorPool2 = self.modulePool2(tensorConv2)

		tensorConv3 = self.moduleConv3(tensorPool2)
		tensorPool3 = self.modulePool3(tensorConv3)

		tensorConv4 = self.moduleConv4(tensorPool3)
		tensorPool4 = self.modulePool4(tensorConv4)

		tensorConv5 = self.moduleConv5(tensorPool4)
		tensorPool5 = self.modulePool5(tensorConv5)

		tensorDeconv5 = self.moduleDeconv5(tensorPool5)
		tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

		tensorCombine = tensorUpsample5 + tensorConv5

		tensorDeconv4 = self.moduleDeconv4(tensorCombine)
		tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

		tensorCombine = tensorUpsample4 + tensorConv4

		tensorDeconv3 = self.moduleDeconv3(tensorCombine)
		tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

		tensorCombine = tensorUpsample3 + tensorConv3

		tensorDeconv2 = self.moduleDeconv2(tensorCombine)
		tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

		tensorCombine = tensorUpsample2 + tensorConv2

		tensorDot1 = sepconv.FunctionSepconv()(self.modulePad(tensorFirst), self.moduleVertical1(tensorCombine), self.moduleHorizontal1(tensorCombine))
		tensorDot2 = sepconv.FunctionSepconv()(self.modulePad(tensorSecond), self.moduleVertical2(tensorCombine), self.moduleHorizontal2(tensorCombine))

		return tensorDot1 + tensorDot2
	# end
# end

moduleNetwork = Network().cuda().eval()

##########################################################

def estimate(tensorFirst, tensorSecond):
	tensorOutput = torch.FloatTensor()

	assert(tensorFirst.size(1) == tensorSecond.size(1))
	assert(tensorFirst.size(2) == tensorSecond.size(2))

	intWidth = tensorFirst.size(2)
	intHeight = tensorFirst.size(1)

	assert(intWidth <= 1280) # while our approach works with larger images, we do not recommend it unless you are aware of the implications
	assert(intHeight <= 720) # while our approach works with larger images, we do not recommend it unless you are aware of the implications

	intPaddingLeft = int(math.floor(51 / 2.0))
	intPaddingTop = int(math.floor(51 / 2.0))
	intPaddingRight = int(math.floor(51 / 2.0))
	intPaddingBottom = int(math.floor(51 / 2.0))
	modulePaddingInput = torch.nn.Sequential()
	modulePaddingOutput = torch.nn.Sequential()

	if True:
		intPaddingWidth = intPaddingLeft + intWidth + intPaddingRight
		intPaddingHeight = intPaddingTop + intHeight + intPaddingBottom

		if intPaddingWidth != ((intPaddingWidth >> 7) << 7):
			intPaddingWidth = (((intPaddingWidth >> 7) + 1) << 7) # more than necessary
		# end
		
		if intPaddingHeight != ((intPaddingHeight >> 7) << 7):
			intPaddingHeight = (((intPaddingHeight >> 7) + 1) << 7) # more than necessary
		# end

		intPaddingWidth = intPaddingWidth - (intPaddingLeft + intWidth + intPaddingRight)
		intPaddingHeight = intPaddingHeight - (intPaddingTop + intHeight + intPaddingBottom)

		modulePaddingInput = torch.nn.ReplicationPad2d(padding=[ intPaddingLeft, intPaddingRight + intPaddingWidth, intPaddingTop, intPaddingBottom + intPaddingHeight ])
		modulePaddingOutput = torch.nn.ReplicationPad2d(padding=[ 0 - intPaddingLeft, 0 - intPaddingRight - intPaddingWidth, 0 - intPaddingTop, 0 - intPaddingBottom - intPaddingHeight ])
	# end

	if True:
		tensorFirst = tensorFirst.cuda()
		tensorSecond = tensorSecond.cuda()
		tensorOutput = tensorOutput.cuda()

		modulePaddingInput = modulePaddingInput.cuda()
		modulePaddingOutput = modulePaddingOutput.cuda()
	# end

	if True:
		tensorPreprocessedFirst = modulePaddingInput(tensorFirst.view(1, 3, intHeight, intWidth))
		tensorPreprocessedSecond = modulePaddingInput(tensorSecond.view(1, 3, intHeight, intWidth))

		tensorOutput.resize_(3, intHeight, intWidth).copy_(modulePaddingOutput(moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond))[0, :, :, :])
	# end

	if True:
		tensorFirst = tensorFirst.cpu()
		tensorSecond = tensorSecond.cpu()
		tensorOutput = tensorOutput.cpu()
	# end

	return tensorOutput
# end

##########################################################

if __name__ == '__main__':
	tensorFirst = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
	tensorSecond = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

	tensorOutput = estimate(tensorFirst, tensorSecond)

	PIL.Image.fromarray((tensorOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(arguments_strOut)
# end