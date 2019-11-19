#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import random
import shutil
import sys
import tempfile

try:
	from sepconv import sepconv # the custom separable convolution layer
except:
	sys.path.insert(0, './sepconv'); import sepconv # you should consider upgrading python
# end

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:3]).split('+')[0]) >= 41) # requires at least pytorch version 0.4.1

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'lf'
arguments_strPadding = 'improved'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strVideo = './videos/car-turn.mp4'
arguments_strOut = './out.png'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use, l1 or lf, please see our paper for more details
	if strOption == '--padding' and strArgument != '': arguments_strPadding = strArgument # which padding to use, the one used in the paper or the improved one
	if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument # path to the first frame
	if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument # path to the second frame
	if strOption == '--video' and strArgument != '': arguments_strVideo = strArgument # path to a video
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

		def Upsample(intInput, intOutput):
			return torch.nn.Sequential(
				torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
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
		self.moduleConv2 = Basic(32, 64)
		self.moduleConv3 = Basic(64, 128)
		self.moduleConv4 = Basic(128, 256)
		self.moduleConv5 = Basic(256, 512)

		self.moduleDeconv5 = Basic(512, 512)
		self.moduleDeconv4 = Basic(512, 256)
		self.moduleDeconv3 = Basic(256, 128)
		self.moduleDeconv2 = Basic(128, 64)

		self.moduleUpsample5 = Upsample(512, 512)
		self.moduleUpsample4 = Upsample(256, 256)
		self.moduleUpsample3 = Upsample(128, 128)
		self.moduleUpsample2 = Upsample(64, 64)

		self.moduleVertical1 = Subnet()
		self.moduleVertical2 = Subnet()
		self.moduleHorizontal1 = Subnet()
		self.moduleHorizontal2 = Subnet()

		self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))
	# end

	def forward(self, tensorFirst, tensorSecond):
		tensorConv1 = self.moduleConv1(torch.cat([ tensorFirst, tensorSecond ], 1))
		tensorConv2 = self.moduleConv2(torch.nn.functional.avg_pool2d(input=tensorConv1, kernel_size=2, stride=2, count_include_pad=False))
		tensorConv3 = self.moduleConv3(torch.nn.functional.avg_pool2d(input=tensorConv2, kernel_size=2, stride=2, count_include_pad=False))
		tensorConv4 = self.moduleConv4(torch.nn.functional.avg_pool2d(input=tensorConv3, kernel_size=2, stride=2, count_include_pad=False))
		tensorConv5 = self.moduleConv5(torch.nn.functional.avg_pool2d(input=tensorConv4, kernel_size=2, stride=2, count_include_pad=False))

		tensorDeconv5 = self.moduleUpsample5(self.moduleDeconv5(torch.nn.functional.avg_pool2d(input=tensorConv5, kernel_size=2, stride=2, count_include_pad=False)))
		tensorDeconv4 = self.moduleUpsample4(self.moduleDeconv4(tensorDeconv5 + tensorConv5))
		tensorDeconv3 = self.moduleUpsample3(self.moduleDeconv3(tensorDeconv4 + tensorConv4))
		tensorDeconv2 = self.moduleUpsample2(self.moduleDeconv2(tensorDeconv3 + tensorConv3))

		tensorCombine = tensorDeconv2 + tensorConv2

		tensorFirst = torch.nn.functional.pad(input=tensorFirst, pad=[ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ], mode='replicate')
		tensorSecond = torch.nn.functional.pad(input=tensorSecond, pad=[ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ], mode='replicate')

		tensorDot1 = sepconv.FunctionSepconv(tensorInput=tensorFirst, tensorVertical=self.moduleVertical1(tensorCombine), tensorHorizontal=self.moduleHorizontal1(tensorCombine))
		tensorDot2 = sepconv.FunctionSepconv(tensorInput=tensorSecond, tensorVertical=self.moduleVertical2(tensorCombine), tensorHorizontal=self.moduleHorizontal2(tensorCombine))

		return tensorDot1 + tensorDot2
	# end
# end

moduleNetwork = Network().cuda().eval()

##########################################################

def estimate(tensorFirst, tensorSecond):
	assert(tensorFirst.size(1) == tensorSecond.size(1))
	assert(tensorFirst.size(2) == tensorSecond.size(2))

	intWidth = tensorFirst.size(2)
	intHeight = tensorFirst.size(1)

	assert(intWidth <= 1280) # while our approach works with larger images, we do not recommend it unless you are aware of the implications
	assert(intHeight <= 720) # while our approach works with larger images, we do not recommend it unless you are aware of the implications

	tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
	tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)

	if arguments_strPadding == 'paper':
		intPaddingLeft, intPaddingTop, intPaddingBottom, intPaddingRight = int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ,int(math.floor(51 / 2.0))

	elif arguments_strPadding == 'improved':
		intPaddingLeft, intPaddingTop, intPaddingBottom, intPaddingRight = 0, 0, 0, 0

	# end

	intPreprocessedWidth = intPaddingLeft + intWidth + intPaddingRight
	intPreprocessedHeight = intPaddingTop + intHeight + intPaddingBottom

	if intPreprocessedWidth != ((intPreprocessedWidth >> 7) << 7):
		intPreprocessedWidth = (((intPreprocessedWidth >> 7) + 1) << 7) # more than necessary
	# end
	
	if intPreprocessedHeight != ((intPreprocessedHeight >> 7) << 7):
		intPreprocessedHeight = (((intPreprocessedHeight >> 7) + 1) << 7) # more than necessary
	# end

	intPaddingRight = intPreprocessedWidth - intWidth - intPaddingLeft
	intPaddingBottom = intPreprocessedHeight - intHeight - intPaddingTop

	tensorPreprocessedFirst = torch.nn.functional.pad(input=tensorPreprocessedFirst, pad=[ intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom ], mode='replicate')
	tensorPreprocessedSecond = torch.nn.functional.pad(input=tensorPreprocessedSecond, pad=[ intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom ], mode='replicate')

	return torch.nn.functional.pad(input=moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond), pad=[ 0 - intPaddingLeft, 0 - intPaddingRight, 0 - intPaddingTop, 0 - intPaddingBottom ], mode='replicate')[0, :, :, :].cpu()
# end

##########################################################

if __name__ == '__main__':
	if arguments_strOut.split('.')[-1] in [ 'bmp', 'jpg', 'jpeg', 'png' ]:
		tensorFirst = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
		tensorSecond = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

		tensorOutput = estimate(tensorFirst, tensorSecond)

		PIL.Image.fromarray((tensorOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(arguments_strOut)

	elif arguments_strOut.split('.')[-1] in [ 'avi', 'mp4', 'webm', 'wmv' ]:
		import moviepy
		import moviepy.editor

		strTempdir = tempfile.gettempdir() + '/' + str.join('', [ random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for intCount in range(20) ]); os.makedirs(strTempdir + '/')

		intFrames = 0
		tensorFrames = [ None, None, None, None, None ]

		for intFrame, numpyFrame in enumerate(numpyFrame[:, :, ::-1] for numpyFrame in moviepy.editor.VideoFileClip(filename=arguments_strVideo).iter_frames()):
			tensorFrames[4] = torch.FloatTensor(numpyFrame.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

			if tensorFrames[0] is not None:
				tensorFrames[2] = estimate(tensorFrames[0], tensorFrames[4])
				tensorFrames[1] = estimate(tensorFrames[0], tensorFrames[2])
				tensorFrames[3] = estimate(tensorFrames[2], tensorFrames[4])

				PIL.Image.fromarray((tensorFrames[0].clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(strTempdir + '/' + str(intFrames).zfill(5) + '.png'); intFrames += 1
				PIL.Image.fromarray((tensorFrames[1].clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(strTempdir + '/' + str(intFrames).zfill(5) + '.png'); intFrames += 1
				PIL.Image.fromarray((tensorFrames[2].clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(strTempdir + '/' + str(intFrames).zfill(5) + '.png'); intFrames += 1
				PIL.Image.fromarray((tensorFrames[3].clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(strTempdir + '/' + str(intFrames).zfill(5) + '.png'); intFrames += 1
			# end

			tensorFrames[0] = torch.FloatTensor(numpyFrame.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
		# end

		moviepy.editor.ImageSequenceClip(sequence=strTempdir + '/', fps=25).write_videofile(arguments_strOut)

		shutil.rmtree(strTempdir + '/')

	# end
# end