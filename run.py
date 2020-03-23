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
	from .sepconv import sepconv # the custom separable convolution layer
except:
	sys.path.insert(0, './sepconv'); import sepconv # you should consider upgrading python
# end

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

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

		self.netConv1 = Basic(6, 32)
		self.netConv2 = Basic(32, 64)
		self.netConv3 = Basic(64, 128)
		self.netConv4 = Basic(128, 256)
		self.netConv5 = Basic(256, 512)

		self.netDeconv5 = Basic(512, 512)
		self.netDeconv4 = Basic(512, 256)
		self.netDeconv3 = Basic(256, 128)
		self.netDeconv2 = Basic(128, 64)

		self.netUpsample5 = Upsample(512, 512)
		self.netUpsample4 = Upsample(256, 256)
		self.netUpsample3 = Upsample(128, 128)
		self.netUpsample2 = Upsample(64, 64)

		self.netVertical1 = Subnet()
		self.netVertical2 = Subnet()
		self.netHorizontal1 = Subnet()
		self.netHorizontal2 = Subnet()

		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load(__file__.replace('run.py', 'network-' + arguments_strModel + '.pytorch')).items() })
	# end

	def forward(self, tenFirst, tenSecond):
		tenConv1 = self.netConv1(torch.cat([ tenFirst, tenSecond ], 1))
		tenConv2 = self.netConv2(torch.nn.functional.avg_pool2d(input=tenConv1, kernel_size=2, stride=2, count_include_pad=False))
		tenConv3 = self.netConv3(torch.nn.functional.avg_pool2d(input=tenConv2, kernel_size=2, stride=2, count_include_pad=False))
		tenConv4 = self.netConv4(torch.nn.functional.avg_pool2d(input=tenConv3, kernel_size=2, stride=2, count_include_pad=False))
		tenConv5 = self.netConv5(torch.nn.functional.avg_pool2d(input=tenConv4, kernel_size=2, stride=2, count_include_pad=False))

		tenDeconv5 = self.netUpsample5(self.netDeconv5(torch.nn.functional.avg_pool2d(input=tenConv5, kernel_size=2, stride=2, count_include_pad=False)))
		tenDeconv4 = self.netUpsample4(self.netDeconv4(tenDeconv5 + tenConv5))
		tenDeconv3 = self.netUpsample3(self.netDeconv3(tenDeconv4 + tenConv4))
		tenDeconv2 = self.netUpsample2(self.netDeconv2(tenDeconv3 + tenConv3))

		tenCombine = tenDeconv2 + tenConv2

		tenFirst = torch.nn.functional.pad(input=tenFirst, pad=[ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ], mode='replicate')
		tenSecond = torch.nn.functional.pad(input=tenSecond, pad=[ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ], mode='replicate')

		tenDot1 = sepconv.FunctionSepconv(tenInput=tenFirst, tenVertical=self.netVertical1(tenCombine), tenHorizontal=self.netHorizontal1(tenCombine))
		tenDot2 = sepconv.FunctionSepconv(tenInput=tenSecond, tenVertical=self.netVertical2(tenCombine), tenHorizontal=self.netHorizontal2(tenCombine))

		return tenDot1 + tenDot2
	# end
# end

netNetwork = None

##########################################################

def estimate(tenFirst, tenSecond):
	global netNetwork

	if netNetwork is None:
		netNetwork = Network().cuda().eval()
	# end

	assert(tenFirst.shape[1] == tenSecond.shape[1])
	assert(tenFirst.shape[2] == tenSecond.shape[2])

	intWidth = tenFirst.shape[2]
	intHeight = tenFirst.shape[1]

	assert(intWidth <= 1280) # while our approach works with larger images, we do not recommend it unless you are aware of the implications
	assert(intHeight <= 720) # while our approach works with larger images, we do not recommend it unless you are aware of the implications

	tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
	tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

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

	tenPreprocessedFirst = torch.nn.functional.pad(input=tenPreprocessedFirst, pad=[ intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom ], mode='replicate')
	tenPreprocessedSecond = torch.nn.functional.pad(input=tenPreprocessedSecond, pad=[ intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom ], mode='replicate')

	return torch.nn.functional.pad(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), pad=[ 0 - intPaddingLeft, 0 - intPaddingRight, 0 - intPaddingTop, 0 - intPaddingBottom ], mode='replicate')[0, :, :, :].cpu()
# end

##########################################################

if __name__ == '__main__':
	if arguments_strOut.split('.')[-1] in [ 'bmp', 'jpg', 'jpeg', 'png' ]:
		tenFirst = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
		tenSecond = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

		tenOutput = estimate(tenFirst, tenSecond)

		PIL.Image.fromarray((tenOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(arguments_strOut)

	elif arguments_strOut.split('.')[-1] in [ 'avi', 'mp4', 'webm', 'wmv' ]:
		import moviepy
		import moviepy.editor

		strTempdir = tempfile.gettempdir() + '/' + str.join('', [ random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for intCount in range(20) ]); os.makedirs(strTempdir + '/')

		intFrames = 0
		tenFrames = [ None, None, None, None, None ]

		for intFrame, npyFrame in enumerate(npyFrame[:, :, ::-1] for npyFrame in moviepy.editor.VideoFileClip(filename=arguments_strVideo).iter_frames()):
			tenFrames[4] = torch.FloatTensor(npyFrame.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

			if tenFrames[0] is not None:
				tenFrames[2] = estimate(tenFrames[0], tenFrames[4])
				tenFrames[1] = estimate(tenFrames[0], tenFrames[2])
				tenFrames[3] = estimate(tenFrames[2], tenFrames[4])

				PIL.Image.fromarray((tenFrames[0].clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(strTempdir + '/' + str(intFrames).zfill(5) + '.png'); intFrames += 1
				PIL.Image.fromarray((tenFrames[1].clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(strTempdir + '/' + str(intFrames).zfill(5) + '.png'); intFrames += 1
				PIL.Image.fromarray((tenFrames[2].clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(strTempdir + '/' + str(intFrames).zfill(5) + '.png'); intFrames += 1
				PIL.Image.fromarray((tenFrames[3].clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(strTempdir + '/' + str(intFrames).zfill(5) + '.png'); intFrames += 1
			# end

			tenFrames[0] = torch.FloatTensor(npyFrame.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
		# end

		moviepy.editor.ImageSequenceClip(sequence=strTempdir + '/', fps=25).write_videofile(arguments_strOut)

		shutil.rmtree(strTempdir + '/')

	# end
# end