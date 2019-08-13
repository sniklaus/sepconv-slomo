#!/usr/bin/env python

import torch

import glob
import numpy
import PIL
import PIL.Image
import skimage
import skimage.measure

import run

##########################################################

run.moduleNetwork.load_state_dict(torch.load('./network-l1.pytorch')) # manually loading the l1 model since it is the one that should be used for quantiative evaluations

##########################################################

if __name__ == '__main__':
	dblPsnr = []
	dblSsim = []

	for strTruth in sorted(glob.glob('./middlebury/*/frame10i11.png')):
		tensorFirst = torch.FloatTensor(numpy.array(PIL.Image.open(strTruth.replace('frame10i11', 'frame10')))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
		tensorSecond = torch.FloatTensor(numpy.array(PIL.Image.open(strTruth.replace('frame10i11', 'frame11')))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
		
		numpyEstimate = (run.estimate(tensorFirst, tensorSecond).clamp(0.0, 1.0).numpy().transpose(1, 2, 0) * 255.0).astype(numpy.uint8)

		dblPsnr.append(skimage.measure.compare_psnr(im_true=numpy.array(PIL.Image.open(strTruth))[:, :, ::-1], im_test=numpyEstimate, data_range=255))
		dblSsim.append(skimage.measure.compare_ssim(X=numpy.array(PIL.Image.open(strTruth))[:, :, ::-1], Y=numpyEstimate, data_range=255, multichannel=True))
	# end

	print('computed average psnr', numpy.mean(dblPsnr))
	print('computed average ssim', numpy.mean(dblSsim))
	print('')
	print('see table below for reference results')
	print('')
	print('+---------+------------+---------+---------+')
	print('| model   | padding    | psnr    | ssim    |')
	print('+---------+------------+---------+---------+')
	print('| l1      | paper      | 35.73   | 0.959   |')
	print('| lf      | paper      | 35.03   | 0.954   |')
	print('| l1      | improved   | 35.85   | 0.959   |')
	print('| lf      | improved   | 35.16   | 0.954   |')
	print('+---------+------------+---------+---------+')
# end