#!/usr/bin/env python

import getopt
import math
import numpy
import PIL
import PIL.Image
import sys
import torch

import sepconv # the custom separable convolution layer

##########################################################

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

args_strModel = 'lf'
args_strPadding = 'improved'
args_strOne = './images/one.png'
args_strTwo = './images/two.png'
args_strVideo = './videos/car-turn.mp4'
args_strOut = './out.png'

for strOption, strArg in getopt.getopt(sys.argv[1:], '', [
    'model=',
    'padding=',
    'one=',
    'two=',
    'vidoe=',
    'out=',
])[0]:
    if strOption == '--model' and strArg != '': args_strModel = strArg # which model to use, l1 or lf, please see our paper for more details
    if strOption == '--padding' and strArg != '': args_strPadding = strArg # which padding to use, the one used in the paper or the improved one
    if strOption == '--one' and strArg != '': args_strOne = strArg # path to the first frame
    if strOption == '--two' and strArg != '': args_strTwo = strArg # path to the second frame
    if strOption == '--video' and strArg != '': args_strVideo = strArg # path to a video
    if strOption == '--out' and strArg != '': args_strOut = strArg # path to where the output should be stored
# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

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

        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/sepconv/network-' + args_strModel + '.pytorch', file_name='sepconv-' + args_strModel).items() })
    # end

    def forward(self, tenOne, tenTwo):
        tenConv1 = self.netConv1(torch.cat([tenOne, tenTwo], 1))
        tenConv2 = self.netConv2(torch.nn.functional.avg_pool2d(input=tenConv1, kernel_size=2, stride=2, count_include_pad=False))
        tenConv3 = self.netConv3(torch.nn.functional.avg_pool2d(input=tenConv2, kernel_size=2, stride=2, count_include_pad=False))
        tenConv4 = self.netConv4(torch.nn.functional.avg_pool2d(input=tenConv3, kernel_size=2, stride=2, count_include_pad=False))
        tenConv5 = self.netConv5(torch.nn.functional.avg_pool2d(input=tenConv4, kernel_size=2, stride=2, count_include_pad=False))

        tenDeconv5 = self.netUpsample5(self.netDeconv5(torch.nn.functional.avg_pool2d(input=tenConv5, kernel_size=2, stride=2, count_include_pad=False)))
        tenDeconv4 = self.netUpsample4(self.netDeconv4(tenDeconv5 + tenConv5))
        tenDeconv3 = self.netUpsample3(self.netDeconv3(tenDeconv4 + tenConv4))
        tenDeconv2 = self.netUpsample2(self.netDeconv2(tenDeconv3 + tenConv3))

        tenCombine = tenDeconv2 + tenConv2

        tenVerone = self.netVertical1(tenCombine)
        tenVertwo = self.netVertical2(tenCombine)
        tenHorone = self.netHorizontal1(tenCombine)
        tenHortwo = self.netHorizontal2(tenCombine)

        return sum([
            sepconv.sepconv_func.apply(torch.nn.functional.pad(input=tenOne, pad=[int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0))], mode='replicate'), tenVerone, tenHorone),
            sepconv.sepconv_func.apply(torch.nn.functional.pad(input=tenTwo, pad=[int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0))], mode='replicate'), tenVertwo, tenHortwo)
        ])
    # end
# end

netNetwork = None

##########################################################

def estimate(tenOne, tenTwo):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().train(False)
    # end

    assert(tenOne.shape[1] == tenTwo.shape[1])
    assert(tenOne.shape[2] == tenTwo.shape[2])

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    assert(intWidth <= 1280) # while our approach works with larger images, we do not recommend it unless you are aware of the implications
    assert(intHeight <= 720) # while our approach works with larger images, we do not recommend it unless you are aware of the implications

    tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    if args_strPadding == 'paper':
        intPaddingLeft, intPaddingTop, intPaddingBottom, intPaddingRight = int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ,int(math.floor(51 / 2.0))

    elif args_strPadding == 'improved':
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

    tenPreprocessedOne = torch.nn.functional.pad(input=tenPreprocessedOne, pad=[intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom], mode='replicate')
    tenPreprocessedTwo = torch.nn.functional.pad(input=tenPreprocessedTwo, pad=[intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom], mode='replicate')

    return torch.nn.functional.pad(input=netNetwork(tenPreprocessedOne, tenPreprocessedTwo), pad=[0 - intPaddingLeft, 0 - intPaddingRight, 0 - intPaddingTop, 0 - intPaddingBottom], mode='replicate')[0, :, :, :].cpu()
# end

##########################################################

if __name__ == '__main__':
    if args_strOut.split('.')[-1] in ['bmp', 'jpg', 'jpeg', 'png']:
        tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(args_strOne))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
        tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(args_strTwo))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

        tenOutput = estimate(tenOne, tenTwo)

        PIL.Image.fromarray((tenOutput.clip(0.0, 1.0).numpy(force=True).transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(args_strOut)

    elif args_strOut.split('.')[-1] in ['avi', 'mp4', 'webm', 'wmv']:
        import moviepy

        objVideoreader = moviepy.VideoFileClip(filename=args_strVideo)

        intWidth = objVideoreader.w
        intHeight = objVideoreader.h

        tenFrames = [None, None, None, None, None]

        with moviepy.video.io.ffmpeg_writer.FFMPEG_VideoWriter(filename=args_strOut, size=(intWidth, intHeight), fps=objVideoreader.fps) as objVideowriter:
            for npyFrame in objVideoreader.iter_frames():
                tenFrames[4] = torch.FloatTensor(numpy.ascontiguousarray(npyFrame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

                if tenFrames[0] is not None:
                    tenFrames[2] = estimate(tenFrames[0], tenFrames[4])
                    tenFrames[1] = estimate(tenFrames[0], tenFrames[2])
                    tenFrames[3] = estimate(tenFrames[2], tenFrames[4])

                    objVideowriter.write_frame((tenFrames[0].clip(0.0, 1.0).numpy(force=True).transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8))
                    objVideowriter.write_frame((tenFrames[1].clip(0.0, 1.0).numpy(force=True).transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8))
                    objVideowriter.write_frame((tenFrames[2].clip(0.0, 1.0).numpy(force=True).transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8))
                    objVideowriter.write_frame((tenFrames[3].clip(0.0, 1.0).numpy(force=True).transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8))
                # end

                tenFrames[0] = torch.FloatTensor(numpy.ascontiguousarray(npyFrame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
            # end
        # end

    # end
# end