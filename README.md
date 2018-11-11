# pytorch-sepconv
**News:** Check our new CVPR 2018 paper on a [faster and higher-quality](http://web.cecs.pdx.edu/~fliu/project/ctxsyn) frame interpolation method.

This is a reference implementation of Video Frame Interpolation via Adaptive Separable Convolution [1] using PyTorch. Given two frames, it will make use of [adaptive convolution](http://graphics.cs.pdx.edu/project/adaconv) [2] in a separable manner to interpolate the intermediate frame. Should you be making use of our work, please cite our paper [1].

<a href="https://arxiv.org/abs/1708.01692" rel="Paper"><img src="http://content.sniklaus.com/sepconv/paper.jpg" alt="Paper" width="100%"></a>

For a reimplemntation of our work, see: https://github.com/martkartasev/sepconv
<br />
For a third-party fork with video support, consider: https://github.com/dagf2101/pytorch-sepconv

## setup
To download the pre-trained models, run `bash download.bash`.

The separable convolution layer is implemented in CUDA using CuPy, which is why CuPy is a required dependency. It can be installed using `pip install cupy` or alternatively using one of the provided binary packages as outlined in the CuPy repository.

## usage
To run it on your own pair of frames, use the following command. You can either select the `l1` or the `lf` model, please see our paper for more details. In short, the `l1` model should be used for quantitative evaluations and the `lf` model for qualitative comparisons.

```
python run.py --model lf --first ./images/first.png --second ./images/second.png --out ./out.png
```

## video
<a href="http://web.cecs.pdx.edu/~fliu/project/sepconv/demo.mp4" rel="Video"><img src="http://web.cecs.pdx.edu/~fliu/project/sepconv/screen.jpg" alt="Video" width="100%"></a>

## license
The provided implementation is strictly for academic purposes only. Should you be interested in using our technology for any commercial use, please feel free to contact us.

## references
```
[1]  @inproceedings{Niklaus_ICCV_2017,
         author = {Simon Niklaus and Long Mai and Feng Liu},
         title = {Video Frame Interpolation via Adaptive Separable Convolution},
         booktitle = {IEEE International Conference on Computer Vision},
         year = {2017}
     }
```

```
[2]  @inproceedings{Niklaus_CVPR_2017,
         author = {Simon Niklaus and Long Mai and Feng Liu},
         title = {Video Frame Interpolation via Adaptive Convolution},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2017}
     }
```

## acknowledgment
This work was supported by NSF IIS-1321119. The video above uses materials under a Creative Common license or with the owner's permission, as detailed at the end.