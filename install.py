import os
import torch
import torch.utils.ffi

strBasepath = os.path.split(os.path.abspath(__file__))[0] + '/'
strHeaders = []
strSources = []
strDefines = []
strObjects = []

if torch.cuda.is_available() == True:
	strHeaders += ['src/SeparableConvolution_cuda.h']
	strSources += ['src/SeparableConvolution_cuda.c']
	strDefines += [('WITH_CUDA', None)]
	strObjects += ['src/SeparableConvolution_kernel.o']
# end

objectExtension = torch.utils.ffi.create_extension(
	name='_ext.cunnex',
	headers=strHeaders,
	sources=strSources,
	verbose=False,
	with_cuda=any(strDefine[0] == 'WITH_CUDA' for strDefine in strDefines),
	package=False,
	relative_to=strBasepath,
	include_dirs=[os.path.expandvars('$CUDA_HOME') + '/include'],
	define_macros=strDefines,
	extra_objects=[os.path.join(strBasepath, strObject) for strObject in strObjects]
)

if __name__ == '__main__':
	objectExtension.build()
# end