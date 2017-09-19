import torch

import _ext.cunnex

class SeparableConvolution(torch.autograd.Function):
	def __init__(self):
		super(SeparableConvolution, self).__init__()
	# end

	def forward(self, input, vertical, horizontal):
		intBatches = input.size(0)
		intInputDepth = input.size(1)
		intInputHeight = input.size(2)
		intInputWidth = input.size(3)
		intFilterSize = min(vertical.size(1), horizontal.size(1))
		intOutputHeight = min(vertical.size(2), horizontal.size(2))
		intOutputWidth = min(vertical.size(3), horizontal.size(3))

		assert(intInputHeight - 51 == intOutputHeight - 1)
		assert(intInputWidth - 51 == intOutputWidth - 1)
		assert(intFilterSize == 51)

		assert(input.is_contiguous() == True)
		assert(vertical.is_contiguous() == True)
		assert(horizontal.is_contiguous() == True)

		output = input.new().resize_(intBatches, intInputDepth, intOutputHeight, intOutputWidth).zero_()

		if input.is_cuda == True:
			_ext.cunnex.SeparableConvolution_cuda_forward(
				input,
				vertical,
				horizontal,
				output
			)

		elif input.is_cuda == False:
			raise NotImplementedError() # CPU VERSION NOT IMPLEMENTED

		# end

		return output
	# end

	def backward(self, gradOutput):
		raise NotImplementedError() # BACKPROPAGATION NOT IMPLEMENTED
	# end
# end
