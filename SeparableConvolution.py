import torch

import _ext.cunnex

class SeparableConvolution(torch.autograd.Function):
	def __init__(self):
		super(SeparableConvolution, self).__init__()
	# end

	def forward(self, input1, input2, input3):
		intBatches = input1.size(0)
		intInputDepth = input1.size(1)
		intInputHeight = input1.size(2)
		intInputWidth = input1.size(3)
		intFilterSize = min(input2.size(1), input3.size(1))
		intOutputHeight = min(input2.size(2), input3.size(2))
		intOutputWidth = min(input2.size(3), input3.size(3))

		assert(intInputHeight - 51 == intOutputHeight - 1)
		assert(intInputWidth - 51 == intOutputWidth - 1)
		assert(intFilterSize == 51)

		assert(input1.is_contiguous() == True)
		assert(input2.is_contiguous() == True)
		assert(input3.is_contiguous() == True)

		output = input1.new().resize_(intBatches, intInputDepth, intOutputHeight, intOutputWidth).zero_()

		if input1.is_cuda == True:
			_ext.cunnex.SeparableConvolution_cuda_forward(
				input1,
				input2,
				input3,
				output
			)

		elif input1.is_cuda == False:
			raise NotImplementedError()

		# end

		return output
	# end

	def backward(self, gradOutput):
		raise NotImplementedError()
	# end
# end
