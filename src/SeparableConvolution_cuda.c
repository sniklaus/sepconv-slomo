#include <THC.h>
#include <THCGeneral.h>

#include "SeparableConvolution_kernel.h"

extern THCState* state;

int SeparableConvolution_cuda_forward(
	THCudaTensor* input1,
	THCudaTensor* input2,
	THCudaTensor* input3,
	THCudaTensor* output
) {
	SeparableConvolution_kernel_forward(
		state,
		input1,
		input2,
		input3,
		output
	);

	return 1;
}