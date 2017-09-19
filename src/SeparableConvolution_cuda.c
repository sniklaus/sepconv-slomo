#include <THC.h>
#include <THCGeneral.h>

#include "SeparableConvolution_kernel.h"

extern THCState* state;

int SeparableConvolution_cuda_forward(
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* output
) {
	SeparableConvolution_kernel_forward(
		state,
		input,
		vertical,
		horizontal,
		output
	);

	return 1;
}