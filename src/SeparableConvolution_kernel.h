#ifdef __cplusplus
	extern "C" {
#endif

void SeparableConvolution_kernel_forward(
	THCState* state,
	THCudaTensor* input1,
	THCudaTensor* input2,
	THCudaTensor* input3,
	THCudaTensor* output
);

#ifdef __cplusplus
	}
#endif