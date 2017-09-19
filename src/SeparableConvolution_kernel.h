#ifdef __cplusplus
	extern "C" {
#endif

void SeparableConvolution_kernel_forward(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* output
);

#ifdef __cplusplus
	}
#endif