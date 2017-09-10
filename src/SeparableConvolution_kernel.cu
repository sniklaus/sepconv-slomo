#include <THC.h>
#include <THCGeneral.h>

#define VEC_0(ARRAY) ((ARRAY).x)
#define VEC_1(ARRAY) ((ARRAY).y)
#define VEC_2(ARRAY) ((ARRAY).z)
#define VEC_3(ARRAY) ((ARRAY).w)

#define IDX_1(ARRAY, X)          ((ARRAY)[((X) * (ARRAY##_stride.x))])
#define IDX_2(ARRAY, X, Y)       ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y))])
#define IDX_3(ARRAY, X, Y, Z)    ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y)) + ((Z) * (ARRAY##_stride.z))])
#define IDX_4(ARRAY, X, Y, Z, W) ((ARRAY)[((X) * (ARRAY##_stride.x)) + ((Y) * (ARRAY##_stride.y)) + ((Z) * (ARRAY##_stride.z)) + ((W) * (ARRAY##_stride.w))])

#ifdef __cplusplus
	extern "C" {
#endif

__global__ void kernel_SeparableConvolution_updateOutput(
	const int n,
	const float* input1, const long4 input1_size, const long4 input1_stride,
	const float* input2, const long4 input2_size, const long4 input2_stride,
	const float* input3, const long4 input3_size, const long4 input3_stride,
	float* output, const long4 output_size, const long4 output_stride
) {
	int intIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (intIndex >= n) {
		return;
	}

	float dblOutput = 0.0;

	int intBatch = ( intIndex / VEC_3(output_size) / VEC_2(output_size) / VEC_1(output_size) ) % VEC_0(output_size);
	int intDepth = ( intIndex / VEC_3(output_size) / VEC_2(output_size)                      ) % VEC_1(output_size);
	int intY     = ( intIndex / VEC_3(output_size)                                           ) % VEC_2(output_size);
	int intX     = ( intIndex                                                                ) % VEC_3(output_size);

	for (int intFilterY = 0; intFilterY < 51; intFilterY += 1) {
		for (int intFilterX = 0; intFilterX < 51; intFilterX += 1) {
			dblOutput += IDX_4(input1, intBatch, intDepth, intY + intFilterY, intX + intFilterX) * IDX_4(input2, intBatch, intFilterY, intY, intX) * IDX_4(input3, intBatch, intFilterX, intY, intX);
		}
	}

	output[intIndex] = dblOutput;
}

void SeparableConvolution_kernel_forward(
	THCState* state,
	THCudaTensor* input1,
	THCudaTensor* input2,
	THCudaTensor* input3,
	THCudaTensor* output
) {
	int n = 0;

	n = THCudaTensor_nElement(state, output);
	kernel_SeparableConvolution_updateOutput<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		THCudaTensor_data(state, input1), make_long4(input1->size[0], input1->size[1], input1->size[2], input1->size[3]), make_long4(input1->stride[0], input1->stride[1], input1->stride[2], input1->stride[3]),
		THCudaTensor_data(state, input2), make_long4(input2->size[0], input2->size[1], input2->size[2], input2->size[3]), make_long4(input2->stride[0], input2->stride[1], input2->stride[2], input2->stride[3]),
		THCudaTensor_data(state, input3), make_long4(input3->size[0], input3->size[1], input3->size[2], input3->size[3]), make_long4(input3->stride[0], input3->stride[1], input3->stride[2], input3->stride[3]),
		THCudaTensor_data(state, output), make_long4(output->size[0], output->size[1], output->size[2], output->size[3]), make_long4(output->stride[0], output->stride[1], output->stride[2], output->stride[3])
	);

	THCudaCheck(cudaGetLastError());
}

#ifdef __cplusplus
	}
#endif