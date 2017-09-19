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
	const float* input, const long4 input_size, const long4 input_stride,
	const float* vertical, const long4 vertical_size, const long4 vertical_stride,
	const float* horizontal, const long4 horizontal_size, const long4 horizontal_stride,
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
			dblOutput += IDX_4(input, intBatch, intDepth, intY + intFilterY, intX + intFilterX) * IDX_4(vertical, intBatch, intFilterY, intY, intX) * IDX_4(horizontal, intBatch, intFilterX, intY, intX);
		}
	}

	output[intIndex] = dblOutput;
}

void SeparableConvolution_kernel_forward(
	THCState* state,
	THCudaTensor* input,
	THCudaTensor* vertical,
	THCudaTensor* horizontal,
	THCudaTensor* output
) {
	int n = 0;

	n = THCudaTensor_nElement(state, output);
	kernel_SeparableConvolution_updateOutput<<< (n + 512 - 1) / 512, 512, 0, THCState_getCurrentStream(state) >>>(
		n,
		THCudaTensor_data(state, input), make_long4(input->size[0], input->size[1], input->size[2], input->size[3]), make_long4(input->stride[0], input->stride[1], input->stride[2], input->stride[3]),
		THCudaTensor_data(state, vertical), make_long4(vertical->size[0], vertical->size[1], vertical->size[2], vertical->size[3]), make_long4(vertical->stride[0], vertical->stride[1], vertical->stride[2], vertical->stride[3]),
		THCudaTensor_data(state, horizontal), make_long4(horizontal->size[0], horizontal->size[1], horizontal->size[2], horizontal->size[3]), make_long4(horizontal->stride[0], horizontal->stride[1], horizontal->stride[2], horizontal->stride[3]),
		THCudaTensor_data(state, output), make_long4(output->size[0], output->size[1], output->size[2], output->size[3]), make_long4(output->stride[0], output->stride[1], output->stride[2], output->stride[3])
	);

	THCudaCheck(cudaGetLastError());
}

#ifdef __cplusplus
	}
#endif