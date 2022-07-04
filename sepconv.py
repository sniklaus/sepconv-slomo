#!/usr/bin/env python

import collections
import cupy
import os
import re
import torch
import typing


##########################################################


objCudacache = {}


def cuda_int32(intIn:int):
    return cupy.int32(intIn)
# end


def cuda_float32(fltIn:float):
    return cupy.float32(fltIn)
# end


def cuda_kernel(strFunction:str, strKernel:str, objVariables:typing.Dict):
    if 'device' not in objCudacache:
        objCudacache['device'] = torch.cuda.get_device_name()
    # end

    strKey = strFunction

    for strVariable in objVariables:
        objValue = objVariables[strVariable]

        strKey += strVariable

        if objValue is None:
            continue

        elif type(objValue) == int:
            strKey += str(objValue)

        elif type(objValue) == float:
            strKey += str(objValue)

        elif type(objValue) == bool:
            strKey += str(objValue)

        elif type(objValue) == str:
            strKey += objValue

        elif type(objValue) == torch.Tensor:
            strKey += str(objValue.dtype)
            strKey += str(objValue.shape)
            strKey += str(objValue.stride())

        elif True:
            print(strVariable, type(objValue))
            assert(False)

        # end
    # end

    strKey += objCudacache['device']

    if strKey not in objCudacache:
        for strVariable in objVariables:
            objValue = objVariables[strVariable]

            if objValue is None:
                continue

            elif type(objValue) == int:
                strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))

            elif type(objValue) == float:
                strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))

            elif type(objValue) == bool:
                strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))

            elif type(objValue) == str:
                strKernel = strKernel.replace('{{' + strVariable + '}}', objValue)

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.uint8:
                strKernel = strKernel.replace('{{type}}', 'unsigned char')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float16:
                strKernel = strKernel.replace('{{type}}', 'half')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float32:
                strKernel = strKernel.replace('{{type}}', 'float')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float64:
                strKernel = strKernel.replace('{{type}}', 'double')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.int32:
                strKernel = strKernel.replace('{{type}}', 'int')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.int64:
                strKernel = strKernel.replace('{{type}}', 'long')

            elif type(objValue) == torch.Tensor:
                print(strVariable, objValue.dtype)
                assert(False)

            elif True:
                print(strVariable, type(objValue))
                assert(False)

            # end
        # end

        while True:
            objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

            if objMatch is None:
                break
            # end

            intArg = int(objMatch.group(2))

            strTensor = objMatch.group(4)
            intSizes = objVariables[strTensor].size()

            strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg] if torch.is_tensor(intSizes[intArg]) == False else intSizes[intArg].item()))
        # end

        while True:
            objMatch = re.search('(OFFSET_)([0-4])(\()', strKernel)

            if objMatch is None:
                break
            # end

            intStart = objMatch.span()[1]
            intStop = objMatch.span()[1]
            intParentheses = 1

            while True:
                intParentheses += 1 if strKernel[intStop] == '(' else 0
                intParentheses -= 1 if strKernel[intStop] == ')' else 0

                if intParentheses == 0:
                    break
                # end

                intStop += 1
            # end

            intArgs = int(objMatch.group(2))
            strArgs = strKernel[intStart:intStop].split(',')

            assert(intArgs == len(strArgs) - 1)

            strTensor = strArgs[0]
            intStrides = objVariables[strTensor].stride()

            strIndex = []

            for intArg in range(intArgs):
                strIndex.append('((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()) + ')')
            # end

            strKernel = strKernel.replace('OFFSET_' + str(intArgs) + '(' + strKernel[intStart:intStop] + ')', '(' + str.join('+', strIndex) + ')')
        # end

        while True:
            objMatch = re.search('(VALUE_)([0-4])(\()', strKernel)

            if objMatch is None:
                break
            # end

            intStart = objMatch.span()[1]
            intStop = objMatch.span()[1]
            intParentheses = 1

            while True:
                intParentheses += 1 if strKernel[intStop] == '(' else 0
                intParentheses -= 1 if strKernel[intStop] == ')' else 0

                if intParentheses == 0:
                    break
                # end

                intStop += 1
            # end

            intArgs = int(objMatch.group(2))
            strArgs = strKernel[intStart:intStop].split(',')

            assert(intArgs == len(strArgs) - 1)

            strTensor = strArgs[0]
            intStrides = objVariables[strTensor].stride()

            strIndex = []

            for intArg in range(intArgs):
                strIndex.append('((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()) + ')')
            # end

            strKernel = strKernel.replace('VALUE_' + str(intArgs) + '(' + strKernel[intStart:intStop] + ')', strTensor + '[' + str.join('+', strIndex) + ']')
        # end

        objCudacache[strKey] = {
            'strFunction': strFunction,
            'strKernel': strKernel
        }
    # end

    return strKey
# end


@cupy.memoize(for_each_device=True)
def cuda_launch(strKey:str):
    if 'CUDA_HOME' not in os.environ:
        os.environ['CUDA_HOME'] = '/usr/local/cuda/'
    # end

    return cupy.cuda.compile_with_cache(objCudacache[strKey]['strKernel'], tuple(['-I ' + os.environ['CUDA_HOME'], '-I ' + os.environ['CUDA_HOME'] + '/include'])).get_function(objCudacache[strKey]['strFunction'])
# end


##########################################################


class sepconv_func(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, tenIn, tenVer, tenHor):
        tenOut = tenIn.new_empty([tenIn.shape[0], tenIn.shape[1], tenVer.shape[2] and tenHor.shape[2], tenVer.shape[3] and tenHor.shape[3]])

        if tenIn.is_cuda == True:
            cuda_launch(cuda_kernel('sepconv_out', '''
                extern "C" __global__ void __launch_bounds__(512) sepconv_out(
                    const int n,
                    const {{type}}* __restrict__ tenIn,
                    const {{type}}* __restrict__ tenVer,
                    const {{type}}* __restrict__ tenHor,
                    {{type}}* __restrict__ tenOut
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenOut) / SIZE_2(tenOut) / SIZE_1(tenOut) ) % SIZE_0(tenOut);
                    const int intC = ( intIndex / SIZE_3(tenOut) / SIZE_2(tenOut)                  ) % SIZE_1(tenOut);
                    const int intY = ( intIndex / SIZE_3(tenOut)                                   ) % SIZE_2(tenOut);
                    const int intX = ( intIndex                                                    ) % SIZE_3(tenOut);

                    {{type}} fltOut = 0.0f;

                    {{type}} fltKahanc = 0.0f;
                    {{type}} fltKahany = 0.0f;
                    {{type}} fltKahant = 0.0f;

                    for (int intFy = 0; intFy < SIZE_1(tenVer); intFy += 1) {
                        for (int intFx = 0; intFx < SIZE_1(tenHor); intFx += 1) {
                            fltKahany = VALUE_4(tenIn, intN, intC, intY + intFy, intX + intFx) * VALUE_4(tenVer, intN, intFy, intY, intX) * VALUE_4(tenHor, intN, intFx, intY, intX);
                            fltKahany = fltKahany - fltKahanc;
                            fltKahant = fltOut + fltKahany;
                            fltKahanc = (fltKahant - fltOut) - fltKahany;
                            fltOut = fltKahant;
                        }
                    }

                    tenOut[intIndex] = fltOut;
                } }
            ''', {
                'tenIn': tenIn,
                'tenVer': tenVer,
                'tenHor': tenHor,
                'tenOut': tenOut
            }))(
                grid=tuple([int((tenOut.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[cuda_int32(tenOut.nelement()), tenIn.data_ptr(), tenVer.data_ptr(), tenHor.data_ptr(), tenOut.data_ptr()],
                stream=collections.namedtuple('Stream', 'ptr')(torch.cuda.current_stream().cuda_stream)
            )

        elif tenIn.is_cuda != True:
            assert(False)

        # end

        self.save_for_backward(tenIn, tenVer, tenHor)

        return tenOut
    # end

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(self, tenOutgrad):
        tenIn, tenVer, tenHor = self.saved_tensors

        tenOutgrad = tenOutgrad.contiguous(); assert(tenOutgrad.is_cuda == True)

        tenIngrad = tenIn.new_empty([tenIn.shape[0], tenIn.shape[1], tenIn.shape[2], tenIn.shape[3]]) if self.needs_input_grad[0] == True else None
        tenVergrad = tenVer.new_empty([tenVer.shape[0], tenVer.shape[1], tenVer.shape[2], tenVer.shape[3]]) if self.needs_input_grad[1] == True else None
        tenHorgrad = tenHor.new_empty([tenHor.shape[0], tenHor.shape[1], tenHor.shape[2], tenHor.shape[3]]) if self.needs_input_grad[2] == True else None

        if tenIngrad is not None:
            cuda_launch(cuda_kernel('sepconv_ingrad', '''
                extern "C" __global__ void __launch_bounds__(512) sepconv_ingrad(
                    const int n,
                    const {{type}}* __restrict__ tenIn,
                    const {{type}}* __restrict__ tenVer,
                    const {{type}}* __restrict__ tenHor,
                    const {{type}}* __restrict__ tenOutgrad,
                    {{type}}* __restrict__ tenIngrad,
                    {{type}}* __restrict__ tenVergrad,
                    {{type}}* __restrict__ tenHorgrad
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenIngrad) / SIZE_2(tenIngrad) / SIZE_1(tenIngrad) ) % SIZE_0(tenIngrad);
                    const int intC = ( intIndex / SIZE_3(tenIngrad) / SIZE_2(tenIngrad)                     ) % SIZE_1(tenIngrad);
                    const int intY = ( intIndex / SIZE_3(tenIngrad)                                         ) % SIZE_2(tenIngrad);
                    const int intX = ( intIndex                                                             ) % SIZE_3(tenIngrad);

                    {{type}} fltIngrad = 0.0f;

                    {{type}} fltKahanc = 0.0f;
                    {{type}} fltKahany = 0.0f;
                    {{type}} fltKahant = 0.0f;

                    for (int intFy = 0; intFy < SIZE_1(tenVer); intFy += 1) {
                        int intKy = intY + intFy - (SIZE_1(tenVer) - 1);

                        if (intKy < 0) { continue; }
                        if (intKy >= SIZE_2(tenVer)) { continue; }

                        for (int intFx = 0; intFx < SIZE_1(tenHor); intFx += 1) {
                            int intKx = intX + intFx - (SIZE_1(tenHor) - 1);

                            if (intKx < 0) { continue; }
                            if (intKx >= SIZE_3(tenHor)) { continue; }

                            fltKahany = VALUE_4(tenVer, intN, (SIZE_1(tenVer) - 1) - intFy, intKy, intKx) * VALUE_4(tenHor, intN, (SIZE_1(tenHor) - 1) - intFx, intKy, intKx) * VALUE_4(tenOutgrad, intN, intC, intKy, intKx);
                            fltKahany = fltKahany - fltKahanc;
                            fltKahant = fltIngrad + fltKahany;
                            fltKahanc = (fltKahant - fltIngrad) - fltKahany;
                            fltIngrad = fltKahant;
                        }
                    }

                    tenIngrad[intIndex] = fltIngrad;
                } }
            ''', {
                'tenIn': tenIn,
                'tenVer': tenVer,
                'tenHor': tenHor,
                'tenOutgrad': tenOutgrad,
                'tenIngrad': tenIngrad,
                'tenVergrad': tenVergrad,
                'tenHorgrad': tenHorgrad
            }))(
                grid=tuple([int((tenIngrad.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[cuda_int32(tenIngrad.nelement()), tenIn.data_ptr(), tenVer.data_ptr(), tenHor.data_ptr(), tenOutgrad.data_ptr(), tenIngrad.data_ptr(), None, None],
                stream=collections.namedtuple('Stream', 'ptr')(torch.cuda.current_stream().cuda_stream)
            )
        # end

        if tenVergrad is not None:
            cuda_launch(cuda_kernel('sepconv_vergrad', '''
                extern "C" __global__ void __launch_bounds__(512) sepconv_vergrad(
                    const int n,
                    const {{type}}* __restrict__ tenIn,
                    const {{type}}* __restrict__ tenVer,
                    const {{type}}* __restrict__ tenHor,
                    const {{type}}* __restrict__ tenOutgrad,
                    {{type}}* __restrict__ tenIngrad,
                    {{type}}* __restrict__ tenVergrad,
                    {{type}}* __restrict__ tenHorgrad
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenVergrad) / SIZE_2(tenVergrad) / SIZE_1(tenVergrad) ) % SIZE_0(tenVergrad);
                    const int intC = ( intIndex / SIZE_3(tenVergrad) / SIZE_2(tenVergrad)                      ) % SIZE_1(tenVergrad);
                    const int intY = ( intIndex / SIZE_3(tenVergrad)                                           ) % SIZE_2(tenVergrad);
                    const int intX = ( intIndex                                                                ) % SIZE_3(tenVergrad);

                    {{type}} fltVergrad = 0.0f;

                    {{type}} fltKahanc = 0.0f;
                    {{type}} fltKahany = 0.0f;
                    {{type}} fltKahant = 0.0f;

                    for (int intI = 0; intI < SIZE_1(tenIn); intI += 1) {
                        for (int intFx = 0; intFx < SIZE_1(tenHor); intFx += 1) {
                            fltKahany = VALUE_4(tenHor, intN, intFx, intY, intX) * VALUE_4(tenIn, intN, intI, intY + intC, intX + intFx) * VALUE_4(tenOutgrad, intN, intI, intY, intX);
                            fltKahany = fltKahany - fltKahanc;
                            fltKahant = fltVergrad + fltKahany;
                            fltKahanc = (fltKahant - fltVergrad) - fltKahany;
                            fltVergrad = fltKahant;
                        }
                    }

                    tenVergrad[intIndex] = fltVergrad;
                } }
            ''', {
                'tenIn': tenIn,
                'tenVer': tenVer,
                'tenHor': tenHor,
                'tenOutgrad': tenOutgrad,
                'tenIngrad': tenIngrad,
                'tenVergrad': tenVergrad,
                'tenHorgrad': tenHorgrad
            }))(
                grid=tuple([int((tenVergrad.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[cuda_int32(tenVergrad.nelement()), tenIn.data_ptr(), tenVer.data_ptr(), tenHor.data_ptr(), tenOutgrad.data_ptr(), None, tenVergrad.data_ptr(), None],
                stream=collections.namedtuple('Stream', 'ptr')(torch.cuda.current_stream().cuda_stream)
            )
        # end

        if tenHorgrad is not None:
            cuda_launch(cuda_kernel('sepconv_horgrad', '''
                extern "C" __global__ void __launch_bounds__(512) sepconv_horgrad(
                    const int n,
                    const {{type}}* __restrict__ tenIn,
                    const {{type}}* __restrict__ tenVer,
                    const {{type}}* __restrict__ tenHor,
                    const {{type}}* __restrict__ tenOutgrad,
                    {{type}}* __restrict__ tenIngrad,
                    {{type}}* __restrict__ tenVergrad,
                    {{type}}* __restrict__ tenHorgrad
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenHorgrad) / SIZE_2(tenHorgrad) / SIZE_1(tenHorgrad) ) % SIZE_0(tenHorgrad);
                    const int intC = ( intIndex / SIZE_3(tenHorgrad) / SIZE_2(tenHorgrad)                      ) % SIZE_1(tenHorgrad);
                    const int intY = ( intIndex / SIZE_3(tenHorgrad)                                           ) % SIZE_2(tenHorgrad);
                    const int intX = ( intIndex                                                                ) % SIZE_3(tenHorgrad);

                    {{type}} fltHorgrad = 0.0f;

                    {{type}} fltKahanc = 0.0f;
                    {{type}} fltKahany = 0.0f;
                    {{type}} fltKahant = 0.0f;

                    for (int intI = 0; intI < SIZE_1(tenIn); intI += 1) {
                        for (int intFy = 0; intFy < SIZE_1(tenVer); intFy += 1) {
                            fltKahany = VALUE_4(tenVer, intN, intFy, intY, intX) * VALUE_4(tenIn, intN, intI, intY + intFy, intX + intC) * VALUE_4(tenOutgrad, intN, intI, intY, intX);
                            fltKahany = fltKahany - fltKahanc;
                            fltKahant = fltHorgrad + fltKahany;
                            fltKahanc = (fltKahant - fltHorgrad) - fltKahany;
                            fltHorgrad = fltKahant;
                        }
                    }

                    tenHorgrad[intIndex] = fltHorgrad;
                } }
            ''', {
                'tenIn': tenIn,
                'tenVer': tenVer,
                'tenHor': tenHor,
                'tenOutgrad': tenOutgrad,
                'tenIngrad': tenIngrad,
                'tenVergrad': tenVergrad,
                'tenHorgrad': tenHorgrad
            }))(
                grid=tuple([int((tenHorgrad.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[cuda_int32(tenHorgrad.nelement()), tenIn.data_ptr(), tenVer.data_ptr(), tenHor.data_ptr(), tenOutgrad.data_ptr(), None, None, tenHorgrad.data_ptr()],
                stream=collections.namedtuple('Stream', 'ptr')(torch.cuda.current_stream().cuda_stream)
            )
        # end

        return tenIngrad, tenVergrad, tenHorgrad
    # end
# end
