//
//  NITI_CPULoss_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/08/28.
//  
//

#include <math.h>
#include "backend/cpu/NITI_CPULoss_Int8.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "CPUTensorConvert.hpp"
#include "math.h"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

void NITI_CPULoss_Int8::_softmax1(const float *srcData, float *dstData, int outside, int channel, int threadNum) {
    MNN_CONCURRENCY_BEGIN(tId, threadNum)
    {
        const float *srcY = srcData + tId * channel;
        float *dstY       = dstData + tId * channel;
        for (int y = (int)tId; y < outside; y += threadNum, srcY += channel * threadNum, dstY += channel * threadNum) {
            MNNSoftmax(dstY, srcY, channel);
        }
    }
    MNN_CONCURRENCY_END();
    return;
}

ErrorCode NITI_CPULoss_Int8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    auto input = inputs[0];

    input_float.reset(Tensor::createDevice<float>({input->batch(), input->channel(), input->height(), input->width()}));
    bool success = backend()->onAcquireBuffer(input_float.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    input_softmax.reset(Tensor::createDevice<float>({input->batch(), input->channel(), input->height(), input->width()}));
    success = backend()->onAcquireBuffer(input_softmax.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    

    input_sum.reset(Tensor::createDevice<float>({input->batch(), 1, input->height(), input->width()}));
    success = backend()->onAcquireBuffer(input_sum.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    backend()->onReleaseBuffer(input_float.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(input_softmax.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(input_sum.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode NITI_CPULoss_Int8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    auto input = inputs[0];
    auto output = outputs[0];
    int8_t ascale_input = *(inputs[1]->host<int8_t>());
    auto target = inputs[2];

    int ib = inputs[0]->batch();
    int ic = inputs[0]->channel();
    int ih = inputs[0]->height();
    int iw = inputs[0]->width();

    int inputSize = ib*ic*ih*iw;

    int8_t* inputPtr = input->host<int8_t>();
    float* input_floatPtr = input_float->host<float>();
    float* input_softmaxPtr = input_softmax->host<float>();
    float* input_sumPtr = input_sum->host<float>();
    float* outputPtr = output->host<float>();
    int32_t* targetPtr = target->host<int32_t>();

    float ascale = 1.0f*ascale_input;


    for(int i=0;i<inputSize;i++)
        input_floatPtr[i] = powf(2.0,ascale)*inputPtr[i];

    int threadNum = ((CPUBackend *)backend())->threadNumber();
    _softmax1(input_floatPtr, input_softmaxPtr, ib ,ic, threadNum);
    

    int tc = target->elementSize()/target->batch();

    for(int i=0;i<ib;i++) {

        for(int j=0;j<ic;j++) {
            float temp = logf(input_softmaxPtr[i*ic+j]);
            if(j >= tc) {
                input_softmaxPtr[i*ic+j] = 0;
                continue;
            }
            input_softmaxPtr[i*ic+j] = temp * targetPtr[i*tc+j];
        }
    }
    
    for(int i=0;i<ib;i++) {
        float sum = 0;
        for(int j=0;j<ic;j++) {
            sum += input_softmaxPtr[i*ic+j];
        }
        input_sumPtr[i] = sum;
    }

    float average_sum = 0;
    for(int i=0;i<ib;i++)
        average_sum += input_sumPtr[i];
    
    *outputPtr = average_sum/ib*(-1);

    return NO_ERROR;
}

NITI_CPULoss_Int8::NITI_CPULoss_Int8(Backend *b, const MNN::Op *op) : MNN::Execution(b), mOp(op) {
    // nothing to do
}

Execution* NITI_CPULoss_Int8::create(const MNN::Op *op, Backend *backend) {
    return new NITI_CPULoss_Int8(backend, op);
}

class NITI_CPULoss_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return NITI_CPULoss_Int8::create(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(NITI_CPULoss_Int8_Creator, OpType_NITI_LOSS_Int8);

} // namespace MNN
