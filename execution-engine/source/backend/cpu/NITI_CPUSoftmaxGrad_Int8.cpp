//
//  NITI_CPUSoftmaxGrad_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/08/28.
//  
//

#include <math.h>
#include "backend/cpu/NITI_CPUSoftmaxGrad_Int8.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "CPUTensorConvert.hpp"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {

ErrorCode NITI_CPUSoftmaxGrad_Int8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    return NO_ERROR;
}

ErrorCode NITI_CPUSoftmaxGrad_Int8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    int ib = inputs[0]->batch();
    int ic = inputs[0]->channel();
    int ih = inputs[0]->height();
    int iw = inputs[0]->width();

    int32_t* inputPtr = inputs[0]->host<int32_t>();
    int8_t* outputPtr = outputs[0]->host<int8_t>();

    int inputSize = ib*ic*ih*iw;

    for(int i=0;i<inputSize;i++) {
        outputPtr[i] = inputPtr[i];
    }

    return NO_ERROR;
}

NITI_CPUSoftmaxGrad_Int8::NITI_CPUSoftmaxGrad_Int8(Backend *b, int axis) : MNN::Execution(b), mAxis(axis) {
    // nothing to do
}

Execution* NITI_CPUSoftmaxGrad_Int8::create(const MNN::Op *op, Backend *backend) {
    auto axis = op->main_as_Axis()->axis();
    return new NITI_CPUSoftmaxGrad_Int8(backend, axis);
}

class NITI_CPUSoftmaxGrad_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return NITI_CPUSoftmaxGrad_Int8::create(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(NITI_CPUSoftmaxGrad_Int8_Creator, OpType_NITI_SOFTMAX_Grad_Int8);

} // namespace MNN
