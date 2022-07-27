//
//  NITI_DSPWeightRotateRef_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/11/17.
//
//

#include <math.h>
#include "backend/cpu/NITI_DSPWeightRotateRef_Int8.hpp"
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

ErrorCode NITI_DSPWeightRotateRef_Int8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    return NO_ERROR;
}


void NITI_DSPWeightRotateRef_Int8::rotate180(int8_t arraySrc[], int8_t arrayDes[], int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols ; j++)
    arrayDes[(rows - i - 1) * cols + (cols - j - 1)] = arraySrc[i * cols + j];
}

ErrorCode NITI_DSPWeightRotateRef_Int8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    auto weight = inputs[0];
    auto weight180 = outputs[0];

    int wbatch = weight->batch();
    int wchannel = weight->channel();
    int wwidth = weight->width();
    int wheight = weight->height();

    const int threads = std::max(static_cast<CPUBackend*>(backend())->threadNumber(), 1);

    MNN_CONCURRENCY_BEGIN(tId, threads) {
       int total = wbatch*wheight;

        int begin = tId * (total/threads);
        int end = begin + total/threads;
        if(tId == threads-1 )
            end = total;
        for (int i = begin;i < end; i++) {
            rotate180(weight->host<int8_t>()+i*wwidth*wchannel,  weight180->host<int8_t>()+i*wwidth*wchannel, wwidth, wchannel);
        }
    }
    MNN_CONCURRENCY_END();
    
    return NO_ERROR;
}

NITI_DSPWeightRotateRef_Int8::NITI_DSPWeightRotateRef_Int8(Backend *b) : MNN::Execution(b) {
    // nothing to do
}


class NITI_DSPWeightRotateRef_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new NITI_DSPWeightRotateRef_Int8(backend);
    }
};

REGISTER_CPU_OP_CREATOR(NITI_DSPWeightRotateRef_Int8_Creator, OpType_NITI_DSP_WEIGHTROTATE180_REF_Int8);

} // namespace MNN
