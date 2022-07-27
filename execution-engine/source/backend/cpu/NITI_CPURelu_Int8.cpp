//
//  NITI_CPURelu_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/09/17.
//
//

#include <math.h>
#include "backend/cpu/NITI_CPURelu_Int8.hpp"
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

ErrorCode NITI_CPURelu_Int8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    return NO_ERROR;
}

ErrorCode NITI_CPURelu_Int8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    auto input = inputs[0];
    auto output = outputs[0];

    int obatch = output->batch();
    int ochannel = output->channel();
    int owidth = output->width();
    int oheight = output->height();

    int osize = obatch*ochannel*owidth*oheight;

    const int threads = std::max(static_cast<CPUBackend*>(backend())->threadNumber(), 1);

    MNN_CONCURRENCY_BEGIN(tId, threads) {
        int total = obatch*ochannel*owidth*oheight;

        int begin = tId * (total/threads);
        int end = begin + total/threads;
        if(tId == threads-1 )
            end = total;

        for(int i=begin;i<end;i++) {
            int8_t temp = input->host<int8_t>()[i];
            if(temp >= 0)
                output->host<int8_t>()[i] = temp;
            else 
                output->host<int8_t>()[i] = 0;
        }
    }
    MNN_CONCURRENCY_END();

    return NO_ERROR;
}

NITI_CPURelu_Int8::NITI_CPURelu_Int8(Backend *b) : MNN::Execution(b) {
    // nothing to do
}


class NITI_CPURelu_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new NITI_CPURelu_Int8(backend);
    }
};

REGISTER_CPU_OP_CREATOR(NITI_CPURelu_Int8_Creator, OpType_NITI_Relu_Int8);

} // namespace MNN
