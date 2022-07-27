//
//  NITI_CPULeftPoolGrad_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/11/06.
//  
//

#include "backend/cpu/NITI_CPULeftPoolGrad_Int8.hpp"
#include "core/Macro.h"
#include "math/Vec.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"

using Vecint8 = MNN::Math::Vec<int8_t, 4>;
namespace MNN {

ErrorCode NITI_CPULeftPoolGrad_Int8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    auto outputDiff    = inputs[0];
    auto output   = outputs[0];

    auto ow = output->width();
    auto oh = output->height();
    auto iw = outputDiff->width();
    auto ih = outputDiff->height();


    auto channelC4 = UP_DIV(outputDiff->channel(), 4);
    auto batch     = outputDiff->batch();
    auto totalChannelC4 = batch * channelC4;
    auto threadNumber = ((CPUBackend*)(backend()))->threadNumber();
    MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
        for (int z = tId; z < totalChannelC4; z+=threadNumber) {
            auto inputZ1    = outputDiff->host<int8_t>() + z * iw * ih * 4;
            auto outputZ    = output->host<int8_t>() + z * ow * oh * 4;

            ::memset(outputZ, 0, sizeof(int8_t) * ow * oh * 4);
            for (int y = 0; y < oh; y+=mStrideY) {
                for (int x = 0; x < ow; x+=mStrideX) {
                    Vecint8 diffValue   = Vecint8::load(inputZ1 + 4 * (x/mStrideX + y/mStrideY * iw));

                    auto dst = outputZ + 4 * (x + y * ow);
                    Vecint8::save(dst, diffValue);
                }
            }
        }
    };
    MNN_CONCURRENCY_END();

    return NO_ERROR;
}

ErrorCode NITI_CPULeftPoolGrad_Int8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    
    return NO_ERROR;
}


class NITI_CPULeftPoolGrad_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto pool = op->main_as_NITI_Pool_Int8();
        return new NITI_CPULeftPoolGrad_Int8(backend, op->main_as_NITI_Pool_Int8());
    }
};

REGISTER_CPU_OP_CREATOR(NITI_CPULeftPoolGrad_Int8_Creator, OpType_NITI_LeftPoolGrad_Int8);
} // namespace MNN
