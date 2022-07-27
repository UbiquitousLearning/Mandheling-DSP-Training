//
//  NITI_Pool_Int8.cpp
//  MNN
//
//  Created by jiangxiaotang on 2019/4/19.
//  Copyright © 2019 Alibaba. All rights reserved.
//

#include "backend/cpu/NITI_CPUPoolGrad_Int8.hpp"
#include "core/Macro.h"
#include "math/Vec.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"

using Vecint8 = MNN::Math::Vec<int8_t, 4>;
namespace MNN {
class NITI_CPUMaxPoolGrad_Int8 : public NITI_CPUPoolGrad_Int8 {
public:
    NITI_CPUMaxPoolGrad_Int8(Backend *b, const NITI_Pool_Int8 *parameter) : NITI_CPUPoolGrad_Int8(b, parameter) {}

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto origin       = inputs[0];
        auto outputOrigin = inputs[1];
        auto inputDiff    = inputs[2];
        auto outputDiff   = outputs[0];

        auto ow = inputDiff->width();
        auto oh = inputDiff->height();
        auto iw = origin->width();
        auto ih = origin->height();

        auto channelC4 = UP_DIV(inputDiff->channel(), 4);
        auto batch     = inputDiff->batch();
        auto totalChannelC4 = batch * channelC4;
        auto threadNumber = ((CPUBackend*)(backend()))->threadNumber();
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            for (int z = tId; z < totalChannelC4; z+=threadNumber) {
                auto inputZ0    = origin->host<int8_t>() + z * iw * ih * 4;
                auto inputZ1    = inputDiff->host<int8_t>() + z * ow * oh * 4;
                auto outputOriZ = outputOrigin->host<int8_t>() + z * ow * oh * 4;
                auto outputZ    = outputDiff->host<int8_t>() + z * iw * ih * 4;

                ::memset(outputZ, 0, sizeof(int8_t) * iw * ih * 4);
                for (int y = 0; y < oh; ++y) {
                    for (int x = 0; x < ow; ++x) {
                        Vecint8 maxValue = Vecint8::load(outputOriZ + 4 * (x + y * ow));
                        Vecint8 diffValue   = Vecint8::load(inputZ1 + 4 * (x + y * ow));
                        bool unfinished[4] = {true, true, true, true};
                        for (int ky = 0; ky < mKernelY; ++ky) {
                            auto sy = y * mStrideY + ky - mPadY;
                            if (sy < 0 || sy >= ih) {
                                continue;
                            }
                            for (int kx = 0; kx < mKernelX; ++kx) {
                                auto sx = x * mStrideX + kx - mPadX;
                                if (sx < 0 || sx >= iw) {
                                    continue;
                                }
                                Vecint8 originValue = Vecint8::load(inputZ0 + 4 * (sx + sy * iw));
                                auto dst         = outputZ + 4 * (sx + sy * iw);
                                for (int j = 0; j < 4; ++j) {
                                    if (unfinished[j] && originValue[j] >= maxValue[j]) {
                                        unfinished[j] = false;
                                        dst[j] = dst[j] + diffValue[j];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
        MNN_CONCURRENCY_END();


        return NO_ERROR;
    }
};


class NITI_CPUPoolGrad_Int8Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto pool = op->main_as_NITI_Pool_Int8();
        return new NITI_CPUMaxPoolGrad_Int8(backend, op->main_as_NITI_Pool_Int8());
    }
};

REGISTER_CPU_OP_CREATOR(NITI_CPUPoolGrad_Int8Creator, OpType_NITI_PoolGrad_Int8);
} // namespace MNN
