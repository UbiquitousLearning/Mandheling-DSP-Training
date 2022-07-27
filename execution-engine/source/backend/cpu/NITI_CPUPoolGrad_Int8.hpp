//
//  NITI_CPUPoolGrad_Int8.hpp
//  MNN
//
//  Created by jiangxiaotang on 2019/4/19.
//  Copyright © 2019 Alibaba. All rights reserved.
//

#ifndef NITI_CPUPoolGrad_Int8_hpp
#define NITI_CPUPoolGrad_Int8_hpp

#include "backend/cpu/CPUBackend.hpp"

namespace MNN {
class NITI_CPUPoolGrad_Int8 : public Execution {
public:
    virtual ~ NITI_CPUPoolGrad_Int8() = default;
    NITI_CPUPoolGrad_Int8(Backend *b, const NITI_Pool_Int8 *parameter) : Execution(b) {
        mStrideX = parameter->strideX();
        mStrideY = parameter->strideY();
        mKernelX = parameter->kernelX();
        mKernelY = parameter->kernelY();
        mGlobal  = parameter->isGlobal();
        mParameter = parameter;
    }
    
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        if (mGlobal) {
            mKernelX = inputs[0]->width();
            mKernelY = inputs[0]->height();
        }
        int padWidth     = mParameter->padX();
        int padHeight    = mParameter->padY();
        int strideWidth  = mParameter->strideX();
        int strideHeight = mParameter->strideY();

        // edit const if global
        auto input       = inputs[0];
        auto output      = outputs[0];
        int kernelWidth  = std::min(mParameter->kernelX(), input->width());
        int kernelHeight = std::min(mParameter->kernelY(), input->height());
        if (mParameter->isGlobal()) {
            kernelWidth  = input->width();
            kernelHeight = input->height();
            strideWidth  = input->width();
            strideHeight = input->height();
            padWidth     = 0;
            padHeight    = 0;
        }
        if (mParameter->padType() == PoolPadType_SAME) {
            int padNeededWidth  = (output->width() - 1) * strideWidth + kernelWidth - input->width();
            int padNeededHeight = (output->height() - 1) * strideHeight + kernelHeight - input->height();
            padWidth            = padNeededWidth > 0 ? padNeededWidth / 2 : 0;
            padHeight           = padNeededHeight > 0 ? padNeededHeight / 2 : 0;
        } else if (mParameter->padType() == PoolPadType_VALID) {
            padWidth = padHeight = 0;
        }
        mPadX = padWidth;
        mPadY = padHeight;
        return NO_ERROR;
    }

protected:
    int mStrideX;
    int mStrideY;
    int mKernelX;
    int mKernelY;
    bool mGlobal;
    int mPadX;
    int mPadY;
    const NITI_Pool_Int8* mParameter;
};
} // namespace MNN
#endif /* CPUPoolGrad_hpp */
