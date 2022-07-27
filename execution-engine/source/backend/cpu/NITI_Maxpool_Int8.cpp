//
//  DeConvInt8.cpp
//  MNN
//
//  Created by xudaliang on 2021/8/17.
//  
//

#include "backend/cpu/NITI_Maxpool_Int8.hpp"
#include "backend/cpu/CPUBackend.hpp"

#include "core/Macro.h"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#include "core/Concurrency.h"

#define DST_TILE 16
#define CACHE_SIZE 128

namespace MNN {

static void poolingMaxNHWCInt8(const Tensor *src, Tensor *dst, int sx, int sy, int kx, int ky, int px, int py) {
    const int inputHeight  = src->length(1);
    const int inputWidth   = src->length(2);
    const int outputHeight = dst->length(1);
    const int outputWidth  = dst->length(2);
    const int channel      = dst->length(3);

    const int batch = src->length(0);
    int8_t result[CACHE_SIZE];

    for (int b=0;b<batch;b++) {
        const auto srcPtr = src->host<int8_t>() + b*inputHeight*inputWidth*channel;
        auto dstPtr       = dst->host<int8_t>() + b*outputHeight*outputWidth*channel;

        for (int oc = 0; oc < channel; oc += CACHE_SIZE) {
            const int realChannel = std::min(channel - oc, CACHE_SIZE);

            for (int oy = 0; oy < outputHeight; ++oy) {
                for (int ox = 0; ox < outputWidth; ++ox) {
                    const int srcOriginX = ox * sx - px;
                    const int srcOriginY = oy * sy - py;
                    const int kxs        = std::max(0, -srcOriginX);
                    const int kxe        = std::min(kx, inputWidth - srcOriginX);
                    const int kys        = std::max(0, -srcOriginY);
                    const int kye        = std::min(ky, inputHeight - srcOriginY);

                    const int8_t *srcCurPtr = srcPtr + oc + (srcOriginX + srcOriginY * inputWidth) * channel;
                    memset(result, INT8_MIN, sizeof(int8_t) * realChannel);
                    for (int y = kys; y < kye; ++y) {
                        const int8_t *srcCurRowPtr = srcCurPtr + (y * inputWidth + kxs) * channel;
                        for (int x = kxs; x < kxe; ++x) {
                            const int8_t *srcCurChannlePtr = srcCurRowPtr;
                            int index                      = 0;
                            for (; index < realChannel; ++index) {
                                result[index] = std::max(result[index], *srcCurChannlePtr++);
                            }
                            srcCurRowPtr += channel;
                        }
                    }

                    int8_t *dstCurPtr = dstPtr + oc + (ox + oy * outputWidth) * channel;
                    memcpy(dstCurPtr, result, sizeof(int8_t) * realChannel);
                }
            }
        }
    }

    
}

ErrorCode NITI_Maxpool_Int8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    
    const auto input = inputs[0];
    auto output      = outputs[0];

    int strideWidth  = mParameter->strideX();
    int strideHeight = mParameter->strideY();
    int padWidth     = mParameter->padX();
    int padHeight    = mParameter->padY();
    int kernelWidth  = mParameter->kernelX();
    int kernelHeight = mParameter->kernelY();

    const int inputWidth   = input->width();
    const int inputHeight  = input->height();
    const int outputWidth  = output->width();
    const int outputHeight = output->height();

    kernelWidth  = std::min(kernelWidth, inputWidth);
    kernelHeight = std::min(kernelHeight, inputHeight);
    if (mParameter->isGlobal()) {
        kernelWidth  = inputWidth;
        kernelHeight = inputHeight;
        strideWidth  = inputWidth;
        strideHeight = inputHeight;
        padWidth     = 0;
        padHeight    = 0;
    }
    if (mParameter->padType() == PoolPadType_SAME) {
        int padNeededWidth  = (outputWidth - 1) * strideWidth + kernelWidth - inputWidth;
        int padNeededHeight = (outputHeight - 1) * strideHeight + kernelHeight - inputHeight;
        padWidth            = padNeededWidth > 0 ? padNeededWidth / 2 : 0;
        padHeight           = padNeededHeight > 0 ? padNeededHeight / 2 : 0;
    }

    const int channel = input->channel();
    auto poolFunc     = poolingMaxNHWCInt8;
    // if (mParameter->type() == MNN::PoolType_AVEPOOL) {
    //     poolFunc = poolingAvgNHWCInt8;
    // }
    mInputTemp.reset(Tensor::createDevice<int8_t>({input->batch(), inputHeight, inputWidth, channel}));
    mOutputTemp.reset(Tensor::createDevice<int8_t>({output->batch(), outputHeight, outputWidth, channel}));

    bool allocSucc = backend()->onAcquireBuffer(mInputTemp.get(), Backend::DYNAMIC);
    allocSucc      = allocSucc && backend()->onAcquireBuffer(mOutputTemp.get(), Backend::DYNAMIC);
    if (!allocSucc) {
        return OUT_OF_MEMORY;
    }


    backend()->onReleaseBuffer(mInputTemp.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mOutputTemp.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode NITI_Maxpool_Int8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    auto input  = inputs[0];
    auto output = outputs[0];
    
    int ibatch = input->batch();
    int ichannel = input->channel();
    int iwidth = input->width();
    int iheight = input->height();

    int isize = ibatch*ichannel*iwidth*iheight;
    memcpy(mInputTemp->host<int8_t>(), input->host<int8_t>(), isize);

    int strideWidth  = mParameter->strideX();
    int strideHeight = mParameter->strideY();
    int padWidth     = mParameter->padX();
    int padHeight    = mParameter->padY();
    int kernelWidth  = mParameter->kernelX();
    int kernelHeight = mParameter->kernelY();

    const int inputWidth   = input->width();
    const int inputHeight  = input->height();
    const int outputWidth  = output->width();
    const int outputHeight = output->height();

    kernelWidth  = std::min(kernelWidth, inputWidth);
    kernelHeight = std::min(kernelHeight, inputHeight);
    if (mParameter->isGlobal()) {
        kernelWidth  = inputWidth;
        kernelHeight = inputHeight;
        strideWidth  = inputWidth;
        strideHeight = inputHeight;
        padWidth     = 0;
        padHeight    = 0;
    }
    if (mParameter->padType() == PoolPadType_SAME) {
        int padNeededWidth  = (outputWidth - 1) * strideWidth + kernelWidth - inputWidth;
        int padNeededHeight = (outputHeight - 1) * strideHeight + kernelHeight - inputHeight;
        padWidth            = padNeededWidth > 0 ? padNeededWidth / 2 : 0;
        padHeight           = padNeededHeight > 0 ? padNeededHeight / 2 : 0;
    }


    poolingMaxNHWCInt8(mInputTemp.get(), mOutputTemp.get(), strideWidth, strideHeight, kernelWidth, kernelHeight, padWidth, padHeight);

    *(outputs[1]->host<int8_t>()) = *(inputs[1]->host<int8_t>());

    int obatch = output->batch();
    int ochannel = output->channel();
    int owidth = output->width();
    int oheight = output->height();

    int osize = obatch*ochannel*owidth*oheight;
    memcpy(output->host<int8_t>(), mOutputTemp->host<int8_t>(), osize);

    return NO_ERROR;
}


NITI_Maxpool_Int8::~NITI_Maxpool_Int8() {
    // Do nothing
}
NITI_Maxpool_Int8::NITI_Maxpool_Int8(Backend* backend, const NITI_Pool_Int8* pool) : MNN::Execution(backend) {
    mParameter = pool;
}

class NITI_Maxpool_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {

        return new NITI_Maxpool_Int8(backend, op->main_as_NITI_Pool_Int8());
    }
};

REGISTER_CPU_OP_CREATOR(NITI_Maxpool_Int8_Creator, OpType_NITI_Maxpool_Int8);

} // namespace MNN
