//
//  NITI_GradientSplitConv_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/1/17.
//  
//

#include "backend/cpu/NITI_GradientSplitConv_Int8.hpp"
#include "backend/cpu/CPUBackend.hpp"

#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include <math.h>

#include "MNN/hexagon/hexagon_nn/HexagonRunningUtils.hpp"

namespace MNN {

bool NITI_GradientSplitConv_Int8::reorderWeight(Backend* bn, const Convolution2DCommon* common,
                          Tensor* weightOrigin,
                          std::shared_ptr<Tensor>& weight,
                          int split,
                          int pos) {
    auto core = static_cast<CPUBackend*>(bn)->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    // reorder weight, [oc, ic, k^2] => [oc/unit, ((ic/unit)*k^2)/(src_unit/unit), unit(oc), (src_unit/unit), unit(ic)]
    int oc = common->outputCount()/split, ic = common->inputCount(), kernelCount = common->kernelX() * common->kernelY();

    auto weightSrc = weightOrigin->host<int8_t>();
    auto weightDst = weight->host<int8_t>();
    memset(weightDst, 0, weight->size());
    for (int k = 0; k < kernelCount; ++k) {
        const auto srcK = weightSrc + k;
        const int threads = std::max(static_cast<CPUBackend*>(backend())->threadNumber(), 1);
        MNN_CONCURRENCY_BEGIN(tId, threads) {

            int total = ic;
            
            int begin = tId * (total/threads);
            int end = begin + total/threads;
            if(tId == threads-1 )
                end = total;

        for (int y = begin; y < end; ++y) {
            const int yOutSide    = y / UNIT;
            const int yInSide     = y % UNIT;
            const int yIndex      = yOutSide + k * UP_DIV(ic, UNIT);
            const int ySubOutSide = yIndex / (SRC_UNIT / UNIT);
            const int ySubInSide  = yIndex % (SRC_UNIT / UNIT);
            
            auto dstY       = weightDst + ySubOutSide * weight->stride(1) + ySubInSide * UNIT + yInSide;
            const auto srcY = srcK + y * kernelCount;
            for (int x = 0; x < oc; ++x) {
                const int xOutSide = x / UNIT;
                const int xInSide  = x % UNIT;
                const int dstIndex = xOutSide * weight->stride(0) + xInSide * SRC_UNIT;
                const int srcIndex = x * kernelCount * ic;
                dstY[dstIndex]     = srcY[srcIndex];
            }
        }
        }
        MNN_CONCURRENCY_END();
    }
    return true;
}

NITI_GradientSplitConv_Int8::~NITI_GradientSplitConv_Int8() {
    // Do nothing
}
NITI_GradientSplitConv_Int8::NITI_GradientSplitConv_Int8(Backend* backend, const NITI_CONV_Int8* convOp) : CPUConvolution(convOp->common(), backend) {

    // choose int8 gemm kernel
    auto core = static_cast<CPUBackend*>(backend)->int8Functions();
    mGemmKernel = core->NITI_Int8GemmKernel;
    mNITIInt32toInt8 = core->NITI_MNNInt32ToInt8;
}


ErrorCode NITI_GradientSplitConv_Int8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // MNN_PRINT("NITI_Conv_Int8 resize\n");

    CPUConvolution::onResize(inputs, outputs);
    auto input  = inputs[0];
    auto output = outputs[0];

    auto weightInt8 = inputs[1];
    
    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    auto convCommon = mCommon;
    const auto kernelCount = convCommon->kernelX() * convCommon->kernelY();
    const auto srcCountUnit = UP_DIV(input->channel(), UNIT);
    const auto totalKernelCountD8Div2 = UP_DIV(srcCountUnit * kernelCount, SRC_UNIT / UNIT);

    mIm2ColParamter.dilateX         = convCommon->dilateX();
    mIm2ColParamter.dilateY         = convCommon->dilateY();
    mIm2ColParamter.strideX         = convCommon->strideX();
    mIm2ColParamter.strideY         = convCommon->strideY();
    mIm2ColParamter.padX            = convCommon->padX();
    mIm2ColParamter.padY            = convCommon->padY();
    mIm2ColParamter.icDiv4          = srcCountUnit;
    mIm2ColParamter.kernelX         = convCommon->kernelX();
    mIm2ColParamter.kernelY         = convCommon->kernelY();
    mIm2ColParamter.kernelCountUnit = totalKernelCountD8Div2;
    mIm2ColParamter.padX = mPadX;
    mIm2ColParamter.padY = mPadY;

    mIm2ColParamter.ih = input->height();
    mIm2ColParamter.iw = input->width();
    mIm2ColParamter.oh = output->height();
    mIm2ColParamter.ow = output->width();
    mIm2ColParamter.srcZStep = input->stride(1) * UNIT * input->batch();
    mIm2ColParamter.srcYStep = input->stride(2) * UNIT;

    mTileCount        = UP_DIV(output->height() * output->width(), DST_XUNIT);
    const int threads = std::max(static_cast<CPUBackend*>(backend())->threadNumber(), 1);
    mThreadNums       = std::min(threads, mTileCount);

    // set im2col tensor info
    mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({threads, DST_XUNIT, mIm2ColParamter.kernelCountUnit * SRC_UNIT}));
    bool success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    int obatch = output->batch();
    int ochannel = output->channel();
    int owidth = output->width();
    int oheight = output->height();

    acc_int32.reset(Tensor::createDevice<int32_t>({obatch, ochannel, owidth, oheight}));
    success = backend()->onAcquireBuffer(acc_int32.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    output_int32.reset(Tensor::createDevice<int32_t>({obatch, ochannel, owidth, oheight}));
    success = backend()->onAcquireBuffer(output_int32.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }


    int align32 = weightInt8->batch()%32;
    if(align32 != 0)
        align32 = (weightInt8->batch()/32 +1)*32;

    int high = 8192/weightInt8->width()/weightInt8->height();

    int realwbatch= weightInt8->batch();
    while(align32 > high) {
        align32 = align32 / 2;
        realwbatch = realwbatch / 2;
    }

    int split =  weightInt8->batch() / realwbatch;

    // reorder weight, [oc, ic, k^2] => [oc/unit, ((ic/unit)*k^2)/(src_unit/unit), unit(oc), (src_unit/unit), unit(ic)]
    int oc = convCommon->outputCount()/split, ic = convCommon->inputCount();
    std::vector<int> shape = {UP_DIV(oc, UNIT), UP_DIV(UP_DIV(ic, UNIT) * kernelCount, SRC_UNIT / UNIT), UNIT, SRC_UNIT};
    
    mWeightInt8.reset(Tensor::createDevice<int8_t>(shape));
    
    bool succ = backend()->onAcquireBuffer(mWeightInt8.get(), Backend::DYNAMIC);
    if (!succ) {
        MNN_ERROR("Memory not enough");
        return OUT_OF_MEMORY;
    }

    splitWeight.reset(Tensor::createDevice<int32_t>({realwbatch, weightInt8->channel(), weightInt8->height(), weightInt8->width()}));
    success = backend()->onAcquireBuffer(splitWeight.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    tempOutput.reset(Tensor::createDevice<int8_t>({obatch, ochannel, owidth, oheight}));
    success = backend()->onAcquireBuffer(tempOutput.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(acc_int32.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(output_int32.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mWeightInt8.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(splitWeight.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(tempOutput.get(), Backend::DYNAMIC);
    
    // MNN_PRINT("NITI_Conv_Int8 resize end\n");
    return NO_ERROR;
}

ErrorCode NITI_GradientSplitConv_Int8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto weight = inputs[1];


    const int threads = std::max(static_cast<CPUBackend*>(backend())->threadNumber(), 1);

    int ibatch = input->batch();
    int ichannel = input->channel();
    int iwidth = input->width();
    int iheight = input->height();
    auto TinputDataPtr = input->host<int8_t>();

    auto output      = outputs[0];

    int align32 = weight->batch()%32;
    if(align32 != 0)
        align32 = (weight->batch()/32 +1)*32;

    int high = 8192/weight->width()/weight->height();

    int realwbatch= weight->batch();
    while(align32 > high) {
        align32 = align32 / 2;
        realwbatch = realwbatch / 2;
    }

    int split =  weight->batch() / realwbatch;
    scale = new int[split];

    for(int s = 0; s<split; s++) {
        memcpy(splitWeight->host<int8_t>(), weight->host<int8_t>() + s*splitWeight->elementSize(), splitWeight->elementSize());
        auto mValid = reorderWeight(backend(), mCommon, splitWeight.get(), mWeightInt8, split, s);
        if(!mValid) {
            return NOT_SUPPORT;
        }

        auto core = static_cast<CPUBackend*>(backend())->int8Functions();
        
        int UNIT, SRC_UNIT, DST_XUNIT;
        core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
        
        auto im2ColProcess = core->chooseIm2Col(&mIm2ColParamter, input->channel());

        const int outputPlaneLen = output->height() * output->width();
        const int dstZStep = outputPlaneLen * UNIT * output->batch();
        const int inputPlaneLen = input->width() * input->height();

        const int batch = input->batch();
        const int ocDiv4 = UP_DIV(output->channel()/split, UNIT);
        const auto kernelCountUnitDouble = mIm2ColParamter.kernelCountUnit;

        const auto inputDataPtr = input->host<int8_t>();
        const auto weightDataPtr = mWeightInt8->host<int8_t>();
        
        auto im2colPtr           = mTempIm2ColBuffer->host<int8_t>();

        int obatch = output->batch();
        int ochannel = output->channel();
        int owidth = output->width();
        int oheight = output->height();

        int osize = obatch*ochannel*owidth*oheight;
        
        auto outputDataPtr       = acc_int32->host<int32_t>() + acc_int32->elementSize()/split*s;
        memset(outputDataPtr, 0, osize*4/split);
        
        const int in_bytes = 1; // int8_t or float
        const int out_bytes = 4;

        if (batch >= 4) {
            auto threadFunction = [&](int tId) {
                auto colAddr        = im2colPtr + tId * mTempIm2ColBuffer->stride(0);
                int total = batch;

                int begin = tId * (total/threads);
                int end = begin + total/threads;
                if(tId == threads-1 )
                    end = total;

                for (int bIndex = begin; bIndex < end; ++bIndex) {
                    const auto srcPtr = inputDataPtr + bIndex * UNIT * in_bytes * inputPlaneLen;
                    auto dstPtr       = outputDataPtr + bIndex * UNIT * outputPlaneLen;

                    for (int tIndex = 0; tIndex < mTileCount; tIndex += 1) {
                        const int xIndexStart  = tIndex * DST_XUNIT;
                        const int realDstCount = ALIMIN(outputPlaneLen - xIndexStart, DST_XUNIT);
                        // im2col
                        im2ColProcess(colAddr, srcPtr, 0, &mIm2ColParamter, xIndexStart, realDstCount);

                        auto outputInTilePtr = dstPtr + xIndexStart * UNIT;
                        mGemmKernel(outputInTilePtr, colAddr, weightDataPtr, kernelCountUnitDouble, dstZStep, ocDiv4, realDstCount);
                    }
                }
            };
            MNN_CONCURRENCY_BEGIN(tId, threads) {
                threadFunction((int)tId);
            }
            MNN_CONCURRENCY_END();
        } else {
            auto threadFunction = [&](int tId) {
                auto colAddr        = im2colPtr + tId * mTempIm2ColBuffer->stride(0);
                for (int bIndex = 0; bIndex < batch; ++bIndex) {
                    const auto srcPtr = inputDataPtr + bIndex * UNIT * in_bytes * inputPlaneLen;
                    auto dstPtr       = outputDataPtr + bIndex * UNIT * outputPlaneLen;

                    for (int tIndex = tId; tIndex < mTileCount; tIndex += mThreadNums) {
                        const int xIndexStart  = tIndex * DST_XUNIT;
                        const int realDstCount = ALIMIN(outputPlaneLen - xIndexStart, DST_XUNIT);
                        // im2col
                        im2ColProcess(colAddr, srcPtr, 0, &mIm2ColParamter, xIndexStart, realDstCount);

                        auto outputInTilePtr = dstPtr + xIndexStart * UNIT;
                        mGemmKernel(outputInTilePtr, colAddr, weightDataPtr, kernelCountUnitDouble, dstZStep, ocDiv4, realDstCount);
                    }
                }
            };
            MNN_CONCURRENCY_BEGIN(tId, mThreadNums) {
                threadFunction((int)tId);
            }
            MNN_CONCURRENCY_END();
        }

        
        auto output_int32_ptr = output_int32->host<int32_t>()+output_int32->elementSize()/split*s;

        int int32_bitwidth = NITI_RangeEstimate(outputDataPtr, osize/split);

        scale[s] = int32_bitwidth;

        auto output_ptr = tempOutput->host<int8_t>()+tempOutput->elementSize()/split*s; 
        if(int32_bitwidth == 0) {
            
            memset(output_ptr, 0, osize/split);
        } else {
            MNN_CONCURRENCY_BEGIN(tId, threads) {
                int total = obatch*ochannel*owidth*oheight/split;

                int begin = tId * (total/threads);
                int end = begin + total/threads;
                if(tId == threads-1 )
                    end = total;

                NITI_MNNPstoShiftInt32(outputDataPtr+begin, int32_bitwidth-3, output_int32_ptr+begin,
                    end - begin);
                
                for(int i=begin;i<end;i++) {
                    output_ptr[i] = output_int32_ptr[i];
                }
            }
            MNN_CONCURRENCY_END();
        }

    }

    int maxScale = scale[0];
    for(int s=1;s<split; s++) {
        if(maxScale < scale[s])
            maxScale = scale[s];
    }
    
    for(int s=0;s<split; s++) {
        if(maxScale > scale[s]) {
            auto rptr = tempOutput->host<int8_t>()+tempOutput->elementSize()/split*s;
            for(int r = 0; r < tempOutput->elementSize()/split; r++) {
                *(rptr + r) = *(rptr + r) / (1<< (maxScale - scale[s]));
            }
        }
    }
    

    
    memcpy(output->host<int8_t>(), tempOutput->host<int8_t>() , tempOutput->elementSize());


    delete []scale;

    
    return NO_ERROR;
}

class NITI_GradientSplitConv_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {

        auto convOp = op->main_as_NITI_CONV_Int8();
        return new NITI_GradientSplitConv_Int8(backend, convOp);
    }
};

REGISTER_CPU_OP_CREATOR(NITI_GradientSplitConv_Int8_Creator, OpType_NITI_GradientSplitCONV_Int8);

} // namespace MNN
