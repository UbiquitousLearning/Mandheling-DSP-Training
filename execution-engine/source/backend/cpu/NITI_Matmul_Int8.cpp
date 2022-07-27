//
//  NITI_Matmul_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/8/17.
//  
//

#include "backend/cpu/NITI_Matmul_Int8.hpp"
#include "backend/cpu/CPUBackend.hpp"

#include "core/Concurrency.h"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/TensorUtils.hpp"
#include <math.h>


namespace MNN {

bool NITI_Matmul_Int8::reorderWeight(Backend* bn, int oc, int ic,
                          Tensor* weightOrigin,
                          std::shared_ptr<Tensor>& weight) {
    auto core = static_cast<CPUBackend*>(bn)->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    // reorder weight, [oc, ic, k^2] => [oc/unit, ((ic/unit)*k^2)/(src_unit/unit), unit(oc), (src_unit/unit), unit(ic)]
    int kernelCount = 1 * 1;

    auto weightSrc = weightOrigin->host<int8_t>();
    auto weightDst = weight->host<int8_t>();
    memset(weightDst, 0, weight->size());
    for (int k = 0; k < kernelCount; ++k) {
        const auto srcK = weightSrc + k;
        for (int y = 0; y < ic; ++y) {
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
    return true;
}

NITI_Matmul_Int8::~NITI_Matmul_Int8() {
    // Do nothing
}
NITI_Matmul_Int8::NITI_Matmul_Int8(Backend* backend, bool transposeA, bool transposeB, bool transposeC) 
    : Execution(backend), mTransposeA(transposeA), mTransposeB(transposeB), mTransposeC(transposeC){
}


ErrorCode NITI_Matmul_Int8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {

    auto gradient = inputs[0];
    auto input = inputs[0];
    auto output = outputs[0];
    auto outdiff = inputs[1];

    auto core = static_cast<CPUBackend*>(backend())->int8Functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);

    const auto kernelCount = 1* 1;
    const auto srcCountUnit = UP_DIV(input->channel(), UNIT);
    const auto totalKernelCountD8Div2 = UP_DIV(srcCountUnit * kernelCount, SRC_UNIT / UNIT);

    mIm2ColParamter.dilateX         = 1;
    mIm2ColParamter.dilateY         = 1;
    mIm2ColParamter.strideX         = 1;
    mIm2ColParamter.strideY         = 1;
    mIm2ColParamter.padX            = 0;
    mIm2ColParamter.padY            = 0;
    mIm2ColParamter.icDiv4          = srcCountUnit;
    mIm2ColParamter.kernelX         = 1;
    mIm2ColParamter.kernelY         = 1;
    mIm2ColParamter.kernelCountUnit = totalKernelCountD8Div2;

    mIm2ColParamter.ih = 1;
    mIm2ColParamter.iw = 1;
    mIm2ColParamter.oh = 1;
    mIm2ColParamter.ow = 1;
    mIm2ColParamter.srcZStep = 1* UNIT * input->batch();
    mIm2ColParamter.srcYStep = 1 * UNIT;

    mTileCount        = UP_DIV(1 * 1, DST_XUNIT);
    const int threads = std::max(static_cast<CPUBackend*>(backend())->threadNumber(), 1);
    mThreadNums       = std::min(threads, mTileCount);

    // set im2col tensor info
    mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({mThreadNums, DST_XUNIT, outdiff->length(1) * SRC_UNIT}));
    bool success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    gradientInt8Transpose.reset(Tensor::createDevice<int32_t>({output->batch(), output->channel(),  1, 1}));
    success = backend()->onAcquireBuffer(gradientInt8Transpose.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    int obatch = output->batch();
    int ochannel = output->channel();
    int owidth = 1;
    int oheight = 1;

    acc_int32.reset(Tensor::createDevice<int32_t>({obatch, ochannel, 1, 1}));
    success = backend()->onAcquireBuffer(acc_int32.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    int oc = output->channel(), ic = input->channel();
    std::vector<int> shape = {UP_DIV(oc, UNIT), UP_DIV(UP_DIV(ic, UNIT) * kernelCount, SRC_UNIT / UNIT), UNIT, SRC_UNIT};
    
    outdiffReorder.reset(Tensor::createDevice<int8_t>(shape));
    
    bool succ = backend()->onAcquireBuffer(outdiffReorder.get(), Backend::STATIC);
    if (!succ) {
        MNN_ERROR("Memory not enough");
        return OUT_OF_MEMORY;
    }

    backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode NITI_Matmul_Int8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    
    auto outdiff = inputs[1];
    auto outdiffDataPtr = outdiff->host<int8_t>();

    auto output = outputs[0];
    auto input = inputs[0];

    auto core = static_cast<CPUBackend*>(backend())->int8Functions();


    auto mValid = reorderWeight(backend(), output->channel(), input->channel(), outdiff, outdiffReorder);
    if(!mValid) {
        return NOT_SUPPORT;
    }

    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);

    auto im2ColProcess = core->chooseIm2Col(&mIm2ColParamter, input->channel());

    const int outputPlaneLen = 1 * 1;
    const int dstZStep = outputPlaneLen * UNIT * output->batch();
    const int inputPlaneLen = 1 * 1;

    const int batch = input->batch();
    const int ocDiv4 = UP_DIV(output->channel(), UNIT);

    const auto kernelCount = 1 * 1;
    const auto srcCountUnit = UP_DIV(input->channel(), UNIT);
    const auto totalKernelCountD8Div2 = UP_DIV(srcCountUnit * kernelCount, SRC_UNIT / UNIT);

    const auto kernelCountUnitDouble = totalKernelCountD8Div2;

    const auto inputDataPtr = input->host<int8_t>();
    const auto weightDataPtr = outdiffReorder->host<int8_t>();
    
    auto im2colPtr           = mTempIm2ColBuffer->host<int8_t>();

    int obatch = output->batch();
    int ochannel = output->channel();
    int owidth = 1;
    int oheight = 1;
    int osize = obatch*ochannel*owidth*oheight;

    auto outputDataPtr       = acc_int32->host<int32_t>();
    memset(outputDataPtr, 0, osize*4);
    
    const int in_bytes = 1; // int8_t or float
    const int out_bytes = 4;

    auto threadFunction = [&](int tId) {
        auto colAddr        = im2colPtr + tId * mTempIm2ColBuffer->stride(0);
        for (int bIndex = 0; bIndex < batch; ++bIndex) {
            const auto srcPtr = inputDataPtr + bIndex * UNIT * in_bytes * inputPlaneLen;
            auto dstPtr       = outputDataPtr + bIndex * UNIT * outputPlaneLen;

            for (int tIndex = tId; tIndex < mTileCount; tIndex += mThreadNums) {
                const int xIndexStart  = tIndex * DST_XUNIT;
                const int realDstCount = ALIMIN(outputPlaneLen - xIndexStart, DST_XUNIT);

                im2ColProcess(colAddr, srcPtr, 0, &mIm2ColParamter, xIndexStart, realDstCount);

                auto outputInTilePtr = dstPtr + xIndexStart * UNIT;
                core->NITI_Int8GemmKernel(outputInTilePtr, colAddr, weightDataPtr, kernelCountUnitDouble, dstZStep, ocDiv4, realDstCount);
            }
        }
    };
    MNN_CONCURRENCY_BEGIN(tId, mThreadNums) {
        threadFunction((int)tId);
    }
    MNN_CONCURRENCY_END();

    auto gradientInt8TransposePtr = gradientInt8Transpose->host<int32_t>();


    int int32_bitwidth = NITI_RangeEstimate(outputDataPtr, osize);
    auto output_ptr = output->host<int8_t>(); 
    if(int32_bitwidth == 0) {
        
        memset(output_ptr, 0, osize);
    } else {
        NITI_MNNPstoShiftInt32(outputDataPtr, int32_bitwidth-3, gradientInt8TransposePtr,
            obatch*ochannel*1*1);
        
        for(int i=0;i<osize;i++) {
            output_ptr[i] = (int8_t)gradientInt8TransposePtr[i];
        }
    }

    return NO_ERROR;
}

class NITI_Matmul_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {

        auto param = op->main_as_MatMul();
        return new NITI_Matmul_Int8(backend, param->transposeA(), param->transposeB(), true);
    }
};

REGISTER_CPU_OP_CREATOR(NITI_Matmul_Int8_Creator, OpType_NITI_MatMul_Int8);

} // namespace MNN
