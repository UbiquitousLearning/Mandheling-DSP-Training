//
//  NITI_DSPMaxPoolGradRef_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/11/19.
//  
//

#include "backend/cpu/NITI_DSPMaxPoolGradRef_Int8.hpp"
#include "core/Macro.h"
#include "math/Vec.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"

namespace MNN {

ErrorCode NITI_DSPMaxPoolGradRef_Int8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto origin       = inputs[0];
    auto outputOrigin = inputs[1];
    auto outputDiff    = inputs[2];
    auto output   = outputs[0];

    // 基于 ih iw进行遍历
    auto iw = origin->height();
    auto ih = origin->batch();

    auto ib = origin->width();
    auto ic = origin->channel();

    int8_t* originPtr = origin->host<int8_t>();
    int8_t* outputOriginPtr = outputOrigin->host<int8_t>();

    int8_t* outputDiffPtr = outputDiff->host<int8_t>();
    int8_t* outputPtr = output->host<int8_t>();

    for(int i=0;i<ih;i+=mStrideX) {
        for(int j=0; j<iw; j+=mStrideY) {

            int32_t offset = i*iw + j;
            
            int32_t bc = ib*ic;

            for(int m=0;m<bc/128;m++) {
                int8_t outputoriginbuf[128];
                int8_t originbuf[128];

                int8_t samebuf[128];
                int8_t finishbuf[128];

                int8_t outputdiffbuf[128];
                int8_t outputbuf[128];

                for(int n=0;n<128;n++) {
                    outputoriginbuf[n] = outputOriginPtr[offset*bc/mKernelX/mKernelY + m*128 + n];
                    outputdiffbuf[n] = outputDiffPtr[offset*bc/mKernelX/mKernelY + m*128 + n];
                    finishbuf[n] = 0;
                }

                for(int ky = 0; ky<mKernelY; ky++) {

                    for(int kx = 0; kx<mKernelX; kx++) {

                        int32_t koffset = ky*iw + kx;

                        for(int n=0;n<128;n++) {
                            originbuf[n] = originPtr[offset*bc + koffset*bc + m*128 + n];
                        }
                        
                        for(int n=0;n<128;n++) {
                            if(outputoriginbuf[n] == originbuf[n])
                                samebuf[n] = 1;
                            else
                                samebuf[n] = 0;
                        }

                        for(int n=0;n<128;n++) {
                            if(samebuf[n] == 1 && finishbuf[n] == 0) {
                                outputbuf[n] = outputdiffbuf[n];
                                finishbuf[n] = 1;
                            } else {
                                outputbuf[n] = 0;
                            }

                        }

                        for(int n=0;n<128;n++) {
                            outputPtr[offset*bc + koffset*bc + m*128 + n] = outputbuf[n];
                        }
                    }

                    
                }

                
            }

        }
    }

    

    return NO_ERROR;
}


class NITI_DSPMaxPoolGradRef_Int8Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto pool = op->main_as_NITI_Pool_Int8();
        return new NITI_DSPMaxPoolGradRef_Int8(backend, op->main_as_NITI_Pool_Int8());
    }
};

REGISTER_CPU_OP_CREATOR(NITI_DSPMaxPoolGradRef_Int8Creator, OpType_NITI_DSP_MAXPOOLGRAD_REF_Int8);
} // namespace MNN
