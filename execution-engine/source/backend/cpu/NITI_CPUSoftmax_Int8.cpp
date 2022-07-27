//
//  NITI_CPUSoftmax_Int8.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>
#include "backend/cpu/NITI_CPUSoftmax_Int8.hpp"
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

ErrorCode NITI_CPUSoftmax_Int8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // MNN_PRINT("NITI_CPU_Softmax resize begin\n");

    auto input           = inputs[0];
    const int dimensions = input->buffer().dimensions;
    int axis = mAxis;
    if (axis < 0) {
        axis += dimensions;
    }

    auto output = outputs[0];

    s.reset(Tensor::createDevice<int32_t>({output->batch(), output->channel(), output->height(), output->width()}));
    bool success = backend()->onAcquireBuffer(s.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    out_max.reset(Tensor::createDevice<int32_t>({output->batch(), 1, output->height(), output->width()}));
    success = backend()->onAcquireBuffer(out_max.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    return NO_ERROR;
}

ErrorCode NITI_CPUSoftmax_Int8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    
    int8_t* inputDataPtr = inputs[0]->host<int8_t>();
    int8_t ascale = *(inputs[1]->host<int8_t>());

    int32_t* outputDataPtr = outputs[0]->host<int32_t>();

    int* sPtr = s->host<int32_t>();
    int* out_maxPtr = out_max->host<int32_t>();

    int ib = inputs[0]->batch();
    int ic = inputs[0]->channel();
    int ih = inputs[0]->height();
    int iw = inputs[0]->width();

    int inputSize = ib*ic*ih*iw;

    if(ascale > -7) {

        if(ascale >= 0) {
            for(int i=0;i<inputSize;i++) {
                int temp = (int)inputDataPtr[i] * 47274;
                temp = temp >> 15;
                sPtr[i] = temp << ascale;
            }

        } else {
            int shift = 15 - ascale;
            for(int i=0;i<inputSize;i++) {
                int temp = (int)inputDataPtr[i] * 47274;
                sPtr[i] = temp >> shift;
            }
        }

        for(int i=0;i<ib;i++) {
            int max = sPtr[i*ic];
            for(int j=1;j<ic;j++) {
                if(max < sPtr[i*ic+j])
                    max = sPtr[i*ic+j];
            }
            out_maxPtr[i] = max - 10;
        }

        for(int i=0;i<ib;i++) {
            for(int j=0;j<ic;j++) {
                int temp = sPtr[i*ic+j];
                temp -= out_maxPtr[i];
                temp = (temp > 0)? temp : 0;
                outputDataPtr[i*ic+j] = (1<<temp) - 1;
            }
        }
        

    } else {
        int base = 1 << (1 - 2*ascale);
        int shiftbase = 1 << (1 - ascale);

        for(int i=0;i<inputSize;i++) {
            int temp = inputDataPtr[i];
            outputDataPtr[i] = base + temp*shiftbase + temp * temp;
        }
    }

    return NO_ERROR;
}

NITI_CPUSoftmax_Int8::NITI_CPUSoftmax_Int8(Backend *b, int axis) : MNN::Execution(b), mAxis(axis) {
    // nothing to do
}

Execution* NITI_CPUSoftmax_Int8::create(const MNN::Op *op, Backend *backend) {
    auto axis = op->main_as_Axis()->axis();
    return new NITI_CPUSoftmax_Int8(backend, axis);
}

class NITI_CPUSoftmax_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return NITI_CPUSoftmax_Int8::create(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(NITI_CPUSoftmax_Int8_Creator, OpType_NITI_SOFTMAX_Int8);

} // namespace MNN
