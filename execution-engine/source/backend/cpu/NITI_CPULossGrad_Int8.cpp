//
//  NITI_CPULoss_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/08/28.
//  
//

#include <math.h>
#include "backend/cpu/NITI_CPULossGrad_Int8.hpp"
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

ErrorCode NITI_CPULossGrad_Int8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    
    auto input = inputs[0];
    auto output = outputs[0];

    s.reset(Tensor::createDevice<int64_t>({input->batch(), input->channel(), input->height(), input->width()}));
    bool success = backend()->onAcquireBuffer(s.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    out_max.reset(Tensor::createDevice<int64_t>({input->batch(), 1, input->height(), input->width()}));
    success = backend()->onAcquireBuffer(out_max.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    output_softmax.reset(Tensor::createDevice<int64_t>({input->batch(), input->channel(), input->height(), input->width()}));
    success = backend()->onAcquireBuffer(output_softmax.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    out_sum.reset(Tensor::createDevice<int64_t>({output->batch(), 1, output->height(), output->width()}));
    success = backend()->onAcquireBuffer(out_sum.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    target_max.reset(Tensor::createDevice<int32_t>({output->batch(), 1, output->height(), output->width()}));
    success = backend()->onAcquireBuffer(target_max.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    out_grad.reset(Tensor::createDevice<int64_t>({output->batch(), output->channel(), output->height(), output->width()}));
    success = backend()->onAcquireBuffer(out_grad.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    out_grad_final.reset(Tensor::createDevice<int32_t>({output->batch(), output->channel(), output->height(), output->width()}));
    success = backend()->onAcquireBuffer(out_grad_final.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    backend()->onReleaseBuffer(s.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(out_max.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(output_softmax.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(out_sum.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(target_max.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(out_grad.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(out_grad_final.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode NITI_CPULossGrad_Int8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    int8_t* inputDataPtr = inputs[0]->host<int8_t>();
    int8_t ascale = *(inputs[1]->host<int8_t>());

    int64_t* sPtr = s->host<int64_t>();
    int64_t* out_maxPtr = out_max->host<int64_t>();
    int64_t* outputDataPtr = output_softmax->host<int64_t>();

    int ib = inputs[0]->batch();
    int ic = inputs[0]->channel();
    int ih = inputs[0]->height();
    int iw = inputs[0]->width();

    int inputSize = ib*ic*ih*iw;
        

    if(ascale > -7) {

        if(ascale >= 0) {
            for(int i=0;i<inputSize;i++) {
                int64_t temp = (int64_t)inputDataPtr[i] * 47274;
                temp = temp / (1 << 15);
                sPtr[i] = temp * (1 << ascale);
            }

        } else {
            for(int i=0;i<inputSize;i++) {
                int64_t temp = (int64_t)inputDataPtr[i] * 47274;
                temp = temp / (1 << 15);
                sPtr[i] = temp / ( 1 << (-ascale));
            }
        }

        for(int i=0;i<ib;i++) {
            int64_t max = sPtr[i*ic];
            for(int j=1;j<ic;j++) {
                if(max < sPtr[i*ic+j])
                    max = sPtr[i*ic+j];
            }
            out_maxPtr[i] = max - 10;
        }

        for(int i=0;i<ib;i++) {
            for(int j=0;j<ic;j++) {
                int64_t temp = sPtr[i*ic+j];
                temp -= out_maxPtr[i];
                temp = (temp > 0)? temp : 0;
                outputDataPtr[i*ic+j] = (1<<temp) - 1;
            }
        }
        

    } else {
        int64_t base = 1 << (1 - 2*(int64_t)ascale);
        int64_t shiftbase = 1 << (1 - (int64_t)ascale);

        for(int i=0;i<inputSize;i++) {
            int64_t temp = (int64_t)inputDataPtr[i];
            outputDataPtr[i] = base + temp*shiftbase + temp * temp;
        }
    }

    int64_t* out_sumPtr = out_sum->host<int64_t>();
    int32_t* target_maxPtr = target_max->host<int32_t>();
    int64_t* out_gradPtr = out_grad->host<int64_t>();

    int32_t* target_Ptr = inputs[2]->host<int32_t>();
    int8_t*  outPtr = outputs[0]->host<int8_t>();

    for(int i=0;i<ib;i++) {
        int64_t sum = 0;
        for(int j=0;j<ic;j++) {
            sum += outputDataPtr[i*ic+j];
        }
        out_sumPtr[i] = sum;
    }

    for(int i=0;i<ib;i++) {
        int64_t base = out_sumPtr[i];
        for(int j=0;j<ic;j++) {
            int64_t temp = outputDataPtr[i*ic+j];
            temp = temp * (1 << 11);
            temp = temp / base;
            out_gradPtr[i*ic+j] = temp;
        }
    }

    int tc = inputs[2]->elementSize()/inputs[2]->batch();

    for(int i = 0; i<ib;i++) {
        for(int j=0; j<tc; j++) {
            if(target_Ptr[i*tc+j] == 1) {
                target_maxPtr[i] = j;
                break;
            }
        }
    }

    for(int i=0;i<ib;i++) {
        int64_t sum = 0;
        for(int j=0;j<ic;j++) {
            sum += (int64_t)out_gradPtr[i*ic+j];
        }
        out_sumPtr[i] = sum;
    }

    int32_t* out_grad_finalPtr = out_grad_final->host<int32_t>();
    for(int i=0; i<ib; i++) {
        for(int j=0; j<ic; j++) {
            out_grad_finalPtr[i*ic+j] = (int32_t)out_gradPtr[i*ic+j];
        }
    } 
    for(int i=0; i<ib; i++) {
        out_grad_finalPtr[i*ic + target_maxPtr[i]] = (int32_t)(out_gradPtr[i*ic + target_maxPtr[i]] - out_sumPtr[i]);
    }

    NITI_MNNPstoShiftInt32ToInt8(out_grad_finalPtr, 4, outPtr, inputSize);

    return NO_ERROR;
}

NITI_CPULossGrad_Int8::NITI_CPULossGrad_Int8(Backend *b, const MNN::Op *op) : MNN::Execution(b), mOp(op) {
    // nothing to do
}

NITI_CPULossGrad_Int8::~NITI_CPULossGrad_Int8()  {
    // nothing to do
}


Execution* NITI_CPULossGrad_Int8::create(const MNN::Op *op, Backend *backend) {
    return new NITI_CPULossGrad_Int8(backend, op);
}

class NITI_CPULossGrad_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return NITI_CPULossGrad_Int8::create(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(NITI_CPULossGrad_Int8_Creator, OpType_NITI_LOSS_Grad_Int8);

} // namespace MNN
