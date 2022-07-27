//
//  NITI_PstoShiftInt32_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/8/17.
//  
//

#include "backend/cpu/NITI_PstoShiftInt32_Int8.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/CPUBackend.hpp"

namespace MNN {

NITI_PstoShiftInt32_Int8::~NITI_PstoShiftInt32_Int8() {
    // Do nothing
}
NITI_PstoShiftInt32_Int8::NITI_PstoShiftInt32_Int8(Backend* backend, const Op *op) : MNN::Execution(backend) {

}


ErrorCode NITI_PstoShiftInt32_Int8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return NO_ERROR;
}

ErrorCode NITI_PstoShiftInt32_Int8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto inputTensor        = inputs[0];
    auto shiftTensor        = inputs[1];
    auto outputTensor       = outputs[0];
    auto inputDataPtr = inputTensor->host<int32_t>();
    auto shiftDataPtr = inputTensor->host<int32_t>();
    auto outputDataPtr      = outputTensor->host<int32_t>();

    auto input  = inputs[0];
    
    int obatch = input->batch();
    int ochannel = input->channel();
    int owidth = input->width();
    int oheight = input->height();

    NITI_MNNPstoShiftInt32(inputDataPtr, *shiftDataPtr, 
        outputDataPtr, obatch*ochannel*owidth*oheight);

    return NO_ERROR;
}

class NITI_PstoShiftInt32_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {

        return new NITI_PstoShiftInt32_Int8(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(NITI_PstoShiftInt32_Int8_Creator, OpType_NITI_PstoShiftInt32toInt8);

} // namespace MNN
