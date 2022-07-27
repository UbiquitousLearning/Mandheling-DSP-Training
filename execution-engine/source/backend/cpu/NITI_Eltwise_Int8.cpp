//
//  NITI_Eltwise_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/8/17.
//  
//

#include "backend/cpu/NITI_Eltwise_Int8.hpp"
#include "backend/cpu/CPUBackend.hpp"

namespace MNN {

NITI_Eltwise_Int8::~NITI_Eltwise_Int8() {
    // Do nothing
}
NITI_Eltwise_Int8::NITI_Eltwise_Int8(Backend* backend, const Op *op) : MNN::Execution(backend) {

}


ErrorCode NITI_Eltwise_Int8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return NO_ERROR;
}

ErrorCode NITI_Eltwise_Int8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return NO_ERROR;
}

class NITI_Eltwise_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {

        return new NITI_Eltwise_Int8(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(NITI_Eltwise_Int8_Creator, OpType_NITI_ELTWISE_Int8);

} // namespace MNN
