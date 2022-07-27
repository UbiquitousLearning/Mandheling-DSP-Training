//
//  NITI_DSPNop_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/12/17.
//
//

#include <math.h>
#include "backend/cpu/NITI_DSPNop_Int8.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "CPUTensorConvert.hpp"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

#include "MNN/hexagon/hexagon_nn/hexagon_nn_ops.h"
#include "MNN/hexagon/hexagon_nn/hexagon_nn.h"
#include "MNN/hexagon/hexagon_nn/HexagonRunningUtils.hpp"

namespace MNN {


ErrorCode NITI_DSPNop_Int8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    return NO_ERROR;
}

ErrorCode NITI_DSPNop_Int8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    if(getDSPExecuteMode())
        return GlobalExecute(inputs,outputs);

    memcpy(outputs[0]->host<int8_t>(), inputs[0]->host<int8_t>(), inputs[0]->elementSize());
    
    return NO_ERROR;
}

ErrorCode NITI_DSPNop_Int8::GlobalExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    if(static_cast<CPUBackend*>(backend())->build_graph) {
        return NO_ERROR;
    }

    int input_op_id = backend()->get_Op_id(inputs[0], OpType_NITI_DSP_NOP_Int8);

    static_cast<CPUBackend*>(backend())->insert_Op_id(outputs[0], input_op_id );

    return NO_ERROR;
}

NITI_DSPNop_Int8::NITI_DSPNop_Int8(Backend *b, const MNN::Op *op) : MNN::Execution(b) {
    // nothing to do
    mOp = op;
}


class NITI_DSPNop_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new NITI_DSPNop_Int8(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR(NITI_DSPNop_Int8_Creator, OpType_NITI_DSP_NOP_Int8);

} // namespace MNN
