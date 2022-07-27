//
//  ShapePool.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <math.h>

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {
class NITI_SoftmaxSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {

        auto output = outputs[0];
        auto input  = inputs[0];

        TensorUtils::copyShape(input, output, true);

        outputs[0]->buffer().type = halide_type_of<int32_t>();

        return true;
    }
};

class NITI_LossSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {

        // MNN_PRINT("NITI_LossSizeComputer onComputeSize\n");
        
        auto output = outputs[0];
        auto input  = inputs[0];

        TensorUtils::copyShape(input, output, true);

        output->buffer().dimensions = 4;
        outputs[0]->buffer().dim[0].extent = 1;
        outputs[0]->buffer().dim[1].extent = 1;
        outputs[0]->buffer().dim[2].extent = 1;
        outputs[0]->buffer().dim[3].extent = 1;

        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;

        outputs[0]->buffer().type = halide_type_of<float>();

        return true;
    }
};

class NITI_LossGradSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        
        auto output = outputs[0];
        auto input  = inputs[0];

        TensorUtils::copyShape(input, output, true);

        outputs[0]->buffer().type = halide_type_of<int8_t>();

        return true;
    }
};

class NITI_SoftmaxGradSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        
        auto output = outputs[0];
        auto input  = inputs[0];

        TensorUtils::copyShape(input, output, true);

        outputs[0]->buffer().type = halide_type_of<int8_t>();

        return true;
    }
};

REGISTER_SHAPE(NITI_SoftmaxSizeComputer, OpType_NITI_SOFTMAX_Int8);
REGISTER_SHAPE(NITI_LossSizeComputer, OpType_NITI_LOSS_Int8);
REGISTER_SHAPE(NITI_LossGradSizeComputer, OpType_NITI_LOSS_Grad_Int8);
REGISTER_SHAPE(NITI_SoftmaxGradSizeComputer, OpType_NITI_SOFTMAX_Grad_Int8);

} // namespace MNN
