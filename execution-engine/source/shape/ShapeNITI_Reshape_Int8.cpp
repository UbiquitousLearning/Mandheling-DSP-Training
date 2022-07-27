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
class NITI_ReshapeSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {

        auto output = outputs[0];
        auto input  = inputs[0];

        auto dims = op->main_as_Reshape()->dims();

        output->buffer().dimensions = input->buffer().dimensions;
        output->buffer().type = input->getType();
        output->buffer().dim[0].extent = input->buffer().dim[0].extent;
        output->buffer().dim[1].extent = dims->data()[1];
        output->buffer().dim[2].extent = dims->data()[2];
        output->buffer().dim[3].extent = dims->data()[3];
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;

        return true;
    }
};

REGISTER_SHAPE(NITI_ReshapeSizeComputer, OpType_NITI_DSP_RESHAPE_Int8);
REGISTER_SHAPE(NITI_ReshapeSizeComputer, OpType_NITI_DSP_RESHAPEGrad_Int8);

} // namespace MNN
