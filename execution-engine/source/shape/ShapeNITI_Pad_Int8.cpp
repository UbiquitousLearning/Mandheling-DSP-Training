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
class NITI_PadSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        
        auto output = outputs[0];
        auto input  = inputs[0];

        MNN_ASSERT(TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NCHW);

        auto pad = op->main_as_NITI_PAD_Int8();

        output->buffer().dimensions = input->buffer().dimensions;
        output->buffer().type = input->getType();
        output->buffer().dim[0].extent = input->buffer().dim[0].extent;
        output->buffer().dim[1].extent = input->buffer().dim[1].extent;
        output->buffer().dim[2].extent = input->buffer().dim[2].extent+2*pad->pad();
        output->buffer().dim[3].extent = input->buffer().dim[3].extent+2*pad->pad();

        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;

        outputs[0]->buffer().type = halide_type_of<int8_t>();

        return true;
    }
};

class NITI_DSPPadSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        
        auto output = outputs[0];
        auto input  = inputs[0];

        MNN_ASSERT(TensorUtils::getDescribe(input)->dimensionFormat == MNN_DATA_FORMAT_NHWC);

        auto pad = op->main_as_NITI_PAD_Int8();

        output->buffer().dimensions = input->buffer().dimensions;
        output->buffer().type = input->getType();
        output->buffer().dim[0].extent = input->buffer().dim[0].extent;
        output->buffer().dim[1].extent = input->buffer().dim[1].extent+2*pad->pad();
        output->buffer().dim[2].extent = input->buffer().dim[2].extent+2*pad->pad();
        output->buffer().dim[3].extent = input->buffer().dim[3].extent;

        MNN_PRINT("PAD SHAPE %d %d \n", output->buffer().dim[2].extent, output->buffer().dim[3].extent);
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;

        outputs[0]->buffer().type = halide_type_of<int8_t>();

        return true;
    }
};

REGISTER_SHAPE(NITI_PadSizeComputer, OpType_NITI_PAD_Int8);
REGISTER_SHAPE(NITI_DSPPadSizeComputer, OpType_NITI_DSP_PAD_Int8);

} // namespace MNN
