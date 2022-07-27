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
class NITI_C1Computer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {

        auto output = outputs[0];
        auto input  = inputs[0];


        TensorUtils::copyShape(input, output, true);

        TensorUtils::getDescribe(output)->dimensionFormat = MNN_DATA_FORMAT_NCHW;

        return true;
    }
};

REGISTER_SHAPE(NITI_C1Computer, OpType_NITI_ConvertChannel1NC4HW4ToNCHW_Int8);

} // namespace MNN
