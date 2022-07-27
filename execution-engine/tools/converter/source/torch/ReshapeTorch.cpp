//
//  ReshapeTorch.cpp
//  MNNConverter
//
//  Created by MNN on 2021/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "torchOpConverter.hpp"

DECLARE_OP_CONVERTER(ReshapeTorch);

MNN::OpType ReshapeTorch::opType() {
    return MNN::OpType_Reshape;
}
MNN::OpParameter ReshapeTorch::type() {
    return MNN::OpParameter_Reshape;
}
std::vector<int> ReshapeTorch::inputTensorIdx() {
    return {0, 1};
}

void ReshapeTorch::run(MNN::OpT* dstOp, const torch::jit::Node* node, torchContext* context) {
    auto param = new MNN::ReshapeT;
    dstOp->main.value = param;
}

REGISTER_CONVERTER(ReshapeTorch, reshape);
REGISTER_CONVERTER(ReshapeTorch, view);
