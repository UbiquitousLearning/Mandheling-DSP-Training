//
//  SizeComputer.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include <stdlib.h>
#include <mutex>
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
//#define MNN_DEBUG_TENSOR_SIZE
namespace MNN {
void registerShapeOps();
SizeComputerSuite* SizeComputerSuite::gInstance = nullptr;

SizeComputerSuite::~SizeComputerSuite() {
    for (auto& iter : mRegistry) {
        delete iter.second;
    }
}

void SizeComputerSuite::init() {
    if (nullptr != gInstance) {
        return;
    }
    gInstance = new SizeComputerSuite;
    registerShapeOps();
}

SizeComputerSuite* SizeComputerSuite::get() {
    return gInstance;
}

void SizeComputerSuite::insert(SizeComputer* t, OpType type) {
    mRegistry.insert(std::make_pair(type, t));
}

SizeComputer* SizeComputerSuite::search(OpType name) {
    auto iter = mRegistry.find(name);
    if (iter == mRegistry.end()) {
        return nullptr;
    }
    return iter->second;
}
float SizeComputer::onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                   const std::vector<Tensor*>& outputs) const {
    MNN_ASSERT(outputs.size() >= 1);
    return (float)outputs[0]->elementSize() / 1024.0f / 1024.0f;
}

float SizeComputer::computeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) {
    auto computeFactory = SizeComputerSuite::get();
    auto computer       = computeFactory->search(op->type());
    if (nullptr != computer) {
        return computer->onComputeFlops(op, inputs, outputs);
    }
    if (op->type() == OpType_While && op->main_type() == OpParameter_LoopParam) {
        auto sumFlops = 0.0f;
        auto loop = op->main_as_LoopParam();
        auto cmdSize = loop->commands()->size();
        for (int i=0; i<cmdSize; ++i) {
            auto cmd = loop->commands()->GetAs<RegionCommand>(i);
            auto size = cmd->size()->data();
            sumFlops += (float)size[0] * (float)size[1] * (float)size[2] / 1024.0f / 1024.0f;
        }
        return sumFlops * (float)loop->loopNumber();
    }
    auto sumFlops = 0.0f;
    for (auto output : outputs) {
        sumFlops += (float)output->elementSize() / 1024.0f / 1024.0f;
    }
    return sumFlops;
}

bool SizeComputer::computeOutputSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                     const std::vector<Tensor*>& outputs) {
    auto computeFactory = SizeComputerSuite::get();
    // When op is nullptr, it means a copy op
    if (nullptr != op) {
        // Don't support compute shape for control flow op
        if (op->type() == OpType_While || op->type() == OpType_If) {
            return false;
        }
        // Check -1 input
        for (auto& t : inputs) {
            for (int i=0; i < t->dimensions(); ++i) {
                if (t->length(i) < 0) {
                    return false;
                }
            }
        }
        auto computer = computeFactory->search(op->type());
        if (nullptr != computer) {
            bool ret = computer->onComputeSize(op, inputs, outputs);
#ifdef MNN_DEBUG_TENSOR_SIZE
            if (op->name() != nullptr) {
                MNN_PRINT("\t===> compute shape: %s, [%s]\n", op->name()->c_str(), MNN::EnumNameOpType(op->type()));
            } else {
                MNN_PRINT("\t===> compute shape:[%s]\n", MNN::EnumNameOpType(op->type()));
            }
            if (inputs.size()) {
                MNN_PRINT("Inputs:\n");
                for (auto o : inputs) {
                    MNN_PRINT("\tformat=%d\t", TensorUtils::getDescribe(o)->dimensionFormat);
                    if (o->dimensions() == 0) {
                        MNN_PRINT("\t*Scalar*");
                    }
                    for (int i = 0; i < o->dimensions(); ++i) {
                        MNN_PRINT("%d, ", o->length(i));
                    }
                    MNN_PRINT("\n");
                }
            }
            MNN_PRINT("Outputs:\n");
            for (auto o : outputs) {
                MNN_PRINT("\tformat=%d\t", TensorUtils::getDescribe(o)->dimensionFormat);
                if (o->dimensions() == 0) {
                    MNN_PRINT("\t*Scalar*");
                }
                for (int i = 0; i < o->dimensions(); ++i) {
                    MNN_PRINT("%d, ", o->length(i));
                }
                MNN_PRINT("\n");
            }
#endif
            return ret;
        }
    }

    // Default Set to the same
    if (inputs.size() >= 1 && outputs.size() == 1) {
        if (inputs[0] == outputs[0]) {
            return true;
        }
        const auto& ib = inputs[0]->buffer();
        auto& ob       = outputs[0]->buffer();
        memcpy(ob.dim, ib.dim, sizeof(halide_dimension_t) * ib.dimensions);
        ob.dimensions                                         = ib.dimensions;
        ob.type                                               = ib.type;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        return true;
    }
    // Not Support
    MNN_PRINT("Can't compute size for %d, name=%s\n", op->type(), op->name() ? op->name()->c_str() : "");

    return false;
}

std::vector<int> SizeComputer::needInputContent(const MNN::Op* op, int inputSize) {
    auto computeFactory = SizeComputerSuite::get();
    // When op is nullptr, it means a copy op
    if (nullptr != op) {
        // when hasOutputShape = true, deconv last is outputShape
        if (op->type() == OpType_Deconvolution && op->main_as_Convolution2D() && op->main_as_Convolution2D()->common()) {
            if (op->main_as_Convolution2D()->common()->hasOutputShape()) {
                return std::vector<int>{ inputSize - 1 };
            }
        }
        auto computer = computeFactory->search(op->type());
        if (nullptr != computer) {
            return computer->mNeedContentInputIndex;
        }
    }
    return std::vector<int>{};
}
bool SizeComputer::computeBroadCastDims(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) {
    int maxDimensions = inputs[0]->dimensions();
    int maxIndex = 0;
    for (int index=1; index < inputs.size(); ++index) {
        if (inputs[index]->dimensions() > maxDimensions) {
            maxDimensions = inputs[index]->dimensions();
            maxIndex = index;
        }
    }
    int outputDims[MNN_MAX_TENSOR_DIM];
    for (int i = 0; i < maxDimensions; i++) {
        outputDims[i] = inputs[maxIndex]->length(i);
    }
    for (int index=0; index < inputs.size(); ++index) {
        if (index == maxIndex) {
            continue;
        }
        auto input1 = inputs[index];
        auto input0 = inputs[maxIndex];
        const int diffDimension = maxDimensions - input1->dimensions();
        for (int i = diffDimension; i < maxDimensions; i++) {
            const int input1Index = i - diffDimension;
            int dim1 = input1->buffer().dim[input1Index].extent;
            if (dim1 != outputDims[i] && (dim1 != 1 && outputDims[i] != 1)) {
                MNN_ERROR("Broad cast error, dim1 = %d, dim2 = %d\n", dim1, outputDims[i]);
                return false;
            }
            if (dim1 == outputDims[i]) {
                continue;
            }
            if (dim1 != outputDims[i] && (dim1 == 1 || outputDims[i] == 1)) {
                outputDims[i] = outputDims[i] * dim1;
            } else {
                return false;
            }
        }
    }
    auto& ob       = outputs[0]->buffer();
    ob.dimensions = maxDimensions;
    for (int i = 0; i < maxDimensions; i++) {
        ob.dim[i].extent = outputDims[i];
    }
    return true;
}
} // namespace MNN
