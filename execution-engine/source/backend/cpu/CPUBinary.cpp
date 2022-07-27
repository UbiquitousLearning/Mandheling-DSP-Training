//
//  CPUBinary.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUBinary.hpp"
#include "CPUBackend.hpp"
#include "compute/CommonOptFunction.h"
#include "compute/ConvOpt.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/OpCommonUtils.hpp"
#include "BinaryUtils.hpp"
#include "math/Vec.hpp"
using Vec4 = MNN::Math::Vec<float, 4>;

namespace MNN {

ErrorCode CPUBinary::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const int input0DataCount = inputs[0]->elementSize();
    const int input1DataCount = inputs[1]->elementSize();
    if (input1DataCount == input0DataCount) {
        mNeedBroadcastIndex = -1;
        mTotalSize = input1DataCount;
    } else if (input0DataCount == 1) {
        mNeedBroadcastIndex = 0;
        mTotalSize = input1DataCount;
    } else {
        mNeedBroadcastIndex = 1;
        mTotalSize = input0DataCount;
    }
    MNN_ASSERT(mTotalSize == outputs[0]->elementSize());
    return NO_ERROR;
}

ErrorCode CPUBinary::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const int input0DataCount = ((CPUBackend*)backend())->getTensorSize(inputs[0]);
    const int input1DataCount = ((CPUBackend*)backend())->getTensorSize(inputs[1]);
//    inputs[0]->printShape();
//    inputs[1]->printShape();
//    MNN_PRINT("%d - %d\n", input0DataCount, input1DataCount);
    if (input1DataCount == input0DataCount) {
        mNeedBroadcastIndex = -1;
        mTotalSize = input1DataCount;
    } else if (input0DataCount == 1) {
        mNeedBroadcastIndex = 0;
        mTotalSize = input1DataCount;
    } else {
        mNeedBroadcastIndex = 1;
        mTotalSize = input0DataCount;
    }
    auto input  = inputs[0];
    auto input1 = inputs[1];
    auto output = outputs[0];
    
    auto schedule = ((CPUBackend*)backend())->multiThreadDivide(mTotalSize);
    auto input0Ptr = input->host<uint8_t>();
    auto input1Ptr = input1->host<uint8_t>();
    auto outputPtr = output->host<uint8_t>();
    int inpBytes = input->getType().bytes();
    int outBytes = output->getType().bytes();
    if (halide_type_float == input->getType().code) {
        inpBytes = static_cast<CPUBackend*>(backend())->functions()->bytes;
    }
    if (halide_type_float == output->getType().code) {
        outBytes = static_cast<CPUBackend*>(backend())->functions()->bytes;
    }
    auto precision = static_cast<CPUBackend*>(backend())->precisionMode();
    MNN_CONCURRENCY_BEGIN(tId, schedule.second) {
        int start = schedule.first * (int)tId;
        int realSize = schedule.first;
        if (tId == schedule.second -1 ) {
            realSize = mTotalSize - start;
        }
        if (realSize > 0) {
            auto inp0 = input0Ptr + start * inpBytes;
            auto inp1 = input1Ptr + start * inpBytes;
            if (mNeedBroadcastIndex == 0) {
                inp0 = input0Ptr;
            } else if (mNeedBroadcastIndex == 1) {
                inp1 = input1Ptr;
            }
            auto out = outputPtr + start * outBytes;
            mProc(out, inp0, inp1, realSize, mNeedBroadcastIndex);
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

MNNBinaryExecute CPUBinary::selectForFloat(int type) {
    auto vecFunction = selectVector<Vec4, 4>(type);
    if (nullptr != vecFunction) {
        return vecFunction;
    }
    switch (type) {
        case BinaryOpOperation_REALDIV:
            return execute<float, float, BinaryRealDiv<float, float, float>>;
        case BinaryOpOperation_FLOORDIV:
            return execute<float, float, BinaryFloorDiv<float, float, float>>;
        case BinaryOpOperation_FLOORMOD:
            return execute<float, float, BinaryFloorMod<float, float, float>>;
        case BinaryOpOperation_POW:
            return execute<float, float, BinaryPow<float, float, float>>;
        case BinaryOpOperation_ATAN2:
            return execute<float, float, BinaryAtan2<float, float, float>>;
        case BinaryOpOperation_MOD:
            return execute<float, float, BinaryMod<float, float, float>>;
        case BinaryOpOperation_GREATER:
            return execute<float, int32_t, BinaryGreater<float, float, int32_t>>;
        case BinaryOpOperation_LESS:
            return execute<float, int32_t, BinaryLess<float, float, int32_t>>;
        case BinaryOpOperation_LESS_EQUAL:
            return execute<float, int32_t, BinaryLessEqual<float, float, int32_t>>;
        case BinaryOpOperation_GREATER_EQUAL:
            return execute<float, int32_t, BinaryGreaterEqual<float, float, int32_t>>;
        case BinaryOpOperation_EQUAL:
            return execute<float, int32_t, BinaryEqual<float, float, int32_t>>;
        case BinaryOpOperation_NOTEQUAL:
            return execute<float, int32_t, BinaryNotEqual<float, float, int32_t>>;
        default:
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}

static MNNBinaryExecute selectForInt(int type) {
    switch (type) {
        case BinaryOpOperation_MUL:
            return execute<int32_t, int32_t, BinaryMul<int32_t, int32_t, int32_t>>;
        case BinaryOpOperation_ADD:
            return execute<int32_t, int32_t, BinaryAdd<int32_t, int32_t, int32_t>>;
        case BinaryOpOperation_SUB:
            return execute<int32_t, int32_t, BinarySub<int32_t, int32_t, int32_t>>;
        case BinaryOpOperation_REALDIV:
            return execute<int32_t, int32_t, BinaryRealDiv<int32_t, int32_t, int32_t>>;
        case BinaryOpOperation_MINIMUM:
            return execute<int32_t, int32_t, BinaryMin<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_MAXIMUM:
            return execute<int32_t, int32_t, BinaryMax<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_GREATER:
            return execute<int32_t, int32_t, BinaryGreater<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_LESS:
            return execute<int32_t, int32_t, BinaryLess<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_LESS_EQUAL:
            return execute<int32_t, int32_t, BinaryLessEqual<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_GREATER_EQUAL:
            return execute<int32_t, int32_t, BinaryGreaterEqual<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_EQUAL:
            return execute<int32_t, int32_t, BinaryEqual<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_FLOORDIV:
            return execute<int32_t, int32_t, BinaryFloorDiv<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_FLOORMOD:
            return execute<int32_t, int32_t, BinaryFloorMod<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_SquaredDifference:
            return execute<int32_t, int32_t, BinarySquaredDifference<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_LOGICALOR:
            return execute<int32_t, int32_t, BinaryLogicalOr<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_NOTEQUAL:
            return execute<int32_t, int32_t, BinaryNotEqual<int32_t, int32_t, int32_t>>;
            break;
        case BinaryOpOperation_MOD:
            return execute<int32_t, int32_t, BinaryMod<int32_t, int32_t, int32_t>>;
            break;
        default:
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}

/**
 * @brief 
 * 所有的算数运算均是float op int8_t，如果不是反着算
 */
static MNNBinaryExecute selectForNITI_Float_Int8(int type, int order) {
     if(order == 1) {
        switch (type) {
            case BinaryOpOperation_MUL:
                return niti_execute<int32_t, int8_t, BinaryMul<float, float, int32_t>>;
            case BinaryOpOperation_ADD:
                return niti_execute<int32_t, int8_t, BinaryAdd<float, float, int32_t>>;
            case BinaryOpOperation_SUB:
                return niti_execute<int32_t, int8_t, BinarySub<float, float, int32_t>>;
            default:
                MNN_PRINT("Error NITI_Float_Int8 Op = %d\n", type);
                break;
        }
    } else if(order == 2) {
        switch (type) {
            case BinaryOpOperation_MUL:
                return niti_execute<float, int8_t, BinaryMul<float, float, int32_t>>;
            case BinaryOpOperation_ADD:
                return niti_execute<float, int8_t, BinaryAdd<float, float, int32_t>>;
            case BinaryOpOperation_SUB:
                return niti_execute<float, int8_t, BinarySub<float, float, int32_t>>;
            default:
                MNN_PRINT("Error NITI_Float_Int8 Op = %d\n", type);
                break;
        }
    } else if(order == 3) {
        switch (type) {
            case BinaryOpOperation_MUL:
                return niti_execute<int8_t, int32_t, BinaryMul<float, float, int32_t>>;
            case BinaryOpOperation_ADD:
                return niti_execute<int8_t, int32_t, BinaryAdd<float, float, int32_t>>;
            case BinaryOpOperation_SUB:
                return niti_execute<int8_t, int32_t, BinarySub<float, float, int32_t>>;
            default:
                MNN_PRINT("Error NITI_Float_Int8 Op = %d\n", type);
                break;
        }
    } else if(order == 4) {
        switch (type) {
            case BinaryOpOperation_MUL:
                return niti_execute<int8_t, float, BinaryMul<float, float, int32_t>>;
            case BinaryOpOperation_ADD:
                return niti_execute<int8_t, float, BinaryAdd<float, float, int32_t>>;
            case BinaryOpOperation_SUB:
                return niti_execute<int8_t, float, BinarySub<float, float, int32_t>>;
            default:
                MNN_PRINT("Error NITI_Float_Int8 Op = %d\n", type);
                break;
        }
    } else {
        MNN_PRINT("error order\n");
    }
    
}

static MNNBinaryExecute selectForNITI_Int8_Int8(int type) {
    switch (type) {
        case BinaryOpOperation_MUL:
            return niti_execute<int8_t, int8_t, BinaryMul<int32_t, int32_t, int32_t>>;
        case BinaryOpOperation_ADD:
            return niti_execute<int8_t, int8_t, BinaryAdd<int32_t, int32_t, int32_t>>;
        case BinaryOpOperation_SUB:
            return niti_execute<int8_t, int8_t, BinarySub<int32_t, int32_t, int32_t>>;
        default:
            MNN_PRINT("Error NITI_Float_Int8 Op = %d\n", type);
            break;
    }
}


class NITI_CPUBinary_Int8 : public Execution {
public:
    NITI_CPUBinary_Int8(Backend *b, MNNBinaryExecute proc, int32_t type) : Execution(b) {
        mProc = proc;
        mType = type;
    }
    virtual ~NITI_CPUBinary_Int8() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

        // MNN_PRINT("NITI_CPUBinary_Int8 resize begin\n");

        const int input0DataCount = inputs[0]->elementSize();
        const int input1DataCount = inputs[1]->elementSize();
        if (input1DataCount == input0DataCount) {
            mNeedBroadcastIndex = -1;
            mTotalSize = input1DataCount;
        } else if (input0DataCount == 1) {
            mNeedBroadcastIndex = 0;
            mTotalSize = input1DataCount;
        } else {
            mNeedBroadcastIndex = 1;
            mTotalSize = input0DataCount;
        }
        MNN_ASSERT(mTotalSize == outputs[0]->elementSize());

        // MNN_PRINT("NITI_CPUBinary_Int8 resize end\n");
        return NO_ERROR;
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
        // MNN_PRINT("NITI_CPUBinary_Int8 execute begin\n");

        if(getDSPExecuteMode()) {
            return GlobalExecute(inputs, outputs);
        }

        const int input0DataCount = ((CPUBackend*)backend())->getTensorSize(inputs[0]);
        const int input1DataCount = ((CPUBackend*)backend())->getTensorSize(inputs[1]);
    //    inputs[0]->printShape();
    //    inputs[1]->printShape();
    //    MNN_PRINT("%d - %d\n", input0DataCount, input1DataCount);
        if (input1DataCount == input0DataCount) {
            mNeedBroadcastIndex = -1;
            mTotalSize = input1DataCount;
        } else if (input0DataCount == 1) {
            mNeedBroadcastIndex = 0;
            mTotalSize = input1DataCount;
        } else {
            mNeedBroadcastIndex = 1;
            mTotalSize = input0DataCount;
        }
        auto input  = inputs[0];
        auto input1 = inputs[1];
        auto output = outputs[0];

        auto input0Ptr = input->host<uint8_t>();
        auto input1Ptr = input1->host<uint8_t>();
        auto outputPtr = output->host<uint8_t>();
        
        

        mProc(outputPtr, input0Ptr, input1Ptr, mTotalSize, mNeedBroadcastIndex);

        return NO_ERROR;
    }

    ErrorCode GlobalExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
        if(static_cast<CPUBackend*>(backend())->build_graph) {
            return NO_ERROR;
        }

        hexagon_nn_ = static_cast<CPUBackend*>(backend())->global_hexagon_nn_;
        graph_id_ = static_cast<CPUBackend*>(backend())->global_graph_id_;

        op_id = static_cast<CPUBackend*>(backend())->global_op_id;

        int input_op_id = backend()->get_Op_id(inputs[0], OpType_NITI_DSP_BINARY_Int8);

        binaylayer_input.push_back(hexagon_nn_input());
        binaylayer_input.back().src_id = input_op_id;
        binaylayer_input.back().output_idx = 0;

        input_op_id = backend()->get_Op_id(inputs[1], OpType_NITI_DSP_BINARY_Int8);

        binaylayer_input.push_back(hexagon_nn_input());
        binaylayer_input.back().src_id = input_op_id;
        binaylayer_input.back().output_idx = 0;

        op_id++;
        addConstInputTensor(hexagon_nn_, op_id, 0, binaylayer_input, (uint8_t*)&input_min, graph_id_, 1, sizeof(float));
        addConstInputTensor(hexagon_nn_, op_id, 0, binaylayer_input, (uint8_t*)&input_max, graph_id_, 1, sizeof(float));

        addConstInputTensor(hexagon_nn_, op_id, 0, binaylayer_input, (uint8_t*)&input_min, graph_id_, 1, sizeof(float));
        addConstInputTensor(hexagon_nn_, op_id, 0, binaylayer_input, (uint8_t*)&input_max, graph_id_, 1, sizeof(float));

        addConstInputTensor(hexagon_nn_, op_id, 0, binaylayer_input, (uint8_t*)&input_min, graph_id_, 1, sizeof(float));
        addConstInputTensor(hexagon_nn_, op_id, 0, binaylayer_input, (uint8_t*)&input_max, graph_id_, 1, sizeof(float));

        binaylayer_output.push_back(hexagon_nn_output());
        binaylayer_output.back().rank = 4;
        auto& max_sizeso = binaylayer_output.back().max_sizes;
        for (int i = 0; i < 4; ++i) {
            max_sizeso[i] = outputs[0]->buffer().dim[i].extent;
        }
        binaylayer_output.back().elementsize = sizeof(uint8_t);

        std::vector<int> max_size;
        max_size.push_back(1);
        max_size.push_back(1);
        max_size.push_back(1);
        max_size.push_back(1);
        addOutputTensor(max_size,binaylayer_output, sizeof(float));
        addOutputTensor(max_size,binaylayer_output, sizeof(float));

        if(mType == BinaryOpOperation_ADD){
            hexagon_nn_->hexagon_nn_append_node(graph_id_,     
                op_id,   OP_QuantizedAdd_8p8to8,   NN_PAD_NA,   binaylayer_input.data(), binaylayer_input.size(),   binaylayer_output.data(), binaylayer_output.size());
            MNN_PRINT("binary add DSP\n");
        } else {
            MNN_ERROR("no support");
            return NOT_SUPPORT;
        }

        backend()->insert_Op_id(outputs[0], op_id);

        op_id++;
        static_cast<CPUBackend*>(backend())->global_op_id = op_id;

        return NO_ERROR;
    }
private:
    MNNBinaryExecute mProc;
    int mNeedBroadcastIndex = -1;
    int mTotalSize;

    int32_t mType;
    int op_id = 0x1000;

    float input_min = -128.0f;
    float input_max = 127.0f;

    const HexagonNN*  hexagon_nn_;
    hexagon_nn_nn_id graph_id_;

    std::vector<hexagon_nn_output> inputlayer_output;

    std::vector<hexagon_nn_input> binaylayer_input;
    std::vector<hexagon_nn_output> binaylayer_output;

    std::vector<hexagon_nn_input> outputlayer_input;

    std::vector<hexagon_nn_input> empty_input;

};

class CPUBinaryCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        int32_t type = op->main_as_BinaryOp()->opType();
        auto dataType = inputs[0]->getType();
        auto dataType2 = inputs[1]->getType();
        auto core = static_cast<CPUBackend*>(backend)->functions();
        
        if( (dataType.bits == 8 && dataType2.bits == 8) ) {
            auto func = selectForNITI_Int8_Int8(type);
            return new NITI_CPUBinary_Int8(backend, func, type);
        }

        if (dataType.bits == 32) {
            if (dataType.code == halide_type_int) {
                auto func = selectForInt(type);
                if (nullptr == func) {
                    return nullptr;
                }
                return new CPUBinary(backend, func);
            } else if (dataType.code == halide_type_float) {
                auto func = core->MNNSelectBinaryFunctionForFloat(type);
                if (nullptr == func) {
                    return nullptr;
                }
                return new CPUBinary(backend, func);
            }
        }
        MNN_ERROR("CpuBinary: unsupported data type (bits: %d, code: %d)\n",
                  dataType.bits, dataType.code);
        return nullptr;
    }
};

REGISTER_CPU_OP_CREATOR(CPUBinaryCreator, OpType_BinaryOp);

} // namespace MNN
