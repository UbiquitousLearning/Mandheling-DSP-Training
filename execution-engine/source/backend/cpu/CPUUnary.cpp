//
//  CPUUnary.cpp
//  MNN
//
//  Created by MNN on 2018/08/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUUnary.hpp"
#include "UnaryUtils.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "compute/ConvOpt.h"
#include "compute/CommonOptFunction.h"
#include <MNN/AutoTime.hpp>

namespace MNN {
CPUUnary::CPUUnary(Backend *b, MNNUnaryExecute proc) : MNN::Execution(b), mProc(proc) {
    // nothing to do
}

ErrorCode CPUUnary::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    // MNN_PRINT("CPUUnary resize begin\n");
    MNN_ASSERT(1 == outputs.size());
    MNN_ASSERT(inputs[0]->getType() == halide_type_of<float>() || inputs[0]->getType() == halide_type_of<int32_t>());
    // 我不能完全理解。。但是type是int32_t 但是bits = 8
    // 也不能完全懂，到底是32位的，还是8位的
    // MNN_PRINT("CPUUnary resize end %d \n", inputs[0]->getType().bits);
    return NO_ERROR;
}

static void _Neg(void* out, const void* inp, int realSize) {
    MNNScaleAndAddBiasScalar((float*)out, (const float*)inp, 0.0f, -1.0f, realSize);
}

static void _ABS(void* out, const void* inp, int realSize) {
    MNNReluWithSlopeCommon((float*)out, (const float*)inp, realSize, -1.0f);
}
static void _Square(void* out, const void* inp, int realSize) {
    MNNMatrixProdCommon((float*)out, (const float*)inp, (const float*)inp, realSize, 0, 0, 0, 1);
}

static void _EXP(void* outRaw, const void* inpRaw, int realSize) {
    auto out = (float*)outRaw;
    auto inp = (const float*)inpRaw;
    MNNScaleAndAddBiasScalar(out, inp, 0.0f, -1.0f, realSize);
    MNNExp(out, out, realSize);
}
static void _EXPM1(void* outRaw, const void* inpRaw, int realSize) {
    auto out = (float*)outRaw;
    auto inp = (const float*)inpRaw;
    MNNScaleAndAddBiasScalar(out, inp, 0.0f, -1.0f, realSize);
    MNNExp(out, out, realSize);
    for (int i=0; i<realSize; ++i) {
        out[i] = out[i] - 1.0f;
    }
}

MNNUnaryExecute CPUUnary::selectForFloat(int type, int precision) {
    switch (type) {
        case UnaryOpOperation_ABS:
            return _ABS;
        case UnaryOpOperation_SQUARE:
            return _Square;
        case UnaryOpOperation_NEG:
            return _Neg;
        case UnaryOpOperation_RSQRT:
            return _unaryOp<UnaryRsqrt<float>, float>;
        case UnaryOpOperation_EXP:
            return _EXP;
        case UnaryOpOperation_COS:
            return _unaryOp<UnaryCos<float>, float>;
        case UnaryOpOperation_SIN:
            return (MNNUnaryExecute)MNNSin;
        case UnaryOpOperation_SIGMOID:
            if (BackendConfig::Precision_Low == precision) {
                return (MNNUnaryExecute)MNNSigmoidLowp;
            } else {
                return (MNNUnaryExecute)MNNSigmoid;
            }
            break;
        case UnaryOpOperation_TANH:
            return (MNNUnaryExecute)MNNTanh;
        case UnaryOpOperation_TAN:
            return _unaryOp<UnaryTan<float>, float>;
        case UnaryOpOperation_ATAN:
            return _unaryOp<UnaryATan<float>, float>;
        case UnaryOpOperation_SQRT:
            return _unaryOp<UnarySqrt<float>, float>;
        case UnaryOpOperation_CEIL:
            return _unaryOp<UnaryCeil<float>, float>;
        case UnaryOpOperation_RECIPROCAL:
            return _unaryOp<UnaryRecipocal<float>, float>;
        case UnaryOpOperation_LOG1P:
            return _unaryOp<UnaryLog1p<float>, float>;
        case UnaryOpOperation_LOG:
            return _unaryOp<UnaryLog<float>, float>;
        case UnaryOpOperation_LOG2:
            return _unaryOp<UnaryLog2<float>, float>;
        case UnaryOpOperation_FLOOR:
            return _unaryOp<UnaryFloor<float>, float>;
        case UnaryOpOperation_BNLL:
            return _unaryOp<UnaryBNLL<float>, float>;
        case UnaryOpOperation_ACOSH:
            return _unaryOp<UnaryAcosh<float>, float>;
        case UnaryOpOperation_SINH:
            return _unaryOp<UnarySinh<float>, float>;
        case UnaryOpOperation_ASINH:
            return _unaryOp<UnaryAsinh<float>, float>;
        case UnaryOpOperation_ATANH:
            return _unaryOp<UnaryAtanh<float>, float>;
        case UnaryOpOperation_SIGN:
            return _unaryOp<UnarySign<float>, float>;
        case UnaryOpOperation_ROUND:
            return _unaryOp<UnaryRound<float>, float>;
        case UnaryOpOperation_COSH:
            return _unaryOp<UnaryCosh<float>, float>;
        case UnaryOpOperation_ERF:
            return _unaryOp<UnaryErf<float>, float>;
        case UnaryOpOperation_ERFC:
            return _unaryOp<UnaryErfc<float>, float>;
        case UnaryOpOperation_ERFINV:
            return _unaryOp<UnaryErfinv<float>, float>;
        case UnaryOpOperation_EXPM1:
            return _EXPM1;
        case UnaryOpOperation_ASIN:
            return _unaryOp<UnaryAsin<float>, float>;
        case UnaryOpOperation_ACOS:
            return _unaryOp<UnaryAcos<float>, float>;
        case UnaryOpOperation_HARDSWISH:
            return (MNNUnaryExecute)MNNHardSwishCommon;
        case UnaryOpOperation_GELU:
            return (MNNUnaryExecute)MNNGeluCommon;
        default:
            MNN_ASSERT(false);
            break;
    }
    return nullptr;
}

static MNNUnaryExecute selectForInt(int type) {
    MNN_PRINT("UNARY OP type = %d\n",type);
    // NITI交叉熵用的是int8的，然后用的是迭代法近似求解
    switch (type) {
        case UnaryOpOperation_ABS:
            return _unaryOp<UnaryAbs<int32_t>, int32_t>;
        case UnaryOpOperation_NEG:
            return _unaryOp<UnaryNeg<int32_t>, int32_t>;
        case UnaryOpOperation_SQUARE:
            return _unaryOp<UnarySquare<int32_t>, int32_t>;
        case UnaryOpOperation_SIGN:
            return _unaryOp<UnarySign<int32_t>, int32_t>;
        case UnaryOpOperation_LOG:
            return _unaryOp<NITI_UnaryLog_Int8<int32_t>, int32_t>;
        default:
            break;
    }
    return nullptr;
}
ErrorCode CPUUnary::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto size = static_cast<CPUBackend*>(backend())->getTensorSize(input);
    auto schedule = ((CPUBackend*)backend())->multiThreadDivide(size);
    auto inputPtr = input->host<uint8_t>();
    auto outputPtr = output->host<uint8_t>();
    int outBytes = output->getType().bytes();
    if (halide_type_float == output->getType().code) {
        outBytes = static_cast<CPUBackend*>(backend())->functions()->bytes;
    }
    MNN_CONCURRENCY_BEGIN(tId, schedule.second) {
        int start = schedule.first * (int)tId;
        int realSize = schedule.first;
        if (tId == schedule.second -1 ) {
            realSize = size - start;
        }
        if (realSize > 0) {
            auto inp = inputPtr + start * outBytes;
            auto out = outputPtr + start * outBytes;
            mProc(out, inp, realSize);
        }
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}

class CPUUnaryCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto precision = static_cast<CPUBackend*>(backend)->precisionMode();
        auto type = inputs[0]->getType();
        MNNUnaryExecute proc = nullptr;
        if (type.code == halide_type_int) {
            MNN_PRINT("UNARY Create for int begin\n");
            proc = selectForInt(op->main_as_UnaryOp()->opType());
            MNN_PRINT("UNARY Create for int end\n");
        } else if (type.code == halide_type_float) {
            proc = static_cast<CPUBackend*>(backend)->functions()->MNNSelectUnaryFunctionForFloat(op->main_as_UnaryOp()->opType(), static_cast<CPUBackend*>(backend)->precisionMode());
        }
        if (nullptr == proc) {
            MNN_PRINT("UNARY Create return nullptr\n");
            return nullptr;
        }
        return new CPUUnary(backend, proc);
    }
};

REGISTER_CPU_OP_CREATOR(CPUUnaryCreator, OpType_UnaryOp);

} // namespace MNN
