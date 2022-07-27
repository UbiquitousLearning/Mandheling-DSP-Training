//
//  BinaryGrad.cpp
//  MNN
//
//  Created by MNN on 2019/05/04.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "BinaryGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;
class NITI_DSPBinaryGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<VARP> res;
        auto inputs = expr->inputs();
        res.resize(inputs.size());
        auto op         = expr->get();
        auto outputDiff = backwardOutput[0];
        std::vector<VARP> output(expr->outputSize());
        for (int i = 0; i < expr->outputSize(); ++i) {
            output[i] = Variable::create(expr, i);
        }
        switch (op->main_as_BinaryOp()->opType()) {
            case BinaryOpOperation_ADD: {
                res[0] = _NITI_DSP_NOP(outputDiff);
                res[1] = _NITI_DSP_NOP(outputDiff);
                break;
            }
            default:
                return res;
        }
        return res;
    }
};

static const auto gRegister = []() {
    static NITI_DSPBinaryGrad _c;
    OpGrad::insert((int)OpType_NITI_DSP_BINARY_Int8, &_c);
    return true;
}();
