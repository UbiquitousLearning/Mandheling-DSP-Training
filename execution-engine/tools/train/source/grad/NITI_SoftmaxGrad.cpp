//
//  SoftmaxGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include "core/Macro.h"
#include <MNN/expr/ExprCreator.hpp>

#include "MNN/hexagon/hexagon_nn/HexagonRunningUtils.hpp"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class NITI_SoftmaxGrad : public OpGrad {
public:
    NITI_SoftmaxGrad() {
        mType = NO_LINEAR;
    }
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result(2, nullptr);
        auto outputDiff = backwardOutput[0];
        std::unique_ptr<OpT> forwardOp(expr->get()->UnPack());
        unique_ptr<OpT> newOp(new OpT);
        newOp->type       = OpType_NITI_SOFTMAX_Grad_Int8;
        auto copyP        = new AxisT(*forwardOp->main.AsAxis());
        newOp->main.type  = OpParameter_Axis;
        newOp->main.value = copyP;

        result[0] = Variable::create(
            Expr::create(std::move(newOp), {outputDiff, expr->inputs()[0], expr->inputs()[1]}));
        result[1] = _Scalar<int8_t>(0);
        return result;
    }
};

class NITI_LossGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result(1, nullptr);
        auto outputDiff = backwardOutput[0];
        std::unique_ptr<OpT> forwardOp(expr->get()->UnPack());
        unique_ptr<OpT> newOp(new OpT);

        int mode;
        FILE *f = fopen("DSP.txt","r");
        fscanf(f, "%d",&mode);
        fclose(f);

        if(mode > 1)
            newOp->type       = OpType_NITI_LOSS_Grad_Int8;
        else
            newOp->type       = OpType_NITI_DSP_LOSSGRAD_Int8;
        auto copyP        = new NITI_LOSS_Int8T(*forwardOp->main.AsNITI_LOSS_Int8());
        newOp->main.type  = OpParameter_NITI_LOSS_Int8;
        newOp->main.value = copyP;

        result[0] = Variable::create(
            Expr::create(std::move(newOp), {expr->inputs()[0], expr->inputs()[1], expr->inputs()[2], outputDiff}));
        return result;
    }
};

static const auto gRegister = []() {
    static NITI_SoftmaxGrad _c;
    OpGrad::insert(OpType_NITI_SOFTMAX_Int8, &_c);

    static NITI_LossGrad _d;
    OpGrad::insert(OpType_NITI_LOSS_Int8, &_d);
    return true;
}();
