//
//  PoolGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class NITI_Pool_Int8_Grad : public OpGrad {
public:
    NITI_Pool_Int8_Grad() {
        mType = SEMI_LINEAR;
    }

    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        std::vector<Express::VARP> result(2, nullptr);
        auto outputDiff = backwardOutput[0];
        std::unique_ptr<OpT> forwardOp(expr->get()->UnPack());
        unique_ptr<OpT> newOp(new OpT);
        newOp->type       = OpType_NITI_PoolGrad_Int8;
        auto copyP        = new NITI_Pool_Int8T(*forwardOp->main.AsNITI_Pool_Int8());
        newOp->main.type  = OpParameter_NITI_Pool_Int8;
        newOp->main.value = copyP;

        result[0] = Variable::create(
            Expr::create(std::move(newOp), { _Convert(expr->inputs()[0],  NC4HW4), _Convert(Variable::create(expr, 0), NC4HW4) , _Convert(outputDiff, NC4HW4) }));
        result[0] = _Convert(result[0],  NHWC);
        result[1] = _Scalar<int8_t>(0);
        return result;
    }
};

class NITI_DSPMaxPoolRef_Int8_Grad : public OpGrad {
public:
    NITI_DSPMaxPoolRef_Int8_Grad() {
        mType = SEMI_LINEAR;
    }

    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {

        std::vector<Express::VARP> result(2, nullptr);
        auto outputDiff = backwardOutput[0];
        std::unique_ptr<OpT> forwardOp(expr->get()->UnPack());
        unique_ptr<OpT> newOp(new OpT);
        newOp->type       = OpType_NITI_DSP_MAXPOOLGRAD_Int8;
        auto copyP        = new NITI_Pool_Int8T(*forwardOp->main.AsNITI_Pool_Int8());
        newOp->main.type  = OpParameter_NITI_Pool_Int8;
        newOp->main.value = copyP;

        result[0] = Variable::create(
            Expr::create(std::move(newOp), { expr->inputs()[0], Variable::create(expr, 0) , outputDiff }));
            
        result[1] = _Scalar<int8_t>(0);
        return result;
    }
};

static const auto gRegister = []() {
    static NITI_Pool_Int8_Grad _c;
    OpGrad::insert(OpType_NITI_Maxpool_Int8, &_c);

    static NITI_DSPMaxPoolRef_Int8_Grad _d;
    OpGrad::insert(OpType_NITI_DSP_MAXPOOL_Int8, &_d);
    return true;
}();
