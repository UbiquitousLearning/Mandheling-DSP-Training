//
//  ReluGrad.cpp
//  MNN
//
//  Created by MNN on 2019/04/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ReluGrad.hpp"
#include "core/Macro.h"
#include <string.h>
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class NITI_ReluGrad : public OpGrad {
public:
    NITI_ReluGrad() {
        mType = SEMI_LINEAR;
    }
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        
        auto outputDiff = backwardOutput[0];                                            
        return {_NITI_ReluGrad_Int8(expr->inputs()[0], outputDiff)};
    }
};

class NITI_ReluDSP_Grad : public OpGrad {
public:
    NITI_ReluDSP_Grad() {
        mType = SEMI_LINEAR;
    }
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        
        auto outputDiff = backwardOutput[0];                                            
        return {_NITI_DSP_ReluGrad_Int8(expr->inputs()[0], outputDiff)};
    }
};

static const auto gRegister = []() {
    static NITI_ReluGrad _c;
    OpGrad::insert(OpType_NITI_Relu_Int8, &_c);

    static NITI_ReluDSP_Grad _d;
    OpGrad::insert(OpType_NITI_DSP_RELU_Int8, &_d);
    
    return true;
}();
