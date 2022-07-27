//
//  NITI_ReshapeGrad.cpp
//  MNN
//
//  Created by xudaliang on 2021/11/22.
//  
//

#include "OpGrad.hpp"
#include "core/Macro.h"
#include <MNN/expr/ExprCreator.hpp>

#include <Utils.hpp>
using namespace std;
using namespace MNN;
using namespace MNN::Express;

class NITI_ReshapeGrad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {

        auto inputs = expr->inputs();
        std::vector<VARP> result(inputs.size(), nullptr);

        auto info = inputs[0]->getInfo();

        result[0] = _NITI_DSP_ReshapeGrad(backwardOutput[0], info->dim);

        return result;
    }
};


static const auto gRegister = []() {
    static NITI_ReshapeGrad _c;
    OpGrad::insert(OpType_NITI_DSP_RESHAPE_Int8, &_c);
    return true;
}();
