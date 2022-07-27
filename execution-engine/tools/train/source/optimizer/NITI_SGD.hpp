//
//  SGD.cpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SGD.hpp"
#include "OpGrad.hpp"
using namespace MNN::Express;

namespace MNN {
namespace Train {

class MNN_PUBLIC NITI_SGD : public SGD {
public:
    NITI_SGD(std::shared_ptr<Express::Module> module): SGD(module) { }

    virtual std::map<Express::VARP, Express::VARP> onGetNextParameter(Express::VARP loss) override {
        auto grad = OpGrad::grad(loss, trainable(), mGradBlockExprName);
        auto parameters = module()->parameters();
        std::vector<VARP> prepareCompute;
        for (auto iter : parameters) {
            if (iter->expr().first->get() != nullptr) {
                prepareCompute.emplace_back(iter);
            }
        }
        for (auto& iter : grad) {
            prepareCompute.emplace_back(iter.second);
        }
        Variable::prepareCompute(prepareCompute);
        std::vector<VARP> replaceOp(prepareCompute.size());
        for (int i=0; i<prepareCompute.size(); ++i) {
            auto info = prepareCompute[i]->getInfo();
            auto ptr = prepareCompute[i]->readMap<void>();
            if (nullptr == ptr) {
                MNN_ERROR("Compute error in SGD\n");
                exit(-1);
                return {};
            }
            auto newVar = _Const(ptr, info->dim, info->order, info->type);
            replaceOp[i]= newVar;
        }
        for (int i=0; i<prepareCompute.size(); ++i) {
            Variable::replace(prepareCompute[i], replaceOp[i]);
        }

        for (auto& iter : grad) {
            auto newParameter = iter.first - iter.second;
            iter.second       = newParameter;
        }
        return grad;
    }


};


} // namespace Train
} // namespace MNN
