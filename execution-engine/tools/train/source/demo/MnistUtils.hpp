//
//  MnistUtils.hpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MnistUtils_hpp
#define MnistUtils_hpp
#include <MNN/expr/Module.hpp>
class MnistUtils {
public:
    static void train(std::shared_ptr<MNN::Express::Module> model, std::string root);
    static void dsp_train(std::shared_ptr<MNN::Express::Module> model, std::string root);
    static void float_train(std::shared_ptr<MNN::Express::Module> model, std::string root);
};
#endif
