//
//  NITI_PstoShiftInt32_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2019/08/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef NITI_PstoShiftInt32_Int8_hpp
#define NITI_PstoShiftInt32_Int8_hpp

#include "core/Execution.hpp"

namespace MNN {

class NITI_PstoShiftInt32_Int8 : public Execution {
public:
    NITI_PstoShiftInt32_Int8(Backend *backend, const Op *op);
    virtual ~NITI_PstoShiftInt32_Int8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* NITI_PstoShiftInt32_Int8_hpp */
