//
//  NITI_Eltwise_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/08/22.
//  
//

#ifndef CPUEltwiseInt8_hpp
#define CPUEltwiseInt8_hpp

#include "core/Execution.hpp"

namespace MNN {

class NITI_Eltwise_Int8 : public Execution {
public:
    NITI_Eltwise_Int8(Backend *backend, const Op *op);
    virtual ~NITI_Eltwise_Int8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    bool isEltwiseInt8 = true;
};

} // namespace MNN

#endif /* CPUEltwiseInt8_hpp */
