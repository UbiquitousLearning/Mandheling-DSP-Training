//
//  NITI_CPUSoftmaxGrad_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/08/28.
//
//

#ifndef NITI_CPUSoftmaxGrad_Int8_hpp
#define NITI_CPUSoftmaxGrad_Int8_hpp

#include "core/Execution.hpp"

namespace MNN {
class NITI_CPUSoftmaxGrad_Int8 : public Execution {
public:
    NITI_CPUSoftmaxGrad_Int8(Backend *b, int axis);
    virtual ~NITI_CPUSoftmaxGrad_Int8() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    static Execution* create(const MNN::Op *op, Backend *backend);
    
private:
    int mAxis;
};
} // namespace MNN

#endif /* NITI_CPUSoftmaxGrad_Int8_hpp */
