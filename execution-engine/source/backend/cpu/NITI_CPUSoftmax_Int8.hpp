//
//  NITI_CPUSoftmax_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/08/22.
//
//

#ifndef NITI_CPUSoftmax_Int8_hpp
#define NITI_CPUSoftmax_Int8_hpp

#include "core/Execution.hpp"

namespace MNN {
class NITI_CPUSoftmax_Int8 : public Execution {
public:
    NITI_CPUSoftmax_Int8(Backend *b, int axis);
    virtual ~NITI_CPUSoftmax_Int8() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    static Execution* create(const MNN::Op *op, Backend *backend);
    
private:
    int mAxis;
    std::shared_ptr<Tensor> s;
    std::shared_ptr<Tensor> out_max;
};
} // namespace MNN

#endif /* NITI_CPUSoftmax_Int8_hpp */
