//
//  NITI_CPULoss_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/08/28.
//
//

#ifndef NITI_CPULossGrad_Int8_hpp
#define NITI_CPULossGrad_Int8_hpp

#include "core/Execution.hpp"

namespace MNN {
class NITI_CPULossGrad_Int8 : public Execution {
public:
    NITI_CPULossGrad_Int8(Backend *b, const MNN::Op *op);
    virtual ~NITI_CPULossGrad_Int8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    static Execution* create(const MNN::Op *op, Backend *backend);
    
private:
    const MNN::Op *mOp;
    std::shared_ptr<Tensor> s;
    std::shared_ptr<Tensor> out_max;

    std::shared_ptr<Tensor> output_softmax;

    std::shared_ptr<Tensor> out_grad;
    std::shared_ptr<Tensor> out_grad_final;
    std::shared_ptr<Tensor> out_sum;
    std::shared_ptr<Tensor> target_max;
};
} // namespace MNN

#endif /* NITI_CPULossGrad_Int8_hpp */
