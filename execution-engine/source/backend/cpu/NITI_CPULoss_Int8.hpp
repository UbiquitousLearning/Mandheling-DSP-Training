//
//  NITI_CPULoss_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/08/28.
//
//

#ifndef NITI_CPULoss_Int8_hpp
#define NITI_CPULoss_Int8_hpp

#include "core/Execution.hpp"

namespace MNN {
class NITI_CPULoss_Int8 : public Execution {
public:
    NITI_CPULoss_Int8(Backend *b, const MNN::Op *op);
    virtual ~NITI_CPULoss_Int8() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    static Execution* create(const MNN::Op *op, Backend *backend);
    
private:

    void _softmax1(const float *srcData, float *dstData, int outside, int channel, int threadNum);

    const MNN::Op *mOp;

    std::shared_ptr<Tensor> input_float;
    std::shared_ptr<Tensor> input_softmax;
    std::shared_ptr<Tensor> input_sum;

};
} // namespace MNN

#endif /* NITI_CPULoss_Int8_hpp */
