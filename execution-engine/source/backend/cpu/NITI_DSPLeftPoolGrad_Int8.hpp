//
//  NITI_DSPLeftPoolGrad_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/12/17.
//
//

#ifndef NITI_DSPLeftPoolGrad_Int8_hpp
#define NITI_DSPLeftPoolGrad_Int8_hpp

#include "core/Execution.hpp"

#include "MNN/hexagon/hexagon_nn/hexagon_nn_ops.h"
#include "MNN/hexagon/hexagon_nn/hexagon_nn.h"
#include "MNN/hexagon/hexagon_nn/HexagonRunningUtils.hpp"

namespace MNN {
class NITI_DSPLeftPoolGrad_Int8 : public Execution {
public:
    NITI_DSPLeftPoolGrad_Int8(Backend *b, const Op* op);
    virtual ~NITI_DSPLeftPoolGrad_Int8() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    ErrorCode GlobalExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    
private:
    int mAxis;
    std::shared_ptr<Tensor> s;
    std::shared_ptr<Tensor> out_max;

    int op_id = 0x1000;

    float input_min = -128.0f;
    float input_max = 127.0f;

    const HexagonNN*  hexagon_nn_;
    hexagon_nn_nn_id graph_id_;

    std::vector<hexagon_nn_output> inputlayer_output;

    std::vector<hexagon_nn_input> leftpoollayer_input;
    std::vector<hexagon_nn_output> leftpoollayer_output;

    std::vector<hexagon_nn_input> outputlayer_input;

    std::vector<hexagon_nn_input> empty_input;

    std::vector<hexagon_nn_tensordef> input_tensors;
    std::vector<hexagon_nn_tensordef> output_tensors;

    std::vector<int32_t> output_shape;

    std::shared_ptr<Tensor> inputUINT8Tensor;

    const Op* mOp;

};
} // namespace MNN

#endif /* NITI_DSPRelu_Int8_hpp */
