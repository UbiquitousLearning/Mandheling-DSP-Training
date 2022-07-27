//
//  NITI_DSPReluGrad_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/11/17.
//
//

#ifndef NITI_DSPReluGrad_Int8_hpp
#define NITI_DSPReluGrad_Int8_hpp

#include "core/Execution.hpp"

#include "MNN/hexagon/hexagon_nn/hexagon_nn_ops.h"
#include "MNN/hexagon/hexagon_nn/hexagon_nn.h"
#include "MNN/hexagon/hexagon_nn/HexagonRunningUtils.hpp"
namespace MNN {
class NITI_DSPReluGrad_Int8 : public Execution {
public:
    NITI_DSPReluGrad_Int8(Backend *b);
    virtual ~NITI_DSPReluGrad_Int8() = default;
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

    std::vector<hexagon_nn_input> relulayer_input;
    std::vector<hexagon_nn_output> relulayer_output;

    std::vector<hexagon_nn_input> outputlayer_input;

    std::vector<hexagon_nn_input> empty_input;

    std::vector<hexagon_nn_tensordef> input_tensors;
    std::vector<hexagon_nn_tensordef> output_tensors;

    std::shared_ptr<Tensor> inputUINT8Tensor;

};
} // namespace MNN

#endif /* NITI_DSPRelu_Int8_hpp */
