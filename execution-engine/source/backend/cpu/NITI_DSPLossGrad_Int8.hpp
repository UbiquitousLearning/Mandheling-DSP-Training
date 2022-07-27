//
//  NITI_DSPLossGrad_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/11/28.
//
//

#ifndef NITI_DSPLossGrad_Int8_hpp
#define NITI_DSPLossGrad_Int8_hpp

#include "core/Execution.hpp"

#include "MNN/hexagon/hexagon_nn/hexagon_nn_ops.h"
#include "MNN/hexagon/hexagon_nn/hexagon_nn.h"
#include "MNN/hexagon/hexagon_nn/HexagonRunningUtils.hpp"


namespace MNN {
class NITI_DSPLossGrad_Int8 : public Execution {
public:
    NITI_DSPLossGrad_Int8(Backend *b, const MNN::Op *op);
    virtual ~NITI_DSPLossGrad_Int8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    ErrorCode GlobalExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    static Execution* create(const MNN::Op *op, Backend *backend);
    
private:
    int op_id = 0x1000;

    int8_t ascale;

    const HexagonNN*  hexagon_nn_;
    hexagon_nn_nn_id graph_id_;

    std::vector<hexagon_nn_output> inputlayer_output;

    std::vector<hexagon_nn_input> lossgradlayer_input;
    std::vector<hexagon_nn_output> lossgradlayer_output;

    std::vector<hexagon_nn_input> outputlayer_input;

    std::vector<hexagon_nn_input> empty_input;

    std::vector<hexagon_nn_tensordef> input_tensors;
    std::vector<hexagon_nn_tensordef> output_tensors;

    std::shared_ptr<Tensor> out_grad_final;

    static int32_t* target_dsp_buffer;

};
} // namespace MNN

#endif /* NITI_DSPLossGrad_Int8_hpp */
