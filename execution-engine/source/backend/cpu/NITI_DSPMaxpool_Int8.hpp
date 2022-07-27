//
//  NITI_DSP_Maxpool_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/8/17.
//  
//

#ifndef NITI_DSP_Maxpool_Int8_hpp
#define NITI_DSP_Maxpool_Int8_hpp

#include "core/Execution.hpp"
#include "compute/Int8FunctionsOpt.h"

#include "MNN/hexagon/hexagon_nn/hexagon_nn_ops.h"
#include "MNN/hexagon/hexagon_nn/hexagon_nn.h"
#include "MNN/hexagon/hexagon_nn/HexagonRunningUtils.hpp"

namespace MNN {

class NITI_DSP_Maxpool_Int8 : public Execution {
public:
    
    NITI_DSP_Maxpool_Int8(Backend *backend, const NITI_Pool_Int8* pool);
    virtual ~NITI_DSP_Maxpool_Int8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    ErrorCode GlobalExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    // static std::shared_ptr<ResourceInt8> makeResource(Backend *backend, const MNN::Convolution2D *convOp);
    // virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    const NITI_Pool_Int8 *mParameter;
    
    int op_id = 0x1000;

    float input_min = -128.0f;
    float input_max = 127.0f;

    const HexagonNN*  hexagon_nn_;
    hexagon_nn_nn_id graph_id_;

    std::vector<hexagon_nn_output> inputlayer_output;

    std::vector<hexagon_nn_input> maxpoollayer_input;
    std::vector<hexagon_nn_output> maxpoollayer_output;

    std::vector<hexagon_nn_input> outputlayer_input;

    std::vector<hexagon_nn_input> empty_input;

    std::vector<hexagon_nn_tensordef> input_tensors;
    std::vector<hexagon_nn_tensordef> output_tensors;

    std::shared_ptr<Tensor> inputUINT8Tensor;
};

} // namespace MNN

#endif /* NITI_DSP_Maxpool_Int8 */
