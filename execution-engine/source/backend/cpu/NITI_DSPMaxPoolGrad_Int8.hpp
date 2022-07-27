//
//  NITI_DSPMaxPoolGrad_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/11/19.
//  
//

#ifndef NITI_DSPMaxPoolGrad_Int8_hpp
#define NITI_DSPMaxPoolGrad_Int8_hpp

#include "backend/cpu/CPUBackend.hpp"

#include "MNN/hexagon/hexagon_nn/hexagon_nn_ops.h"
#include "MNN/hexagon/hexagon_nn/hexagon_nn.h"
#include "MNN/hexagon/hexagon_nn/HexagonRunningUtils.hpp"

namespace MNN {
class NITI_DSPMaxPoolGrad_Int8 : public Execution {
public:
    virtual ~NITI_DSPMaxPoolGrad_Int8() = default;
    NITI_DSPMaxPoolGrad_Int8(Backend *b, const NITI_Pool_Int8 *parameter) : Execution(b) {
        mStrideX = parameter->strideX();
        mStrideY = parameter->strideY();
        mKernelX = parameter->kernelX();
        mKernelY = parameter->kernelY();
        mGlobal  = parameter->isGlobal();
        mParameter = parameter;

        if(!getDSPExecuteMode())
            hexagon_nn_ = generate_interface();
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    ErrorCode GlobalExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);

protected:
    int mStrideX;
    int mStrideY;
    int mKernelX;
    int mKernelY;
    bool mGlobal;
    int mPadX;
    int mPadY;
    const NITI_Pool_Int8* mParameter;

private:
    int op_id = 0x1000;

    const HexagonNN*  hexagon_nn_;
    hexagon_nn_nn_id graph_id_;

    float input_min = -128.0f;
    float input_max = 127.0f;

    std::vector<hexagon_nn_output> inputlayer_output;

    std::vector<hexagon_nn_input> maxpoolgradlayer_input;
    std::vector<hexagon_nn_output> maxpoolgradlayer_output;

    std::vector<hexagon_nn_input> outputlayer_input;

    std::vector<hexagon_nn_input> empty_input;

    std::vector<hexagon_nn_tensordef> input_tensors;
    std::vector<hexagon_nn_tensordef> output_tensors;

    std::vector<hexagon_nn_input> transposeinputlayer_input;
    std::vector<hexagon_nn_output> transposeinputlayer_output;
    std::vector<int32_t> transposeinput_shape;


    std::vector<hexagon_nn_input> transposeoutputlayer_input;
    std::vector<hexagon_nn_output> transposeoutputlayer_output;
    std::vector<int32_t> transposeoutput_shape;

    std::vector<hexagon_nn_input> transposeoriginoutputlayer_input;
    std::vector<hexagon_nn_output> transposeoriginoutputlayer_output;

    std::vector<hexagon_nn_input> transposeoutdifflayer_input;
    std::vector<hexagon_nn_output> transposeoutdifflayer_output;

    std::shared_ptr<Tensor> inputUINT8Tensor;
    std::shared_ptr<Tensor> originOutputUINT8Tensor;
    std::shared_ptr<Tensor> outdiffUINT8Tensor;

    uint8_t padValue = 128;

    int32_t pad = 0;

};
} // namespace MNN
#endif /* CPUPoolGrad_hpp */
