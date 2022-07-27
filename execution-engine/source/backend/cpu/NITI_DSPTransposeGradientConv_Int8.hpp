//
//  NITI_DSPTransposeGradientConv_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2022/2/20.
//  
//

#ifndef NITI_DSPTransposeGradientConv_Int8_hpp
#define NITI_DSPTransposeGradientConv_Int8_hpp

#include "backend/cpu/CPUConvolution.hpp"
#include "core/Execution.hpp"
#include "compute/Int8FunctionsOpt.h"

#include "MNN/hexagon/hexagon_nn/hexagon_nn_ops.h"
#include "MNN/hexagon/hexagon_nn/hexagon_nn.h"
#include "MNN/hexagon/hexagon_nn/HexagonRunningUtils.hpp"
#include <limits>

namespace MNN {

class NITI_DSPTransposeGradientConv_Int8 : public CPUConvolution {
public:
    
    NITI_DSPTransposeGradientConv_Int8(Backend *backend, const NITI_CONV_Int8* common,  const MNN::Op* op);
    virtual ~NITI_DSPTransposeGradientConv_Int8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    ErrorCode GlobalExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    // static std::shared_ptr<ResourceInt8> makeResource(Backend *backend, const MNN::Convolution2D *convOp);
    // virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

    static int conv_num;

private:
    std::vector<hexagon_nn_input> _inputs;
    std::vector<hexagon_nn_input> _inputs0;
    std::vector<hexagon_nn_input> _inputs1;
    std::vector<hexagon_nn_input> _inputs_output;
    std::vector<hexagon_nn_input> _inputs_output1;
    std::vector<hexagon_nn_input> _inputs_output2;

    std::vector<hexagon_nn_output> _outputs;
    std::vector<hexagon_nn_output> _outputs0;
    std::vector<hexagon_nn_output> _outputs1;

    std::vector<hexagon_nn_input> _pad_inputs;
    std::vector<hexagon_nn_input> _padweight_inputs;
    std::vector<hexagon_nn_output> _pad_outputs;
    std::vector<hexagon_nn_output> _padweight_outputs;

    uint8_t padValue = 128;
    std::vector<int32_t> padInputShape;
    std::vector<int32_t> padOutputShape;
    std::vector<int32_t> padWeightShape;
    std::vector<int32_t> padWeightOutputShape;

    int op_id = 0x1000;

    float input_min = -128.0f;
    float input_max = 127.0f;

    float weights_min = -128.0f;
    float weights_max = 127.0f;

    std::vector<uint8_t> strides_tensor;
    uint8_t* bias = nullptr;


    float bias_min = 0;
    float bias_max = 0;
    
    float output_min = -128.0f;
    float output_max = 127.0f;

    float scale = 1.0f;

    const HexagonNN*  hexagon_nn_;
    hexagon_nn_nn_id graph_id_;

    float minmax[2];

    int total_int32_num;

    std::shared_ptr<Tensor> biasTensor;
    std::shared_ptr<Tensor> weightUINT8Tensor;
    std::shared_ptr<Tensor> inputUINT8Tensor;
    std::vector<int32_t> output32TensorShape;
    
    std::vector<hexagon_nn_input> padlayer_input;
    std::vector<hexagon_nn_output> padlayer_output;

    const MNN::Op* mOp;

    std::vector<int32_t> padtensor;

    std::vector<int32_t> outputTransposeShape;
    std::vector<hexagon_nn_input> output_transposelayer_input;
    std::vector<hexagon_nn_output> output_transposelayer_output;

    std::vector<int32_t> inputTransposeShape;
    std::vector<hexagon_nn_input> input_transposelayer_input;
    std::vector<hexagon_nn_output> input_transposelayer_output;

    std::vector<int32_t> outdiffTransposeShape;
    std::vector<hexagon_nn_input> outdiff_transposelayer_input;
    std::vector<hexagon_nn_output> outdiff_transposelayer_output;


    std::vector<int32_t> shapeValue;
    std::vector<int32_t> inputShapeValue;

    std::vector<hexagon_nn_input> castlayer_input;
    std::vector<hexagon_nn_output> castlayer_output;

};

} // namespace MNN

#endif /* NITIDSPGradientConvInt8_hpp */
