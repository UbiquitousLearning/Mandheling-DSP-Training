//
//  NITI_DSPTranspose_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/11/17.
//
//

#include <math.h>
#include "backend/cpu/NITI_DSPTranspose_Int8.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "CPUTensorConvert.hpp"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

#include "MNN/hexagon/hexagon_nn/hexagon_nn_ops.h"
#include "MNN/hexagon/hexagon_nn/hexagon_nn.h"
#include "MNN/hexagon/hexagon_nn/HexagonRunningUtils.hpp"

namespace MNN {

ErrorCode NITI_DSPTranspose_Int8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    auto input = inputs[0];

    return NO_ERROR;
}

ErrorCode NITI_DSPTranspose_Int8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    initDspGraph(hexagon_nn_, graph_id_);

    auto input = inputs[0];
    auto shapeValue = inputs[1]->host<int32_t>();
    auto output = outputs[0];
    

    int obatch = output->batch();
    int ochannel = output->channel();
    int owidth = output->width();
    int oheight = output->height();

    int osize = obatch*ochannel*owidth*oheight;

    // input layer
    inputlayer_output.push_back(hexagon_nn_output());
    inputlayer_output.back().rank = 4;
    auto& max_sizes0 = inputlayer_output.back().max_sizes;
    for (int i = 0; i < 4; ++i) {
        max_sizes0[i] = inputs[0]->buffer().dim[i].extent;
    }
    inputlayer_output.back().elementsize = sizeof(uint8_t);

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_INPUT,   NN_PAD_NA,   empty_input.data(), 0,   inputlayer_output.data(), inputlayer_output.size());


    // relu layer
    transposelayer_input.push_back(hexagon_nn_input());
    transposelayer_input.back().src_id = op_id;
    transposelayer_input.back().output_idx = 0;

    op_id++;

    // control tensor
    addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)shapeValue, graph_id_, 4, sizeof(int32_t)*4);

    addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)&input_min, graph_id_, 1, sizeof(float));
    addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)&input_max, graph_id_, 1, sizeof(float));

    transposelayer_output.push_back(hexagon_nn_output());
    transposelayer_output.back().rank = 4;
    auto& max_sizes1 = transposelayer_output.back().max_sizes;
    for (int i = 0; i < 4; ++i) {
        max_sizes1[i] = outputs[0]->buffer().dim[i].extent;
    }
    transposelayer_output.back().elementsize = sizeof(uint8_t);

    std::vector<int> max_size;
    max_size.push_back(1);
    max_size.push_back(1);
    max_size.push_back(1);
    max_size.push_back(1);
    addOutputTensor(max_size,transposelayer_output, sizeof(float));
    addOutputTensor(max_size,transposelayer_output, sizeof(float));

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_Transpose_8,   NN_PAD_NA,   transposelayer_input.data(), transposelayer_input.size(),   transposelayer_output.data(), transposelayer_output.size());

    // output layer
    outputlayer_input.push_back(hexagon_nn_input());
    outputlayer_input.back().src_id = op_id;
    outputlayer_input.back().output_idx = 0;

    op_id++;

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_OUTPUT,   NN_PAD_NA,   outputlayer_input.data(), outputlayer_input.size(),   NULL, 0);
    

    int error = hexagon_nn_->hexagon_nn_prepare(graph_id_);
    if(error!=0){
        MNN_ERROR("Whoops... Cannot prepare: %d\n", error);
        return NOT_SUPPORT;
    }
    
    {
        input_tensors.emplace_back();
        auto& input_tensor = input_tensors.back();
        input_tensor.data = inputs[0]->host<uint8_t>();
        input_tensor.dataLen = inputs[0]->elementSize();
        input_tensor.data_valid_len = inputs[0]->elementSize();
        input_tensor.batches = inputs[0]->batch();
        input_tensor.height = inputs[0]->height();
        input_tensor.width = inputs[0]->width();
        input_tensor.depth = inputs[0]->channel();
    }


    output_tensors.emplace_back();
    auto& output_tensor = output_tensors.back();
    output_tensor.data = outputs[0]->host<uint8_t>();
    output_tensor.dataLen = outputs[0]->elementSize();
    output_tensor.batches = outputs[0]->batch();
    output_tensor.height = outputs[0]->height();
    output_tensor.width = outputs[0]->width();
    output_tensor.depth = outputs[0]->channel();

    error = hexagon_nn_->hexagon_nn_execute_new( graph_id_, input_tensors.data(), input_tensors.size(),
      output_tensors.data(), output_tensors.size());
    
    if(error!=0){
        MNN_ERROR("Whoops... run failed: %d\n", error);
        return NOT_SUPPORT;
    }

    hexagon_nn_->hexagon_nn_teardown(graph_id_);


    static_cast<CPUBackend*>(backend())->gradient_map.insert(std::make_pair(outputs[0]->host<int8_t>(), outputs[0]->elementSize()));

    return NO_ERROR;
}

NITI_DSPTranspose_Int8::NITI_DSPTranspose_Int8(Backend *b) : MNN::Execution(b) {
    // nothing to do
    hexagon_nn_ = generate_interface();
}


class NITI_DSPTranspose_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new NITI_DSPTranspose_Int8(backend);
    }
};

REGISTER_CPU_OP_CREATOR(NITI_DSPTranspose_Int8_Creator, OpType_NITI_DSP_TRANSPOSE_Int8);

} // namespace MNN
