//
//  NITI_DSPLossGrad_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/11/28.
//  
//

#include <math.h>
#include "backend/cpu/NITI_DSPLossGrad_Int8.hpp"
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

ErrorCode NITI_DSPLossGrad_Int8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    
    auto input = inputs[0];
    auto output = outputs[0];

    if(getDSPExecuteMode() && !static_cast<CPUBackend*>(backend())->build_graph) {

        // target
        static_cast<CPUBackend*>(backend())->_outputs0.push_back(hexagon_nn_output());
        static_cast<CPUBackend*>(backend())->_outputs0.back().rank = 4;
        auto& max_sizesw = static_cast<CPUBackend*>(backend())->_outputs0.back().max_sizes;
        for (int i = 0; i < 4; ++i) {
            max_sizesw[i] = inputs[2]->buffer().dim[i].extent;
        }
        static_cast<CPUBackend*>(backend())->_outputs0.back().elementsize = sizeof(int32_t);

        static_cast<CPUBackend*>(backend())->total_input_num++;

        return NO_ERROR;

     }


    return NO_ERROR;
}

ErrorCode NITI_DSPLossGrad_Int8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    if(getDSPExecuteMode())
        return GlobalExecute(inputs,outputs);

    initDspGraph(hexagon_nn_, graph_id_);

    auto input = inputs[0];
    auto target = inputs[2];
    auto output = outputs[0];

    ascale = *(inputs[1]->host<int8_t>());

    int obatch = output->batch();
    int ochannel = output->channel();
    int owidth = output->width();
    int oheight = output->height();

    int osize = obatch*ochannel*owidth*oheight;

    inputlayer_output.push_back(hexagon_nn_output());
    inputlayer_output.back().rank = 4;
    auto& max_sizes0 = inputlayer_output.back().max_sizes;
    for (int i = 0; i < 4; ++i) {
        max_sizes0[i] = inputs[0]->buffer().dim[i].extent;
    }
    inputlayer_output.back().elementsize = sizeof(uint8_t);

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_INPUT,   NN_PAD_NA,   empty_input.data(), 0,   inputlayer_output.data(), inputlayer_output.size());

    lossgradlayer_input.push_back(hexagon_nn_input());
    lossgradlayer_input.back().src_id = op_id;
    lossgradlayer_input.back().output_idx = 0;
    
    op_id++;
    
    lossgradlayer_input.push_back(hexagon_nn_input());
    lossgradlayer_input.back().src_id = op_id;
    lossgradlayer_input.back().output_idx = 0;

    hexagon_nn_->hexagon_nn_append_const_node(
        graph_id_,                   // Graph handle we're appending into
        op_id++,                    // Node identifier (a unique uint32)
        target->batch(),                          // size: batches
        target->height(),                          // size: height
        target->width(),                          // size: width
        target->channel(),                          // size: depth
        target->host<uint8_t>(), // Pointer to data
        target->elementSize()*sizeof(int32_t)  // Length of data to copy
        );

    addConstInputTensor(hexagon_nn_, op_id, 0, lossgradlayer_input, (uint8_t*)&ascale, graph_id_, 1, sizeof(uint8_t));

    lossgradlayer_output.push_back(hexagon_nn_output());
    lossgradlayer_output.back().rank = 4;
    auto& max_sizes1 = lossgradlayer_output.back().max_sizes;
    for (int i = 0; i < 4; ++i) {
        max_sizes1[i] = outputs[0]->buffer().dim[i].extent;
    }
    lossgradlayer_output.back().elementsize = sizeof(uint8_t);

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_QuantizedLoss_8,   NN_PAD_NA,   lossgradlayer_input.data(), lossgradlayer_input.size(),   lossgradlayer_output.data(), lossgradlayer_output.size());

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

    uint8_t* inputPtr = inputs[0]->host<uint8_t>();
    for(int i=0;i<inputs[0]->elementSize();i++)
        inputPtr[i] = inputs[0]->host<int8_t>()[i] + 128;


    {
        input_tensors.emplace_back();
        auto& input_tensor = input_tensors.back();
        input_tensor.data = input->host<uint8_t>();
        input_tensor.dataLen = inputs[0]->elementSize();
        input_tensor.data_valid_len = inputs[0]->elementSize();
        input_tensor.batches = inputs[0]->batch();
        input_tensor.height = inputs[0]->height();
        input_tensor.width = inputs[0]->width();
        input_tensor.depth = inputs[0]->channel();
    }

    {
        output_tensors.emplace_back();
        auto& output_tensor = output_tensors.back();
        output_tensor.data = outputs[0]->host<uint8_t>();
        output_tensor.dataLen = outputs[0]->elementSize();
        output_tensor.batches = outputs[0]->batch();
        output_tensor.height = outputs[0]->height();
        output_tensor.width = outputs[0]->width();
        output_tensor.depth = outputs[0]->channel();
    }

    error = hexagon_nn_->hexagon_nn_execute_new( graph_id_, input_tensors.data(), input_tensors.size(),
      output_tensors.data(), output_tensors.size());
    
    if(error!=0){
        MNN_ERROR("Whoops... run failed: %d\n", error);
        return NOT_SUPPORT;
    }

    for(int i=0;i<outputs[0]->elementSize();i++)
        outputs[0]->host<int8_t>()[i] = outputs[0]->host<uint8_t>()[i] - 128;

    int64_t* out_grad_final_ptr = out_grad_final->host<int64_t>();

    hexagon_nn_->hexagon_nn_teardown(graph_id_);

    return NO_ERROR;
}

ErrorCode NITI_DSPLossGrad_Int8::GlobalExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    if(static_cast<CPUBackend*>(backend())->build_graph) {
        
        memcpy(target_dsp_buffer, inputs[2]->host<int32_t>() ,inputs[2]->elementSize()*sizeof(int32_t));

        {
            static_cast<CPUBackend*>(backend())->input_tensors.emplace_back();
            auto& input_tensor = static_cast<CPUBackend*>(backend())->input_tensors.back();
            input_tensor.data = (uint8_t*)target_dsp_buffer;
            input_tensor.dataLen = inputs[2]->elementSize()*sizeof(int32_t);
            input_tensor.data_valid_len = inputs[2]->elementSize()*sizeof(int32_t);
            input_tensor.batches = inputs[2]->batch();
            input_tensor.height = inputs[2]->height();
            input_tensor.width = inputs[2]->width();
            input_tensor.depth = inputs[2]->channel();
        }
        return NO_ERROR;
    }

    hexagon_nn_ = static_cast<CPUBackend*>(backend())->global_hexagon_nn_;
    graph_id_ = static_cast<CPUBackend*>(backend())->global_graph_id_;

    op_id = static_cast<CPUBackend*>(backend())->global_op_id;

    int input_op_id = backend()->get_Op_id(inputs[0], OpType_NITI_DSP_LOSSGRAD_Int8);

    lossgradlayer_input.push_back(hexagon_nn_input());
    lossgradlayer_input.back().src_id = input_op_id;
    lossgradlayer_input.back().output_idx = 0;

    lossgradlayer_input.push_back(hexagon_nn_input());
    lossgradlayer_input.back().src_id = 0x1000;
    lossgradlayer_input.back().output_idx = static_cast<CPUBackend*>(backend())->current_input_num;

    static_cast<CPUBackend*>(backend())->current_input_num++;

    op_id++;
    ascale = -4;
    addConstInputTensor(hexagon_nn_, op_id, 0, lossgradlayer_input, (uint8_t*)&ascale, graph_id_, 1, sizeof(uint8_t));


    op_id++;

    lossgradlayer_output.push_back(hexagon_nn_output());
    lossgradlayer_output.back().rank = 4;
    auto& max_sizes1 = lossgradlayer_output.back().max_sizes;
    for (int i = 0; i < 4; ++i) {
        max_sizes1[i] = outputs[0]->buffer().dim[i].extent;
    }
    lossgradlayer_output.back().elementsize = sizeof(uint8_t);

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_QuantizedLoss_8,   NN_PAD_NA,   lossgradlayer_input.data(), lossgradlayer_input.size(),   lossgradlayer_output.data(), lossgradlayer_output.size());

    static_cast<CPUBackend*>(backend())->opadr_opid_map.insert(std::make_pair(outputs[0]->host<int8_t>(), op_id ));

    op_id++;
    static_cast<CPUBackend*>(backend())->global_op_id = op_id;

    {
        if(target_dsp_buffer == NULL) 
        {
            target_dsp_buffer = (int32_t*)malloc(inputs[2]->elementSize()*sizeof(int32_t));
        }
        memcpy(target_dsp_buffer, inputs[2]->host<int32_t>() ,inputs[2]->elementSize()*sizeof(int32_t));
        
        static_cast<CPUBackend*>(backend())->input_tensors.emplace_back();
        auto& input_tensor = static_cast<CPUBackend*>(backend())->input_tensors.back();
        input_tensor.data = (uint8_t*)target_dsp_buffer;
        input_tensor.dataLen = inputs[2]->elementSize()*sizeof(int32_t);
        input_tensor.data_valid_len = inputs[2]->elementSize()*sizeof(int32_t);
        input_tensor.batches = inputs[2]->batch();
        input_tensor.height = inputs[2]->height();
        input_tensor.width = inputs[2]->width();
        input_tensor.depth = inputs[2]->channel();
    }

    return NO_ERROR;
}

NITI_DSPLossGrad_Int8::NITI_DSPLossGrad_Int8(Backend *b, const MNN::Op *op) : MNN::Execution(b) {
    if(!getDSPExecuteMode())
        hexagon_nn_ = generate_interface();
}

NITI_DSPLossGrad_Int8::~NITI_DSPLossGrad_Int8()  {
    // nothing to do
}


int32_t* NITI_DSPLossGrad_Int8::target_dsp_buffer = NULL;

Execution* NITI_DSPLossGrad_Int8::create(const MNN::Op *op, Backend *backend) {
    return new NITI_DSPLossGrad_Int8(backend, op);
}

class NITI_DSPLossGrad_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return NITI_DSPLossGrad_Int8::create(op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(NITI_DSPLossGrad_Int8_Creator, OpType_NITI_DSP_LOSSGRAD_Int8);

} // namespace MNN
