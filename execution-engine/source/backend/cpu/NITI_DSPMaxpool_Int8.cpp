//
//  NITI_DSP_Maxpool_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/8/17.
//  
//

#include "backend/cpu/NITI_DSPMaxpool_Int8.hpp"
#include "backend/cpu/CPUBackend.hpp"

#include "core/Macro.h"

#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#include "core/Concurrency.h"

#define DST_TILE 16
#define CACHE_SIZE 128

namespace MNN {

ErrorCode NITI_DSP_Maxpool_Int8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    auto input = inputs[0];

    inputUINT8Tensor.reset(Tensor::createDevice<uint8_t>({input->batch(), input->height(), input->width(), input->channel()}));
    bool success = backend()->onAcquireBuffer(inputUINT8Tensor.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    backend()->onReleaseBuffer(inputUINT8Tensor.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode NITI_DSP_Maxpool_Int8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    if(getDSPExecuteMode())
        return GlobalExecute(inputs,outputs);

    initDspGraph(hexagon_nn_, graph_id_);

    auto input = inputs[0];
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

    maxpoollayer_input.push_back(hexagon_nn_input());
    maxpoollayer_input.back().src_id = op_id;
    maxpoollayer_input.back().output_idx = 0;

    op_id++;
    addConstInputTensor(hexagon_nn_, op_id, 0, maxpoollayer_input, (uint8_t*)&input_min, graph_id_, 1, sizeof(float));
    addConstInputTensor(hexagon_nn_, op_id, 0, maxpoollayer_input, (uint8_t*)&input_max, graph_id_, 1, sizeof(float));

    maxpoollayer_input.push_back(hexagon_nn_input());
    maxpoollayer_input.back().src_id = op_id;
    maxpoollayer_input.back().output_idx = 0;

    hexagon_nn_->hexagon_nn_append_const_node(
        graph_id_,                   // Graph handle we're appending into
        op_id++,                    // Node identifier (a unique uint32)
        1,                          // size: batches
        mParameter->kernelX(),                          // size: height
        mParameter->kernelY(),                          // size: width
        1,                          // size: depth
        NULL, // Pointer to data
        0  // Length of data to copy
        );
    // window shape
    maxpoollayer_input.push_back(hexagon_nn_input());
    maxpoollayer_input.back().src_id = op_id;
    maxpoollayer_input.back().output_idx = 0;

    hexagon_nn_->hexagon_nn_append_const_node(
        graph_id_,                   // Graph handle we're appending into
        op_id++,                    // Node identifier (a unique uint32)
        1,                          // size: batches
        mParameter->strideX(),                          // size: height
        mParameter->strideY(),                          // size: width
        1,                          // size: depth
        NULL, // Pointer to data
        0  // Length of data to copy
        );

    maxpoollayer_output.push_back(hexagon_nn_output());
    maxpoollayer_output.back().rank = 4;
    auto& max_sizes1 = maxpoollayer_output.back().max_sizes;
    for (int i = 0; i < 4; ++i) {
        max_sizes1[i] = outputs[0]->buffer().dim[i].extent;
    }
    maxpoollayer_output.back().elementsize = sizeof(uint8_t);

    std::vector<int> max_size;
    max_size.push_back(1);
    max_size.push_back(1);
    max_size.push_back(1);
    max_size.push_back(1);
    addOutputTensor(max_size,maxpoollayer_output, sizeof(float));
    addOutputTensor(max_size,maxpoollayer_output, sizeof(float));

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_QuantizedMaxPool_8,   NN_PAD_NA,   maxpoollayer_input.data(), maxpoollayer_input.size(),   maxpoollayer_output.data(), maxpoollayer_output.size());

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

    uint8_t* inputPtr = inputUINT8Tensor->host<uint8_t>();
    for(int i=0;i<inputs[0]->elementSize();i++)
        inputPtr[i] = inputs[0]->host<int8_t>()[i] + 128;
    
    {
        input_tensors.emplace_back();
        auto& input_tensor = input_tensors.back();
        input_tensor.data = inputPtr;
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

    for(int i=0;i<outputs[0]->elementSize();i++)
        outputs[0]->host<int8_t>()[i] = outputs[0]->host<uint8_t>()[i] - 128;

    
    *(outputs[1]->host<int8_t>()) = *(inputs[1]->host<int8_t>());

    return NO_ERROR;
}
ErrorCode NITI_DSP_Maxpool_Int8::GlobalExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    if(static_cast<CPUBackend*>(backend())->build_graph) {
        return NO_ERROR;
    }

    hexagon_nn_ = static_cast<CPUBackend*>(backend())->global_hexagon_nn_;
    graph_id_ = static_cast<CPUBackend*>(backend())->global_graph_id_;

    op_id = static_cast<CPUBackend*>(backend())->global_op_id;

    int input_op_id = backend()->get_Op_id(inputs[0], OpType_NITI_DSP_MAXPOOL_Int8);

    // maxpool layer
    maxpoollayer_input.push_back(hexagon_nn_input());
    maxpoollayer_input.back().src_id = input_op_id;
    maxpoollayer_input.back().output_idx = 0;

    op_id++;
    addConstInputTensor(hexagon_nn_, op_id, 0, maxpoollayer_input, (uint8_t*)&input_min, graph_id_, 1, sizeof(float));
    addConstInputTensor(hexagon_nn_, op_id, 0, maxpoollayer_input, (uint8_t*)&input_max, graph_id_, 1, sizeof(float));

    // stride shape
    maxpoollayer_input.push_back(hexagon_nn_input());
    maxpoollayer_input.back().src_id = op_id;
    maxpoollayer_input.back().output_idx = 0;

    hexagon_nn_->hexagon_nn_append_const_node(
        graph_id_,                   // Graph handle we're appending into
        op_id++,                    // Node identifier (a unique uint32)
        1,                          // size: batches
        mParameter->kernelX(),                          // size: height
        mParameter->kernelY(),                          // size: width
        1,                          // size: depth
        NULL, // Pointer to data
        0  // Length of data to copy
        );
    // window shape
    maxpoollayer_input.push_back(hexagon_nn_input());
    maxpoollayer_input.back().src_id = op_id;
    maxpoollayer_input.back().output_idx = 0;

    hexagon_nn_->hexagon_nn_append_const_node(
        graph_id_,                   // Graph handle we're appending into
        op_id++,                    // Node identifier (a unique uint32)
        1,                          // size: batches
        mParameter->strideX(),                          // size: height
        mParameter->strideY(),                          // size: width
        1,                          // size: depth
        NULL, // Pointer to data
        0  // Length of data to copy
        );

    maxpoollayer_output.push_back(hexagon_nn_output());
    maxpoollayer_output.back().rank = 4;
    auto& max_sizes1 = maxpoollayer_output.back().max_sizes;
    for (int i = 0; i < 4; ++i) {
        max_sizes1[i] = outputs[0]->buffer().dim[i].extent;
    }
    maxpoollayer_output.back().elementsize = sizeof(uint8_t);

    std::vector<int> max_size;
    max_size.push_back(1);
    max_size.push_back(1);
    max_size.push_back(1);
    max_size.push_back(1);
    addOutputTensor(max_size,maxpoollayer_output, sizeof(float));
    addOutputTensor(max_size,maxpoollayer_output, sizeof(float));

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_QuantizedMaxPool_8,   NN_PAD_NA,   maxpoollayer_input.data(), maxpoollayer_input.size(),   maxpoollayer_output.data(), maxpoollayer_output.size());


    static_cast<CPUBackend*>(backend())->opadr_opid_map.insert(std::make_pair(outputs[0]->host<int8_t>(), op_id ));

    op_id++;
    static_cast<CPUBackend*>(backend())->global_op_id = op_id;

    return NO_ERROR;
}

NITI_DSP_Maxpool_Int8::~NITI_DSP_Maxpool_Int8() {
    // Do nothing
}
NITI_DSP_Maxpool_Int8::NITI_DSP_Maxpool_Int8(Backend* backend, const NITI_Pool_Int8* pool) : MNN::Execution(backend) {
    mParameter = pool;
    if(!getDSPExecuteMode())
        hexagon_nn_ = generate_interface();
}

class NITI_DSP_Maxpool_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {

        return new NITI_DSP_Maxpool_Int8(backend, op->main_as_NITI_Pool_Int8());
    }
};

REGISTER_CPU_OP_CREATOR(NITI_DSP_Maxpool_Int8_Creator, OpType_NITI_DSP_MAXPOOL_Int8);

} // namespace MNN
