//
//  NITI_DSPMaxPoolGrad_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/11/19.
//  
//

#include "backend/cpu/NITI_DSPMaxPoolGrad_Int8.hpp"
#include "core/Macro.h"
#include "math/Vec.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"

#include "MNN/hexagon/hexagon_nn/hexagon_nn_ops.h"
#include "MNN/hexagon/hexagon_nn/hexagon_nn.h"
#include "MNN/hexagon/hexagon_nn/HexagonRunningUtils.hpp"

namespace MNN {

ErrorCode NITI_DSPMaxPoolGrad_Int8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    auto input = inputs[0];
    auto originOutput = inputs[1];

    inputUINT8Tensor.reset(Tensor::createDevice<uint8_t>({input->batch(), input->height(), input->width(), input->channel()}));
    bool success = backend()->onAcquireBuffer(inputUINT8Tensor.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    originOutputUINT8Tensor.reset(Tensor::createDevice<uint8_t>({originOutput->batch(), originOutput->height(), originOutput->width(), originOutput->channel()}));
    success = backend()->onAcquireBuffer(originOutputUINT8Tensor.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    outdiffUINT8Tensor.reset(Tensor::createDevice<uint8_t>({originOutput->batch(), originOutput->height(), originOutput->width(), originOutput->channel()}));
    success = backend()->onAcquireBuffer(outdiffUINT8Tensor.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    backend()->onReleaseBuffer(inputUINT8Tensor.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(originOutputUINT8Tensor.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(outdiffUINT8Tensor.get(), Backend::DYNAMIC);

    pad = 32 - input->channel() % 32;
    if(pad == 32)
        pad = 0;

    return NO_ERROR;
}

ErrorCode NITI_DSPMaxPoolGrad_Int8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    if(getDSPExecuteMode())
        return GlobalExecute(inputs,outputs);

    initDspGraph(hexagon_nn_, graph_id_);

    auto origin       = inputs[0];
    auto outputOrigin = inputs[1];
    auto outputDiff    = inputs[2];
    auto output   = outputs[0];

    auto iw = origin->height();
    auto ih = origin->batch();

    auto ib = origin->width();
    auto ic = origin->channel();

    uint8_t* originPtr = inputUINT8Tensor->host<uint8_t>();
    uint8_t* outputOriginPtr = originOutputUINT8Tensor->host<uint8_t>();

    uint8_t* outputDiffPtr = outdiffUINT8Tensor->host<uint8_t>();
    uint8_t* outputPtr = output->host<uint8_t>();

    for(int i=0;i<inputs[0]->elementSize();i++)
        originPtr[i] = inputs[0]->host<int8_t>()[i] + 128;

    for(int i=0;i<inputs[1]->elementSize();i++)
        outputOriginPtr[i] = inputs[1]->host<int8_t>()[i] + 128;

    for(int i=0;i<inputs[2]->elementSize();i++)
        outputDiffPtr[i] = inputs[2]->host<int8_t>()[i] + 128;

    // input layer
    for(int n=0; n<3;n++) {
        inputlayer_output.push_back(hexagon_nn_output());
        inputlayer_output.back().rank = 4;
        auto& max_sizes0 = inputlayer_output.back().max_sizes;
        for (int i = 0; i < 4; ++i) {
            max_sizes0[i] = inputs[n]->buffer().dim[i].extent;
        }
        inputlayer_output.back().elementsize = sizeof(uint8_t);
    }
    

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_INPUT,   NN_PAD_NA,   empty_input.data(), 0,   inputlayer_output.data(), inputlayer_output.size());

    int input_op_id = op_id;

    // transpose input layer
    transposeinput_shape.push_back(0);
    transposeinput_shape.push_back(0);
    transposeinput_shape.push_back(0);
    transposeinput_shape.push_back(0);
    transposeinput_shape.push_back(0);
    transposeinput_shape.push_back(0);
    transposeinput_shape.push_back(0);
    transposeinput_shape.push_back(pad);
    // addInputTransposeLayer(hexagon_nn_, graph_id_, op_id, inputs[0], transposeinputlayer_input, transposeinputlayer_output, transposeinput_shape, &input_min, &input_max);
    addInputPadLayer(hexagon_nn_, graph_id_, input_op_id, op_id, inputs[0], transposeinputlayer_input, transposeinputlayer_output, transposeinput_shape, &input_min, &input_max, &padValue, 0);


    // relu layer
    maxpoolgradlayer_input.push_back(hexagon_nn_input());
    maxpoolgradlayer_input.back().src_id = op_id;
    maxpoolgradlayer_input.back().output_idx = 0;

    op_id++;

    addInputPadLayer(hexagon_nn_, graph_id_, input_op_id, op_id, inputs[1], transposeoriginoutputlayer_input, transposeoriginoutputlayer_output, transposeinput_shape, &input_min, &input_max, &padValue, 1);

    maxpoolgradlayer_input.push_back(hexagon_nn_input());
    maxpoolgradlayer_input.back().src_id = op_id;
    maxpoolgradlayer_input.back().output_idx = 0;


    op_id++;

    addInputPadLayer(hexagon_nn_, graph_id_, input_op_id, op_id, inputs[2], transposeoutdifflayer_input, transposeoutdifflayer_output, transposeinput_shape, &input_min, &input_max, &padValue, 2);

    maxpoolgradlayer_input.push_back(hexagon_nn_input());
    maxpoolgradlayer_input.back().src_id = op_id;
    maxpoolgradlayer_input.back().output_idx = 0;

    op_id++;

    maxpoolgradlayer_input.push_back(hexagon_nn_input());
    maxpoolgradlayer_input.back().src_id = op_id;
    maxpoolgradlayer_input.back().output_idx = 0;

    hexagon_nn_->hexagon_nn_append_const_node(
        graph_id_,                   // Graph handle we're appending into
        op_id++,                    // Node identifier (a unique uint32)
        mStrideX,                          // size: batches
        mStrideY,                          // size: height
        mKernelX,                          // size: width
        mKernelY,                          // size: depth
        NULL, // Pointer to data
        0  // Length of data to copy
        );

    addConstInputTensor(hexagon_nn_, op_id, 0, maxpoolgradlayer_input, (uint8_t*)&pad, graph_id_, 1, sizeof(int32_t));


    maxpoolgradlayer_output.push_back(hexagon_nn_output());
    maxpoolgradlayer_output.back().rank = 4;
    auto& max_sizes1 = maxpoolgradlayer_output.back().max_sizes;
    // for (int i = 0; i < 4; ++i) {
        max_sizes1[0] = outputs[0]->buffer().dim[0].extent;
        max_sizes1[1] = outputs[0]->buffer().dim[1].extent;
        max_sizes1[2] = outputs[0]->buffer().dim[2].extent + transposeinput_shape[5];
        max_sizes1[3] = outputs[0]->buffer().dim[3].extent + transposeinput_shape[7];
    // }
    maxpoolgradlayer_output.back().elementsize = sizeof(uint8_t);


    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_QuantizedMaxPoolGrad_8,   NN_PAD_NA,   maxpoolgradlayer_input.data(), maxpoolgradlayer_input.size(),   maxpoolgradlayer_output.data(), maxpoolgradlayer_output.size());

    if (pad != 0) {
        transposeoutputlayer_input.push_back(hexagon_nn_input());
        transposeoutputlayer_input.back().src_id = op_id;
        transposeoutputlayer_input.back().output_idx = 0;

        transposeoutputlayer_output.push_back(hexagon_nn_output());
        transposeoutputlayer_output.back().rank = 4;
        auto& max_sizes3 = transposeoutputlayer_output.back().max_sizes;
        // for (int i = 0; i < 4; ++i) {
            max_sizes3[0] = outputs[0]->buffer().dim[0].extent;
            max_sizes3[1] = outputs[0]->buffer().dim[1].extent;
            max_sizes3[2] = outputs[0]->buffer().dim[2].extent;
            max_sizes3[3] = outputs[0]->buffer().dim[3].extent;
        // }
        transposeoutputlayer_output.back().elementsize = sizeof(uint8_t);

        op_id++;

        hexagon_nn_->hexagon_nn_append_node(graph_id_,     
                op_id,   OP_Convert_from_d32,   NN_PAD_NA,   transposeoutputlayer_input.data(), transposeoutputlayer_input.size(),   transposeoutputlayer_output.data(), transposeoutputlayer_output.size());
    
    }
    
    
    
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
        input_tensor.data = originPtr;
        input_tensor.dataLen = inputs[0]->elementSize();
        input_tensor.data_valid_len = inputs[0]->elementSize();
        input_tensor.batches = inputs[0]->batch();
        input_tensor.height = inputs[0]->height();
        input_tensor.width = inputs[0]->width();
        input_tensor.depth = inputs[0]->channel();
    }

    {
        input_tensors.emplace_back();
        auto& input_tensor = input_tensors.back();
        input_tensor.data = outputOriginPtr;
        input_tensor.dataLen = inputs[1]->elementSize();
        input_tensor.data_valid_len = inputs[1]->elementSize();
        input_tensor.batches = inputs[1]->batch();
        input_tensor.height = inputs[1]->height();
        input_tensor.width = inputs[1]->width();
        input_tensor.depth = inputs[1]->channel();
    }

    {
        input_tensors.emplace_back();
        auto& input_tensor = input_tensors.back();
        input_tensor.data = outputDiffPtr;
        input_tensor.dataLen = inputs[2]->elementSize();
        input_tensor.data_valid_len = inputs[2]->elementSize();
        input_tensor.batches = inputs[2]->batch();
        input_tensor.height = inputs[2]->height();
        input_tensor.width = inputs[2]->width();
        input_tensor.depth = inputs[2]->channel();
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

    return NO_ERROR;
}

ErrorCode NITI_DSPMaxPoolGrad_Int8::GlobalExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {

    if(static_cast<CPUBackend*>(backend())->build_graph) {
        return NO_ERROR;
    }

    hexagon_nn_ = static_cast<CPUBackend*>(backend())->global_hexagon_nn_;
    graph_id_ = static_cast<CPUBackend*>(backend())->global_graph_id_;

    op_id = static_cast<CPUBackend*>(backend())->global_op_id;

    int input_op_id = static_cast<CPUBackend*>(backend())->opadr_opid_map[inputs[0]->host<int8_t>()];

    auto origin       = inputs[0];
    auto outputOrigin = inputs[1];
    auto outputDiff    = inputs[2];
    auto output   = outputs[0];


    int ichannel = origin->channel();

    if(pad > 0) {
        // transpose input layer
        transposeinput_shape.push_back(0);
        transposeinput_shape.push_back(0);
        transposeinput_shape.push_back(0);
        transposeinput_shape.push_back(0);
        transposeinput_shape.push_back(0);
        transposeinput_shape.push_back(0);
        transposeinput_shape.push_back(0);
        transposeinput_shape.push_back(pad);
        // addInputTransposeLayer(hexagon_nn_, graph_id_, op_id, inputs[0], transposeinputlayer_input, transposeinputlayer_output, transposeinput_shape, &input_min, &input_max);
        addInputPadLayer(hexagon_nn_, graph_id_, input_op_id, op_id, inputs[0], transposeinputlayer_input, transposeinputlayer_output, transposeinput_shape, &input_min, &input_max, &padValue, 0);


        // relu layer
        maxpoolgradlayer_input.push_back(hexagon_nn_input());
        maxpoolgradlayer_input.back().src_id = op_id;
        maxpoolgradlayer_input.back().output_idx = 0;

        op_id++;

        int originougtput_op_id = static_cast<CPUBackend*>(backend())->opadr_opid_map[inputs[1]->host<int8_t>()];

        addInputPadLayer(hexagon_nn_, graph_id_, originougtput_op_id, op_id, inputs[1], transposeoriginoutputlayer_input, transposeoriginoutputlayer_output, transposeinput_shape, &input_min, &input_max, &padValue, 0);

        maxpoolgradlayer_input.push_back(hexagon_nn_input());
        maxpoolgradlayer_input.back().src_id = op_id;
        maxpoolgradlayer_input.back().output_idx = 0;


        op_id++;

        int outdiff_op_id = static_cast<CPUBackend*>(backend())->opadr_opid_map[inputs[2]->host<int8_t>()];

        addInputPadLayer(hexagon_nn_, graph_id_, outdiff_op_id, op_id, inputs[2], transposeoutdifflayer_input, transposeoutdifflayer_output, transposeinput_shape, &input_min, &input_max, &padValue, 0);

        maxpoolgradlayer_input.push_back(hexagon_nn_input());
        maxpoolgradlayer_input.back().src_id = op_id;
        maxpoolgradlayer_input.back().output_idx = 0;
    } else {

        // relu layer
        maxpoolgradlayer_input.push_back(hexagon_nn_input());
        maxpoolgradlayer_input.back().src_id = input_op_id;
        maxpoolgradlayer_input.back().output_idx = 0;

        int originougtput_op_id = static_cast<CPUBackend*>(backend())->opadr_opid_map[inputs[1]->host<int8_t>()];

        maxpoolgradlayer_input.push_back(hexagon_nn_input());
        maxpoolgradlayer_input.back().src_id = originougtput_op_id;
        maxpoolgradlayer_input.back().output_idx = 0;

        int outdiff_op_id = static_cast<CPUBackend*>(backend())->opadr_opid_map[inputs[2]->host<int8_t>()];

        maxpoolgradlayer_input.push_back(hexagon_nn_input());
        maxpoolgradlayer_input.back().src_id = outdiff_op_id;
        maxpoolgradlayer_input.back().output_idx = 0;
    }

    

    op_id++;

    maxpoolgradlayer_input.push_back(hexagon_nn_input());
    maxpoolgradlayer_input.back().src_id = op_id;
    maxpoolgradlayer_input.back().output_idx = 0;

    hexagon_nn_->hexagon_nn_append_const_node(
        graph_id_,                   // Graph handle we're appending into
        op_id++,                    // Node identifier (a unique uint32)
        mStrideX,                          // size: batches
        mStrideY,                          // size: height
        mKernelX,                          // size: width
        mKernelY,                          // size: depth
        NULL, // Pointer to data
        0  // Length of data to copy
        );

    addConstInputTensor(hexagon_nn_, op_id, 0, maxpoolgradlayer_input, (uint8_t*)&pad, graph_id_, 1, sizeof(int32_t));


    maxpoolgradlayer_output.push_back(hexagon_nn_output());
    maxpoolgradlayer_output.back().rank = 4;
    auto& max_sizes1 = maxpoolgradlayer_output.back().max_sizes;
    // for (int i = 0; i < 4; ++i) {
        max_sizes1[0] = outputs[0]->buffer().dim[0].extent;
        max_sizes1[1] = outputs[0]->buffer().dim[1].extent;
        max_sizes1[2] = outputs[0]->buffer().dim[2].extent;
        if(pad>0)
            max_sizes1[3] = outputs[0]->buffer().dim[3].extent + transposeinput_shape[7];
        else
             max_sizes1[3] = outputs[0]->buffer().dim[3].extent;
    // }
    maxpoolgradlayer_output.back().elementsize = sizeof(uint8_t);


    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_QuantizedMaxPoolGrad_8,   NN_PAD_NA,   maxpoolgradlayer_input.data(), maxpoolgradlayer_input.size(),   maxpoolgradlayer_output.data(), maxpoolgradlayer_output.size());

    // from d32 layer
    transposeoutputlayer_input.push_back(hexagon_nn_input());
    transposeoutputlayer_input.back().src_id = op_id;
    transposeoutputlayer_input.back().output_idx = 0;

    transposeoutputlayer_output.push_back(hexagon_nn_output());
    transposeoutputlayer_output.back().rank = 4;
    auto& max_sizes3 = transposeoutputlayer_output.back().max_sizes;
    // for (int i = 0; i < 4; ++i) {
        max_sizes3[0] = outputs[0]->buffer().dim[0].extent;
        max_sizes3[1] = outputs[0]->buffer().dim[1].extent;
        max_sizes3[2] = outputs[0]->buffer().dim[2].extent;
        max_sizes3[3] = outputs[0]->buffer().dim[3].extent;
    // }
    transposeoutputlayer_output.back().elementsize = sizeof(uint8_t);


    op_id++;

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_Convert_from_d32,   NN_PAD_NA,   transposeoutputlayer_input.data(), transposeoutputlayer_input.size(),   transposeoutputlayer_output.data(), transposeoutputlayer_output.size());
    

    static_cast<CPUBackend*>(backend())->opadr_opid_map.insert(std::make_pair(outputs[0]->host<int8_t>(), op_id ));

    op_id++;
    static_cast<CPUBackend*>(backend())->global_op_id = op_id;

    return NO_ERROR;
}

class NITI_DSPMaxPoolGrad_Int8Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto pool = op->main_as_NITI_Pool_Int8();
        return new NITI_DSPMaxPoolGrad_Int8(backend, op->main_as_NITI_Pool_Int8());
    }
};

REGISTER_CPU_OP_CREATOR(NITI_DSPMaxPoolGrad_Int8Creator, OpType_NITI_DSP_MAXPOOLGRAD_Int8);
} // namespace MNN
