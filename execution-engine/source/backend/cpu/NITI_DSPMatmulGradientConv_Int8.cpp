//
//  NITI_DSPMatmulGradientConv_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/12/20.
//  
//

#include "backend/cpu/NITI_DSPMatmulGradientConv_Int8.hpp"
#include "backend/cpu/CPUBackend.hpp"

#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include <math.h>

#include "MNN/hexagon/hexagon_nn/hexagon_nn_ops.h"
#include "MNN/hexagon/hexagon_nn/hexagon_nn.h"
#include "MNN/hexagon/hexagon_nn/HexagonRunningUtils.hpp"

#include <MNN/AutoTime.hpp>
#include <iostream>

namespace MNN {

NITI_DSPMatmulGradientConv_Int8::~NITI_DSPMatmulGradientConv_Int8() {
    // Do nothing
    // hexagon_nn_->hexagon_nn_global_teardown();
}
NITI_DSPMatmulGradientConv_Int8::NITI_DSPMatmulGradientConv_Int8(Backend* backend, const NITI_CONV_Int8* convOp, const MNN::Op* op) : CPUConvolution(convOp->common(), backend) {
    if(!getDSPExecuteMode())
        hexagon_nn_ = generate_interface();
    mOp = op;
}

#define LPAD 2 
#define roundup(a, p2)       (((a)+(p2)-1)&~((p2)-1))

ErrorCode NITI_DSPMatmulGradientConv_Int8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {

    int ichannel = inputs[0]->channel();
    auto weight = inputs[1];
    auto input = inputs[0];

    biasTensor.reset(Tensor::createDevice<int32_t>({1, 1, 1, outputs[0]->channel()}));
    bool success = backend()->onAcquireBuffer(biasTensor.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    weightUINT8Tensor.reset(Tensor::createDevice<uint8_t>({weight->batch(), weight->height(), weight->width(), weight->channel()}));
    success = backend()->onAcquireBuffer(weightUINT8Tensor.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    inputUINT8Tensor.reset(Tensor::createDevice<uint8_t>({input->batch(), input->height(), input->width(), input->channel()}));
    success = backend()->onAcquireBuffer(inputUINT8Tensor.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    {   
		int32_t in_width = inputs[0]->width();
		int32_t in_height = inputs[0]->height();
        if(mCommon->padX() == 1) {
            in_width +=2;
            in_height +=2;
        }
		int32_t required_w_before, required_h_before, required_h_after;
		int32_t out_width = outputs[0]->height();
		int32_t out_width_pad = roundup(out_width, 4);
		int32_t out_height = outputs[0]->batch();
        int32_t out_depth = outputs[0]->channel();
		required_w_before += LPAD;

		int32_t out_left_pad = 4;
		int32_t out_right_pad = out_width_pad-out_width;
		int32_t out_top_pad = 4;
		int32_t out_bottom_pad = out_top_pad;
		int32_t out_depth_before_pad = 0;
		int32_t out_depth_after_pad = (-out_depth) & 31; // padding amount in case out depth != 32

		int32_t out_depth_total = out_depth + out_depth_before_pad + out_depth_after_pad;
		int32_t out_width_total = out_width + out_left_pad + out_right_pad;
		int32_t out_height_total = out_height + out_top_pad + out_bottom_pad;
		int32_t out_batch = outputs[0]->width();

		total_int32_num = out_batch*out_height_total*out_width_total*out_depth_total;

		output32TensorShape.push_back(out_batch);
        output32TensorShape.push_back(out_height_total);
        output32TensorShape.push_back(out_width_total);
        output32TensorShape.push_back(out_depth_total);
	}

    backend()->onReleaseBuffer(biasTensor.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(weightUINT8Tensor.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(inputUINT8Tensor.get(), Backend::DYNAMIC);
    
    
    return NO_ERROR;
}

ErrorCode NITI_DSPMatmulGradientConv_Int8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {

    if(getDSPExecuteMode())
        return GlobalExecute(inputs,outputs);

    const auto input = inputs[0];
    const auto weight = inputs[1];

    initDspGraph(hexagon_nn_, graph_id_);

    int ibatch = input->batch();
    int iheight = input->height();
    int iwidth = input->width();
    int ichannel = input->channel();


    int wbatch = weight->batch();
    int wheight = weight->height();
    int wwidth = weight->width();
    int wchannel = weight->channel();

    int obatch = outputs[0]->batch();
    int oheight = outputs[0]->height();
    int owidth = outputs[0]->width();
    int ochannel = outputs[0]->channel();

    uint8_t* weightTensorPtr = weightUINT8Tensor->host<uint8_t>();

    bias = biasTensor->host<uint8_t>();
    int channelout = ochannel;
    memset(bias, 0, channelout*sizeof(int32_t));

    _outputs0.push_back(hexagon_nn_output());
    _outputs0.back().rank = 4;
    auto& max_sizes0 = _outputs0.back().max_sizes;
    for (int i = 0; i < 4; ++i) {
        max_sizes0[i] = inputs[0]->buffer().dim[i].extent;
    }
    _outputs0.back().elementsize = sizeof(uint8_t);

    _outputs0.push_back(hexagon_nn_output());
    _outputs0.back().rank = 4;
    auto& max_sizesw = _outputs0.back().max_sizes;
    for (int i = 0; i < 4; ++i) {
        max_sizesw[i] = inputs[1]->buffer().dim[i].extent;
    }
    _outputs0.back().elementsize = sizeof(uint8_t);


    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_INPUT,   NN_PAD_NA,   empty_input_list.data(), 0,   _outputs0.data(), _outputs0.size());

    int input_op_id = op_id;

    pad = mCommon->padX();
    if(mCommon->padX() == 1) {

        padtensor.push_back(0);
        padtensor.push_back(0);
        padtensor.push_back(1);
        padtensor.push_back(1);
        padtensor.push_back(1);
        padtensor.push_back(1);
        padtensor.push_back(0);
        padtensor.push_back(0);

        padlayer_input.push_back(hexagon_nn_input());
        padlayer_input.back().src_id = op_id;
        padlayer_input.back().output_idx = 0;

        op_id++;

        addConstInputTensor(hexagon_nn_, op_id, 0, padlayer_input, (uint8_t*)&input_min, graph_id_, 1, sizeof(float));
        addConstInputTensor(hexagon_nn_, op_id, 0, padlayer_input, (uint8_t*)&input_max, graph_id_, 1, sizeof(float));

        // control tensor
        padlayer_input.push_back(hexagon_nn_input());
        padlayer_input.back().src_id = op_id;
        padlayer_input.back().output_idx = 0;

        hexagon_nn_->hexagon_nn_append_const_node(
            graph_id_,                   // Graph handle we're appending into
            op_id++,                    // Node identifier (a unique uint32)
            1,                          // size: batches
            1,                          // size: height
            4,                          // size: width
            2,                          // size: depth
            (uint8_t*)padtensor.data(), // Pointer to data
            8*sizeof(int32_t)  // Length of data to copy
            );
        
        addConstInputTensor(hexagon_nn_, op_id, 0, padlayer_input, (uint8_t*)&padValue, graph_id_, 1, sizeof(uint8_t));

        padlayer_output.push_back(hexagon_nn_output());
        padlayer_output.back().rank = 4;
        auto& max_sizes1 = padlayer_output.back().max_sizes;
        
        max_sizes1[0] = inputs[0]->buffer().dim[0].extent;
        max_sizes1[1] = inputs[0]->buffer().dim[1].extent+2;
        max_sizes1[2] = inputs[0]->buffer().dim[2].extent+2;
        max_sizes1[3] = inputs[0]->buffer().dim[3].extent;
        
        padlayer_output.back().elementsize = sizeof(uint8_t);

        std::vector<int> max_size;
        max_size.push_back(1);
        max_size.push_back(1);
        max_size.push_back(1);
        max_size.push_back(1);
        addOutputTensor(max_size,padlayer_output, sizeof(float));
        addOutputTensor(max_size,padlayer_output, sizeof(float));

        hexagon_nn_->hexagon_nn_append_node(graph_id_,     
                op_id,   OP_QuantizedPad_V2_8,   NN_PAD_NA,   padlayer_input.data(), padlayer_input.size(),   padlayer_output.data(), padlayer_output.size());
    }

    im2collayer_input.push_back(hexagon_nn_input());
    im2collayer_input.back().src_id = op_id;
    im2collayer_input.back().output_idx = 0;

    op_id++;

    // kernel tensor
    im2collayer_input.push_back(hexagon_nn_input());
    im2collayer_input.back().src_id = op_id;
    im2collayer_input.back().output_idx = 0;

    hexagon_nn_->hexagon_nn_append_const_node(
        graph_id_,                   // Graph handle we're appending into
        op_id++,                    // Node identifier (a unique uint32)
        1,                          // size: batches
        wheight,                          // size: height
        wwidth,                          // size: width
        1,                          // size: depth
        NULL, // Pointer to data
        0  // Length of data to copy
        );


    im2collayer_output.push_back(hexagon_nn_output());
    im2collayer_output.back().rank = 4;
    auto& max_sizes1 = im2collayer_output.back().max_sizes;
    
    max_sizes1[0] = ibatch;
    max_sizes1[1] = obatch*oheight;
    max_sizes1[2] = wheight*wwidth;
    max_sizes1[3] = ichannel;
        
    im2collayer_output.back().elementsize = sizeof(uint8_t);

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_QuantizedIm2col_8,   NN_PAD_NA,   im2collayer_input.data(), im2collayer_input.size(),   im2collayer_output.data(), im2collayer_output.size());

    inputTransposeShape.push_back(3);
    inputTransposeShape.push_back(1);
    inputTransposeShape.push_back(2);
    inputTransposeShape.push_back(0);

    std::vector<int> tempshape;
    for(int i=0;i<4;i++)
        tempshape.push_back(max_sizes1[i]);
    addInputTransposeLayerWithShape(hexagon_nn_,graph_id_,op_id,inputs[0], input_transposelayer_input, input_transposelayer_output, inputTransposeShape, &input_min, &input_max, tempshape);

    
    reshapelayer_input.push_back(hexagon_nn_input());
    reshapelayer_input.back().src_id = op_id;
    reshapelayer_input.back().output_idx = 0;

    op_id++;

    desired_shape.push_back(1);
    desired_shape.push_back(1);
    desired_shape.push_back(ichannel*obatch*oheight);
    desired_shape.push_back(ibatch*wheight*wwidth);

    addConstInputTensor(hexagon_nn_, op_id, 0, reshapelayer_input, (uint8_t*)desired_shape.data(), graph_id_, 4, desired_shape.size()*sizeof(int32_t));

    addConstInputTensor(hexagon_nn_, op_id, 0, reshapelayer_input, (uint8_t*)&input_min, graph_id_, 1, sizeof(float));
    addConstInputTensor(hexagon_nn_, op_id, 0, reshapelayer_input, (uint8_t*)&input_max, graph_id_, 1, sizeof(float));

    reshapelayer_output.push_back(hexagon_nn_output());
    reshapelayer_output.back().rank = 4;
    auto& max_sizes2 = reshapelayer_output.back().max_sizes;
    for (int i = 0; i < 4; ++i) {
        max_sizes2[i] =desired_shape[i];
    }
    reshapelayer_output.back().elementsize = sizeof(uint8_t);

    std::vector<int> max_size;
    max_size.push_back(1);
    max_size.push_back(1);
    max_size.push_back(1);
    max_size.push_back(1);
    addOutputTensor(max_size,reshapelayer_output, sizeof(float));
    addOutputTensor(max_size,reshapelayer_output, sizeof(float));

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_QuantizedReshape,   NN_PAD_NA,   reshapelayer_input.data(), reshapelayer_input.size(),   reshapelayer_output.data(), reshapelayer_output.size());

    
    _inputs.push_back(hexagon_nn_input());
    _inputs.back().src_id = op_id;
    _inputs.back().output_idx = 0;

    op_id++;

    outdiffTransposeShape.push_back(1);
    outdiffTransposeShape.push_back(2);
    outdiffTransposeShape.push_back(0);
    outdiffTransposeShape.push_back(3);
    addInputTransposeLayerWithInputOpIdWithIdx(hexagon_nn_,graph_id_, input_op_id, op_id,inputs[1], 1, outdiff_transposelayer_input, outdiff_transposelayer_output, outdiffTransposeShape, &input_min, &input_max);


    addInputCastLayerWithIdxWithTranspose(hexagon_nn_, graph_id_, op_id, op_id, inputs[1], 0, castlayer_input, castlayer_output, OP_Quantized_CastInt8ToUInt8, outdiffTransposeShape);

    weight_reshapelayer_input.push_back(hexagon_nn_input());
    weight_reshapelayer_input.back().src_id = op_id;
    weight_reshapelayer_input.back().output_idx = 0;

    op_id++;

    weight_desired_shape.push_back(1);
    weight_desired_shape.push_back(1);
    weight_desired_shape.push_back(wbatch*wheight*wwidth);
    weight_desired_shape.push_back(wchannel);

    addConstInputTensor(hexagon_nn_, op_id, 0, weight_reshapelayer_input, (uint8_t*)weight_desired_shape.data(), graph_id_, 4, weight_desired_shape.size()*sizeof(int32_t));

    addConstInputTensor(hexagon_nn_, op_id, 0, weight_reshapelayer_input, (uint8_t*)&input_min, graph_id_, 1, sizeof(float));
    addConstInputTensor(hexagon_nn_, op_id, 0, weight_reshapelayer_input, (uint8_t*)&input_max, graph_id_, 1, sizeof(float));

    weight_reshapelayer_output.push_back(hexagon_nn_output());
    weight_reshapelayer_output.back().rank = 4;
    auto& max_sizes3 = weight_reshapelayer_output.back().max_sizes;
    for (int i = 0; i < 4; ++i) {
        max_sizes3[i] = weight_desired_shape[i];
    }
    weight_reshapelayer_output.back().elementsize = sizeof(uint8_t);

    addOutputTensor(max_size,weight_reshapelayer_output, sizeof(float));
    addOutputTensor(max_size,weight_reshapelayer_output, sizeof(float));

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_QuantizedReshape,   NN_PAD_NA,   weight_reshapelayer_input.data(), weight_reshapelayer_input.size(),   weight_reshapelayer_output.data(), weight_reshapelayer_output.size());



    _inputs.push_back(hexagon_nn_input());
    _inputs.back().src_id = op_id;
    _inputs.back().output_idx = 0;

    op_id++;

    addConstInputTensor(hexagon_nn_, op_id, 0, _inputs, (uint8_t*)&input_min, graph_id_, 1, sizeof(float));
    addConstInputTensor(hexagon_nn_, op_id, 0, _inputs, (uint8_t*)&input_max, graph_id_, 1, sizeof(float));

    addConstInputTensor(hexagon_nn_, op_id, 0, _inputs, (uint8_t*)&weights_min, graph_id_, 1, sizeof(float));
    addConstInputTensor(hexagon_nn_, op_id, 0, _inputs, (uint8_t*)&weights_max, graph_id_, 1, sizeof(float));

    
    _outputs.push_back(hexagon_nn_output());
    _outputs.back().rank = 4;
    auto& max_sizes = _outputs.back().max_sizes;
    
        max_sizes[0] = 1;
        max_sizes[1] = 1;
        max_sizes[2] = outputs[0]->buffer().dim[2].extent*outputs[0]->buffer().dim[0].extent*outputs[0]->buffer().dim[1].extent;
        max_sizes[3] = outputs[0]->buffer().dim[3].extent;
    
    _outputs.back().elementsize = sizeof(int32_t);

    addOutputTensor(max_size,_outputs, sizeof(float));
    addOutputTensor(max_size,_outputs, sizeof(float));

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
        op_id,   OP_QuantizedMatMul_8x8to32,   NN_PAD_VALID,   _inputs.data(), _inputs.size(),   _outputs.data(), _outputs.size());
    

    output_reshapelayer_input.push_back(hexagon_nn_input());
    output_reshapelayer_input.back().src_id = op_id;
    output_reshapelayer_input.back().output_idx = 0;

    op_id++;
    output_desired_shape.push_back(ichannel);
    output_desired_shape.push_back(obatch);
    output_desired_shape.push_back(oheight);
    output_desired_shape.push_back(wchannel);

    addConstInputTensor(hexagon_nn_, op_id, 0, output_reshapelayer_input, (uint8_t*)output_desired_shape.data(), graph_id_, 4, output_desired_shape.size()*sizeof(int32_t));

    addConstInputTensor(hexagon_nn_, op_id, 0, output_reshapelayer_input, (uint8_t*)&input_min, graph_id_, 1, sizeof(float));
    addConstInputTensor(hexagon_nn_, op_id, 0, output_reshapelayer_input, (uint8_t*)&input_max, graph_id_, 1, sizeof(float));

    output_reshapelayer_output.push_back(hexagon_nn_output());
    output_reshapelayer_output.back().rank = 4;
    auto& max_sizes4 = output_reshapelayer_output.back().max_sizes;
    for (int i = 0; i < 4; ++i) {
        max_sizes4[i] = output_desired_shape[i];
    }
    output_reshapelayer_output.back().elementsize = sizeof(int32_t);

    addOutputTensor(max_size,output_reshapelayer_output, sizeof(float));
    addOutputTensor(max_size,output_reshapelayer_output, sizeof(float));

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_QuantizedReshape,   NN_PAD_NA,   output_reshapelayer_input.data(), output_reshapelayer_input.size(),   output_reshapelayer_output.data(), output_reshapelayer_output.size());


    requantizelayer_input.push_back(hexagon_nn_input());
    requantizelayer_input.back().src_id = op_id;
    requantizelayer_input.back().output_idx = 0;

    
    op_id++;

    addConstInputTensor(hexagon_nn_, op_id, 0, requantizelayer_input, (uint8_t*)&input_min, graph_id_, 1, sizeof(float));
    addConstInputTensor(hexagon_nn_, op_id, 0, requantizelayer_input, (uint8_t*)&input_max, graph_id_, 1, sizeof(float));

    addConstInputTensor(hexagon_nn_, op_id, 0, requantizelayer_input, (uint8_t*)&output_min, graph_id_, 1, sizeof(float));
    addConstInputTensor(hexagon_nn_, op_id, 0, requantizelayer_input, (uint8_t*)&output_max, graph_id_, 1, sizeof(float));

    requantizelayer_output.push_back(hexagon_nn_output());
    requantizelayer_output.back().rank = 4;
    auto& max_sizes5 = requantizelayer_output.back().max_sizes;
    for (int i = 0; i < 4; ++i) {
        max_sizes5[i] = output_desired_shape[i];
    }
    requantizelayer_output.back().elementsize = sizeof(uint8_t);

    addOutputTensor(max_size,requantizelayer_output, sizeof(float));
    addOutputTensor(max_size,requantizelayer_output, sizeof(float));

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_Requantize_32to8,   NN_PAD_NA,   requantizelayer_input.data(), requantizelayer_input.size(),   requantizelayer_output.data(), requantizelayer_output.size());

    
    outputTransposeShape.push_back(1);
    outputTransposeShape.push_back(2);
    outputTransposeShape.push_back(0);
    outputTransposeShape.push_back(3);
    addOutputTransposeLayer(hexagon_nn_,graph_id_,op_id, outputs[0], output_transposelayer_input, output_transposelayer_output, outputTransposeShape, &input_min, &input_max);

    op_id++;

    
    // CONV output
    _inputs_output.push_back(hexagon_nn_input());
    _inputs_output.back().src_id = op_id-1;
    _inputs_output.back().output_idx = 0;
    

    // CONV output min
    _inputs_output.push_back(hexagon_nn_input());
    _inputs_output.back().src_id = op_id-1;
    _inputs_output.back().output_idx = 1;
   

    // CONV output max
    _inputs_output.push_back(hexagon_nn_input());
    _inputs_output.back().src_id = op_id-1;
    _inputs_output.back().output_idx = 2;

   
    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_OUTPUT,   NN_PAD_NA,   _inputs_output.data(), _inputs_output.size(),   NULL, 0);
    

    int error = hexagon_nn_->hexagon_nn_prepare(graph_id_);
    if(error!=0){
        MNN_ERROR("Whoops... Cannot prepare: %d\n", error);
        return NOT_SUPPORT;
    }

    uint8_t* inputPtr = inputUINT8Tensor->host<uint8_t>();
    for(int i=0;i<inputs[0]->elementSize();i++)
        inputPtr[i] = inputs[0]->host<int8_t>()[i] + 128;

    std::vector<hexagon_nn_tensordef> input_tensors;
    std::vector<hexagon_nn_tensordef> output_tensors;
    
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

    {
        input_tensors.emplace_back();
        auto& input_tensor = input_tensors.back();
        input_tensor.data = inputs[1]->host<uint8_t>();
        input_tensor.dataLen = inputs[1]->elementSize();
        input_tensor.data_valid_len = inputs[1]->elementSize();
        input_tensor.batches = inputs[1]->batch();
        input_tensor.height = inputs[1]->height();
        input_tensor.width = inputs[1]->width();
        input_tensor.depth = inputs[1]->channel();
    }

    output_tensors.emplace_back();
    auto& output_tensor = output_tensors.back();
     output_tensor.data = outputs[0]->host<uint8_t>();
    output_tensor.dataLen = outputs[0]->elementSize();
    output_tensor.batches = outputs[0]->batch();
    output_tensor.height = outputs[0]->height();
    output_tensor.width = outputs[0]->width();
    output_tensor.depth = outputs[0]->channel();


    
    for(int i=0;i<2;i++) {
        output_tensors.emplace_back();
        auto& output_tensor = output_tensors.back();
        output_tensor.data = reinterpret_cast<unsigned char*>(minmax+i);
        output_tensor.dataLen = sizeof(float);
        output_tensor.batches = 1;
        output_tensor.height = 1;
        output_tensor.width = 1;
        output_tensor.depth = 1;
    }

    error = hexagon_nn_->hexagon_nn_execute_new( graph_id_, input_tensors.data(), input_tensors.size(),
      output_tensors.data(), output_tensors.size());
    
    if(error!=0){
        MNN_ERROR("Whoops... run failed: %d\n", error);
        exit(-1);
        return NOT_SUPPORT;
    }

    hexagon_nn_->hexagon_nn_teardown(graph_id_);


    for(int i=0;i<outputs[0]->elementSize();i++)
        outputs[0]->host<int8_t>()[i] = outputs[0]->host<uint8_t>()[i] - 128;

    
    
        for(int i=0;i<outputs[0]->elementSize();i++) {
            outputs[0]->host<int8_t>()[i] = outputs[0]->host<int8_t>()[i] / 16;
        }

    return NO_ERROR;
}

ErrorCode NITI_DSPMatmulGradientConv_Int8::GlobalExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {

    // if(static_cast<CPUBackend*>(backend())->build_graph) {
        
    //     {
    //         static_cast<CPUBackend*>(backend())->output_tensors.emplace_back();
    //         auto& output_tensor = static_cast<CPUBackend*>(backend())->output_tensors.back();
    //             output_tensor.data = outputs[0]->host<uint8_t>();
    //         output_tensor.dataLen = outputs[0]->elementSize();
    //         output_tensor.batches = outputs[0]->batch();
    //         output_tensor.height = outputs[0]->height();
    //         output_tensor.width = outputs[0]->width();
    //         output_tensor.depth = outputs[0]->channel();

    //         static_cast<CPUBackend*>(backend())->gradient_map.insert(std::make_pair( outputs[0]->host<int8_t>(), outputs[0]->elementSize() ));
    //     }
    //     return NO_ERROR;
    // }

    // // if(conv_num == 1) {

    // //     // AUTOTIME;
    // //     // Timer _100Time;
    // //     // _100Time.reset();

    // //     static_cast<CPUBackend*>(backend())->global_hexagon_nn_->hexagon_nn_append_node(static_cast<CPUBackend*>(backend())->global_graph_id_,     
    // //         static_cast<CPUBackend*>(backend())->global_op_id,   OP_OUTPUT,   NN_PAD_NA,   static_cast<CPUBackend*>(backend())->outputlayer_input.data(), static_cast<CPUBackend*>(backend())->outputlayer_input.size(),   NULL, 0);
    
    // //     int error = static_cast<CPUBackend*>(backend())->global_hexagon_nn_->hexagon_nn_prepare(static_cast<CPUBackend*>(backend())->global_graph_id_);
    // //     if(error!=0){
    // //         MNN_ERROR("Whoops... Cannot prepare: %d\n", error);
    // //         exit(-1);
    // //     }

    // //     for(int i=0;i<1;i++) {
    // //         error = static_cast<CPUBackend*>(backend())->global_hexagon_nn_->hexagon_nn_execute_new( static_cast<CPUBackend*>(backend())->global_graph_id_, static_cast<CPUBackend*>(backend())->input_tensors.data(), static_cast<CPUBackend*>(backend())->input_tensors.size(),
    // //         static_cast<CPUBackend*>(backend())->output_tensors.data(), static_cast<CPUBackend*>(backend())->output_tensors.size());
            
    // //         if(error!=0){
    // //             MNN_ERROR("Whoops... run failed: %d\n", error);
    // //             exit(-1);
    // //         }
    // //     }
        
        

    // //     static_cast<CPUBackend*>(backend())->global_hexagon_nn_->hexagon_nn_teardown(static_cast<CPUBackend*>(backend())->global_graph_id_);


    // //     // ??????????????????
    // //     static_cast<CPUBackend*>(backend())->input_tensors.resize(0);
    // //     static_cast<CPUBackend*>(backend())->output_tensors.resize(0);

    // //     static_cast<CPUBackend*>(backend())->outputlayer_input.resize(0);

    // //      initDspGraph(static_cast<CPUBackend*>(backend())->global_hexagon_nn_, static_cast<CPUBackend*>(backend())->global_graph_id_);
    // //      static_cast<CPUBackend*>(backend())->global_op_id = 0x1000;

    // //      conv_num--;

    // //     //  std::cout << "execute forward and backward conv time: " << (float)_100Time.durationInUs() / 1000.0f << " ms / " << std::endl;
    // //     // _100Time.reset();
    // // }

    // // AUTOTIME;
    // // Timer _100Time;
    // // _100Time.reset();

    // hexagon_nn_ = static_cast<CPUBackend*>(backend())->global_hexagon_nn_;
    // graph_id_ = static_cast<CPUBackend*>(backend())->global_graph_id_;

    // op_id = static_cast<CPUBackend*>(backend())->global_op_id;

    
    // const auto input = inputs[0];
    // const auto weight = inputs[1];

    // int ibatch = input->batch();
    // int iheight = input->height();
    // int iwidth = input->width();
    // int ichannel = input->channel();


    // int wbatch = weight->batch();
    // int wheight = weight->height();
    // int wwidth = weight->width();
    // int wchannel = weight->channel();

    // int ochannel = outputs[0]->channel();

    // bias = biasTensor->host<uint8_t>();
    // int channelout = ochannel;
    // memset(bias, 0, channelout*sizeof(int32_t));


    // int input_op_id = backend()->get_Op_id(inputs[0], OpType_NITI_DSP_GRADIENTCONV_Int8);

    // inputTransposeShape.push_back(3);
    // inputTransposeShape.push_back(1);
    // inputTransposeShape.push_back(2);
    // inputTransposeShape.push_back(0);
    // addInputTransposeLayerWithInputOpId(hexagon_nn_,graph_id_, input_op_id, op_id, inputs[0], input_transposelayer_input, input_transposelayer_output, inputTransposeShape, &input_min, &input_max);

    // // MNN_PRINT("pad = %d\n", mCommon->padX());
    // if(mCommon->padX() == 1) {

    //     padtensor.push_back(0);
    //     padtensor.push_back(0);
    //     padtensor.push_back(1);
    //     padtensor.push_back(1);
    //     padtensor.push_back(1);
    //     padtensor.push_back(1);
    //     padtensor.push_back(0);
    //     padtensor.push_back(0);

    //     padlayer_input.push_back(hexagon_nn_input());
    //     padlayer_input.back().src_id = op_id;
    //     padlayer_input.back().output_idx = 0;

    //     op_id++;

    //     addConstInputTensor(hexagon_nn_, op_id, 0, padlayer_input, (uint8_t*)&input_min, graph_id_, 1, sizeof(float));
    //     addConstInputTensor(hexagon_nn_, op_id, 0, padlayer_input, (uint8_t*)&input_max, graph_id_, 1, sizeof(float));

    //     // control tensor
    //     padlayer_input.push_back(hexagon_nn_input());
    //     padlayer_input.back().src_id = op_id;
    //     padlayer_input.back().output_idx = 0;

    //     hexagon_nn_->hexagon_nn_append_const_node(
    //         graph_id_,                   // Graph handle we're appending into
    //         op_id++,                    // Node identifier (a unique uint32)
    //         1,                          // size: batches
    //         1,                          // size: height
    //         4,                          // size: width
    //         2,                          // size: depth
    //         (uint8_t*)padtensor.data(), // Pointer to data
    //         8*sizeof(int32_t)  // Length of data to copy
    //         );
        
    //     addConstInputTensor(hexagon_nn_, op_id, 0, padlayer_input, (uint8_t*)&padValue, graph_id_, 1, sizeof(uint8_t));

    //     padlayer_output.push_back(hexagon_nn_output());
    //     padlayer_output.back().rank = 4;
    //     auto& max_sizes1 = padlayer_output.back().max_sizes;
        
    //     max_sizes1[0] = inputs[0]->buffer().dim[inputTransposeShape[0]].extent;
    //     max_sizes1[1] = inputs[0]->buffer().dim[inputTransposeShape[1]].extent+2;
    //     max_sizes1[2] = inputs[0]->buffer().dim[inputTransposeShape[2]].extent+2;
    //     max_sizes1[3] = inputs[0]->buffer().dim[inputTransposeShape[3]].extent;
        
    //     padlayer_output.back().elementsize = sizeof(uint8_t);

    //     std::vector<int> max_size;
    //     max_size.push_back(1);
    //     max_size.push_back(1);
    //     max_size.push_back(1);
    //     max_size.push_back(1);
    //     addOutputTensor(max_size,padlayer_output, sizeof(float));
    //     addOutputTensor(max_size,padlayer_output, sizeof(float));

    //     hexagon_nn_->hexagon_nn_append_node(graph_id_,     
    //             op_id,   OP_QuantizedPad_V2_8,   NN_PAD_NA,   padlayer_input.data(), padlayer_input.size(),   padlayer_output.data(), padlayer_output.size());
        
    // } 
        
    // _inputs.push_back(hexagon_nn_input());
    // _inputs.back().src_id = op_id;
    // _inputs.back().output_idx = 0;

    // op_id++;

    // // ??????weight const node
    // int weight_op_id = backend()->get_Op_id(inputs[1], OpType_NITI_DSP_GRADIENTCONV_Int8);

    // outdiffTransposeShape.push_back(1);
    // outdiffTransposeShape.push_back(2);
    // outdiffTransposeShape.push_back(0);
    // outdiffTransposeShape.push_back(3);
    // addInputTransposeLayerWithInputOpId(hexagon_nn_,graph_id_,weight_op_id, op_id,inputs[1], outdiff_transposelayer_input, outdiff_transposelayer_output, outdiffTransposeShape, &input_min, &input_max);


    // _inputs.push_back(hexagon_nn_input());
    // _inputs.back().src_id = op_id;
    // _inputs.back().output_idx = 0;

    // op_id++;

    // // CONV op
    // addConstInputTensor(hexagon_nn_, op_id, 0, _inputs, (uint8_t*)&input_min, graph_id_, 1, sizeof(float));
    // addConstInputTensor(hexagon_nn_, op_id, 0, _inputs, (uint8_t*)&input_max, graph_id_, 1, sizeof(float));

    // addConstInputTensor(hexagon_nn_, op_id, 0, _inputs, (uint8_t*)&weights_min, graph_id_, 1, sizeof(float));
    // addConstInputTensor(hexagon_nn_, op_id, 0, _inputs, (uint8_t*)&weights_max, graph_id_, 1, sizeof(float));

    // addConstInputTensor(hexagon_nn_, op_id, 0, _inputs, strides_tensor.data(), graph_id_, 1, strides_tensor.size()*sizeof(uint8_t));
    // addConstInputTensor(hexagon_nn_, op_id, 0, _inputs, bias, graph_id_, channelout, channelout*sizeof(int32_t));

    // addConstInputTensor(hexagon_nn_, op_id, 0, _inputs, (uint8_t*)&bias_min, graph_id_, 1, sizeof(float));
    // addConstInputTensor(hexagon_nn_, op_id, 0, _inputs, (uint8_t*)&bias_max, graph_id_, 1, sizeof(float));

    // addConstInputTensor(hexagon_nn_, op_id, 0, _inputs, (uint8_t*)&output_min, graph_id_, 1, sizeof(float));
    // addConstInputTensor(hexagon_nn_, op_id, 0, _inputs, (uint8_t*)&output_max, graph_id_, 1, sizeof(float));

    
    // _outputs.push_back(hexagon_nn_output());
    // _outputs.back().rank = 4;
    // auto& max_sizes = _outputs.back().max_sizes;
    
    //     max_sizes[0] = outputs[0]->buffer().dim[2].extent;
    //     max_sizes[1] = outputs[0]->buffer().dim[0].extent;
    //     max_sizes[2] = outputs[0]->buffer().dim[1].extent;
    //     max_sizes[3] = outputs[0]->buffer().dim[3].extent;
    
    // _outputs.back().elementsize = sizeof(uint8_t);

    // std::vector<int> max_size;
    // max_size.push_back(1);
    // max_size.push_back(1);
    // max_size.push_back(1);
    // max_size.push_back(1);
    // addOutputTensor(max_size,_outputs, sizeof(float));
    // addOutputTensor(max_size,_outputs, sizeof(float));

    // // _outputs.push_back(hexagon_nn_output());
    // // _outputs.back().rank = 4;
    // // auto& max_sizes2 = _outputs.back().max_sizes;
    // // for (int i = 0; i < 4; ++i) {
    // //     max_sizes2[i] = output32TensorShape[i];
    // // }
    // // _outputs.back().elementsize = sizeof(int32_t);

   
    // hexagon_nn_->hexagon_nn_append_node(graph_id_,     
    // op_id,   OP_Supernode_8x8p32to8,   NN_PAD_VALID,   _inputs.data(), _inputs.size(),   _outputs.data(), _outputs.size());
    

    // outputTransposeShape.push_back(1);
    // outputTransposeShape.push_back(2);
    // outputTransposeShape.push_back(0);
    // outputTransposeShape.push_back(3);
    // addOutputTransposeLayer(hexagon_nn_,graph_id_,op_id, outputs[0], output_transposelayer_input, output_transposelayer_output, outputTransposeShape, &input_min, &input_max);

    // addInputCastLayer(hexagon_nn_, graph_id_, op_id, op_id, outputs[0], castlayer_input, castlayer_output, OP_Quantized_CastUInt8ToInt8);

    // // static_cast<CPUBackend*>(backend())->opadr_opid_map.insert(std::make_pair(outputs[0]->host<int8_t>(), op_id ));
    // backend()->insert_Op_id(outputs[0], op_id);
    // // MNN_PRINT("gradient output id = %d\n", op_id);

    // static_cast<CPUBackend*>(backend())->outputlayer_input.emplace_back();
    // static_cast<CPUBackend*>(backend())->outputlayer_input.back().src_id = op_id;
    // static_cast<CPUBackend*>(backend())->outputlayer_input.back().output_idx = 0;

    // {
    //     static_cast<CPUBackend*>(backend())->output_tensors.emplace_back();
    //     auto& output_tensor = static_cast<CPUBackend*>(backend())->output_tensors.back();
    //         output_tensor.data = outputs[0]->host<uint8_t>();
    //     output_tensor.dataLen = outputs[0]->elementSize();
    //     output_tensor.batches = outputs[0]->batch();
    //     output_tensor.height = outputs[0]->height();
    //     output_tensor.width = outputs[0]->width();
    //     output_tensor.depth = outputs[0]->channel();

    //     static_cast<CPUBackend*>(backend())->gradient_map.insert(std::make_pair( outputs[0]->host<int8_t>(), outputs[0]->elementSize() ));
    // }

    // op_id++;
    // static_cast<CPUBackend*>(backend())->global_op_id = op_id;

    // // std::cout << "execute gradient conv time: " << (float)_100Time.durationInUs() / 1000.0f << " ms / " << std::endl;
    // // _100Time.reset();

    
    return NO_ERROR;
}

int NITI_DSPMatmulGradientConv_Int8::conv_num = 0;

class NITI_DSPMatmulGradientConv_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {

        auto convOp = op->main_as_NITI_CONV_Int8();
        return new NITI_DSPMatmulGradientConv_Int8(backend, convOp, op);
    }
};

REGISTER_CPU_OP_CREATOR(NITI_DSPMatmulGradientConv_Int8_Creator, OpType_NITI_DSP_MATMUL_GRADIENT_Int8);
} // namespace MNN
