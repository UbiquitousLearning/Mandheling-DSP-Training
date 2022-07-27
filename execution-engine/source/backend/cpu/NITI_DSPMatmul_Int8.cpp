//
//  NITI_DSPMatmul_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/12/20.
//  
//

#include "backend/cpu/NITI_DSPMatmul_Int8.hpp"
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

NITI_DSPMatmul_Int8::~NITI_DSPMatmul_Int8() {
    // Do nothing
    // hexagon_nn_->hexagon_nn_global_teardown();
}
NITI_DSPMatmul_Int8::NITI_DSPMatmul_Int8(Backend* backend, const NITI_CONV_Int8* convOp, const MNN::Op* op) : CPUConvolution(convOp->common(), backend) {
    if(!getDSPExecuteMode())
        hexagon_nn_ = generate_interface();
    mOp = op;
}


#define LPAD 2 
#define roundup(a, p2)       (((a)+(p2)-1)&~((p2)-1))

ErrorCode NITI_DSPMatmul_Int8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {

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

    matmulA.reset(Tensor::createDevice<uint8_t>({1, 1, ichannel*obatch*oheight, wheight*wwidth*ibatch}));
    success = backend()->onAcquireBuffer(matmulA.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    matmulB.reset(Tensor::createDevice<uint8_t>({1, 1, wheight*wwidth*ibatch, ochannel}));
    success = backend()->onAcquireBuffer(matmulB.get(), Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    matmulC.reset(Tensor::createDevice<int32_t>({1, 1, ichannel*obatch*oheight, ochannel}));
    success = backend()->onAcquireBuffer(matmulC.get(), Backend::DYNAMIC);
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
    backend()->onReleaseBuffer(matmulA.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(matmulB.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(matmulC.get(), Backend::DYNAMIC);
    
    
    return NO_ERROR;
}

ErrorCode NITI_DSPMatmul_Int8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {

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

    std::vector<int> max_size;
    max_size.push_back(1);
    max_size.push_back(1);
    max_size.push_back(1);
    max_size.push_back(1);


    // input
    _outputs0.push_back(hexagon_nn_output());
    _outputs0.back().rank = 4;
    auto& max_sizes0 = _outputs0.back().max_sizes;
    for (int i = 0; i < 4; ++i) {
        max_sizes0[i] = matmulA->buffer().dim[i].extent;
    }
    _outputs0.back().elementsize = sizeof(uint8_t);

    _outputs0.push_back(hexagon_nn_output());
    _outputs0.back().rank = 4;
    auto& max_sizesw = _outputs0.back().max_sizes;
    for (int i = 0; i < 4; ++i) {
        max_sizesw[i] = matmulB->buffer().dim[i].extent;
    }
    _outputs0.back().elementsize = sizeof(uint8_t);


    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_INPUT,   NN_PAD_NA,   empty_input_list.data(), 0,   _outputs0.data(), _outputs0.size());

    int input_op_id = op_id;

    _inputs.push_back(hexagon_nn_input());
    _inputs.back().src_id = op_id;
    _inputs.back().output_idx = 0;


    _inputs.push_back(hexagon_nn_input());
    _inputs.back().src_id = op_id;
    _inputs.back().output_idx = 1;

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

    op_id++;

    
    _inputs_output.push_back(hexagon_nn_input());
    _inputs_output.back().src_id = op_id-1;
    _inputs_output.back().output_idx = 0;

    hexagon_nn_->hexagon_nn_append_node(graph_id_,     
            op_id,   OP_OUTPUT,   NN_PAD_NA,   _inputs_output.data(), _inputs_output.size(),   NULL, 0);
    

    int error = hexagon_nn_->hexagon_nn_prepare(graph_id_);
    if(error!=0){
        MNN_ERROR("Whoops... Cannot prepare: %d\n", error);
        return NOT_SUPPORT;
    }

    std::vector<hexagon_nn_tensordef> input_tensors;
    std::vector<hexagon_nn_tensordef> output_tensors;
    
    {
        input_tensors.emplace_back();
        auto& input_tensor = input_tensors.back();
        input_tensor.data = matmulA->host<uint8_t>();
        input_tensor.dataLen = matmulA->elementSize();
        input_tensor.data_valid_len = matmulA->elementSize();
        input_tensor.batches = matmulA->batch();
        input_tensor.height = matmulA->height();
        input_tensor.width = matmulA->width();
        input_tensor.depth = matmulA->channel();
    }

    {
        input_tensors.emplace_back();
        auto& input_tensor = input_tensors.back();
        input_tensor.data = matmulB->host<uint8_t>();
        input_tensor.dataLen = matmulB->elementSize();
        input_tensor.data_valid_len = matmulB->elementSize();
        input_tensor.batches = matmulB->batch();
        input_tensor.height = matmulB->height();
        input_tensor.width = matmulB->width();
        input_tensor.depth = matmulB->channel();
    }

    output_tensors.emplace_back();
    auto& output_tensor = output_tensors.back();
     output_tensor.data = matmulC->host<uint8_t>();
    output_tensor.dataLen = matmulC->elementSize()*sizeof(int32_t);
    output_tensor.batches = matmulC->batch();
    output_tensor.height = matmulC->height();
    output_tensor.width =  matmulC->channel();

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

ErrorCode NITI_DSPMatmul_Int8::GlobalExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return NO_ERROR;
}

int NITI_DSPMatmul_Int8::conv_num = 0;

class NITI_DSPMatmul_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {

        auto convOp = op->main_as_NITI_CONV_Int8();
        return new NITI_DSPMatmul_Int8(backend, convOp, op);
    }
};

REGISTER_CPU_OP_CREATOR(NITI_DSPMatmul_Int8_Creator, OpType_NITI_DSP_MATMUL_Int8);
} // namespace MNN
