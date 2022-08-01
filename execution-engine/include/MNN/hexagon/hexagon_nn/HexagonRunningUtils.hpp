#ifndef HexagonRunningUtils_hpp
#define HexagonRunningUtils_hpp

#include <string>
#include <vector>
#include <algorithm>
#include <climits>

#include "hexagon_nn_ops.h"
#include "hexagon_nn.h"
#include "HexagonRunningUtils.hpp"
#include "hexagon_implementation.h"

#include <backend/cpu/CPUBackend.hpp>


#include <sys/time.h> 

namespace MNN{
    inline void addConstInputTensor(const HexagonNN* hexagon_nn_,  int& src_id, int output_idx, std::vector<hexagon_nn_input>& _inputs, uint8_t* data, hexagon_nn_nn_id graph_id, int shape_size, int size) {
        _inputs.push_back(hexagon_nn_input());
        _inputs.back().src_id = src_id;
        _inputs.back().output_idx = output_idx;

        hexagon_nn_->hexagon_nn_append_const_node(
            graph_id,                   // Graph handle we're appending into
            src_id++,                    // Node identifier (a unique uint32)
            1,                          // size: batches
            1,                          // size: height
            1,                          // size: width
            shape_size,                          // size: depth
            data, // Pointer to data
            size  // Length of data to copy
            );
    }

    inline void addOutputTensor(std::vector<int> max_sizes, std::vector<hexagon_nn_output>& _outputs, int size) {
        _outputs.push_back(hexagon_nn_output());
        _outputs.back().rank = 4;
        auto& omax_sizes = _outputs.back().max_sizes;
        for (int i = 0; i < 4; ++i) {
            omax_sizes[i] = max_sizes[i];
        }  
        _outputs.back().elementsize = size;
    }

    static bool is_global_init = false;
    static const HexagonNN* hexagon_nn = nullptr;

    inline const HexagonNN* generate_interface() {
        if(hexagon_nn == nullptr) {
            hexagon_nn = HexagonNNImplementation();
        }
        
        return hexagon_nn;
    }

    inline void initDspGraph(const HexagonNN*  hexagon_nn_,  hexagon_nn_nn_id& graph_id_) {
        // 以下是DSP的内容

        if (hexagon_nn_ == nullptr) {
            MNN_PRINT("Hexagon interface not available.");
            return ;
        }

        if(!is_global_init) {
            int ret = hexagon_nn_->hexagon_nn_global_init();
            is_global_init = true;

            int error = hexagon_nn_->hexagon_nn_config();
            if (error != 0) {
                MNN_PRINT("hexagon_nn_config failed. Error: %d", error);
                return ;
            }

            error = hexagon_nn_->hexagon_nn_set_powersave_level(0);
            if (error != 0) {
                MNN_PRINT("Failed to set powersave level, error %d",
                                error);
                return ;
            }
        }

        // hexagon_nn_->hexagon_nn_global_init();
        // Ensure Hexagon NNLib is ready to start working.
        
        // Initialize an empty graph.
        int error = hexagon_nn_->hexagon_nn_init(&graph_id_);
        if (error != 0) {
            MNN_PRINT("failed to init");
            return ;
        }
        error =
            hexagon_nn_->hexagon_nn_set_debug_level(graph_id_, 0);
        if (error != 0) {
            MNN_PRINT("Failed to set debug level, error: %d", error);
            return ;
        }
        
    }

    static long getCurrentTime()  
    {  
    struct timeval tv;  
    gettimeofday(&tv,NULL);  
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;  
    } 

    inline void addInputTransposeLayerWithInputOpIdWithIdx(const HexagonNN* hexagon_nn_, hexagon_nn_nn_id graph_id_, int input_op_id, int& op_id, Tensor* input, int idx,
                            std::vector<hexagon_nn_input> transposelayer_input, std::vector<hexagon_nn_output> transposelayer_output, 
                            std::vector<int32_t> transpose_shape, float* input_min, float* input_max)
    {
        transposelayer_input.push_back(hexagon_nn_input());
        transposelayer_input.back().src_id = input_op_id;
        transposelayer_input.back().output_idx = idx;

        op_id++;

        // control tensor
        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)transpose_shape.data(), graph_id_, 4, sizeof(int32_t)*4);

        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)input_min, graph_id_, 1, sizeof(float));
        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)input_max, graph_id_, 1, sizeof(float));

        transposelayer_output.push_back(hexagon_nn_output());
        transposelayer_output.back().rank = 4;
        auto& max_sizes1 = transposelayer_output.back().max_sizes;
        
            max_sizes1[0] = input->buffer().dim[transpose_shape[0]].extent;
            max_sizes1[1] = input->buffer().dim[transpose_shape[1]].extent;
            max_sizes1[2] = input->buffer().dim[transpose_shape[2]].extent;
            max_sizes1[3] = input->buffer().dim[transpose_shape[3]].extent;
        
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

    }

    inline void addInputTransposeLayerWithInputOpIdWithIdxWithShape(const HexagonNN* hexagon_nn_, hexagon_nn_nn_id graph_id_, int input_op_id, int& op_id, Tensor* input, int idx,
                            std::vector<hexagon_nn_input> transposelayer_input, std::vector<hexagon_nn_output> transposelayer_output, 
                            std::vector<int32_t> transpose_shape, float* input_min, float* input_max, std::vector<int> shape)
    {
        transposelayer_input.push_back(hexagon_nn_input());
        transposelayer_input.back().src_id = input_op_id;
        transposelayer_input.back().output_idx = idx;

        op_id++;

        // control tensor
        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)transpose_shape.data(), graph_id_, 4, sizeof(int32_t)*4);

        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)input_min, graph_id_, 1, sizeof(float));
        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)input_max, graph_id_, 1, sizeof(float));

        transposelayer_output.push_back(hexagon_nn_output());
        transposelayer_output.back().rank = 4;
        auto& max_sizes1 = transposelayer_output.back().max_sizes;
        
            max_sizes1[0] = shape[transpose_shape[0]];
            max_sizes1[1] = shape[transpose_shape[1]];
            max_sizes1[2] = shape[transpose_shape[2]];
            max_sizes1[3] = shape[transpose_shape[3]];
        
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

    }


    inline void addInputTransposeLayerWithInputOpId(const HexagonNN* hexagon_nn_, hexagon_nn_nn_id graph_id_, int input_op_id, int& op_id, Tensor* input,
                            std::vector<hexagon_nn_input> transposelayer_input, std::vector<hexagon_nn_output> transposelayer_output, 
                            std::vector<int32_t> transpose_shape, float* input_min, float* input_max)
    {
        transposelayer_input.push_back(hexagon_nn_input());
        transposelayer_input.back().src_id = input_op_id;
        transposelayer_input.back().output_idx = 0;

        op_id++;

        // control tensor
        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)transpose_shape.data(), graph_id_, 4, sizeof(int32_t)*4);

        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)input_min, graph_id_, 1, sizeof(float));
        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)input_max, graph_id_, 1, sizeof(float));

        transposelayer_output.push_back(hexagon_nn_output());
        transposelayer_output.back().rank = 4;
        auto& max_sizes1 = transposelayer_output.back().max_sizes;
        
            max_sizes1[0] = input->buffer().dim[transpose_shape[0]].extent;
            max_sizes1[1] = input->buffer().dim[transpose_shape[1]].extent;
            max_sizes1[2] = input->buffer().dim[transpose_shape[2]].extent;
            max_sizes1[3] = input->buffer().dim[transpose_shape[3]].extent;
        
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

    }

    inline void addInputTransposeLayer(const HexagonNN* hexagon_nn_, hexagon_nn_nn_id graph_id_, int& op_id, Tensor* input,
                            std::vector<hexagon_nn_input> transposelayer_input, std::vector<hexagon_nn_output> transposelayer_output, 
                            std::vector<int32_t> transpose_shape, float* input_min, float* input_max)
    {
        transposelayer_input.push_back(hexagon_nn_input());
        transposelayer_input.back().src_id = op_id;
        transposelayer_input.back().output_idx = 0;

        op_id++;

        // control tensor
        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)transpose_shape.data(), graph_id_, 4, sizeof(int32_t)*4);

        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)input_min, graph_id_, 1, sizeof(float));
        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)input_max, graph_id_, 1, sizeof(float));

        transposelayer_output.push_back(hexagon_nn_output());
        transposelayer_output.back().rank = 4;
        auto& max_sizes1 = transposelayer_output.back().max_sizes;
        
            max_sizes1[0] = input->buffer().dim[transpose_shape[0]].extent;
            max_sizes1[1] = input->buffer().dim[transpose_shape[1]].extent;
            max_sizes1[2] = input->buffer().dim[transpose_shape[2]].extent;
            max_sizes1[3] = input->buffer().dim[transpose_shape[3]].extent;
        
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

    }
    inline void addInputTransposeLayerWithShape(const HexagonNN* hexagon_nn_, hexagon_nn_nn_id graph_id_, int& op_id, Tensor* input,
                            std::vector<hexagon_nn_input> transposelayer_input, std::vector<hexagon_nn_output> transposelayer_output, 
                            std::vector<int32_t> transpose_shape, float* input_min, float* input_max, std::vector<int> shape)
    {
        transposelayer_input.push_back(hexagon_nn_input());
        transposelayer_input.back().src_id = op_id;
        transposelayer_input.back().output_idx = 0;

        op_id++;

        // control tensor
        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)transpose_shape.data(), graph_id_, 4, sizeof(int32_t)*4);

        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)input_min, graph_id_, 1, sizeof(float));
        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)input_max, graph_id_, 1, sizeof(float));

        transposelayer_output.push_back(hexagon_nn_output());
        transposelayer_output.back().rank = 4;
        auto& max_sizes1 = transposelayer_output.back().max_sizes;
        
            max_sizes1[0] = shape[transpose_shape[0]];
            max_sizes1[1] = shape[transpose_shape[1]];
            max_sizes1[2] = shape[transpose_shape[2]];
            max_sizes1[3] = shape[transpose_shape[3]];
        
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

    }

    inline void addOutputTransposeLayer(const HexagonNN* hexagon_nn_, hexagon_nn_nn_id graph_id_, int& op_id, Tensor* output,
                            std::vector<hexagon_nn_input> transposelayer_input, std::vector<hexagon_nn_output> transposelayer_output, 
                            std::vector<int32_t> transpose_shape, float* input_min, float* input_max)
    {
        transposelayer_input.push_back(hexagon_nn_input());
        transposelayer_input.back().src_id = op_id;
        transposelayer_input.back().output_idx = 0;

        op_id++;

        // control tensor
        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)transpose_shape.data(), graph_id_, 4, sizeof(int32_t)*4);

        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)input_min, graph_id_, 1, sizeof(float));
        addConstInputTensor(hexagon_nn_, op_id, 0, transposelayer_input, (uint8_t*)input_max, graph_id_, 1, sizeof(float));

        transposelayer_output.push_back(hexagon_nn_output());
        transposelayer_output.back().rank = 4;
        auto& max_sizes1 = transposelayer_output.back().max_sizes;
        
            max_sizes1[0] = output->buffer().dim[0].extent;
            max_sizes1[1] = output->buffer().dim[1].extent;
            max_sizes1[2] = output->buffer().dim[2].extent;
            max_sizes1[3] = output->buffer().dim[3].extent;
        
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

    }


    inline void addInputPadLayer(const HexagonNN* hexagon_nn_, hexagon_nn_nn_id graph_id_, int input_op_id, int& op_id, Tensor* input,
                            std::vector<hexagon_nn_input> padlayer_input, std::vector<hexagon_nn_output> padlayer_output, 
                            std::vector<int32_t> pad_shape, float* input_min, float* input_max, uint8_t* padValue, int output_idx)
    {
        padlayer_input.push_back(hexagon_nn_input());
        padlayer_input.back().src_id = input_op_id;
        padlayer_input.back().output_idx = output_idx;

        op_id++;

        addConstInputTensor(hexagon_nn_, op_id, 0, padlayer_input, (uint8_t*)input_min, graph_id_, 1, sizeof(float));
        addConstInputTensor(hexagon_nn_, op_id, 0, padlayer_input, (uint8_t*)input_max, graph_id_, 1, sizeof(float));

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
            (uint8_t*)pad_shape.data(), // Pointer to data
            sizeof(int32_t)*8  // Length of data to copy
            );

        addConstInputTensor(hexagon_nn_, op_id, 0, padlayer_input, padValue, graph_id_, 1, sizeof(uint8_t));


        padlayer_output.push_back(hexagon_nn_output());
        padlayer_output.back().rank = 4;
        auto& max_sizes1 = padlayer_output.back().max_sizes;
        
            max_sizes1[0] = input->buffer().dim[0].extent + pad_shape[0] + pad_shape[1];
            max_sizes1[1] = input->buffer().dim[1].extent + pad_shape[2] + pad_shape[3];
            max_sizes1[2] = input->buffer().dim[2].extent + pad_shape[4] + pad_shape[5];
            max_sizes1[3] = input->buffer().dim[3].extent + pad_shape[6] + pad_shape[7];
        
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

    inline void addInputCastLayer(const HexagonNN* hexagon_nn_, hexagon_nn_nn_id graph_id_, int input_op_id, int& op_id, Tensor* input,
                            std::vector<hexagon_nn_input> castlayer_input, std::vector<hexagon_nn_output> castlayer_output, unsigned int operation)
    {
        castlayer_input.push_back(hexagon_nn_input());
        castlayer_input.back().src_id = input_op_id;
        castlayer_input.back().output_idx = 0;

        op_id++;

        castlayer_output.push_back(hexagon_nn_output());
        castlayer_output.back().rank = 4;
        auto& max_sizes1 = castlayer_output.back().max_sizes;
        
            max_sizes1[0] = input->buffer().dim[0].extent;
            max_sizes1[1] = input->buffer().dim[1].extent;
            max_sizes1[2] = input->buffer().dim[2].extent;
            max_sizes1[3] = input->buffer().dim[3].extent;
        
        castlayer_output.back().elementsize = sizeof(uint8_t);

        hexagon_nn_->hexagon_nn_append_node(graph_id_,     
                op_id,   operation,   NN_PAD_NA,   castlayer_input.data(), castlayer_input.size(),   castlayer_output.data(), castlayer_output.size());

    }


    inline void addInputCastLayerWithIdx(const HexagonNN* hexagon_nn_, hexagon_nn_nn_id graph_id_, int input_op_id, int& op_id, Tensor* input, int idx,
                            std::vector<hexagon_nn_input> castlayer_input, std::vector<hexagon_nn_output> castlayer_output, unsigned int operation)
    {
        castlayer_input.push_back(hexagon_nn_input());
        castlayer_input.back().src_id = input_op_id;
        castlayer_input.back().output_idx = idx;

        op_id++;

        castlayer_output.push_back(hexagon_nn_output());
        castlayer_output.back().rank = 4;
        auto& max_sizes1 = castlayer_output.back().max_sizes;
        
            max_sizes1[0] = input->buffer().dim[0].extent;
            max_sizes1[1] = input->buffer().dim[1].extent;
            max_sizes1[2] = input->buffer().dim[2].extent;
            max_sizes1[3] = input->buffer().dim[3].extent;
        
        castlayer_output.back().elementsize = sizeof(uint8_t);

        hexagon_nn_->hexagon_nn_append_node(graph_id_,     
                op_id,   operation,   NN_PAD_NA,   castlayer_input.data(), castlayer_input.size(),   castlayer_output.data(), castlayer_output.size());

    }


    inline void addInputCastLayerWithIdxWithTranspose(const HexagonNN* hexagon_nn_, hexagon_nn_nn_id graph_id_, int input_op_id, int& op_id, Tensor* input, int idx,
                            std::vector<hexagon_nn_input> castlayer_input, std::vector<hexagon_nn_output> castlayer_output, unsigned int operation, std::vector<int32_t> transpose_shape)
    {
        castlayer_input.push_back(hexagon_nn_input());
        castlayer_input.back().src_id = input_op_id;
        castlayer_input.back().output_idx = idx;

        op_id++;

        castlayer_output.push_back(hexagon_nn_output());
        castlayer_output.back().rank = 4;
        auto& max_sizes1 = castlayer_output.back().max_sizes;
        
            max_sizes1[0] = input->buffer().dim[transpose_shape[0]].extent;
            max_sizes1[1] = input->buffer().dim[transpose_shape[1]].extent;
            max_sizes1[2] = input->buffer().dim[transpose_shape[2]].extent;
            max_sizes1[3] = input->buffer().dim[transpose_shape[3]].extent;
        
        castlayer_output.back().elementsize = sizeof(uint8_t);

        hexagon_nn_->hexagon_nn_append_node(graph_id_,     
                op_id,   operation,   NN_PAD_NA,   castlayer_input.data(), castlayer_input.size(),   castlayer_output.data(), castlayer_output.size());

    }

    inline void addInputCastLayerWithIdxWithTransposeWithShape(const HexagonNN* hexagon_nn_, hexagon_nn_nn_id graph_id_, int input_op_id, int& op_id, Tensor* input, int idx,
                            std::vector<hexagon_nn_input> castlayer_input, std::vector<hexagon_nn_output> castlayer_output, unsigned int operation, std::vector<int32_t> transpose_shape, std::vector<int> shape)
    {
        castlayer_input.push_back(hexagon_nn_input());
        castlayer_input.back().src_id = input_op_id;
        castlayer_input.back().output_idx = idx;

        op_id++;

        castlayer_output.push_back(hexagon_nn_output());
        castlayer_output.back().rank = 4;
        auto& max_sizes1 = castlayer_output.back().max_sizes;
        
            max_sizes1[0] = shape[transpose_shape[0]];
            max_sizes1[1] = shape[transpose_shape[1]];
            max_sizes1[2] = shape[transpose_shape[2]];
            max_sizes1[3] = shape[transpose_shape[3]];
        
        castlayer_output.back().elementsize = sizeof(uint8_t);

        hexagon_nn_->hexagon_nn_append_node(graph_id_,     
                op_id,   operation,   NN_PAD_NA,   castlayer_input.data(), castlayer_input.size(),   castlayer_output.data(), castlayer_output.size());

    }
    

    extern bool getDSPExecuteMode();
    extern void setDSPExecuteMode(bool mode);
}
#endif