/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the
 * disclaimer below) provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of The Linux Foundation nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
 * GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
 * HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

/*
 * This is reverse op of pack.
 * 
 */

#include <nn_graph.h>
#include <string.h>
#include <math.h>
#include "hvx_inlines.h"
#include "quantize.h"
#include "nn_asm_ops.h"
#include "nn_bufferpool.h"

// #define TEST_PERFORMANCE 1

static int castint8touint8_execute(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *a_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];

#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif

    int32_t size  = a_tensor->data_size;
    HVX_Vector* input  = (HVX_Vector* )(a_tensor->data);
    HVX_Vector* output  = (HVX_Vector* )(out_tensor->data);
    
    HVX_Vector const2 = Q6_Vb_vsplat_R(128);

    for (int i = 0; i<size/128; i+=2) 
    {
        HVX_Vector t1 = *input;
		HVX_Vector t12 = *(input+1);

        HVX_Vector t2 = Q6_V_vxor_VV( t1, const2);
		HVX_Vector t22 = Q6_V_vxor_VV( t12, const2);
        
        *output = t2;
        output++;
		*output = t22;
		output++;
        input+=2;
    }

	if(size % 256 != 0) {

		int8_t* toutput = out_tensor->data;
		uint8_t* tinput = a_tensor->data;

		int32_t begin = size / 256 * 256;

		for(int i=begin; i<size; i++) {
			toutput[i] = tinput[i] ^ 128;
		}


	}

    // Set the size of the output tensor.
	out_tensor->shape.batches = a_tensor->shape.batches;
	out_tensor->shape.height = a_tensor->shape.height;
	out_tensor->shape.width = a_tensor->shape.width;
	out_tensor->shape.depth = a_tensor->shape.depth;
	out_tensor->data_size = out_tensor->shape.batches *
		out_tensor->shape.height *
		out_tensor->shape.width*
		out_tensor->shape.depth * sizeof(uint8_t);
	out_tensor->format.layout = NN_LAYOUT_PLAIN; //simple in-memory array format
	out_tensor->format.type = 1; // 1=chars, 4=int32/float
	// Note: This format field will change with new interface updates.
	//       e.g. format.type will soon distinguish int32 versus float types.

#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("castuint8toint8 cycles = %d\n", (end_time-start_time));
#endif

	return 0;
}

static int castuint8toint8_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *a_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];

#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif

    int32_t size  = a_tensor->data_size;
    HVX_Vector* input  = (HVX_Vector* )(a_tensor->data);
    HVX_Vector* output  = (HVX_Vector* )(out_tensor->data);
    
    HVX_Vector const2 = Q6_Vb_vsplat_R(128);
	// HVX_Vector const1 = Q6_Vb_vsplat_R(8);


    for (int i = 0; i<size/128; i+=2) 
    {
        HVX_Vector t1 = *input;
		HVX_Vector t12 = *(input+1);

		// t1 = Q6_Vub_vlsr_VubR(t1, 4);
        // HVX_Vector t2 = Q6_Vb_vsub_VbVb( t1, const1);
		
		// t12 = Q6_Vub_vlsr_VubR(t12, 4);
		// HVX_Vector t22 = Q6_Vb_vsub_VbVb( t12, const1);

		HVX_Vector t2 = Q6_V_vxor_VV( t1, const2);
		HVX_Vector t22 = Q6_V_vxor_VV( t12, const2);
        
        *output = t2;
        output++;
		*output = t22;
		output++;
        input+=2;
    }

	if(size % 256 != 0) {

		int8_t* toutput = out_tensor->data;
		uint8_t* tinput = a_tensor->data;

		int32_t begin = size / 256 * 256;

		for(int i=begin; i<size; i++) {
			toutput[i] = tinput[i] ^ 128;
		}


	}

    // Set the size of the output tensor.
	out_tensor->shape.batches = a_tensor->shape.batches;
	out_tensor->shape.height = a_tensor->shape.height;
	out_tensor->shape.width = a_tensor->shape.width;
	out_tensor->shape.depth = a_tensor->shape.depth;
	out_tensor->data_size = out_tensor->shape.batches *
		out_tensor->shape.height *
		out_tensor->shape.width*
		out_tensor->shape.depth * sizeof(uint8_t);
	out_tensor->format.layout = NN_LAYOUT_PLAIN; //simple in-memory array format
	out_tensor->format.type = 1; // 1=chars, 4=int32/float
	// Note: This format field will change with new interface updates.
	//       e.g. format.type will soon distinguish int32 versus float types.

#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("castuint8toint8 cycles = %d\n", (end_time-start_time));
#endif

	return 0;
}

struct nn_node_ops nn_ops_for_Quantized_CastUInt8ToInt8 = {
	.execute = castuint8toint8_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(1),
	.n_outputs = NN_IOCOUNT_GE(1),
};

struct nn_node_ops nn_ops_for_Quantized_CastInt8ToUInt8 = {
	.execute = castint8touint8_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(1),
	.n_outputs = NN_IOCOUNT_GE(1),
};
