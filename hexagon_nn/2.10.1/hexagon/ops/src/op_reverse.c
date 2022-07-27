#include <nn_graph.h>
#include <string.h>
#include <math.h>

#include "hvx_inlines.h"
#include "nn_asm_ops.h"

int reverse_8_execute(struct nn_node *self, struct nn_graph *nn) {

    const struct tensor *origin_tensor = self->inputs[0];

    struct tensor *output_tensor = self->outputs[0];

    

    output_tensor->shape.batches = origin_tensor->shape.batches;
	output_tensor->shape.height = origin_tensor->shape.height;
	output_tensor->shape.width = origin_tensor->shape.width;
	output_tensor->shape.depth = origin_tensor->shape.depth;
	output_tensor->data_size = origin_tensor->shape.batches *
		output_tensor->shape.height *
		output_tensor->shape.width*
		output_tensor->shape.depth * sizeof(int8_t);
	output_tensor->format.layout = NN_LAYOUT_PLAIN; //simple in-memory array format
	output_tensor->format.type = 1; // 1=chars, 4=int32/float

    
    return 0;
}

struct nn_node_ops nn_ops_for_QuantizedReverse_8 = {
	.execute = reverse_8_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_QuantizedReverse_8_ref = {
	.execute = reverse_8_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(1),
};