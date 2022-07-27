#include <nn_graph.h>
#include <string.h>
#include <math.h>


struct loss_info {
    int64_t *sPtr;
    int64_t *out_maxPtr;
    int64_t *outputDataPtr;
    int64_t *out_sumPtr;
    int64_t *out_gradPtr;
    int32_t *out_grad_finalPtr;
    int32_t *target_maxPtr;
};

int loss_8_check_error_return (struct nn_graph *nn, struct loss_info *info, char const * alloctag )
{
	if(info){
		if(info->sPtr != NULL) nn_free(info->sPtr);
		if(info->out_maxPtr != NULL) nn_free(info->out_maxPtr);
		if(info->outputDataPtr != NULL) nn_free(info->outputDataPtr);
		if(info->out_sumPtr != NULL) nn_free(info->out_sumPtr);
		if(info->out_gradPtr != NULL) nn_free(info->out_gradPtr);
		if(info->out_grad_finalPtr != NULL) nn_free(info->out_grad_finalPtr);
		if(info->target_maxPtr != NULL) nn_free(info->target_maxPtr);
		nn_free(info);
	}
	if( alloctag != NULL)
		errlog(nn,"alloc failed for %s", alloctag);
	return -1;
}

int32_t NITI_int8_clip(int32_t a) {
    if(a > 127)
        return 127;
    else if(a<-127)
        return -127;
    else
        return a;
}
int32_t NITI_sign(int32_t a) {
    if(a > 0)
        return 1;
    else if( a < 0)
        return -1;
    else
        return 0;
}

int loss_8_check(struct nn_node *self, struct nn_graph *nn) {

    const struct tensor *in_tensor = self->inputs[0];

    int32_t inputSize = tensor_element_count(in_tensor);
    int32_t ib = in_tensor->shape.batches;

    struct loss_info *info = self->opaque;

    if ((info = nn_calloc(1,sizeof(*info))) == NULL) {
		return errlog(nn,"couldn't allocate info");
	}

    if ((info->sPtr = nn_malloc(inputSize*sizeof(int64_t))) == NULL) {
		return loss_8_check_error_return(nn, info,"sPtr");
	}

    if ((info->out_maxPtr = nn_malloc(ib*sizeof(int64_t))) == NULL) {
		return loss_8_check_error_return(nn, info,"out_maxPtr");
	}

    if ((info->outputDataPtr = nn_malloc(inputSize*sizeof(int64_t))) == NULL) {
		return loss_8_check_error_return(nn, info,"outputDataPtr");
	}


    if ((info->out_sumPtr = nn_malloc(ib*sizeof(int64_t))) == NULL) {
		return loss_8_check_error_return(nn, info,"out_sumPtr");
	}

    if ((info->out_gradPtr = nn_malloc(inputSize*sizeof(int64_t))) == NULL) {
		return loss_8_check_error_return(nn, info,"out_gradPtr");
	}

    if ((info->out_grad_finalPtr = nn_malloc(inputSize*sizeof(int32_t))) == NULL) {
		return loss_8_check_error_return(nn, info,"out_grad_finalPtr");
	}

    if ((info->target_maxPtr = nn_malloc(ib*sizeof(int32_t))) == NULL) {
		return loss_8_check_error_return(nn, info,"target_maxPtr");
	}

    self->opaque = info;

    return 0;
}

void NITI_MNNPstoShiftInt32ToInt8(int32_t* input, int shift, int8_t* dest, int size) {
    if(shift % 2 == 1) {
        for(int i=0;i<size;i++) {
            int32_t round_temp = input[i] / (1 << shift);
            int32_t prob = abs(input[i] - (round_temp * (1 << shift) ) );
            int32_t quantized_prob = prob / (1 << (shift/2) );
            int32_t pseudo_rand_num = prob - (quantized_prob * (1 << (shift/2)) );
            pseudo_rand_num = pseudo_rand_num * 2;

            int32_t round_1 = (quantized_prob > pseudo_rand_num);
            int32_t sign_1 = NITI_sign(input[i]);
            dest[i] = (int8_t)NITI_int8_clip(round_temp + round_1*sign_1);
        }
    } else {
        for(int i=0;i<size;i++) {
            int32_t round_temp = input[i] / (1 << shift);
            int32_t prob = abs(input[i] - (round_temp * (1 << shift) ) );
            int32_t quantized_prob = prob / (1 << (shift/2) );
            int32_t pseudo_rand_num = prob - (quantized_prob * (1 << (shift/2)) );

            int32_t round_1 = (quantized_prob > pseudo_rand_num);
            int32_t sign_1 = NITI_sign(input[i]);
            dest[i] = (int8_t)NITI_int8_clip(round_temp + round_1*sign_1);
        }
    }
}

int loss_8_execute(struct nn_node *self, struct nn_graph *nn) {

    const struct tensor *in_tensor = self->inputs[0];
    const struct tensor *ascale_tensor = self->inputs[2];
    const struct tensor *label_tensor = self->inputs[1];

    struct tensor *out_tensor = self->outputs[0];

    struct loss_info *info = self->opaque;

    int8_t ascale = *(int8_t*)(ascale_tensor->data);
    int32_t* target_Ptr = label_tensor->data;
    uint8_t* outPtr = out_tensor->data;


    int32_t inputSize = tensor_element_count(in_tensor);

    int8_t *inputDataPtr = (int8_t*)(in_tensor->data);

    int32_t ib = in_tensor->shape.batches;
    int32_t ic = in_tensor->shape.depth;

    for(int i=0;i<inputSize;i++) {
        inputDataPtr[i] = *((uint8_t*)inputDataPtr + i)  - 128;
    }

    if(ascale > -7) {

        if(ascale >= 0) {
            for(int i=0;i<inputSize;i++) {
                int64_t temp = (int64_t)inputDataPtr[i] * 47274;
                temp = temp / (1 << 15);
                info->sPtr[i] = temp * (1 << ascale);
            }

        } else {
            for(int i=0;i<inputSize;i++) {
                int64_t temp = (int64_t)inputDataPtr[i] * 47274;
                temp = temp / (1 << 15);
                info->sPtr[i] = temp / ( 1 << (-ascale));
            }
            
        }

        for(int i=0;i<ib;i++) {
            int64_t max = info->sPtr[i*ic];
            for(int j=1;j<ic;j++) {
                if(max < info->sPtr[i*ic+j])
                    max = info->sPtr[i*ic+j];
            }
            info->out_maxPtr[i] = max - 10;
        }

        for(int i=0;i<ib;i++) {
            for(int j=0;j<ic;j++) {
                int64_t temp = info->sPtr[i*ic+j];
                temp -= info->out_maxPtr[i];
                temp = (temp > 0)? temp : 0;
                info->outputDataPtr[i*ic+j] = (1<<temp) - 1;
            }
        }
        

    } else {
        int64_t base = 1 << (1 - 2*(int64_t)ascale);
        int64_t shiftbase = 1 << (1 - (int64_t)ascale);

        for(int i=0;i<inputSize;i++) {
            int64_t temp = (int64_t)inputDataPtr[i];
            info->outputDataPtr[i] = base + temp*shiftbase + temp * temp;
        }
    }

    for(int i=0;i<ib;i++) {
        int64_t sum = 0;
        for(int j=0;j<ic;j++) {
            sum += info->outputDataPtr[i*ic+j];
        }
        info->out_sumPtr[i] = sum;
    }

    for(int i=0;i<ib;i++) {
        int64_t base = info->out_sumPtr[i];
        for(int j=0;j<ic;j++) {
            int64_t temp = info->outputDataPtr[i*ic+j];
            temp = temp * (1 << 11);
            temp = temp / base;
            info->out_gradPtr[i*ic+j] = temp;
        }
    }

    for(int i = 0; i<ib;i++) {
        info->target_maxPtr[i] = 0;
        for(int j=0; j<10; j++) {
            if(target_Ptr[i*10+j] != 0) {
                info->target_maxPtr[i] = j;
                break;
            }
        }
    }

    for(int i=0;i<ib;i++) {
        int64_t sum = 0;
        for(int j=0;j<ic;j++) {
            sum += (int64_t)info->out_gradPtr[i*ic+j];
        }
        info->out_sumPtr[i] = sum;
    }

    for(int i=0; i<ib; i++) {
        for(int j=0; j<ic; j++) {
            info->out_grad_finalPtr[i*ic+j] = (int32_t)info->out_gradPtr[i*ic+j];
        }
    } 

    for(int i=0; i<ib; i++) {
        info->out_grad_finalPtr[i*ic + info->target_maxPtr[i]] = (int32_t)(info->out_gradPtr[i*ic + info->target_maxPtr[i]] - info->out_sumPtr[i]);
    }

    NITI_MNNPstoShiftInt32ToInt8(info->out_grad_finalPtr, 4, (int8_t*)outPtr, inputSize);

    for(int i=0;i<inputSize;i++) {
        outPtr[i] = *((int8_t*)outPtr + i)  + 128;
    }


    out_tensor->shape.batches = in_tensor->shape.batches;
	out_tensor->shape.height = in_tensor->shape.height;
	out_tensor->shape.width = in_tensor->shape.width;
	out_tensor->shape.depth = in_tensor->shape.depth;
	out_tensor->data_size = out_tensor->shape.batches *
		out_tensor->shape.height *
		out_tensor->shape.width*
		out_tensor->shape.depth * sizeof(int8_t);
	out_tensor->format.layout = NN_LAYOUT_PLAIN; //simple in-memory array format
	out_tensor->format.type = 1; // 1=chars, 4=int32/float

    return 0;
}

static int loss_8_dtor(struct nn_node *self, struct nn_graph *nn) {
    struct loss_info *info = self->opaque;
	if (info != NULL) {
		nn_free(info->sPtr);
		nn_free(info->out_maxPtr);
		nn_free(info->outputDataPtr);
		nn_free(info->out_sumPtr);
		nn_free(info->out_gradPtr);
		nn_free(info->out_grad_finalPtr);
		nn_free(info->target_maxPtr);
		nn_free(info);
	}
	self->opaque = NULL;
	return node_free_common(self,nn);
}

struct nn_node_ops nn_ops_for_QuantizedLoss_8 = {
	.execute = loss_8_execute,
	.check = loss_8_check,
	.ctor = node_alloc_common,
	.dtor = loss_8_dtor,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_QuantizedLoss_8_ref = {
	.execute = loss_8_execute,
	.check = loss_8_check,
	.ctor = node_alloc_common,
	.dtor = loss_8_dtor,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(1),
};