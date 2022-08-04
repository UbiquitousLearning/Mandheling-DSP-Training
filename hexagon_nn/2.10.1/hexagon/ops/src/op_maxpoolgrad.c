#include <nn_graph.h>
#include <string.h>
#include <math.h>

#include "hvx_inlines.h"
#include "nn_asm_ops.h"

int maxpoolgrad_8_execute(struct nn_node *self, struct nn_graph *nn) {

    const struct tensor *origin_tensor = self->inputs[0];
    const struct tensor *outputOrigin_tensor = self->inputs[1];
    const struct tensor *outputDiff_tensor = self->inputs[2];
    const struct tensor *parameter_tensor = self->inputs[3];
    const struct tensor *pad_tensor = self->inputs[4];

    int32_t pad = *(int32_t*)(pad_tensor->data);

    struct tensor *output_tensor = self->outputs[0];

    // int32_t mStrideX = parameter_tensor->shape.batches;
    int32_t mStrideY = parameter_tensor->shape.height;

    int32_t mKernelX = parameter_tensor->shape.width;
    int32_t mKernelY = parameter_tensor->shape.depth;

    int32_t ib = origin_tensor->shape.batches;
    int32_t ih = origin_tensor->shape.height;
    int32_t iw = origin_tensor->shape.width;
    int32_t ic = origin_tensor->shape.depth;

    HVX_Vector *originPtr = origin_tensor->data;
    HVX_Vector *outputOriginPtr = outputOrigin_tensor->data;

    HVX_Vector *outputDiffPtr = outputDiff_tensor->data;
    HVX_Vector *outputPtr = output_tensor->data;

    memset(outputPtr, 128, tensor_element_count(origin_tensor));

    int32_t ob = outputDiff_tensor->shape.batches;
    // int32_t oh = outputDiff_tensor->shape.height;
    // int32_t ow = outputDiff_tensor->shape.width;
    int32_t oc = outputDiff_tensor->shape.depth;

    if(oc == 32) {
        for(int i=0;i<ib;i++) {

            for(int j=0; j<ih; j+=mStrideY) {

                int offset = i*ih*iw*ic+ j*iw*ic;
                int ooffset = offset/mKernelY/mKernelX;

                for(int l = 0; l < iw; l+=8) {

                    int woffset = offset + l*ic;
                    int wooffset = ooffset + l*oc/2;

                    HVX_Vector zero = Q6_Vb_vsplat_R(0);

                    HVX_VectorPred finishbuf = Q6_Q_vcmp_eq_VbVb(zero, zero);

                    HVX_Vector outputdiffbuf = outputDiffPtr[wooffset/128];
                    HVX_Vector outputoriginbuf = outputOriginPtr[wooffset/128];

                    for(int k=0; k<mKernelY; k++) {

                        HVX_Vector originbuf = originPtr[woffset/128];
                        HVX_Vector originbuf2 = originPtr[woffset/128+1];

                        HVX_VectorPair originbuf_ACBD =  Q6_W_vdeal_VVR(originbuf, originbuf2, -32);
                        

                        HVX_VectorPred tempbuf;
                        HVX_VectorPred tempbuf2;

                        HVX_VectorPred samebuf = Q6_Q_vcmp_eq_VbVb(Q6_V_lo_W(originbuf_ACBD), outputoriginbuf);

                        tempbuf = samebuf;
                        samebuf = Q6_Q_and_QQ(samebuf, finishbuf);

                        Q6_vmem_QRIV(samebuf, outputPtr + woffset/128, outputdiffbuf);

                        finishbuf = Q6_Q_and_QQn(finishbuf, tempbuf);


                        HVX_VectorPred samebuf2 = Q6_Q_vcmp_eq_VbVb(Q6_V_hi_W(originbuf_ACBD), outputoriginbuf);

                        tempbuf2 = samebuf2;
                        samebuf2 = Q6_Q_and_QQ(samebuf2, finishbuf);

                        Q6_vmem_QRIV(samebuf2, outputPtr + woffset/128 + 1, outputoriginbuf);

                        
                        finishbuf = Q6_Q_and_QQn(finishbuf, tempbuf2);


                        HVX_Vector outputbuf = outputPtr[woffset/128];
                        HVX_Vector outputbuf2 = outputPtr[woffset/128 + 1];


                        HVX_VectorPair rightret = Q6_W_vshuff_VVR(outputbuf, outputbuf2, -32);

                        outputPtr[woffset/128] = Q6_V_lo_W(rightret);
                        outputPtr[woffset/128+1] = Q6_V_hi_W(rightret);

                        woffset += iw*ic;
                    }
                }
                
            }
        }
    } else if(oc == 64) {

        for(int i=0;i<ib;i++) {

            for(int j=0; j<ih; j+=mStrideY) {

                int offset = i*ih*iw*ic+ j*iw*ic;
                int ooffset = offset/mKernelY/mKernelX;

                for(int l = 0; l < iw; l+=4) {

                    int woffset = offset + l*ic;
                    int wooffset = ooffset + l*oc/2;

                    HVX_Vector zero = Q6_Vb_vsplat_R(0);

                    HVX_VectorPred finishbuf = Q6_Q_vcmp_eq_VbVb(zero, zero);

                    HVX_Vector outputdiffbuf = outputDiffPtr[wooffset/128];
                    HVX_Vector outputoriginbuf = outputOriginPtr[wooffset/128];

                    for(int k=0; k<mKernelY; k++) {
                        
                        HVX_Vector originbuf = originPtr[woffset/128];
                        HVX_Vector originbuf2 = originPtr[woffset/128+1];

                        HVX_VectorPair originbuf_ACBD =  Q6_W_vdeal_VVR(originbuf, originbuf2, -64);
                        

                        HVX_VectorPred tempbuf;
                        HVX_VectorPred tempbuf2;

                        HVX_VectorPred samebuf = Q6_Q_vcmp_eq_VbVb(Q6_V_lo_W(originbuf_ACBD), outputoriginbuf);

                        tempbuf = samebuf;
                        samebuf = Q6_Q_and_QQ(samebuf, finishbuf);

                        Q6_vmem_QRIV(samebuf, outputPtr + woffset/128, outputdiffbuf);

                        finishbuf = Q6_Q_and_QQn(finishbuf, tempbuf);

                        HVX_VectorPred samebuf2 = Q6_Q_vcmp_eq_VbVb(Q6_V_hi_W(originbuf_ACBD), outputoriginbuf);

                        tempbuf2 = samebuf2;
                        samebuf2 = Q6_Q_and_QQ(samebuf2, finishbuf);

                        Q6_vmem_QRIV(samebuf2, outputPtr + woffset/128 + 1, outputoriginbuf);

                        
                        finishbuf = Q6_Q_and_QQn(finishbuf, tempbuf2);


                        HVX_Vector outputbuf = outputPtr[woffset/128];
                        HVX_Vector outputbuf2 = outputPtr[woffset/128 + 1];


                        HVX_VectorPair rightret = Q6_W_vshuff_VVR(outputbuf, outputbuf2, -64);

                        outputPtr[woffset/128] = Q6_V_lo_W(rightret);
                        outputPtr[woffset/128+1] = Q6_V_hi_W(rightret);

                        woffset += iw*ic;
                    }


                }
                
            }


        }

    } else if (oc %128 == 0) {

        for(int i=0;i<ib;i++) {

            for(int j=0; j<ih; j+=mStrideY) {

                int offset = i*ih*iw*ic+ j*iw*ic;
                int ooffset = offset/mKernelY/mKernelX;

                for(int l = 0; l < iw; l+= 2 ) {

                    for (int c = 0; c<ic; c+=128) {

                        int woffset = offset + l*ic + c;
                        int wooffset = ooffset + l*oc/2 + c;

                        HVX_Vector zero = Q6_Vb_vsplat_R(0);

                        HVX_VectorPred finishbuf = Q6_Q_vcmp_eq_VbVb(zero, zero);

                        HVX_Vector outputdiffbuf = outputDiffPtr[wooffset/128];
                        HVX_Vector outputoriginbuf = outputOriginPtr[wooffset/128];

                        for(int k=0; k<mKernelY; k++) {

                            HVX_Vector originbuf = originPtr[woffset/128];
                            HVX_Vector originbuf2 = originPtr[woffset/128+oc/128];

                            HVX_VectorPred tempbuf;

                            HVX_VectorPred samebuf = Q6_Q_vcmp_eq_VbVb(outputoriginbuf, originbuf);

                            tempbuf = samebuf;
                            samebuf = Q6_Q_and_QQ(samebuf, finishbuf);

                            Q6_vmem_QRIV(samebuf, outputPtr + woffset/128, outputdiffbuf);
                            finishbuf = Q6_Q_and_QQn(finishbuf, tempbuf);

                            HVX_VectorPred samebuf2 = Q6_Q_vcmp_eq_VbVb(outputoriginbuf, originbuf2);

                            tempbuf = samebuf2;
                            samebuf2 = Q6_Q_and_QQ(samebuf2, finishbuf);

                            Q6_vmem_QRIV(samebuf2, outputPtr + woffset/128 + oc/128, outputdiffbuf);

                            
                            finishbuf = Q6_Q_and_QQn(finishbuf, tempbuf);

                            woffset += iw*ic;
                        }

                    }

                    
                }
                
            }


        }
        
    } else {

    }

    int fail = 0;

    if(pad !=0 ) {
        if ((fail = tensor_out_prepare_padded_d32(output_tensor,
						  ob,
						  ih,0,0,
						  iw,0,0,
						  ic-pad,0,pad,
						  NN_TYPE_QUINT8)) != 0) {
            errlog(nn,"output tensor prep fail %d", fail);
            return fail;
        }
    } else {
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
    }

    

    
    return 0;
}

struct nn_node_ops nn_ops_for_QuantizedMaxPoolGrad_8 = {
	.execute = maxpoolgrad_8_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_QuantizedMaxPoolGrad_8_ref = {
	.execute = maxpoolgrad_8_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(1),
};