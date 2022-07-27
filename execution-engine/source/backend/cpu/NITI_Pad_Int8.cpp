//
//  NITI_Pad_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/9/26.
//  
//

#include "backend/cpu/NITI_Pad_Int8.hpp"
#include "backend/cpu/CPUBackend.hpp"

namespace MNN {

NITI_Pad_Int8::~NITI_Pad_Int8() {
    // Do nothing
}
NITI_Pad_Int8::NITI_Pad_Int8(Backend* backend, int pad) : MNN::Execution(backend), mPad(pad) {

}


ErrorCode NITI_Pad_Int8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return NO_ERROR;
}

ErrorCode NITI_Pad_Int8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    
    auto input = inputs[0];
    auto output = outputs[0];
    
    int ibatch = input->batch();
    int ichannel = input->channel();
    int iwidth = input->width();
    int iheight = input->height();
    
    int obatch = output->batch();
    int ochannel = output->channel();
    int owidth = output->width();
    int oheight = output->height();

    int8_t* inputPtr = input->host<int8_t>();
    int8_t* outputPtr = output->host<int8_t>();

    memset(outputPtr, 0 , output->elementSize());

    for(int i=0; i<obatch; i++) {
        int8_t* bptr = outputPtr+ i*output->stride(0);
        int8_t* biptr = inputPtr+ i*input->stride(0);
        for(int j=0; j<ochannel; j++) {

            int8_t* cptr = bptr + j*output->stride(1);
            int8_t* ciptr = biptr+ j*input->stride(1);
            memset( cptr, 0 , owidth*oheight);

            for(int k = mPad; k < oheight - mPad; k++) {
                
                int8_t* kptr = cptr + k * owidth;
                int8_t* kiptr = ciptr + (k-mPad) * iwidth;

                memcpy(kptr + mPad, kiptr, iwidth );
            }

        }
    }

    return NO_ERROR;
}

class NITI_Pad_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {

        
        return new NITI_Pad_Int8(backend, op->main_as_NITI_PAD_Int8()->pad());
    }
};

REGISTER_CPU_OP_CREATOR(NITI_Pad_Int8_Creator, OpType_NITI_PAD_Int8);

} // namespace MNN
