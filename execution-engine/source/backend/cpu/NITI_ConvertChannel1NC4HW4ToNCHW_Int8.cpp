//
//  NITI_ConvertChannel1NC4HW4ToNCHW_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/9/26.
//  
//

#include "backend/cpu/NITI_ConvertChannel1NC4HW4ToNCHW_Int8.hpp"
#include "backend/cpu/CPUBackend.hpp"

namespace MNN {

NITI_ConvertChannel1NC4HW4ToNCHW_Int8::~NITI_ConvertChannel1NC4HW4ToNCHW_Int8() {
    // Do nothing
}
NITI_ConvertChannel1NC4HW4ToNCHW_Int8::NITI_ConvertChannel1NC4HW4ToNCHW_Int8(Backend* backend) : MNN::Execution(backend) {

}


ErrorCode NITI_ConvertChannel1NC4HW4ToNCHW_Int8::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    return NO_ERROR;
}

ErrorCode NITI_ConvertChannel1NC4HW4ToNCHW_Int8::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    
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

    MNN_ASSERT(input->elementSize() == ibatch * ichannel * iwidth * iheight * 4);
    MNN_ASSERT(ichannel == 1);

    for(int i = 0;i < ibatch; i++) {
        for(int j = 0; j < iwidth * iheight; j++) {
            outputPtr[i*iwidth * iheight + j] = inputPtr[i*iwidth * iheight * 4 + j*4];
        }
    }

    return NO_ERROR;
}

class NITI_ConvertChannel1NC4HW4ToNCHW_Int8_Creator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {

        
        return new NITI_ConvertChannel1NC4HW4ToNCHW_Int8(backend);
    }
};

REGISTER_CPU_OP_CREATOR(NITI_ConvertChannel1NC4HW4ToNCHW_Int8_Creator, OpType_NITI_ConvertChannel1NC4HW4ToNCHW_Int8);

} // namespace MNN
