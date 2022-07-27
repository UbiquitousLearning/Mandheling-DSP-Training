//
//  NITI_CPULeftPoolGrad_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/12/19.
//  
//

#ifndef NITI_CPULeftPoolGrad_Int8_hpp
#define NITI_CPULeftPoolGrad_Int8_hpp

#include "backend/cpu/CPUBackend.hpp"

namespace MNN {
class NITI_CPULeftPoolGrad_Int8 : public Execution {
public:
    virtual ~NITI_CPULeftPoolGrad_Int8() = default;
    NITI_CPULeftPoolGrad_Int8(Backend *b, const NITI_Pool_Int8 *parameter) : Execution(b) {
        mStrideX = parameter->strideX();
        mStrideY = parameter->strideY();
        mKernelX = parameter->kernelX();
        mKernelY = parameter->kernelY();
        mGlobal  = parameter->isGlobal();
        mParameter = parameter;
    }
    
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    int mStrideX;
    int mStrideY;
    int mKernelX;
    int mKernelY;
    bool mGlobal;
    int mPadX;
    int mPadY;
    const NITI_Pool_Int8* mParameter;
};
} // namespace MNN
#endif /* CPUPoolGrad_hpp */
