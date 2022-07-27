//
//  NITIConvInt8.hpp
//  MNN
//
//  Created by xudaliang on 2021/8/17.
//  
//

#ifndef NITI_Conv_Int8_hpp
#define NITI_Conv_Int8_hpp

#include "backend/cpu/CPUConvolution.hpp"
#include "core/Execution.hpp"
#include "compute/Int8FunctionsOpt.h"

namespace MNN {

class NITI_Conv_Int8 : public CPUConvolution {
public:
    
    NITI_Conv_Int8(Backend *backend, const NITI_CONV_Int8* common);
    virtual ~NITI_Conv_Int8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    // static std::shared_ptr<ResourceInt8> makeResource(Backend *backend, const MNN::Convolution2D *convOp);
    // virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    int mTileCount;
    int mThreadNums;
    std::shared_ptr<Tensor> mTempIm2ColBuffer;
    decltype(CoreInt8Functions::NITI_Int8GemmKernel) mGemmKernel;
    decltype(CoreInt8Functions::NITI_MNNInt32ToInt8) mNITIInt32toInt8;
    bool mDoPostProcess = true; //whether quan post process (add bias, min/max then scale to int8)

    bool reorderWeight(Backend* bn, const Convolution2DCommon* common,
                          Tensor* weightOrigin,
                          std::shared_ptr<Tensor>& weight);

    std::shared_ptr<Tensor> acc_int32;
    std::shared_ptr<Tensor> mWeightInt8;

    std::shared_ptr<Tensor> output_int32;
};

} // namespace MNN

#endif /* NITI_Conv_Int8_hpp */
