//
//  NITI_Matmul_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/8/17.
//  
//

#ifndef NITI_Matmul_Int8_hpp
#define NITI_Matmul_Int8_hpp

#include "core/Execution.hpp"
#include "compute/Int8FunctionsOpt.h"
#include "backend/cpu/CPUConvolution.hpp"
#include "CPUConvolution.hpp"

namespace MNN {

class NITI_Matmul_Int8 : public Execution {
public:
    
    NITI_Matmul_Int8(Backend *backend, bool transposeA, bool transposeB, bool transposeC);
    virtual ~NITI_Matmul_Int8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    // static std::shared_ptr<ResourceInt8> makeResource(Backend *backend, const MNN::Convolution2D *convOp);
    // virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    int mTileCount;
    int mThreadNums;
    ConvolutionCommon::Im2ColParameter mIm2ColParamter;

    std::shared_ptr<Tensor> mTempIm2ColBuffer;
    std::shared_ptr<Tensor>  outdiffReorder;

    std::shared_ptr<Tensor> gradientInt8Transpose;

    std::shared_ptr<Tensor> acc_int32;

    bool mTransposeA;
    bool mTransposeB;
    bool mTransposeC;

    bool reorderWeight(Backend* bn, int oc, int ic,
                          Tensor* weightOrigin,
                          std::shared_ptr<Tensor>& weight);
};

} // namespace MNN

#endif /* NITI_Matmul_Int8_hpp */
