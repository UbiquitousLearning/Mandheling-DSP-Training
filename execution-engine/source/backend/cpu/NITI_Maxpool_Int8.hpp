//
//  NITI_Maxpool_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/8/17.
//  
//

#ifndef NITI_Maxpool_Int8_hpp
#define NITI_Maxpool_Int8_hpp

#include "core/Execution.hpp"
#include "compute/Int8FunctionsOpt.h"

namespace MNN {

class NITI_Maxpool_Int8 : public Execution {
public:
    
    NITI_Maxpool_Int8(Backend *backend, const NITI_Pool_Int8* pool);
    virtual ~NITI_Maxpool_Int8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    // static std::shared_ptr<ResourceInt8> makeResource(Backend *backend, const MNN::Convolution2D *convOp);
    // virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    const NITI_Pool_Int8 *mParameter;
    std::function<void(const Tensor *src, Tensor *dst)> mThreadFunction;
    // nhwc buffer
    std::shared_ptr<Tensor> mInputTemp;
    std::shared_ptr<Tensor> mOutputTemp;
};

} // namespace MNN

#endif /* NITI_Conv_Int8_hpp */
