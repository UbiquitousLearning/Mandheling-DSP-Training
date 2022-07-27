//
//  NITI_Pad_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/09/26.
//  
//

#ifndef NITI_Pad_Int8_hpp
#define NITI_Pad_Int8_hpp

#include "core/Execution.hpp"

namespace MNN {

class NITI_Pad_Int8 : public Execution {
public:
    NITI_Pad_Int8(Backend *backend, int pad);
    virtual ~NITI_Pad_Int8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    bool isEltwiseInt8 = true;
    int mPad = 0;
};

} // namespace MNN

#endif /* NITI_Pad_Int8_hpp */
