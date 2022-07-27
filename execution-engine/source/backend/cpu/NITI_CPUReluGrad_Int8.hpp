//
//  NITI_CPUReluGrad_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/09/17.
//
//

#ifndef NITI_CPUReluGrad_Int8_hpp
#define NITI_CPUReluGrad_Int8_hpp

#include "core/Execution.hpp"

namespace MNN {
class NITI_CPUReluGrad_Int8 : public Execution {
public:
    NITI_CPUReluGrad_Int8(Backend *b);
    virtual ~NITI_CPUReluGrad_Int8() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    
private:
    int mAxis;
    std::shared_ptr<Tensor> s;
    std::shared_ptr<Tensor> out_max;
};
} // namespace MNN

#endif /* NITI_CPUReluGrad_Int8_hpp */
