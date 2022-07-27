//
//  NITI_DSPWeightRotateRef_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/11/17.
//
//

#ifndef NITI_DSPWeightRotateRef_Int8_hpp
#define NITI_DSPWeightRotateRef_Int8_hpp

#include "core/Execution.hpp"

namespace MNN {
class NITI_DSPWeightRotateRef_Int8 : public Execution {
public:
    NITI_DSPWeightRotateRef_Int8(Backend *b);
    virtual ~NITI_DSPWeightRotateRef_Int8() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    
private:

    void rotate180(int8_t arraySrc[], int8_t arrayDes[], int rows, int cols);
};
} // namespace MNN

#endif /* NITI_DSPWeightRotateRef_Int8_hpp */
