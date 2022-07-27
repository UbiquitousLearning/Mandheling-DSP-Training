//
//  NITI_DSPNop_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/12/17.
//
//

#ifndef NITI_DSPNop_Int8_hpp
#define NITI_DSPNop_Int8_hpp

#include "core/Execution.hpp"

#include "MNN/hexagon/hexagon_nn/hexagon_nn_ops.h"
#include "MNN/hexagon/hexagon_nn/hexagon_nn.h"
#include "MNN/hexagon/hexagon_nn/HexagonRunningUtils.hpp"

namespace MNN {
class NITI_DSPNop_Int8 : public Execution {
public:
    NITI_DSPNop_Int8(Backend *b, const MNN::Op *op);
    virtual ~NITI_DSPNop_Int8() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    ErrorCode GlobalExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    
private:

    const Op* mOp;

};
} // namespace MNN

#endif /* NITI_DSPBinary_Int8_hpp */
