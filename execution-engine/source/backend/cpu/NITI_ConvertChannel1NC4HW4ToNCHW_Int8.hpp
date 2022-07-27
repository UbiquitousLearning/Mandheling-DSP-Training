//
//  NITI_ConvertChannel1NC4HW4ToNCHW_Int8.hpp
//  MNN
//
//  Created by xudaliang on 2021/09/26.
//  
//

#ifndef NITI_ConvertChannel1NC4HW4ToNCHW_Int8_hpp
#define NITI_ConvertChannel1NC4HW4ToNCHW_Int8_hpp

#include "core/Execution.hpp"

namespace MNN {

class NITI_ConvertChannel1NC4HW4ToNCHW_Int8 : public Execution {
public:
    NITI_ConvertChannel1NC4HW4ToNCHW_Int8(Backend *backend);
    virtual ~NITI_ConvertChannel1NC4HW4ToNCHW_Int8();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif /* NITI_ConvertChannel1NC4HW4ToNCHW_Int8_hpp */
