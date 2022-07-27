//
//  Distributions.hpp
//  MNN
//
//  Created by MNN on 2019/11/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Distributions_hpp
#define Distributions_hpp

#include <MNN/MNNDefine.h>
#include <random>

namespace MNN {
namespace Express {

class Distributions {
public:
    static void uniform(const int count, const float min, const float max, float* r, std::mt19937 gen);
    static void niti_normal_int8(const int count, const float min, const float max, float* temp, int8_t* r, int8_t* wscale);
    static void niti_uniform_int8(const int count, const float min, const float max, float* temp, int8_t* r, int8_t* wscale, std::mt19937 gen);
    static void gaussian(const int count, const float mu, const float sigma, float* r, std::mt19937 gen);
};

} // namespace Express
} // namespace MNN

#endif // Distritutions_hpp
