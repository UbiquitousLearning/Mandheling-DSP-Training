//
//  Distributions.cpp
//  MNN
//
//  Created by MNN on 2019/11/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Distributions.hpp"
#include <cmath>
#include <sys/time.h>

#include <sstream>

namespace MNN {
namespace Express {

void Distributions::uniform(const int count, const float min, const float max, float *r, std::mt19937 gen) {
    std::uniform_real_distribution<float> dis(min, max);
    for (int i = 0; i < count; i++) {
        r[i] = dis(gen);
    }
}

int layer_num=0;
void Distributions::niti_normal_int8(const int count, const float min, const float max, float* temp, int8_t* r, int8_t* wscale) {
    struct timeval time;
    gettimeofday(&time, NULL);
    std::default_random_engine e(time.tv_usec);
    std::normal_distribution<float> dis(min, max); 
    for (int i = 0; i < count; i++) {
        temp[i] = dis(e);
    }

    float range = 0;
    for(int i=0; i<count; i++) {
        if(fabs(temp[i]) > range)
            range = fabs(temp[i]);
    }

    int8_t bitwidth = ceil(log2f(range));
    int8_t act_exp = bitwidth - 7;

    for(int i=0; i<count; i++) {
        r[i] = (int8_t)round(temp[i]/range * 127);
    }

    *wscale = act_exp;


}

void Distributions::niti_uniform_int8(const int count, const float min, const float max, float* temp, int8_t* r, int8_t* wscale, std::mt19937 gen) {
    std::uniform_real_distribution<float> dis(min, max);
    for (int i = 0; i < count; i++) {
        temp[i] = dis(gen);
    }

    float range = 0;
    for(int i=0; i<count; i++) {
        if(abs(temp[i]) > range)
            range = abs(temp[i]);
    }

    int8_t bitwidth = ceil(log2(range));
    int8_t act_exp = bitwidth - 7;

    for(int i=0; i<count; i++) {
        r[i] = (int8_t)round(temp[i]/range * 127);
    }
    *wscale = act_exp;
}

void Distributions::gaussian(const int count, const float mu, const float sigma, float *r, std::mt19937 gen) {
    std::normal_distribution<float> dis(mu, sigma);
    for (int i = 0; i < count; i++) {
        r[i] = dis(gen);
    }
}

} // namespace Express
} // namespace MNN
