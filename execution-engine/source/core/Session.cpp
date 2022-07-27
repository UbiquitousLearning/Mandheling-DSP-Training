//
//  Session.cpp
//  MNN
//
//  Created by MNN on 2018/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/Session.hpp"
#include <string.h>
#include <MNN/AutoTime.hpp>
#include <map>
#include <set>
#include "MNN_generated.h"
#include "core/AutoStorage.h"
#include "core/RuntimeFactory.hpp"
#include "core/TensorUtils.hpp"
#include "core/WrapExecution.hpp"

using namespace std;

namespace MNN {
Session::Session(Schedule::ScheduleInfo&& info, Interpreter::SessionMode callBackMode,
                 Interpreter::SessionMode inputMode, RuntimeInfo&& runtime) {
    mRuntime = std::move(runtime);
    if (info.pipelineInfo.empty()) {
        mValid = false;
        return;
    }
    mTensors       = std::move(info.allTensors);
    auto defaultBn = std::move(info.defaultBackend);
    for (auto& iter : info.pipelineInfo) {
        auto rt    = mRuntime.first.find(iter.first.type)->second.get();
        auto cpuRuntime = mRuntime.second;
        bool specialUsage = false;
        if (iter.first.user != nullptr) {
            specialUsage = iter.first.user->flags > 0;
        }
        std::shared_ptr<Backend> first(rt->onCreate(iter.first.user));
        std::shared_ptr<Backend> second;
        if (first->type() == MNN_FORWARD_CPU && (!specialUsage)) {
            second = first;
        } else {
            BackendConfig defaultConfig;
            defaultConfig.flags = 4;
            second.reset(cpuRuntime->onCreate(&defaultConfig));
        }
        std::shared_ptr<Pipeline> newPipeline(new Pipeline(std::move(iter.second), first, second, defaultBn, inputMode == Interpreter::Session_Input_Inside, rt->onGetCompilerType()));
        mPipelines.emplace_back(std::move(newPipeline));
    }
    mInputs       = std::move(info.inputTensors);
    mOutputs      = std::move(info.outputTensor);
    mCallBackMode = callBackMode;
}

Session::~Session() {
    for (auto& t : mTensors) {
        TensorUtils::clearHandleData(t.get());
    }
    mPipelines.clear();
    mRuntime.first.clear();
    mTensors.clear();
    mRuntime.second = nullptr;
}

bool Session::loadCache(const void* buffer, size_t size) {
    for (auto iter : mRuntime.first) {
        auto res = iter.second->onSetCache(buffer, size);
        if (res) {
            return true;
        }
    }
    return false;
}

std::pair<const void*, size_t> Session::getCache() {
    for (auto iter : mRuntime.first) {
        auto res = iter.second->onGetCache();
        if (res.first != nullptr) {
            return res;
        }
    }
    return std::make_pair(nullptr, 0);
}
void Session::cloneExecution(const std::map<const Op*, std::shared_ptr<Execution>>& cache, int pipelineIndex) {
    mPipelines[pipelineIndex]->cloneExecution(cache);
}
const std::map<const Op*, std::shared_ptr<Execution>>& Session::getExecution(int pipelineIndex) {
    return mPipelines[pipelineIndex]->getCache();
}

ErrorCode Session::run() const {
    if (mNeedResize) {
        MNN_ERROR("Can't run session because not resized\n");
        return COMPUTE_SIZE_ERROR;
    }
    for (auto& iter : mPipelines) {
        auto error = iter->execute();
        if (NO_ERROR != error) {
            return error;
        }
    }
    return NO_ERROR;
}

ErrorCode Session::runWithCallBack(const TensorCallBackWithInfo& before, const TensorCallBackWithInfo& end,
                                   bool sync) const {
    if (mNeedResize) {
        MNN_ERROR("Can't run session because not resized\n");
        return COMPUTE_SIZE_ERROR;
    }
    for (auto& iter : mPipelines) {
        auto error = iter->executeCallBack(before, end);
        if (NO_ERROR != error) {
            return error;
        }
    }
    return NO_ERROR;
}

void Session::_clearCache() {
    for (auto& t : mTensors) {
        auto describe = TensorUtils::getDescribe(t.get());
        if (describe->usage == Tensor::InsideDescribe::TRAINABLE || describe->usage == Tensor::InsideDescribe::CONSTANT) {
            continue;
        }
        TensorUtils::clearHandleData(t.get());
        describe->useCount = 0;
        describe->backend  = nullptr;
        describe->regions.clear();
    }
}

ErrorCode Session::resize(bool isStatic) {
    if (mNeedResize) {
        if (!isStatic) {
            _clearCache();
        }
        bool debug = mCallBackMode == Interpreter::Session_Debug;
        for (auto& iter : mPipelines) {
            auto error = iter->encode(isStatic, debug);
            if (NO_ERROR != error) {
                return error;
            }
        }
        mNeedResize = false;
        mNeedMalloc = true;
    }
    if (mNeedMalloc) {
        // Set needResize = true for easy for judge in runSession when error
        mNeedResize = true;
        // Turn Pipeline to Command Buffer and Malloc resource
        // TODO: Seperate Schedule and Malloc
        for (auto& iter : mPipelines) {
            auto error = iter->allocMemory();
            if (NO_ERROR != error) {
                return error;
            }
        }
        for (auto& iter : mRuntime.first) {
            iter.second->onGabageCollect(0);
        }
        mNeedMalloc = false;
        mNeedResize = false;
    }
    return NO_ERROR;
}
bool Session::getInfo(Interpreter::SessionInfoCode code, void* ptr) const {
    switch (code) {
        case Interpreter::MEMORY: {
            auto dst     = (float*)ptr;
            float summer = mRuntime.second->onGetMemoryInMB();
            for (auto& r : mRuntime.first) {
                if (r.second.get() != mRuntime.second.get()) {
                    summer += r.second->onGetMemoryInMB();
                }
            }
            *dst = summer;
            return true;
        } break;
        case Interpreter::BACKENDS: {
            int pos = 0;
            auto res = (int32_t*)ptr;
            for (auto& r : mRuntime.first) {
                res[pos++] = r.first;
            }
            return true;
        } break;
        case Interpreter::FLOPS: {
            float flo = 0.0f;
            for (auto& iter : mPipelines) {
                flo += iter->flops();
            }
            auto dst     = (float*)ptr;
            *dst = flo;
            return true;
        } break;
        // TODO: Support other debug info
        default:
            break;
    }
    return false;
}

const Backend* Session::getBackEnd(const Tensor* tensor) const {
    return TensorUtils::getDescribe(tensor)->backend;
}

Tensor* Session::getInput(const char* name) const {
    //MNN_ASSERT(!mInputs.empty());
    if (nullptr == name) {
        return mInputs.begin()->second;
    }
    auto iter = mInputs.find(name);
    if (iter == mInputs.end()) {
        MNN_PRINT("Error: can't find input: %s\n", name);
        return nullptr;
    }
    return iter->second;
}

Tensor* Session::getOutput(const char* name) const {
    MNN_ASSERT(!mOutputs.empty());
    if (nullptr == name) {
        return mOutputs.begin()->second;
    }

    auto iter = mOutputs.find(name);
    if (iter == mOutputs.end()) {
        MNN_PRINT("Error: can't find output: %s\n", name);
        return nullptr;
    }
    return iter->second;
}

const std::map<std::string, Tensor*>& Session::getInputAll() const {
    return mInputs;
}

const std::map<std::string, Tensor*>& Session::getOutputAll() const {
    return mOutputs;
}

ErrorCode Session::updateToModel(Net* net) const {
    if (mNeedResize) {
        return NOT_SUPPORT;
    }
    int opSize = net->oplists()->size();
    for (int i = 0; i < opSize; ++i) {
        auto op = net->oplists()->GetAs<Op>(i);
        if ((net->usage() == Usage_INFERENCE || net->usage() == Usage_INFERENCE_STATIC) && op->type() != OpType_Const) {
            continue;
        }
        if (net->usage() == Usage_TRAIN && op->type() != OpType_TrainableParam) {
            continue;
        }
        if (!op->outputIndexes() || op->outputIndexes()->size() != 1) {
            continue;
        }
        auto index = op->outputIndexes()->data()[0];
        auto blob  = op->main_as_Blob();
        if (blob->dataType() != DataType_DT_FLOAT) {
            continue;
        }
        std::shared_ptr<Tensor> tensor = mTensors[index];
        if (tensor->host<void>() == nullptr && tensor->deviceId() != 0) {
            tensor.reset(Tensor::createHostTensorFromDevice(tensor.get(), true));
            if (tensor.get() == nullptr) {
                MNN_ERROR("failed to copy trained param from device to host\n");
                return INVALID_VALUE;
            }
        }
        ::memcpy((void*)blob->float32s()->data(), tensor->host<float>(), tensor->size());
    }

    return NO_ERROR;
}

} // namespace MNN
