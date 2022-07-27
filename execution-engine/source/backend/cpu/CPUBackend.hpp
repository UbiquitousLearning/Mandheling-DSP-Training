//
//  CPUBackend.hpp
//  MNN
//
//  Created by MNN on 2018/07/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUBackend_hpp
#define CPUBackend_hpp

#include <map>
#include <memory>
#include "core/Backend.hpp"
#include "core/Execution.hpp"
#include "MNN_generated.h"

#include "MNN/hexagon/hexagon_nn/hexagon_nn_ops.h"
#include "MNN/hexagon/hexagon_nn/hexagon_nn.h"
#include "MNN/hexagon/hexagon_nn/HexagonRunningUtils.hpp"

#include <thread>

namespace MNN {
class BufferAllocator;
class CPURuntime : public Runtime {
public:
    friend class CPUBackend;
    CPURuntime(const Backend::Info& info);
    virtual ~ CPURuntime();
    virtual Backend* onCreate(const BackendConfig* config) const override;
    virtual void onGabageCollect(int level) override;
    virtual float onGetMemoryInMB() override;
    virtual CompilerType onGetCompilerType() const override {
        return Compiler_Loop;
    }
private:
    std::shared_ptr<BufferAllocator> mStaticAllocator;
    int mThreadNumber;
    int mTaskIndex;
    BackendConfig::MemoryMode mMemory;
    BackendConfig::PowerMode mPower;
    BackendConfig::PrecisionMode mPrecision;

    // Backend features
    // CPU features
    float mFlops = 0.0f;
    static Backend*(*gExtraCreate)(const Runtime* runtime);
    size_t mFlags = 0;
};
struct CoreFunctions;
struct CoreInt8Functions;

class CPUBackend : public Backend {
public:
    CPUBackend(const CPURuntime* runtime, BackendConfig::PrecisionMode precision, MNNForwardType type = MNN_FORWARD_CPU, size_t flags = 0);
    virtual ~CPUBackend();

    // Return sizeDivide, scheduleNumber aligned memory
    std::pair<int, int> multiThreadDivide(int size) const;
public:
    virtual bool onAcquireBuffer(const Tensor* nativeTensor, StorageType storageType) override;
    virtual bool onReleaseBuffer(const Tensor* nativeTensor, StorageType storageType) override;
    virtual bool onClearBuffer() override;
    virtual void onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const override;
    virtual std::pair<float, bool> onMeasure(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                            const MNN::Op* op) override;

    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op) override;
    virtual void onExecuteBegin() const override;
    virtual void onExecuteEnd() const override;

    const CoreFunctions* functions() const {
        return mCoreFunctions;
    }

    // Return element size for Tensor, conside pack
    int getTensorSize(const Tensor* tensor) const;
    const CoreInt8Functions* int8Functions() const {
        return mInt8CoreFunctions;
    }
    Execution* makePostWrapExectuion(Execution* execution) const;
public:
    class Creator {
    public:
        virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                    const MNN::Op* op, Backend* backend) const = 0;
    };

    static bool addCreator(OpType t, Creator* c);

    int threadNumber() const {
        return mRuntime->mThreadNumber;
    }

    BufferAllocator* getBufferAllocator() const {
        return mDynamicAllocator.get();
    }

    BackendConfig::MemoryMode memoryMode() const {
        return mRuntime->mMemory;
    }
    BackendConfig::PrecisionMode precisionMode() const {
        return mPrecisionMode;
    }
    std::map<const Tensor*, const Tensor*>& getCachedCastTensor() {
        return mCachedCastTensor;
    }
#ifdef MNN_USE_THREAD_POOL
    inline int taskIndex() const {return mRuntime->mTaskIndex;}
#endif
    static void initCreatorMap();
    halide_type_t getRunType(const Op* op, halide_type_t qtype, halide_type_t rtype) override;
private:
    OpType getRealOpType(OpType opType, halide_type_t dataType);
protected:
    bool allocBuffer(int size, Tensor* dest,  StorageType storageType);
    const CoreFunctions* mCoreFunctions;
    const CoreInt8Functions* mInt8CoreFunctions;
private:
    std::shared_ptr<BufferAllocator> mStaticAllocator;
    std::shared_ptr<BufferAllocator> mDynamicAllocator;
    bool mCheckNAN = false;
    const CPURuntime* mRuntime;
    BackendConfig::PrecisionMode mPrecisionMode;
    static std::map<OpType, CPUBackend::Creator*>* gCreator;
    std::map<const Tensor*, const Tensor*> mCachedCastTensor;
public:
    
    std::map<int8_t*, int32_t> gradient_map;
    
    std::map<int8_t*, int32_t> weight_map;
    
    std::map<int8_t*, int32_t> opadr_opid_map;

    
    int global_op_id = 0x1000;
    static const HexagonNN*  global_hexagon_nn_;
    static hexagon_nn_nn_id global_graph_id_;

    static hexagon_nn_nn_id global_gradient_graph_id_;

    static hexagon_nn_nn_id split_gradient_graph_id_;

    static bool build_graph;

    static bool execute_mode;

    static bool parallel_mode;
    static bool split_mode;

    std::vector<hexagon_nn_tensordef> input_tensors;
    std::vector<hexagon_nn_tensordef> output_tensors;

    std::vector<hexagon_nn_tensordef> gradient_input_tensors;
    std::vector<hexagon_nn_tensordef> gradient_output_tensors;

    std::vector<hexagon_nn_input> outputlayer_input;
    std::vector<hexagon_nn_input> gradient_outputlayer_input;

    void DSPExecute() override;
    void DSPParallelWait() override;
    bool isDSPParallel() override;
    void CPU_Split_DSPExecute();
    void CPU_P_DSPExecute();
    int get_Op_id(Tensor* t, MNN::OpType type) override;
    void insert_Op_id(Tensor* t, int op_id) override;

    int total_input_num=0;
    int current_input_num=0;

    std::vector<hexagon_nn_output> _outputs0;
    std::vector<hexagon_nn_output> _gradient_outputs0;

    std::thread threads[1];
};

#define REGISTER_CPU_OP_CREATOR(name, opType)     \
    void ___##name##__##opType##__() {            \
        static name _temp;\
        CPUBackend::addCreator(opType, &_temp); \
    }

} // namespace MNN

#endif /* CPUBackend_hpp */
