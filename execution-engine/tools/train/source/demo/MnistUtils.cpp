//
//  MnistUtils.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MnistUtils.hpp"
#include <MNN/expr/Executor.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include "DataLoader.hpp"
#include "DemoUnit.hpp"
#include "MnistDataset.hpp"
#include "NN.hpp"
#include "SGD.hpp"
#include "NITI_SGD.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "ADAM.hpp"
#include "LearningRateScheduler.hpp"
#include "Loss.hpp"
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include "OpGrad.hpp"

#include <MNN/hexagon/hexagon_nn/HexagonRunningUtils.hpp>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

void MnistUtils::train(std::shared_ptr<Module> model, std::string root) {
    // {
    //     // Load snapshot
    //     auto para = Variable::load("mnist.snapshot.mnn");
    //     model->loadParameters(para);
    // }
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);
    std::shared_ptr<SGD> sgd(new NITI_SGD(model));
    sgd->setMomentum(0.9f);
    // sgd->setMomentum2(0.99f);
    sgd->setWeightDecay(0.0005f);

    auto dataset = MnistDataset::create(root, MnistDataset::Mode::TRAIN);
    // the stack transform, stack [1, 28, 28] to [n, 1, 28, 28]
    const size_t batchSize  = 64;
    const size_t numWorkers = 0;
    bool shuffle            = true;

    auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));

    size_t iterations = dataLoader->iterNumber();

    auto testDataset            = MnistDataset::create(root, MnistDataset::Mode::TEST);
    const size_t testBatchSize  = 20;
    const size_t testNumWorkers = 0;
    shuffle                     = false;

    auto testDataLoader = std::shared_ptr<DataLoader>(testDataset.createLoader(testBatchSize, true, shuffle, testNumWorkers));

    size_t testIterations = testDataLoader->iterNumber();

    for (int epoch = 0; epoch < 50; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            AUTOTIME;
            dataLoader->reset();
            model->setIsTraining(true);
            Timer _100Time;
            int lastIndex = 0;
            int moveBatchSize = 0;
            for (int i = 0; i < iterations; i++) {
                // AUTOTIME;
                auto trainData  = dataLoader->next();
                auto example    = trainData[0];
                auto cast       = _Cast<float>(example.first[0]);

                auto mean = _ReduceMean(cast);
                auto std =  _Sqrt(_ReduceSum((cast - mean)*(cast - mean) ) / _Scalar<float>(batchSize *28 *28 *1.0f)  );
                auto Y = (cast - mean)/std;

                auto range = _ReduceMax(_Abs(Y));
                auto bitwidth = _Ceil(_Log(range));
                auto ascale = _Cast<int8_t>(bitwidth - _Scalar<float>(7.0));
                example.first[0] = _Round(Y / range * _Const(127.0f));
                example.first[0] = _Cast<int8_t>(example.first[0]);
                moveBatchSize += example.first[0]->getInfo()->dim[0];

                auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]), _Scalar<int>(10), _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));

                auto predict = model->multioutput_forward({example.first[0], ascale});
                auto loss    = _NITI_LOSS_CrossEntropy(predict[0], predict[1], newTarget);
#ifdef DEBUG_GRAD
                {
                    static bool init = false;
                    if (!init) {
                        init = true;
                        std::set<VARP> para;
                        example.first[0].fix(VARP::INPUT);
                        newTarget.fix(VARP::CONSTANT);
                        auto total = model->parameters();
                        for (auto p :total) {
                            para.insert(p);
                        }
                        auto grad = OpGrad::grad(loss, para);
                        total.clear();
                        for (auto iter : grad) {
                            total.emplace_back(iter.second);
                        }
                        Variable::save(total, ".temp.grad");
                    }
                }
#endif
                float rate   = LrScheduler::inv(0.01, epoch * iterations + i, 0.0001, 0.75);
                sgd->setLearningRate(rate);
                if (moveBatchSize % (1 * batchSize) == 0 || i == iterations - 1) {
                    std::cout << "epoch: " << (epoch);
                    std::cout << "  " << moveBatchSize << " / " << dataLoader->size();
                    FILE *f = fopen("DSP.txt","w");
                    fprintf(f, "%d\n",2);
                    fclose(f);
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    std::cout << " lr: " << rate;
                    if(moveBatchSize % (1 * batchSize) == 0 || i == iterations - 1) {
                        std::cout << " time: " << (float)_100Time.durationInUs() / 1000.0f << " ms / " << (i - lastIndex) <<  " iter"  << std::endl;
                        _100Time.reset();
                    }
                    
                    std::cout.flush();
                    
                    lastIndex = i;
                }
                // _100Time.reset();
                FILE *f = fopen("DSP.txt","w");
                fprintf(f, "%d\n",2);
                fclose(f);
                sgd->step(loss);
                // MNN_PRINT("one iter time: %f ms \n ", (float)_100Time.durationInUs() / 1000.0f);
            }
        }
        
        int correct = 0;
        testDataLoader->reset();
        model->setIsTraining(true);
        int moveBatchSize = 0;
        for (int i = 0; i < testIterations; i++) {
            auto data       = testDataLoader->next();
            auto example    = data[0];
            moveBatchSize += example.first[0]->getInfo()->dim[0];
            if ((i + 1) % 100 == 0) {
                std::cout << "test: " << moveBatchSize << " / " << testDataLoader->size() << std::endl;
            }
           auto cast       = _Cast<float>(example.first[0]);

            auto mean = _ReduceMean(cast);
            auto std =  _Sqrt(_ReduceSum((cast - mean)*(cast - mean) ) / _Scalar<float>(testBatchSize * 28 *28 * 1.0f)  );
            auto Y = (cast - mean)/std;

            // niti float->int8
            auto range = _ReduceMax(_Abs(Y));
            auto bitwidth = _Ceil(_Log(range));
            auto ascale = _Cast<int8_t>(bitwidth - _Scalar<float>(7.0));
            example.first[0] = _Round(Y / range * _Const(127.0f));
            example.first[0] = _Cast<int8_t>(example.first[0]);
            auto predict    = model->multioutput_forward({example.first[0], ascale});
            predict[0]      = _Convert(predict[0], NCHW);
            predict[0]      = _Reshape(predict[0], {0, -1});
            predict[0]         = _ArgMax( _Cast<float>(predict[0]), 1);
            auto accu       = _Cast<int32_t>(_Equal(predict[0], _Cast<int32_t>(example.second[0]))).sum({});
            correct += accu->readMap<int32_t>()[0];
        }
        auto accu = (float)correct / (float)testDataLoader->size();
        std::cout << "epoch: " << epoch << "lr  accuracy: " << accu << std::endl;
        exe->dumpProfile();
    }
}

void MnistUtils::dsp_train(std::shared_ptr<Module> model, std::string root) {
    // {
    //     // Load snapshot
    //     auto para = Variable::load("mnist.snapshot.mnn");
    //     model->loadParameters(para);
    // }
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);
    std::shared_ptr<SGD> sgd(new NITI_SGD(model));
    sgd->setMomentum(0.9f);
    // sgd->setMomentum2(0.99f);
    sgd->setWeightDecay(0.0005f);

    auto dataset = MnistDataset::create(root, MnistDataset::Mode::TRAIN);
    // the stack transform, stack [1, 28, 28] to [n, 1, 28, 28]
    const size_t batchSize  = 64;
    const size_t numWorkers = 0;
    bool shuffle            = true;

    auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));

    size_t iterations = dataLoader->iterNumber();

    auto testDataset            = MnistDataset::create(root, MnistDataset::Mode::TEST);
    const size_t testBatchSize  = 20;
    const size_t testNumWorkers = 0;
    shuffle                     = false;

    auto testDataLoader = std::shared_ptr<DataLoader>(testDataset.createLoader(testBatchSize, true, shuffle, testNumWorkers));

    size_t testIterations = testDataLoader->iterNumber();

    for (int epoch = 0; epoch < 50; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            AUTOTIME;
            dataLoader->reset();
            model->setIsTraining(true);
            Timer _100Time;
            int lastIndex = 0;
            int moveBatchSize = 0;
            for (int i = 0; i < iterations; i++) {
                // AUTOTIME;
                auto trainData  = dataLoader->next();
                auto example    = trainData[0];
                auto cast       = _Cast<float>(example.first[0]);

                auto mean = _ReduceMean(cast);
                auto std =  _Sqrt(_ReduceSum((cast - mean)*(cast - mean) ) / _Scalar<float>(batchSize *28 *28 *1.0f)  );
                auto Y = (cast - mean)/std;

                auto range = _ReduceMax(_Abs(Y));
                auto bitwidth = _Ceil(_Log(range));
                auto ascale = _Cast<int8_t>(bitwidth - _Scalar<float>(7.0));
                example.first[0] = _Round(Y / range * _Const(127.0f));
                example.first[0] = _Cast<int8_t>(example.first[0]);
                moveBatchSize += example.first[0]->getInfo()->dim[0];

                auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]), _Scalar<int>(10), _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));

                auto predict = model->multioutput_forward({example.first[0], ascale});
                auto loss    = _NITI_LOSS_CrossEntropy(predict[0], predict[1], newTarget);
#ifdef DEBUG_GRAD
                {
                    static bool init = false;
                    if (!init) {
                        init = true;
                        std::set<VARP> para;
                        example.first[0].fix(VARP::INPUT);
                        newTarget.fix(VARP::CONSTANT);
                        auto total = model->parameters();
                        for (auto p :total) {
                            para.insert(p);
                        }
                        auto grad = OpGrad::grad(loss, para);
                        total.clear();
                        for (auto iter : grad) {
                            total.emplace_back(iter.second);
                        }
                        Variable::save(total, ".temp.grad");
                    }
                }
#endif
                float rate   = LrScheduler::inv(0.01, epoch * iterations + i, 0.0001, 0.75);
                sgd->setLearningRate(rate);
                if (moveBatchSize % (1 * batchSize) == 0 || i == iterations - 1) {
                    std::cout << "epoch: " << (epoch);
                    std::cout << "  " << moveBatchSize << " / " << dataLoader->size();
                    FILE *f = fopen("DSP.txt","w");
                    fprintf(f, "%d\n",0);
                    fclose(f);
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    std::cout << " lr: " << rate;
                    if(moveBatchSize % (1 * batchSize) == 0 || i == iterations - 1) {
                        std::cout << " time: " << (float)_100Time.durationInUs() / 1000.0f << " ms / " << (i - lastIndex) <<  " iter"  << std::endl;
                        _100Time.reset();
                    }
                    
                    std::cout.flush();
                    
                    lastIndex = i;
                }
                // _100Time.reset();
                FILE *f = fopen("DSP.txt","w");
                fprintf(f, "%d\n",0);
                fclose(f);
                sgd->step(loss);
                // MNN_PRINT("one iter time: %f ms \n ", (float)_100Time.durationInUs() / 1000.0f);
            }
        }
        
        int correct = 0;
        testDataLoader->reset();
        model->setIsTraining(true);
        int moveBatchSize = 0;
        for (int i = 0; i < testIterations; i++) {
            auto data       = testDataLoader->next();
            auto example    = data[0];
            moveBatchSize += example.first[0]->getInfo()->dim[0];
            if ((i + 1) % 100 == 0) {
                std::cout << "test: " << moveBatchSize << " / " << testDataLoader->size() << std::endl;
            }
           auto cast       = _Cast<float>(example.first[0]);

            auto mean = _ReduceMean(cast);
            auto std =  _Sqrt(_ReduceSum((cast - mean)*(cast - mean) ) / _Scalar<float>(testBatchSize * 28 *28 * 1.0f)  );
            auto Y = (cast - mean)/std;

            // niti float->int8
            auto range = _ReduceMax(_Abs(Y));
            auto bitwidth = _Ceil(_Log(range));
            auto ascale = _Cast<int8_t>(bitwidth - _Scalar<float>(7.0));
            example.first[0] = _Round(Y / range * _Const(127.0f));
            example.first[0] = _Cast<int8_t>(example.first[0]);
            auto predict    = model->multioutput_forward({example.first[0], ascale});
            predict[0]      = _Convert(predict[0], NCHW);
            predict[0]      = _Reshape(predict[0], {0, -1});
            predict[0]         = _ArgMax( _Cast<float>(predict[0]), 1);
            auto accu       = _Cast<int32_t>(_Equal(predict[0], _Cast<int32_t>(example.second[0]))).sum({});
            correct += accu->readMap<int32_t>()[0];
        }
        auto accu = (float)correct / (float)testDataLoader->size();
        std::cout << "epoch: " << epoch << "lr  accuracy: " << accu << std::endl;
        exe->dumpProfile();
    }
}


void MnistUtils::float_train(std::shared_ptr<Module> model, std::string root) {
    // {
    //     // Load snapshot
    //     auto para = Variable::load("mnist.snapshot.mnn");
    //     model->loadParameters(para);
    // }
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);
    std::shared_ptr<SGD> sgd(new SGD(model));
    sgd->setMomentum(0.9f);
    // sgd->setMomentum2(0.99f);
    sgd->setWeightDecay(0.0005f);

    auto dataset = MnistDataset::create(root, MnistDataset::Mode::TRAIN);
    // the stack transform, stack [1, 28, 28] to [n, 1, 28, 28]
    const size_t batchSize  = 64;
    const size_t numWorkers = 0;
    bool shuffle            = true;

    auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));

    size_t iterations = dataLoader->iterNumber();

    auto testDataset            = MnistDataset::create(root, MnistDataset::Mode::TEST);
    const size_t testBatchSize  = 20;
    const size_t testNumWorkers = 0;
    shuffle                     = false;

    auto testDataLoader = std::shared_ptr<DataLoader>(testDataset.createLoader(testBatchSize, true, shuffle, testNumWorkers));

    size_t testIterations = testDataLoader->iterNumber();

    for (int epoch = 0; epoch < 50; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            AUTOTIME;
            dataLoader->reset();
            model->setIsTraining(true);
            Timer _100Time;
            int lastIndex = 0;
            int moveBatchSize = 0;
            for (int i = 0; i < iterations; i++) {
                // AUTOTIME;
                auto trainData  = dataLoader->next();
                auto example    = trainData[0];
                auto cast       = _Cast<float>(example.first[0]);
                example.first[0] = cast * _Const(1.0f / 255.0f);
                moveBatchSize += example.first[0]->getInfo()->dim[0];

                // Compute One-Hot
                auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]), _Scalar<int>(10), _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));

                auto predict = model->forward(example.first[0]);
                // auto softmax      = _NITI_Softmax(predict, _Scalar<float>(0.0f), 1);
                auto loss    = _CrossEntropy(predict, newTarget);
                // auto loss = _CrossEntropy(predict, newTarget);
#ifdef DEBUG_GRAD
                {
                    static bool init = false;
                    if (!init) {
                        init = true;
                        std::set<VARP> para;
                        example.first[0].fix(VARP::INPUT);
                        newTarget.fix(VARP::CONSTANT);
                        auto total = model->parameters();
                        for (auto p :total) {
                            para.insert(p);
                        }
                        auto grad = OpGrad::grad(loss, para);
                        total.clear();
                        for (auto iter : grad) {
                            total.emplace_back(iter.second);
                        }
                        Variable::save(total, ".temp.grad");
                    }
                }
#endif
                float rate   = LrScheduler::inv(0.01, epoch * iterations + i, 0.0001, 0.75);
                sgd->setLearningRate(rate);
                if (moveBatchSize % (1 * batchSize) == 0 || i == iterations - 1) {
                    std::cout << "epoch: " << (epoch);
                    std::cout << "  " << moveBatchSize << " / " << dataLoader->size();
                    FILE *f = fopen("DSP.txt","w");
                    fprintf(f, "%d\n",2);
                    fclose(f);
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    std::cout << " lr: " << rate;
                    if(moveBatchSize % (1 * batchSize) == 0 || i == iterations - 1) {
                        std::cout << " time: " << (float)_100Time.durationInUs() / 1000.0f << " ms / " << (i - lastIndex) <<  " iter"  << std::endl;
                        _100Time.reset();
                    }
                    
                    std::cout.flush();
                    
                    lastIndex = i;
                }
                // _100Time.reset();
                FILE *f = fopen("DSP.txt","w");
                fprintf(f, "%d\n",2);
                fclose(f);
                sgd->step(loss);
                // MNN_PRINT("one iter time: %f ms \n ", (float)_100Time.durationInUs() / 1000.0f);
            }
        }

        int correct = 0;
        testDataLoader->reset();
        model->setIsTraining(true);
        int moveBatchSize = 0;
        for (int i = 0; i < testIterations; i++) {
            auto data       = testDataLoader->next();
            auto example    = data[0];
            moveBatchSize += example.first[0]->getInfo()->dim[0];
            if ((i + 1) % 100 == 0) {
                std::cout << "test: " << moveBatchSize << " / " << testDataLoader->size() << std::endl;
            }
           auto cast       = _Cast<float>(example.first[0]);

            auto predict    = model->forward(example.first[0]);
            predict         = _ArgMax( _Cast<float>(predict), 1);
            auto accu       = _Cast<int32_t>(_Equal(predict, _Cast<int32_t>(example.second[0]))).sum({});
            correct += accu->readMap<int32_t>()[0];
        }
        auto accu = (float)correct / (float)testDataLoader->size();
        std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;
        exe->dumpProfile();
    }
}
