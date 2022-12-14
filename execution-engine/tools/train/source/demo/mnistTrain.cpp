//
//  mnistTrain.cpp
//  MNN
//
//  Created by MNN on 2019/11/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/Executor.hpp>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include "DemoUnit.hpp"
#include "Lenet.hpp"
#include "MnistUtils.hpp"
#include "NN.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "module/PipelineModule.hpp"
#include "RandomGenerator.hpp"
#include "Transformer.hpp"

using namespace MNN::Train;
using namespace MNN::Express;
using namespace MNN::Train::Model;

class MnistV2 : public Module {
public:
    MnistV2() {
        NN::ConvOption convOption;
        convOption.kernelSize = {5, 5};
        convOption.channel    = {1, 20};
        convOption.depthwise  = false;
        conv1.reset(NN::Conv(convOption));
        // bn.reset(NN::BatchNorm(20));
        convOption.reset();
        convOption.kernelSize = {5, 5};
        convOption.channel    = {20, 50};
        convOption.depthwise  = false;
        conv2.reset(NN::Conv(convOption));
        convOption.reset();
        convOption.channel    = {800, 500};
        convOption.kernelSize = {1, 1};
        convOption.fusedActivationFunction = NN::Relu6;
        ip1.reset(NN::Conv(convOption));
        convOption.channel    = {500, 10};
        convOption.kernelSize = {1, 1};
        convOption.fusedActivationFunction = NN::None;
        ip2.reset(NN::Conv(convOption));
        registerModel({conv1, conv2, ip1, ip2});
    }

    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override {
        VARP x = inputs[0];
        x      = conv1->forward(x);
        // x      = bn->forward(x);
        x      = _MaxPool(x, {2, 2}, {2, 2});
        x      = conv2->forward(x);
        x      = _MaxPool(x, {2, 2}, {2, 2});
        x      = _Reshape(x, {0, -1, 1, 1});
        //auto info = x->getInfo();
        x      = ip1->forward(x);
        x      = ip2->forward(x);
        x      = _Convert(x, NCHW);
        x      = _Reshape(x, {0, 1, -1});
        x      = _Softmax(x, 2);
        x      = _Reshape(x, {0, -1});
        return {x};
    }
    std::shared_ptr<Module> conv1;
    std::shared_ptr<Module> bn;
    std::shared_ptr<Module> conv2;
    std::shared_ptr<Module> ip1;
    std::shared_ptr<Module> ip2;
};
class MnistInt8 : public Module {
public:
    MnistInt8(int bits) {
        AUTOTIME;
        NN::ConvOption convOption;
        convOption.kernelSize = {5, 5};
        convOption.channel    = {1, 20};
        conv1.reset(NN::ConvInt8(convOption, bits));
        conv1->setName("conv1");
        convOption.reset();
        convOption.kernelSize = {5, 5};
        convOption.channel    = {20, 50};
        convOption.depthwise  = true;
        conv2.reset(NN::ConvInt8(convOption, bits));
        conv2->setName("conv2");
        convOption.reset();
        convOption.kernelSize = {1, 1};
        convOption.channel    = {800, 500};
        convOption.fusedActivationFunction = NN::Relu6;
        ip1.reset(NN::ConvInt8(convOption, bits));
        ip1->setName("ip1");
        convOption.kernelSize = {1, 1};
        convOption.channel    = {500, 10};
        convOption.fusedActivationFunction = NN::None;
        ip2.reset(NN::ConvInt8(convOption, bits));
        ip2->setName("ip2");
        dropout.reset(NN::Dropout(0.5));
        registerModel({conv1, conv2, ip1, ip2, dropout});
    }

    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override {
        VARP x = inputs[0];
        x      = conv1->forward(x);
        x      = _MaxPool(x, {2, 2}, {2, 2});
        x      = conv2->forward(x);
        x      = _MaxPool(x, {2, 2}, {2, 2});
        x      = _Reshape(x, {0, -1, 1, 1});
        x      = ip1->forward(x);
        x      = dropout->forward(x);
        x      = ip2->forward(x);
        x      = _Convert(x, NCHW);
        x      = _Reshape(x, {0, -1});
        x      = _Softmax(x, 1);
        return {x};
    }
    std::shared_ptr<Module> conv1;
    std::shared_ptr<Module> conv2;
    std::shared_ptr<Module> ip1;
    std::shared_ptr<Module> ip2;
    std::shared_ptr<Module> dropout;
};

class NITIInt8 : public Module {
public:
    NITIInt8(int bits) {
        AUTOTIME;
        NN::ConvOption convOption;
        convOption.kernelSize = {5, 5};
        convOption.channel    = {1, 20};
        conv1.reset(NN::NITI_Conv_Int8(convOption, bits, true));
        conv1->setName("conv1");
        convOption.reset();
        convOption.kernelSize = {5, 5};
        convOption.channel    = {20, 52};
        // convOption.depthwise  = true;
        conv2.reset(NN::NITI_Conv_Int8(convOption, bits));
        conv2->setName("conv2");
        convOption.reset();
        convOption.kernelSize = {1, 1};
        convOption.channel    = {832, 500};
        // convOption.fusedActivationFunction = NN::Relu6;
        ip1.reset(NN::NITI_Conv_Int8(convOption, bits, true));
        ip1->setName("ip1");
        convOption.kernelSize = {1, 1};
        convOption.channel    = {500, 12};
        // convOption.fusedActivationFunction = NN::None;
        ip2.reset(NN::NITI_Conv_Int8(convOption, bits, true));
        ip2->setName("ip2");
        // dropout.reset(NN::Dropout(0.5));
        registerModel({conv1, conv2, ip1, ip2});
    }

    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override {
        std::vector<VARP> x = inputs;
        // VARP ascale = _Scalar<int8_t>(1);
        // x.emplace_back(ascale);

        x         = conv1->multioutput_forward(x);
        x[0]      = _NITI_Relu_Int8(x[0]);
        x         = _NITI_Maxpool_Int8(x[0], x[1], {2, 2}, {2, 2});
        x         = conv2->multioutput_forward({x[0], x[1]});
        x[0]      = _NITI_Relu_Int8(x[0]);
        x         = _NITI_Maxpool_Int8(x[0], x[1], {2, 2}, {2, 2});
        x[0]      = _Convert(x[0], NCHW);
        x[0]      = _Reshape(x[0], {0, -1, 1, 1});
        x         = ip1->multioutput_forward(x);
        x[0]      = _NITI_Relu_Int8(x[0]);
        // x      = dropout->forward(x);
        x         = ip2->multioutput_forward(x);
        // x[0]      = _NITI_Relu_Int8(x[0]);
        x[0]      = _Convert(x[0], NCHW);
        x[0]      = _Reshape(x[0], {0, -1});
        // auto softmax      = _NITI_Softmax(x[0], x[1], 1);
        return x;
    }
    std::shared_ptr<Module> conv1;
    std::shared_ptr<Module> conv2;
    std::shared_ptr<Module> ip1;
    std::shared_ptr<Module> ip2;
    // std::shared_ptr<Module> dropout;
};

class NITIDSPInt8 : public Module {
public:
    NITIDSPInt8(int bits) {
        AUTOTIME;
        NN::ConvOption convOption;
        convOption.kernelSize = {5, 5};
        convOption.channel    = {1, 20};
        conv1.reset(NN::NITI_DSP_Conv_Int8(convOption, bits, true));
        conv1->setName("conv1");
        convOption.reset();
        convOption.kernelSize = {5, 5};
        convOption.channel    = {20, 52};
        // convOption.depthwise  = true;
        conv2.reset(NN::NITI_DSP_Conv_Int8(convOption, bits));
        conv2->setName("conv2");
        convOption.reset();
        convOption.kernelSize = {1, 1};
        convOption.channel    = {832, 500};
        // convOption.fusedActivationFunction = NN::Relu6;
        ip1.reset(NN::NITI_DSP_Conv_Int8(convOption, bits, true));
        ip1->setName("ip1");
        convOption.kernelSize = {1, 1};
        convOption.channel    = {500, 12};
        // convOption.fusedActivationFunction = NN::None;
        ip2.reset(NN::NITI_DSP_Conv_Int8(convOption, bits, true));
        ip2->setName("ip2");
        // dropout.reset(NN::Dropout(0.5));
        registerModel({conv1, conv2, ip1, ip2});
    }

    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override {
        std::vector<VARP> x = inputs;
        // VARP ascale = _Scalar<int8_t>(1);
        // x.emplace_back(ascale);

        x         = conv1->multioutput_forward(x);
        x[0]      = _NITI_DSP_Relu_Int8(x[0]);
        x         = _NITI_DSP_Maxpool_Int8(x[0], x[1], {2, 2}, {2, 2});
        x         = conv2->multioutput_forward(x);
        x[0]      = _NITI_DSP_Relu_Int8(x[0]);
        x         = _NITI_DSP_Maxpool_Int8(x[0], x[1], {2, 2}, {2, 2});
        x[0]      = _NITI_DSP_Reshape(x[0], {0, 1, 1, 832});
        x         = ip1->multioutput_forward(x);
        x[0]      = _NITI_DSP_Relu_Int8(x[0]);
        // // // x      = dropout->forward(x);
        x         = ip2->multioutput_forward(x);
        // x[0]      = _NITI_Relu_Int8(x[0]);
        // x[0]      = _NITI_DSP_Reshape(x[0], {0, 1, 1, 12});
        // auto softmax      = _NITI_Softmax(x[0], x[1], 1);
        return x;
    }
    std::shared_ptr<Module> conv1;
    std::shared_ptr<Module> conv2;
    std::shared_ptr<Module> ip1;
    std::shared_ptr<Module> ip2;
    // std::shared_ptr<Module> dropout;
};

static void train(std::shared_ptr<Module> model, std::string root) {
    MnistUtils::train(model, root);
}

static void dsp_train(std::shared_ptr<Module> model, std::string root) {
    MnistUtils::dsp_train(model, root);
}


static void float_train(std::shared_ptr<Module> model, std::string root) {
    MnistUtils::float_train(model, root);
}
class MnistInt8Train : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 2) {
            std::cout << "usage: ./runTrainDemo.out MnistInt8Train /path/to/unzipped/mnist/data/ quantbits"
                      << std::endl;
            return 0;
        }
        // global random number generator, should invoke before construct the model and dataset
        RandomGenerator::generator(17);

        std::string root = argv[1];
        int bits         = 8;
        if (argc >= 3) {
            std::istringstream is(argv[2]);
            is >> bits;
        }
        if (1 > bits || bits > 8) {
            MNN_ERROR("bits must be 2-8, use 8 default\n");
            bits = 8;
        }
        std::shared_ptr<Module> model(new MnistInt8(bits));
        float_train(model, root);
        return 0;
    }
};

class NITIInt8Train : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 2) {
            std::cout << "usage: ./runTrainDemo.out NITIInt8Train /path/to/unzipped/mnist/data/ quantbits"
                      << std::endl;
            return 0;
        }
        // global random number generator, should invoke before construct the model and dataset
        RandomGenerator::generator(17);

        std::string root = argv[1];
        int bits         = 8;
        if (argc >= 3) {
            std::istringstream is(argv[2]);
            is >> bits;
        }
        if (1 > bits || bits > 8) {
            MNN_ERROR("bits must be 2-8, use 8 default\n");
            bits = 8;
        }
        std::shared_ptr<Module> model(new NITIInt8(bits));
        train(model, root);
        return 0;
    }
};

class NITIDSPInt8Train : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 2) {
            std::cout << "usage: ./runTrainDemo.out NITIInt8Train /path/to/unzipped/mnist/data/ quantbits"
                      << std::endl;
            return 0;
        }
        // global random number generator, should invoke before construct the model and dataset
        RandomGenerator::generator(17);

        std::string root = argv[1];
        int bits         = 8;
        if (argc >= 3) {
            std::istringstream is(argv[2]);
            is >> bits;
        }
        if (1 > bits || bits > 8) {
            MNN_ERROR("bits must be 2-8, use 8 default\n");
            bits = 8;
        }
        std::shared_ptr<Module> model(new NITIDSPInt8(bits));
        dsp_train(model, root);
        return 0;
    }
};

class MnistTrain : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 2) {
            std::cout << "usage: ./runTrainDemo.out MnistTrain /path/to/unzipped/mnist/data/  [depthwise]" << std::endl;
            return 0;
        }
        // global random number generator, should invoke before construct the model and dataset
        RandomGenerator::generator(17);

        std::string root = argv[1];
        std::shared_ptr<Module> model(new Lenet);
        if (argc >= 3) {
            model.reset(new MnistV2);
        }
        float_train(model, root);
        return 0;
    }
};

class MnistTrainSnapshot : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc < 2) {
            std::cout << "usage: ./runTrainDemo.out MnistTrainSnapshot /path/to/unzipped/mnist/data/  [depthwise]" << std::endl;
            return 0;
        }
        // global random number generator, should invoke before construct the model and dataset
        RandomGenerator::generator(17);

        std::string root = argv[1];
        std::shared_ptr<Module> model(new Lenet);
        if (argc >= 3) {
            model.reset(new MnistV2);
        }
        auto snapshot = Variable::load("mnist.snapshot.mnn");
        model->loadParameters(snapshot);
        train(model, root);
        return 0;
    }
};
DemoUnitSetRegister(MnistTrain, "MnistTrain");
DemoUnitSetRegister(MnistTrainSnapshot, "MnistTrainSnapshot");
DemoUnitSetRegister(MnistInt8Train, "MnistInt8Train");
DemoUnitSetRegister(NITIInt8Train, "NITIInt8Train");
DemoUnitSetRegister(NITIDSPInt8Train, "NITIDSPInt8Train");