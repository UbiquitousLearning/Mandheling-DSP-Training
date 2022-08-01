//
//  ShapeNITI_Conv_Int8.cpp
//  MNN
//
//  Created by xudaliang on 2021/8/15.
//  
//

#include <math.h>
#include "shape/SizeComputer.hpp"
#include "core/TensorUtils.hpp"
namespace MNN {

class NITI_ConvolutionSizeComputer : public SizeComputer {
public:
    static const Convolution2DCommon* loadCommon(const Op* op) {
        const Convolution2DCommon* layer = nullptr;
        if (op->main_type() == OpParameter_NITI_CONV_Int8) {
            layer = op->main_as_NITI_CONV_Int8()->common();
        // } else {
        //     MNN_ASSERT(op->main_type() == OpParameter_TfQuantizedConv2D);
        //     layer = op->main_as_TfQuantizedConv2D()->common();
        }
        return layer;
    }
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() >= 1);
        // MNN_ASSERT(2 == outputs.size());
        const Convolution2DCommon* layer = loadCommon(op);
        int kX = layer->kernelX();
        int kY = layer->kernelY();
        auto outputCount = layer->outputCount();
        if (inputs.size() > 1 && outputCount == 0) {
            // From TF's multi input convolution
            outputCount = inputs[1]->length(0);
            kX = inputs[1]->length(3);
            kY = inputs[1]->length(2);
        }
        int kernel_width  = layer->dilateX() * (kX - 1) + 1;
        int kernel_height = layer->dilateY() * (kY - 1) + 1;

        int output_width  = 1;
        int output_height = 1;

        auto input = inputs[0];
        if (input->dimensions() <= 1) {
            // Convolution is not valid for dimension <= 1
            return false;
        }
        // For Tensorflow Group Convolution, the inputCount is the size of filter's input count
        if (layer->inputCount() > 0 && input->channel() % layer->inputCount() != 0 && OpType_Convolution == op->type()) {
            MNN_ERROR("Error for compute convolution shape, need channel = %d, input channel = %d\n", layer->inputCount(), input->channel());
            return false;
        }

        if (layer->padMode() == PadMode_SAME) {
            // Tensorflow padding mode SAME
            output_width  = ceil((float)input->width() / (float)layer->strideX());
            output_height = ceil((float)input->height() / (float)layer->strideY());
        } else if (layer->padMode() == PadMode_VALID) {
            // Tensorflow padding mode VALID
            output_width  = ceil((float)(input->width() - kernel_width + 1) / (float)layer->strideX());
            output_height = ceil((float)(input->height() - kernel_height + 1) / (float)layer->strideY());
        } else {
            // Pad_Caffe means User setted padding
            if (nullptr != layer->pads()) {
                MNN_ASSERT(layer->pads()->size() >= 4);
                int input_width  = input->width() + layer->pads()->data()[1] + layer->pads()->data()[3];
                int input_height = input->height() + layer->pads()->data()[0] + layer->pads()->data()[2];
                output_width     = (input_width - kernel_width) / layer->strideX() + 1;
                output_height    = (input_height - kernel_height) / layer->strideY() + 1;
            } else {
                int input_width  = input->width() + layer->padX() * 2;
                int input_height = input->height() + layer->padY() * 2;
                output_width     = (input_width - kernel_width) / layer->strideX() + 1;
                output_height    = (input_height - kernel_height) / layer->strideY() + 1;
            }
        }

        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.dimensions    = input->buffer().dimensions;
        auto format = TensorUtils::getDescribe(input)->dimensionFormat;
        outputBuffer.type = input->getType();
        outputBuffer.dim[0].extent = input->buffer().dim[0].extent;
        if (MNN_DATA_FORMAT_NHWC == format) {
            outputBuffer.dim[3].extent = outputCount;
            outputBuffer.dim[1].extent = output_height;
            outputBuffer.dim[2].extent = output_width;
        } else {
            outputBuffer.dim[1].extent = outputCount;
            outputBuffer.dim[2].extent = output_height;
            outputBuffer.dim[3].extent = output_width;
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        outputs[1]->buffer().dimensions = inputs[2]->buffer().dimensions;
        outputs[1]->buffer().type = inputs[2]->getType();
        outputs[1]->buffer().dim[0].extent = 1;
        outputs[1]->buffer().dim[1].extent = 1;
        outputs[1]->buffer().dim[2].extent = 1;
        outputs[1]->buffer().dim[3].extent = 1;
        TensorUtils::getDescribe(outputs[1])->dimensionFormat = TensorUtils::getDescribe(inputs[2])->dimensionFormat;

        return true;
    }

    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        const Convolution2DCommon* layer = loadCommon(op);
        auto kw    = layer->kernelX();
        auto kh    = layer->kernelY();
        auto group = layer->group();
        auto ic    = inputs[0]->channel();
        auto oc    = outputs[0]->channel();
        auto oSize = outputs[0]->width() * outputs[0]->height() * outputs[0]->batch();
        if (op->type() == OpType_QuantizedDepthwiseConv2D) {
            group = ic;
        }
        if (layer->inputCount() != ic && layer->inputCount() > 0) {
            group = ic / layer->inputCount();
        }
        auto flops = (float)oSize * kw * kh * (ic * oc / group) / FLOPS_M;
        return flops;
    }
};

class NITI_Conv2DBackpropFilter_Int8_SizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto common = op->main_as_NITI_CONV_Int8()->common();
        auto kernel = outputs[0];
        kernel->buffer().dimensions = 4;
        
        kernel->buffer().type = halide_type_of<int8_t>();
        TensorUtils::getDescribe(kernel)->dimensionFormat = MNN_DATA_FORMAT_NCHW;
        kernel->setLength(0, inputs[1]->channel());
        kernel->setLength(1, inputs[0]->channel() / common->group());
        kernel->setLength(2, common->kernelY());
        kernel->setLength(3, common->kernelX());
        return true;
    }
};

class NITI_DSPGradientCONVSizeComputer : public SizeComputer {
public:
    static const Convolution2DCommon* loadCommon(const Op* op) {
        const Convolution2DCommon* layer = nullptr;
        if (op->main_type() == OpParameter_NITI_CONV_Int8) {
            layer = op->main_as_NITI_CONV_Int8()->common();
        // } else {
        //     MNN_ASSERT(op->main_type() == OpParameter_TfQuantizedConv2D);
        //     layer = op->main_as_TfQuantizedConv2D()->common();
        }
        return layer;
    }
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() >= 1);
        // MNN_ASSERT(2 == outputs.size());
        const Convolution2DCommon* layer = loadCommon(op);
        int kX = layer->kernelX();
        int kY = layer->kernelY();
        auto outputCount = layer->outputCount();
        if (inputs.size() > 1 && outputCount == 0) {
            // From TF's multi input convolution
            outputCount = inputs[1]->length(0);
            kX = inputs[1]->length(3);
            kY = inputs[1]->length(2);
        }
        int kernel_width  = layer->dilateX() * (kX - 1) + 1;
        int kernel_height = layer->dilateY() * (kY - 1) + 1;

        int output_width  = 1;
        int output_height = 1;

        auto input = inputs[0];
        if (input->dimensions() <= 1) {
            // Convolution is not valid for dimension <= 1
            return false;
        }
        // For Tensorflow Group Convolution, the inputCount is the size of filter's input count
        if (layer->inputCount() > 0 && input->channel() % layer->inputCount() != 0 && OpType_Convolution == op->type()) {
            MNN_ERROR("Error for compute convolution shape, need channel = %d, input channel = %d\n", layer->inputCount(), input->channel());
            return false;
        }

        if (layer->padMode() == PadMode_SAME) {
            // Tensorflow padding mode SAME
            output_width  = ceil((float)input->width() / (float)layer->strideX());
            output_height = ceil((float)input->height() / (float)layer->strideY());
        } else if (layer->padMode() == PadMode_VALID) {
            // Tensorflow padding mode VALID
            output_width  = ceil((float)(input->width() - kernel_width + 1) / (float)layer->strideX());
            output_height = ceil((float)(input->height() - kernel_height + 1) / (float)layer->strideY());
        } else {
            // Pad_Caffe means User setted padding
            if (nullptr != layer->pads()) {
                MNN_ASSERT(layer->pads()->size() >= 4);
                int input_width  = input->width() + layer->pads()->data()[1] + layer->pads()->data()[3];
                int input_height = input->height() + layer->pads()->data()[0] + layer->pads()->data()[2];
                output_width     = (input_width - kernel_width) / layer->strideX() + 1;
                output_height    = (input_height - kernel_height) / layer->strideY() + 1;
            } else {
                int input_width  = input->width() + layer->padX() * 2;
                int input_height = input->height() + layer->padY() * 2;
                output_width     = (input_width - kernel_width) / layer->strideX() + 1;
                output_height    = (input_height - kernel_height) / layer->strideY() + 1;
            }
        }

        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.dimensions    = input->buffer().dimensions;
        auto format = TensorUtils::getDescribe(input)->dimensionFormat;
        outputBuffer.type = input->getType();
        
        if (MNN_DATA_FORMAT_NHWC == format) {
            outputBuffer.dim[0].extent = output_height;
            outputBuffer.dim[1].extent = output_width;
            outputBuffer.dim[2].extent = input->buffer().dim[3].extent;;
            outputBuffer.dim[3].extent = outputCount;
        } 
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        outputs[1]->buffer().dimensions = inputs[2]->buffer().dimensions;
        outputs[1]->buffer().type = inputs[2]->getType();
        outputs[1]->buffer().dim[0].extent = 1;
        outputs[1]->buffer().dim[1].extent = 1;
        outputs[1]->buffer().dim[2].extent = 1;
        outputs[1]->buffer().dim[3].extent = 1;
        TensorUtils::getDescribe(outputs[1])->dimensionFormat = TensorUtils::getDescribe(inputs[2])->dimensionFormat;

        return true;
    }
};


class NITI_DSPGradientSplitCONVSizeComputer : public SizeComputer {
public:
    static const Convolution2DCommon* loadCommon(const Op* op) {
        const Convolution2DCommon* layer = nullptr;
        if (op->main_type() == OpParameter_NITI_CONV_Int8) {
            layer = op->main_as_NITI_CONV_Int8()->common();
        // } else {
        //     MNN_ASSERT(op->main_type() == OpParameter_TfQuantizedConv2D);
        //     layer = op->main_as_TfQuantizedConv2D()->common();
        }
        return layer;
    }
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(inputs.size() >= 1);
        // MNN_ASSERT(2 == outputs.size());
        const Convolution2DCommon* layer = loadCommon(op);
        int kX = layer->kernelX();
        int kY = layer->kernelY();
        auto outputCount = layer->outputCount();
        if (inputs.size() > 1 && outputCount == 0) {
            // From TF's multi input convolution
            outputCount = inputs[1]->length(0);
            kX = inputs[1]->length(3);
            kY = inputs[1]->length(2);
        }
        int kernel_width  = layer->dilateX() * (kX - 1) + 1;
        int kernel_height = layer->dilateY() * (kY - 1) + 1;

        int output_width  = 1;
        int output_height = 1;

        auto input = inputs[0];
        if (input->dimensions() <= 1) {
            // Convolution is not valid for dimension <= 1
            return false;
        }
        // For Tensorflow Group Convolution, the inputCount is the size of filter's input count
        if (layer->inputCount() > 0 && input->channel() % layer->inputCount() != 0 && OpType_Convolution == op->type()) {
            MNN_ERROR("Error for compute convolution shape, need channel = %d, input channel = %d\n", layer->inputCount(), input->channel());
            return false;
        }

        if (layer->padMode() == PadMode_SAME) {
            // Tensorflow padding mode SAME
            output_width  = ceil((float)input->width() / 1.0);
            output_height = ceil((float)input->height() / 1.0);
        } else if (layer->padMode() == PadMode_VALID) {
            // Tensorflow padding mode VALID
            output_width  = ceil((float)(input->width() - kernel_width + 1) / 1.0);
            output_height = ceil((float)(input->height() - kernel_height + 1) / 1.0);
        } else {
            // Pad_Caffe means User setted padding
            if (nullptr != layer->pads()) {
                MNN_ASSERT(layer->pads()->size() >= 4);
                int input_width  = input->width() + layer->pads()->data()[1] + layer->pads()->data()[3];
                int input_height = input->height() + layer->pads()->data()[0] + layer->pads()->data()[2];
                output_width     = (input_width - kernel_width) / 1.0 + 1;
                output_height    = (input_height - kernel_height) / 1.0 + 1;
            } else {
                int input_width  = input->width() + layer->padX() * 2;
                int input_height = input->height() + layer->padY() * 2;
                output_width     = (input_width - kernel_width) / 1.0 + 1;
                output_height    = (input_height - kernel_height) / 1.0 + 1;
            }
        }

        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.dimensions    = input->buffer().dimensions;
        auto format = TensorUtils::getDescribe(input)->dimensionFormat;
        outputBuffer.type = input->getType();
        
        if (MNN_DATA_FORMAT_NHWC == format) {
            outputBuffer.dim[0].extent = output_height;
            outputBuffer.dim[1].extent = output_width;
            outputBuffer.dim[2].extent = input->buffer().dim[0].extent;;
            outputBuffer.dim[3].extent = outputCount;
        } 
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        outputs[1]->buffer().dimensions = inputs[2]->buffer().dimensions;
        outputs[1]->buffer().type = inputs[2]->getType();
        outputs[1]->buffer().dim[0].extent = 1;
        outputs[1]->buffer().dim[1].extent = 1;
        outputs[1]->buffer().dim[2].extent = 1;
        outputs[1]->buffer().dim[3].extent = 1;
        TensorUtils::getDescribe(outputs[1])->dimensionFormat = TensorUtils::getDescribe(inputs[2])->dimensionFormat;

        return true;
    }
};

class NITI_DeconvolutionSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        auto layer = op->main_as_NITI_CONV_Int8()->common();

        auto inputTensor = inputs[0];
        int outputHeight = 0, outputWidth = 0;
        if (layer->hasOutputShape()) {
            MNN_ASSERT(inputs.size() >= 2);
            auto outputShape = inputs.back();
            outputHeight = outputShape->host<int>()[1];
            outputWidth  = outputShape->host<int>()[2];
        }

        int input_width   = inputTensor->width();
        int input_height  = inputTensor->height();
        int sH            = layer->strideY();
        int sW            = layer->strideX();
        int kH            = layer->kernelY();
        int kW            = layer->kernelX();
        int pH            = layer->padY();
        int pW            = layer->padX();
        int dH            = layer->dilateY();
        int dW            = layer->dilateX();
        int output_width;
        int output_height;
        auto format = TensorUtils::getDescribe(inputTensor)->dimensionFormat;

        // if (outputHeight > 0 && outputWidth > 0) {
        //     output_width = outputWidth;
        //     output_height = outputHeight;
        // } else if (layer->padMode() == PadMode_SAME) { // Tensorflow support
        //     output_width  = input_width * sW;
        //     output_height = input_height * sH;
        // } else {
        //     if (nullptr != layer->pads()) {
        //         MNN_ASSERT(layer->pads()->size() >= 4);
        //         output_width  = (input_width - 1) * sW + dW * (kW - 1) + 1 - layer->pads()->data()[1] - layer->pads()->data()[3];
        //         output_height = (input_height - 1) * sH + dH * (kH - 1) + 1 - layer->pads()->data()[0] - layer->pads()->data()[2];
        //     } else {
        //         output_width  = (input_width - 1) * sW + dW * (kW - 1) + 1 - pW * 2;
        //         output_height = (input_height - 1) * sH + dH * (kH - 1) + 1 - pH * 2;
        //     }
        //     if(nullptr != layer->outPads()) {
        //         output_width  += layer->outPads()->data()[1];
        //         output_height += layer->outPads()->data()[0];
        //     }
        // }

        output_width = input_width + pH*2 - kW + 1;
        output_height  = input_height + pW*2 - kH + 1;

        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.type = inputTensor->getType();
        outputBuffer.dimensions    = inputTensor->buffer().dimensions;
        outputBuffer.dim[0].extent = inputTensor->buffer().dim[0].extent;
        if (MNN_DATA_FORMAT_NHWC == format) {
            outputBuffer.dim[3].extent = op->main_as_NITI_CONV_Int8()->common()->outputCount();
            outputBuffer.dim[1].extent = output_height;
            outputBuffer.dim[2].extent = output_width;
        } else {
            outputBuffer.dim[1].extent = op->main_as_NITI_CONV_Int8()->common()->outputCount();
            outputBuffer.dim[2].extent = output_height;
            outputBuffer.dim[3].extent = output_width;
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = format;

        return true;
    }

    virtual float onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) const override {
        auto layer = op->main_as_Convolution2D()->common();
        auto kw    = layer->kernelX();
        auto kh    = layer->kernelY();
        auto group = layer->group();
        auto ic    = inputs[0]->channel();
        auto oc    = outputs[0]->channel();
        auto oSize = inputs[0]->width() * inputs[0]->height() * inputs[0]->batch();

        return (float)oSize * kw * kh * (ic * oc / group) / FLOPS_M;
    }
};

class NITI_ConvolutionMinMaxSizeComputer : public SizeComputer {
public:
    static const Convolution2DCommon* loadCommon(const Op* op) {
        const Convolution2DCommon* layer = nullptr;
        if (op->main_type() == OpParameter_NITI_CONV_Int8) {
            layer = op->main_as_NITI_CONV_Int8()->common();
        // } else {
        //     MNN_ASSERT(op->main_type() == OpParameter_TfQuantizedConv2D);
        //     layer = op->main_as_TfQuantizedConv2D()->common();
        }
        return layer;
    }

    virtual bool onComputeSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {

        MNN_ASSERT(inputs.size() >= 1);
        // MNN_ASSERT(2 == outputs.size());
        const Convolution2DCommon* layer = loadCommon(op);
        int kX = layer->kernelX();
        int kY = layer->kernelY();
        auto outputCount = layer->outputCount();
        if (inputs.size() > 1 && outputCount == 0) {
            // From TF's multi input convolution
            outputCount = inputs[1]->length(0);
            kX = inputs[1]->length(3);
            kY = inputs[1]->length(2);
        }
        int kernel_width  = layer->dilateX() * (kX - 1) + 1;
        int kernel_height = layer->dilateY() * (kY - 1) + 1;

        int output_width  = 1;
        int output_height = 1;

        auto input = inputs[0];
        if (input->dimensions() <= 1) {
            // Convolution is not valid for dimension <= 1
            return false;
        }
        // For Tensorflow Group Convolution, the inputCount is the size of filter's input count
        if (layer->inputCount() > 0 && input->channel() % layer->inputCount() != 0 && OpType_Convolution == op->type()) {
            MNN_ERROR("Error for compute convolution shape, need channel = %d, input channel = %d\n", layer->inputCount(), input->channel());
            return false;
        }

        if (layer->padMode() == PadMode_SAME) {
            // Tensorflow padding mode SAME
            output_width  = ceil((float)input->width() / (float)layer->strideX());
            output_height = ceil((float)input->height() / (float)layer->strideY());
        } else if (layer->padMode() == PadMode_VALID) {
            // Tensorflow padding mode VALID
            output_width  = ceil((float)(input->width() - kernel_width + 1) / (float)layer->strideX());
            output_height = ceil((float)(input->height() - kernel_height + 1) / (float)layer->strideY());
        } else {
            // Pad_Caffe means User setted padding
            if (nullptr != layer->pads()) {
                MNN_ASSERT(layer->pads()->size() >= 4);
                int input_width  = input->width() + layer->pads()->data()[1] + layer->pads()->data()[3];
                int input_height = input->height() + layer->pads()->data()[0] + layer->pads()->data()[2];
                output_width     = (input_width - kernel_width) / layer->strideX() + 1;
                output_height    = (input_height - kernel_height) / layer->strideY() + 1;
            } else {
                int input_width  = input->width() + layer->padX() * 2;
                int input_height = input->height() + layer->padY() * 2;
                output_width     = (input_width - kernel_width) / layer->strideX() + 1;
                output_height    = (input_height - kernel_height) / layer->strideY() + 1;
            }
        }

        auto& outputBuffer         = outputs[0]->buffer();
        outputBuffer.dimensions    = input->buffer().dimensions;
        auto format = TensorUtils::getDescribe(input)->dimensionFormat;
        outputBuffer.type = input->getType();
        outputBuffer.dim[0].extent = input->buffer().dim[0].extent;
        if (MNN_DATA_FORMAT_NHWC == format) {
            outputBuffer.dim[3].extent = outputCount;
            outputBuffer.dim[1].extent = output_height;
            outputBuffer.dim[2].extent = output_width;
        } else {
            outputBuffer.dim[1].extent = outputCount;
            outputBuffer.dim[2].extent = output_height;
            outputBuffer.dim[3].extent = output_width;
        }
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;

        outputs[1]->buffer().dimensions = 4;
        outputs[1]->buffer().type = halide_type_of<int32_t>();
        outputs[1]->buffer().dim[0].extent = 1;
        outputs[1]->buffer().dim[1].extent = 1;
        outputs[1]->buffer().dim[2].extent = 1;
        outputs[1]->buffer().dim[3].extent = 1;
        TensorUtils::getDescribe(outputs[1])->dimensionFormat = MNN_DATA_FORMAT_NCHW;

        outputs[2]->buffer().dimensions = 4;
        outputs[2]->buffer().type = halide_type_of<int32_t>();
        outputs[2]->buffer().dim[0].extent = 1;
        outputs[2]->buffer().dim[1].extent = 1;
        outputs[2]->buffer().dim[2].extent = 1;
        outputs[2]->buffer().dim[3].extent = 1;
        TensorUtils::getDescribe(outputs[2])->dimensionFormat = MNN_DATA_FORMAT_NCHW;

        return true;
    }
};

REGISTER_SHAPE(NITI_DeconvolutionSizeComputer, OpType_NITI_DeCONV_Int8);
REGISTER_SHAPE(NITI_ConvolutionSizeComputer, OpType_NITI_DSP_DECONV_Int8);
REGISTER_SHAPE(NITI_ConvolutionSizeComputer, OpType_NITI_CONV_Int8);
REGISTER_SHAPE(NITI_ConvolutionSizeComputer, OpType_NITI_DSP_CONV_Int8);
REGISTER_SHAPE(NITI_ConvolutionSizeComputer, OpType_NITI_GradientCONV_Int8);
REGISTER_SHAPE(NITI_ConvolutionSizeComputer, OpType_NITI_GradientSplitCONV_Int8);
REGISTER_SHAPE(NITI_DSPGradientCONVSizeComputer, OpType_NITI_DSP_GRADIENTCONV_Int8);
REGISTER_SHAPE(NITI_DSPGradientCONVSizeComputer, OpType_NITI_DSP_MATMUL_GRADIENT_Int8);
REGISTER_SHAPE(NITI_DSPGradientCONVSizeComputer, OpType_NITI_DSP_PARALLEL_GRADIENTCONV_Int8);
REGISTER_SHAPE(NITI_DSPGradientSplitCONVSizeComputer, OpType_NITI_DSP_GRADIENT_SPLITBatchCONV_Int8);
REGISTER_SHAPE(NITI_DSPGradientSplitCONVSizeComputer, OpType_NITI_DSP_TRANSPOSEGRADIENT_CONV_Int8);
REGISTER_SHAPE(NITI_DSPGradientCONVSizeComputer, OpType_NITI_DSP_MATMUL_Int8);
REGISTER_SHAPE(NITI_Conv2DBackpropFilter_Int8_SizeComputer, OpType_NITI_Conv2DBackPropFilter_Int8);
REGISTER_SHAPE(NITI_ConvolutionMinMaxSizeComputer, OpType_NITI_CONVMaxMin_Int8);
} // namespace MNN
