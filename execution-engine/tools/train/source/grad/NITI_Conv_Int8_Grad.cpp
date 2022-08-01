//
//  NITI_Conv_Int8_Grad.cpp
//  MNN
//
//  Created by xudaliang on 2021/08/12.
//  
//

#include "OpGrad.hpp"
#include "core/Macro.h"
using namespace std;
using namespace MNN::Express;
using namespace MNN;
class NITI_Conv_Int8_Grad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        auto inputs = expr->inputs();
        if (inputs.size() == 1) {
            return std::vector<Express::VARP>{nullptr};
        }
        std::vector<VARP> res(inputs.size(), nullptr);
        auto forwardName = expr->name();
        std::shared_ptr<OpT> forwardOp(expr->get()->UnPack());
        auto outputDiff = backwardOutput[0];
        
        {
            // Create Input Grad
            unique_ptr<OpT> newOp(new OpT);
            // if (forwardOp->type == OpType_NITI_CONV_Int8) {
                newOp->type = OpType_NITI_DeCONV_Int8;
            // } else if (forwardOp->type == OpType_DepthwiseConvInt8 ) {
            //     newOp->type = OpType_DeconvDepthwise;
            // }
            newOp->main.type = OpParameter_NITI_CONV_Int8;
            auto conv2D      = new NITI_CONV_Int8T;
            conv2D->common.reset(new Convolution2DCommonT(*forwardOp->main.AsNITI_CONV_Int8()->common));

            auto inputCount             = conv2D->common->inputCount;
            auto outputCount            = conv2D->common->outputCount;
            auto padMode = conv2D->common->padMode;
            if ((conv2D->common->strideX > 1 || conv2D->common->strideY > 1)) {
                auto inputShape = inputs[0]->getInfo();
                auto outputShape = outputDiff->getInfo();
                if (nullptr == inputShape || nullptr == outputShape) {
                    return {};
                }
                auto iw = inputShape->dim[3];
                auto ih = inputShape->dim[2];
                auto ow = outputShape->dim[3];
                auto oh = outputShape->dim[2];
                auto kW = conv2D->common->kernelX;
                auto kH = conv2D->common->kernelY;
                auto sW = conv2D->common->strideX;
                auto sH = conv2D->common->strideY;
                auto dW = conv2D->common->dilateX;
                auto dH = conv2D->common->dilateY;

                std::vector<int> padding {0, 0, 0, 0};
                int kernelWidthSize = dW * (kW - 1) + 1;
                int kernelHeightSize = dH * (kH - 1) + 1;
                int padNeededWidth  = (ow - 1) * sW + kernelWidthSize - iw;
                int padNeededHeight = (oh - 1) * sH + kernelHeightSize - ih;
                if (padMode == PadMode_SAME) {
                    padding[0] = padNeededHeight / 2;
                    padding[1] = padNeededWidth / 2;
                } else if (padMode == PadMode_CAFFE) {
                    if (conv2D->common->pads.empty()) {
                        padding[0] = conv2D->common->padY;
                        padding[1] = conv2D->common->padX;
                    } else {
                        padding[0] = conv2D->common->pads[0];
                        padding[1] = conv2D->common->pads[1];
                    }
                }
                padding[2] = padNeededHeight - padding[0];
                padding[3] = padNeededWidth - padding[1];
                conv2D->common->pads = padding;
                conv2D->common->padMode = PadMode_CAFFE;
            }
            conv2D->common->inputCount  = outputCount;
            conv2D->common->outputCount = inputCount;

            auto inputShape= inputs[0]->getInfo();
            auto weightShape= inputs[1]->getInfo();

            int pH = conv2D->common->padX;
            int pW = conv2D->common->padY;

            if(conv2D->common->strideX == 2) {
                int ow = inputShape->dim[3] + pW*2 - weightShape->dim[3] + 1;

                int extraPad = (inputShape->dim[3] - (ow + pW*2 - weightShape->dim[3] + 1))/2;

                conv2D->common->strideX = 1;
                conv2D->common->strideY = 1; 

                newOp->main.value           = conv2D;

                if (extraPad != 0) {
                    auto expr = Expr::create(std::move(newOp), { _Convert(_NITI_Pad_Int8(_Convert(_NITI_LeftPoolGrad(outputDiff, {ow, ow}, {2, 2}), NCHW), extraPad), NC4HW4), _NITI_Transpose_INT8(inputs[1] , {1,0,2,3})  , inputs[3]});
                    res[0]    = Variable::create(expr);
                } else {
                    auto expr = Expr::create(std::move(newOp), { _Convert(_NITI_LeftPoolGrad(outputDiff, {ow, ow}, {2, 2}), NC4HW4), _NITI_Transpose_INT8(inputs[1] , {1,0,2,3})  , inputs[3]});
                    res[0]    = Variable::create(expr);
                }

            } else {
                newOp->main.value           = conv2D;

                auto outputDiffShape = outputDiff->getInfo();
            
            
                int extraPad = (inputShape->dim[3] - (outputDiffShape->dim[3] + pW*2 - weightShape->dim[3] + 1))/2;
                
                if (extraPad != 0) {
                    auto expr = Expr::create(std::move(newOp), { _Convert(_NITI_Pad_Int8(_Convert(outputDiff, NCHW), extraPad), NC4HW4), _NITI_Transpose_INT8(inputs[1] , {1,0,2,3})  , inputs[3]});
                    res[0]    = Variable::create(expr);
                } else {
                    auto expr = Expr::create(std::move(newOp), { _Convert(outputDiff, NC4HW4), _NITI_Transpose_INT8(inputs[1] , {1,0,2,3})  , inputs[3]});
                    res[0]    = Variable::create(expr);
                }
            }

            

            
            auto resultShape = res[0]->getInfo();
            
            MNN_ASSERT(resultShape->dim[3] == inputShape->dim[3]);
            MNN_ASSERT(resultShape->dim[2] == inputShape->dim[2]);
        }
        // Add Filter Grad
        {
            auto inputShape= inputs[0]->getInfo();
            auto weightShape= inputs[1]->getInfo();
            // if(inputShape->dim[3] != 1) {
                unique_ptr<OpT> newOp(new OpT);
                newOp->type      = OpType_NITI_GradientCONV_Int8;
                newOp->main.type = OpParameter_NITI_CONV_Int8;
                auto conv2D      = new NITI_CONV_Int8T;
                conv2D->common.reset(new Convolution2DCommonT(*forwardOp->main.AsNITI_CONV_Int8()->common));


                int pH = conv2D->common->padX;
                int pW = conv2D->common->padY;

                if(conv2D->common->strideX == 2) {
                    int ow = inputShape->dim[3] + pW*2 - weightShape->dim[3] + 1;
                    auto outputDiffShape= outputDiff->getInfo();

                    conv2D->common->kernelX = ow;
                    conv2D->common->kernelY = ow;

                    conv2D->common->inputCount = inputShape->dim[0];
                    conv2D->common->outputCount = outputDiffShape->dim[1];

                    conv2D->common->strideX = 1;
                    conv2D->common->strideY = 1; 

                    newOp->main.value = conv2D;

                    auto expr         = Expr::create(std::move(newOp), { _Convert(_NITI_Transpose_INT8(_Convert(inputs[0], NCHW),{1,0,2,3}), NC4HW4), _NITI_Transpose_INT8(_Convert(_NITI_LeftPoolGrad(outputDiff, {ow, ow}, {2, 2}), NCHW), {1,0,2,3}), inputs[2],  inputs[3] }, 2);
                    
                    auto ret = Variable::create(expr, 0);
                    auto transpose = _Transpose(ret, {1,0,2,3});
                    res[1]            = transpose;
                    


                } else {
                    auto outputDiffShape= outputDiff->getInfo();

                    conv2D->common->kernelX = outputDiffShape->dim[2];
                    conv2D->common->kernelY = outputDiffShape->dim[3];

                    conv2D->common->inputCount = inputShape->dim[0];
                    conv2D->common->outputCount = outputDiffShape->dim[1];

                    newOp->main.value = conv2D;
                    auto expr         = Expr::create(std::move(newOp), { _Convert(_NITI_Transpose_INT8(_Convert(inputs[0], NCHW),{1,0,2,3}), NC4HW4), _NITI_Transpose_INT8(_Convert(outputDiff, NCHW), {1,0,2,3}), inputs[2],  inputs[3] }, 2);
                    
                    auto ret = Variable::create(expr, 0);
                    auto convert = _Convert(ret , NCHW);
                    auto transpose = _NITI_Transpose_INT8(convert, {1,0,2,3});
                    res[1]            = transpose;
                    
                }

            
        }
        
        res[2] = _Scalar<int8_t>(0);
        res[3] = _Scalar<int8_t>(0);

        return res;
    }
};

static const auto gRegister = []() {
    static NITI_Conv_Int8_Grad _c;
    OpGrad::insert(OpType_NITI_CONV_Int8, &_c);
    return true;
}();
