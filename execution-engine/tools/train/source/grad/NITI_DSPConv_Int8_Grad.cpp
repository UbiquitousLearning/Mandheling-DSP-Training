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
class NITI_DSPConv_Int8_Grad : public OpGrad {
public:
    virtual std::vector<Express::VARP> onGrad(Express::EXPRP expr,
                                              const std::vector<Express::VARP>& backwardOutput) override {
        // MNN_PRINT("NITI_Conv_Int8_Grad\n onGrad");
        /*
            expr 是 forward的表达式，可以通过input获得当前执行的算子的输入
                以CONV为例，那么就是 x  weight  bias scale
                x 是 输入这个CONV 算子的上一轮激活值！
            backwardOutput 是 backward的输入的 gradient值！
                不能完全确定，但是感觉上默认是res[0]的值就会被认为是下一层的backwardOutput，也就是gradient 
        */
        auto inputs = expr->inputs();
        MNN_PRINT("inputs size = %d\n", inputs.size());
        if (inputs.size() == 1) {
            return std::vector<Express::VARP>{nullptr};
        }
        std::vector<VARP> res(inputs.size(), nullptr);
        auto forwardName = expr->name();
        std::shared_ptr<OpT> forwardOp(expr->get()->UnPack());
        auto outputDiff = backwardOutput[0];
       
        {
            
            unique_ptr<OpT> newOp(new OpT);
            // if (forwardOp->type == OpType_NITI_CONV_Int8) {
                newOp->type = OpType_NITI_DSP_DECONV_Int8;
            // } else if (forwardOp->type == OpType_DepthwiseConvInt8 ) {
            //     newOp->type = OpType_DeconvDepthwise;
            // }
            

            newOp->main.type = OpParameter_NITI_CONV_Int8;
            auto conv2D      = new NITI_CONV_Int8T;
            conv2D->common.reset(new Convolution2DCommonT(*forwardOp->main.AsNITI_CONV_Int8()->common));
            auto inputCount             = conv2D->common->inputCount;
            auto outputCount            = conv2D->common->outputCount;
            auto padMode = conv2D->common->padMode;

            auto inputShape= inputs[0]->getInfo();
            auto weightShape= inputs[1]->getInfo();
            auto outputDiffShape = outputDiff->getInfo();

            int pH = conv2D->common->padX;
            int pW = conv2D->common->padY;


            int extraPad = (inputShape->dim[2] - (outputDiffShape->dim[2] + pW*2 - weightShape->dim[1] + 1))/2;
    

            conv2D->common->inputCount  = outputCount;
            conv2D->common->outputCount = inputCount;
            newOp->main.value           = conv2D;

            if(conv2D->common->strideX == 2) {
                 auto inputShape= inputs[0]->getInfo();
                auto weightShape= inputs[1]->getInfo();

                int pH = conv2D->common->padX;
                int pW = conv2D->common->padY;

                int ow = inputShape->dim[2] + pW*2 - weightShape->dim[1] + 1;
                int extraPad = (inputShape->dim[2] - (ow + pW*2 - weightShape->dim[1] + 1))/2;

                conv2D->common->padX += extraPad;
                conv2D->common->padY += extraPad;
                conv2D->common->padMode = PadMode_CAFFE;

                conv2D->common->strideX = 1;
                conv2D->common->strideY = 1; 

                newOp->main.value           = conv2D;

                auto expr = Expr::create(std::move(newOp), {
                            _NITI_DSP_LeftPoolGrad_Deconv(outputDiff, {ow, ow}, {2, 2}), 
                            _NITI_Transpose_INT8(_NITI_DSP_WeightRotate180_Int8(_NITI_Transpose_INT8(inputs[1] , {2,3,0,1}) ),  {2,3,1,0}),  
                            _Scalar<int8_t>(0),
                            _Scalar<int8_t>(0),
                            }, 2);

                res[0]    = Variable::create(expr, 0);
            } else {
                if (extraPad != 0) {

                    conv2D->common->padX += extraPad;
                    conv2D->common->padY += extraPad;
                    conv2D->common->padMode = PadMode_CAFFE;

                    newOp->main.value           = conv2D;

                    auto expr = Expr::create(std::move(newOp), {
                                outputDiff, 
                                _NITI_Transpose_INT8(_NITI_DSP_WeightRotate180_Int8(_NITI_Transpose_INT8(inputs[1] , {2,3,0,1}) ),  {2,3,1,0}),  
                                _Scalar<int8_t>(0),
                                _Scalar<int8_t>(0),
                                }, 2);

                    res[0]    = Variable::create(expr, 0);
                } else {

                    auto expr = Expr::create(std::move(newOp), {
                                outputDiff,
                                _NITI_Transpose_INT8(_NITI_DSP_WeightRotate180_Int8(_NITI_Transpose_INT8(inputs[1] , {2,3,0,1}) ),  {2,3,1,0}), 
                                _Scalar<int8_t>(0),
                                _Scalar<int8_t>(0),
                                }, 2);

                    res[0]    = Variable::create(expr, 0);
                }

            }
            
            
            auto resultShape = res[0]->getInfo();
            
            MNN_ASSERT(resultShape->dim[3] == inputShape->dim[3]);
            MNN_ASSERT(resultShape->dim[2] == inputShape->dim[2]);
        }
        // Add Filter Grad
        {
            int parallel_mode;

            FILE* f = fopen("parallel.txt","r");
            fscanf(f, "%d\n", &parallel_mode);
            fclose(f);

            auto inputShape= inputs[0]->getInfo();
            // if(inputShape->dim[3] != 1) {
                unique_ptr<OpT> newOp(new OpT);
                if(parallel_mode)
                    newOp->type      = OpType_NITI_DSP_PARALLEL_GRADIENTCONV_Int8;
                else
                    newOp->type      = OpType_NITI_DSP_TRANSPOSEGRADIENT_CONV_Int8;
                newOp->main.type = OpParameter_NITI_CONV_Int8;
                auto conv2D      = new NITI_CONV_Int8T;
                conv2D->common.reset(new Convolution2DCommonT(*forwardOp->main.AsNITI_CONV_Int8()->common));

                auto outputDiffShape= outputDiff->getInfo();

                conv2D->common->kernelX = outputDiffShape->dim[1];
                conv2D->common->kernelY = outputDiffShape->dim[2];

                conv2D->common->inputCount = inputShape->dim[0];
                conv2D->common->outputCount = outputDiffShape->dim[3];

                newOp->main.value = conv2D;

                if(conv2D->common->strideX == 2) {

                    auto weightShape= inputs[1]->getInfo();

                    int pH = conv2D->common->padX;
                    int pW = conv2D->common->padY;

                    int ow = inputShape->dim[2] + pW*2 - weightShape->dim[1] + 1;
                    MNN_PRINT("ow  = %d\n", ow);
                    auto outputDiffShape= outputDiff->getInfo();

                    conv2D->common->kernelX = ow;
                    conv2D->common->kernelY = ow;

                    conv2D->common->strideX = 1;
                    conv2D->common->strideY = 1; 
                    
                    newOp->main.value = conv2D;

                    if(outputDiffShape->dim[2] > 8 && outputDiffShape->dim[2] <=16 && inputShape->dim[3] > 4) {
                        newOp->type = OpType_NITI_DSP_GRADIENT_SPLITBatchCONV_Int8;
                        auto expr            = Expr::create(std::move(newOp), {_NITI_Transpose_INT8(inputs[0], {3,1,2,0}), _NITI_LeftPoolGrad(outputDiff, {ow, ow}, {2, 2}), _Scalar<int8_t>(0),  _Scalar<int8_t>(0) }, 2);
                        res[1]            = Variable::create(expr, 0);
                    } else {


                        auto expr            = Expr::create(std::move(newOp), {_NITI_Transpose_INT8(inputs[0], {3,1,2,0}), _NITI_LeftPoolGrad(outputDiff, {ow, ow}, {2, 2}),  _Scalar<int8_t>(0),  _Scalar<int8_t>(0) }, 2);
                        res[1]            = Variable::create(expr, 0);

                    }                    

                } else {
                    int parallel_mode;

                    FILE* f = fopen("parallel.txt","r");
                    fscanf(f, "%d\n", &parallel_mode);
                    fclose(f);

                    if(parallel_mode) {
                        if(outputDiffShape->dim[3] > 256) {
                            auto expr            = Expr::create(std::move(newOp), {inputs[0], outputDiff, _Scalar<int8_t>(0),  _Scalar<int8_t>(0) }, 2);
                            res[1]            = Variable::create(expr, 0);
                        } else {
                            newOp->type = OpType_NITI_GradientCONV_Int8;
                            auto outputDiffShape= outputDiff->getInfo();

                            conv2D->common->kernelX = outputDiffShape->dim[1];
                            conv2D->common->kernelY = outputDiffShape->dim[2];

                            conv2D->common->inputCount = inputShape->dim[0];
                            conv2D->common->outputCount = outputDiffShape->dim[3];

                            newOp->main.value = conv2D;
                            auto expr         = Expr::create(std::move(newOp), { _Convert(_NITI_Transpose_INT8(_Convert(inputs[0], NCHW),{1,0,2,3}), NC4HW4), _NITI_Transpose_INT8(_Convert(outputDiff, NCHW), {1,0,2,3}), inputs[2],  inputs[3] }, 2);
                            res[1]            = _Convert(_NITI_Transpose_INT8(_Convert(Variable::create(expr, 0), NCHW), {2,1,3,0}), NHWC);
                        }
                    } else {

                            auto expr            = Expr::create(std::move(newOp), {_NITI_Transpose_INT8(inputs[0], {3,1,2,0}), outputDiff, _Scalar<int8_t>(0),  _Scalar<int8_t>(0) }, 2);
                            res[1]            = Variable::create(expr, 0);
                            
                    }

                    
                    
                }
            
        }
        
        res[2] = _Scalar<int8_t>(0);
        res[3] = _Scalar<int8_t>(0);
#ifdef MNN_HEXAGON_DSP_SIMU
        res[4] = _Scalar<int8_t>(0);
        res[5] = _Scalar<int8_t>(0);
#endif
        return res;
    }
};

static const auto gRegister = []() {
    static NITI_DSPConv_Int8_Grad _c;
    OpGrad::insert(OpType_NITI_DSP_CONV_Int8, &_c);
    return true;
}();
