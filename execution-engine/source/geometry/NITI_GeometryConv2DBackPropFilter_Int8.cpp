//
//  NITI_GeometryConv2DBackPropFilter.cpp
//  MNN
//
//  Created by xudaliang on 2021/09/16.
//  
//

#include "ConvertUtils.hpp"
#include "GeometryConvUtils.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
namespace MNN {
class NITI_GeometryConv2DBackPropFilter : public GeometryComputer {
public:
    virtual bool onCompute(const Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                           Context& context, CommandBuffer& res) const override {
                               
        auto common     = op->main_as_NITI_CONV_Int8()->common();
        auto input      = inputs[0];
        auto outputDiff = inputs[2];
        
        auto kw    = common->kernelX();
        auto kh    = common->kernelY();
        auto sw    = common->strideX();
        auto sh    = common->strideY();
        auto dw    = common->dilateX();
        auto dh    = common->dilateY();
        auto batch = outputDiff->batch();
        auto ow    = outputDiff->width();
        auto oh    = outputDiff->height();
        auto oc    = outputDiff->channel();
        auto ic    = input->channel();
        auto iw    = input->width();
        auto ih    = input->height();
        auto pads  = ConvolutionCommon::convolutionPad(input, outputDiff, common);
        
        MNN_ASSERT(TensorUtils::getDescribe(input)->dimensionFormat != MNN_DATA_FORMAT_NHWC);
        MNN_ASSERT(TensorUtils::getDescribe(outputDiff)->dimensionFormat != MNN_DATA_FORMAT_NHWC);
        Tensor* A = nullptr;
        Tensor* B = nullptr;
        {
            // B: Input Im2Col, n, ic, ih, iw -> ic*kh*kw, n*oh*ow
            std::shared_ptr<Tensor> im2Col(new Tensor);
            GeometryConvUtils::im2Col(im2Col.get(), input, ic, kh, kw, batch, oh, ow, ih, iw, sh, sw, dh, dw, pads);
            B = im2Col.get();
            res.extras.emplace_back(im2Col);
        }
        {
            // A: Output n, oc, oh, ow -> oc, n*oh*ow
            std::shared_ptr<Tensor> outputTranspose(new Tensor);
            A                                    = outputTranspose.get();
            outputTranspose->buffer().type       = halide_type_of<int8_t>();
            outputTranspose->buffer().dimensions = 2;
            outputTranspose->setLength(0, oc);
            outputTranspose->setLength(1, batch * ow * oh);
            // outputTranspose->setLength(2, 1);
            // outputTranspose->setLength(3, 1);
            auto des = TensorUtils::getDescribe(outputTranspose.get());
            des->regions.resize(1);
            des->memoryType   = Tensor::InsideDescribe::MEMORY_VIRTUAL;
            auto& reg         = des->regions[0];
            reg.origin        = outputDiff;
            reg.size[0]       = oc;
            reg.size[1]       = batch;
            reg.size[2]       = ow * oh;
            reg.src.offset    = 0;
            reg.src.stride[0] = oh * ow;
            reg.src.stride[1] = oh * ow * oc;
            reg.src.stride[2] = 1;
            reg.dst.offset    = 0;
            reg.dst.stride[0] = oh * ow * batch;
            reg.dst.stride[1] = oh * ow;
            reg.dst.stride[2] = 1;
            res.extras.emplace_back(std::move(outputTranspose));
        }
        {
            // C = MatMul(B, A)
            std::shared_ptr<Tensor> C(new Tensor);
            C->buffer().type       = halide_type_of<int8_t>();
            C->buffer().dimensions = 2;
            C->setLength(0, ic * kw * kh);
            C->setLength(1, oc);
            // C->setLength(2, 1);
            // C->setLength(3, 1);

            std::unique_ptr<OpT> matmul(new OpT);
            matmul->type                   = OpType_NITI_MatMul_Int8;
            matmul->main.type                = OpParameter_MatMul;
            matmul->main.value               = new MatMulT;
            matmul->main.AsMatMul()->transposeA = false;
            matmul->main.AsMatMul()->transposeB  = true;
        
            flatbuffers::FlatBufferBuilder builder;
            auto lastOffset = Op::Pack(builder, matmul.get());
            builder.Finish(lastOffset);
            Command cmd;
            cmd.buffer.resize(builder.GetSize());
            ::memcpy(cmd.buffer.data(), builder.GetBufferPointer(), cmd.buffer.size());
            cmd.inputs = {B, A};
            cmd.outputs = {C.get()};
            cmd.op      = flatbuffers::GetMutableRoot<Op>(cmd.buffer.data());

            auto kernelDiffDes        = TensorUtils::getDescribe(outputs[0]);
            kernelDiffDes->memoryType = Tensor::InsideDescribe::MEMORY_VIRTUAL;

            // Transpose
            auto len0 = kw * kh * ic;
            auto len1 = oc;
            kernelDiffDes->regions.resize(1);
            auto& desReg         = kernelDiffDes->regions[0];
            desReg.size[0]       = 1;
            desReg.size[1]       = len1;
            desReg.size[2]       = len0;
            desReg.dst.offset    = 0;
            desReg.dst.stride[0] = 0;
            desReg.dst.stride[1] = len0;
            desReg.dst.stride[2] = 1;
            desReg.src.offset    = 0;
            desReg.src.stride[0] = 0;
            desReg.src.stride[1] = 1;
            desReg.src.stride[2] = len1;
            desReg.origin        = C.get();
            res.extras.emplace_back(std::move(C));
            res.command.emplace_back(std::move(cmd));
        }
        return true;
    }
};

static void _create() {
    std::shared_ptr<GeometryComputer> comp(new NITI_GeometryConv2DBackPropFilter);
    GeometryComputer::registerGeometryComputer(comp, {OpType_NITI_Conv2DBackPropFilter_Int8});
}

REGISTER_GEOMETRY(NITI_GeometryConv2DBackPropFilter, _create);

} // namespace MNN
