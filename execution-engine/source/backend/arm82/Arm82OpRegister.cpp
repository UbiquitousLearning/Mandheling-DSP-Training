// This file is generated by Shell for ops register
namespace MNN {
extern void ___OpType_Moments__Arm82MomentsCreator__();
extern void ___OpType_InstanceNorm__Arm82InstanceNormCreator__();
extern void ___OpType_Eltwise__Arm82EltwiseCreator__();
extern void ___OpType_Interp__Arm82InterpCreator__();

void registerArm82Ops() {
#if defined(__ANDROID__) || defined(__aarch64__)
___OpType_Moments__Arm82MomentsCreator__();
___OpType_InstanceNorm__Arm82InstanceNormCreator__();
___OpType_Eltwise__Arm82EltwiseCreator__();
___OpType_Interp__Arm82InterpCreator__();
#endif
}
}
