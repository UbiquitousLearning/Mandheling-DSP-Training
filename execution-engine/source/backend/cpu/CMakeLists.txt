# CPU
option(MNN_SUPPORT_BF16 "Enable MNN's bf16 op" OFF)
FILE(GLOB MNN_CPU_SRC ${CMAKE_CURRENT_LIST_DIR}/* ${CMAKE_CURRENT_LIST_DIR}/compute/* ${CMAKE_CURRENT_LIST_DIR}/hexagon/hexagon_nn/*.cpp ${CMAKE_CURRENT_LIST_DIR}/hexagon/hexagon_op_builders/*)
add_library(MNNCPU OBJECT ${MNN_CPU_SRC})
if (MNN_SUPPORT_BF16)
    include(${CMAKE_CURRENT_LIST_DIR}/bf16/CMakeLists.txt)
    list(APPEND MNN_TARGETS MNN_BF16)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_BF16>)
    target_compile_options(MNNCPU PRIVATE -DMNN_SUPPORT_BF16)
endif()
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCPU>)
list(APPEND MNN_TARGETS MNNCPU)
option(MNN_SSE_USE_FP16_INSTEAD "Use fp16 instead of bf16 for x86op" OFF)


if(MNN_USE_SPARSE_COMPUTE)
  target_compile_options(MNNCPU PRIVATE -DMNN_USE_SPARSE_COMPUTE)
endif()

# X86_64 AVX/SSE
if (MNN_USE_SSE)
    include(${CMAKE_CURRENT_LIST_DIR}/x86_x64/CMakeLists.txt)
endif()

# AArch32/64 Assemblies
include(${CMAKE_CURRENT_LIST_DIR}/arm/CMakeLists.txt)

IF(NOT DEFINED IOS_ARCH)
    set(IOS_ARCH "")
ENDIF()

# ARM82 Assemblies
IF(MNN_ARM82)
    target_compile_options(MNNCPU PRIVATE -DENABLE_ARMV82)
    include(${CMAKE_CURRENT_LIST_DIR}/../arm82/CMakeLists.txt)
    list(APPEND MNN_TARGETS MNN_Arm82)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_Arm82>)
ENDIF()

