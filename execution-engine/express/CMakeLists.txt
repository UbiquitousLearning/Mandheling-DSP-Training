file(GLOB_RECURSE MNN_EXPR_SRCS "${CMAKE_CURRENT_LIST_DIR}/*.*")
option(MNN_EXPR_ENABLE_PROFILER "Support profile Expr's op cost" OFF)
option(MNN_EXPR_SHAPE_EAGER "Force compute Expr's shape directly cost" OFF)
IF (MNN_EXPR_ENABLE_PROFILER)
    add_definitions(-DMNN_EXPR_ENABLE_PROFILER)
ENDIF()
IF (MNN_EXPR_SHAPE_EAGER)
    add_definitions(-DMNN_EXPR_SHAPE_EAGER)
ENDIF()
IF(MNN_SEP_BUILD)
    if (MNN_BUILD_FOR_ANDROID_COMMAND)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../")
    endif()
    if (MNN_BUILD_MINI)
        add_library(MNN_Express SHARED ${MNN_EXPR_SRCS} $<TARGET_OBJECTS:MNNTransform>)
        target_compile_options(MNN_Express PRIVATE -DMNN_BUILD_MINI)
    else()
        add_library(MNN_Express SHARED ${MNN_EXPR_SRCS})
    endif()
    target_link_libraries(MNN_Express MNN)
ELSE()
    add_library(MNN_Express OBJECT ${MNN_EXPR_SRCS})
ENDIF()
