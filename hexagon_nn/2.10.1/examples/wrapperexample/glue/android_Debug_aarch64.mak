# pakman tree build file
_@ ?= @

.PHONY: MAKE_D_3_LIBDIR QAIC_DIR tree_MAKE_D_EXT_4_DIR MAKE_D_4_INCDIR_MAKE_D_1_LIBDIR MAKE_D_4_INCDIR tree MAKE_D_3_LIBDIR_clean QAIC_DIR_clean tree_MAKE_D_EXT_4_DIR_clean MAKE_D_4_INCDIR_MAKE_D_1_LIBDIR_clean MAKE_D_4_INCDIR_clean tree_clean

tree: MAKE_D_3_LIBDIR tree_MAKE_D_EXT_4_DIR MAKE_D_4_INCDIR MAKE_D_4_INCDIR
	$(call job,,$(MAKE) V=android_Debug_aarch64,making .)

tree_clean: MAKE_D_3_LIBDIR_clean tree_MAKE_D_EXT_4_DIR_clean MAKE_D_4_INCDIR_clean MAKE_D_4_INCDIR_clean
	$(call job,,$(MAKE) V=android_Debug_aarch64 clean,cleaning .)

MAKE_D_4_INCDIR: tree_MAKE_D_EXT_4_DIR MAKE_D_4_INCDIR_MAKE_D_1_LIBDIR MAKE_D_3_LIBDIR
	$(call job,$(HEXAGON_SDK_ROOT)/libs/common/rpcmem,$(MAKE) V=android_Debug_aarch64,making $(HEXAGON_SDK_ROOT)/libs/common/rpcmem)

MAKE_D_4_INCDIR_clean: tree_MAKE_D_EXT_4_DIR_clean MAKE_D_4_INCDIR_MAKE_D_1_LIBDIR_clean MAKE_D_3_LIBDIR_clean
	$(call job,$(HEXAGON_SDK_ROOT)/libs/common/rpcmem,$(MAKE) V=android_Debug_aarch64 clean,cleaning $(HEXAGON_SDK_ROOT)/libs/common/rpcmem)

MAKE_D_4_INCDIR_MAKE_D_1_LIBDIR: 
	$(call job,$(HEXAGON_SDK_ROOT)/test/common/test_util,$(MAKE) V=android_Debug_aarch64,making $(HEXAGON_SDK_ROOT)/test/common/test_util)

MAKE_D_4_INCDIR_MAKE_D_1_LIBDIR_clean: 
	$(call job,$(HEXAGON_SDK_ROOT)/test/common/test_util,$(MAKE) V=android_Debug_aarch64 clean,cleaning $(HEXAGON_SDK_ROOT)/test/common/test_util)

tree_MAKE_D_EXT_4_DIR: QAIC_DIR

tree_MAKE_D_EXT_4_DIR_clean: QAIC_DIR_clean

QAIC_DIR: 
	$(call job,$(HEXAGON_SDK_ROOT)/tools/qaic,make,making $(HEXAGON_SDK_ROOT)/tools/qaic)

QAIC_DIR_clean: 
	$(call job,$(HEXAGON_SDK_ROOT)/tools/qaic,make clean,cleaning $(HEXAGON_SDK_ROOT)/tools/qaic)

MAKE_D_3_LIBDIR: 
	$(call job,$(HEXAGON_SDK_ROOT)/libs/common/atomic,$(MAKE) V=android_Debug_aarch64,making $(HEXAGON_SDK_ROOT)/libs/common/atomic)

MAKE_D_3_LIBDIR_clean: 
	$(call job,$(HEXAGON_SDK_ROOT)/libs/common/atomic,$(MAKE) V=android_Debug_aarch64 clean,cleaning $(HEXAGON_SDK_ROOT)/libs/common/atomic)

W := $(findstring ECHO,$(shell echo))# W => Windows environment
@LOG = $(if $W,$(TEMP)\\)$@-build.log

C = $(if $1,$(if $W,cd /D,cd) $1 && )$2
job = $(_@)echo $3 && ( $C )> $(@LOG) && $(if $W,del,rm) $(@LOG) || ( echo ERROR $3 && $(if $W,type,cat) $(@LOG) && $(if $W,del,rm) $(@LOG) && exit 1)
ifdef VERBOSE
  job = $(_@)echo $3 && $C
endif
