ifndef SDK_SETUP_ENV
$(error setup_sdk_env must be run prior to building)
endif

ifndef V
$(error Varaint must be provided, pass a variant by adding 'V=<desired varaint>' to your build command)
endif

V_TARGET = $(word 1,$(subst _, ,$(V)))
ifeq ($(wildcard $(V_TARGET)_deps.min $(V_TARGET)_rules.min), )
$(error Unsupported target '$(V)' in variant '$(V)', check that both $(V_TARGET)_deps.min and $(V_TARGET)_rules.min exist)
endif

include $(V_TARGET)_deps.min
include $(HEXAGON_SDK_ROOT)/build/make.d/$(V_TARGET)_vs.min
include $(HEXAGON_SDK_ROOT)/build/defines.min
include $(V_TARGET).min

ifeq (,$(GRAPHINIT))
 GRAPHINIT := test/graphinit_small.c 
endif  

# always last
include $(RULES_MIN)
