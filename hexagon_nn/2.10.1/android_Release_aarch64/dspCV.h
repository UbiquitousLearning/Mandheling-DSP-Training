#ifndef _DSPCV_H
#define _DSPCV_H
#include <AEEStdDef.h>
#include "AEEStdDef.h"
#ifndef __QAIC_HEADER
#define __QAIC_HEADER(ff) ff
#endif //__QAIC_HEADER

#ifndef __QAIC_HEADER_EXPORT
#define __QAIC_HEADER_EXPORT
#endif // __QAIC_HEADER_EXPORT

#ifndef __QAIC_HEADER_ATTRIBUTE
#define __QAIC_HEADER_ATTRIBUTE
#endif // __QAIC_HEADER_ATTRIBUTE

#ifndef __QAIC_IMPL
#define __QAIC_IMPL(ff) ff
#endif //__QAIC_IMPL

#ifndef __QAIC_IMPL_EXPORT
#define __QAIC_IMPL_EXPORT
#endif // __QAIC_IMPL_EXPORT

#ifndef __QAIC_IMPL_ATTRIBUTE
#define __QAIC_IMPL_ATTRIBUTE
#endif // __QAIC_IMPL_ATTRIBUTE
#ifdef __cplusplus
extern "C" {
#endif
#define dspCV_SUCCESS 0
#define dspCV_ERR_REMOTE_HEAP_FAILED 5000
#define dspCV_ERR_HVX_UNSUPPORTED 5001
#define dspCV_ERR_HVX_BUSY 5002
#define dspCV_ERR_UNSUPPORTED_ATTRIBUTE 5003
#define dspCV_ERR_CONSTRUCTOR_FAILED 5004
#define dspCV_ERR_WORKER_POOL_FAILED 5005
#define dspCV_ERR_CLIENT_CLASS_SETTING_FAILED 5006
#define dspCV_ERR_CLOCK_SETTING_FAILED 5007
#define dspCV_ERR_BAD_STATE 5008
enum dspCV_AttributeID {
   MINIMUM_DSP_MHZ,
   MINIMUM_BUS_MHZ,
   LATENCY_TOLERANCE,
   CLOCK_PRESET_MODE,
   RESERVE_HVX_UNITS,
   RESERVED_ENUM_0,
   DISABLE_HVX_USAGE,
   DEFAULT_HVX_MODE,
   DSP_TOTAL_MCPS,
   DSP_MCPS_PER_THREAD,
   PEAK_BUS_BANDWIDTH_MBPS,
   BUS_USAGE_PERCENT,
   AUDIO_MPPS_EVICTION_THRESHOLD_1_STREAMING_HVX,
   AUDIO_MPPS_EVICTION_THRESHOLD_2_STREAMING_HVX,
   _32BIT_PLACEHOLDER_dspCV_AttributeID = 0x7fffffff
};
typedef enum dspCV_AttributeID dspCV_AttributeID;
typedef struct dspCV_Attribute dspCV_Attribute;
struct dspCV_Attribute {
   dspCV_AttributeID ID;
   int value;
};
enum dspCV_ClockPresetMode {
   POWER_SAVING_MODE,
   NORMAL_MODE,
   MAX_PERFORMANCE_MODE,
   NUM_AVAIL_CLOCK_PRESET_MODES,
   _32BIT_PLACEHOLDER_dspCV_ClockPresetMode = 0x7fffffff
};
typedef enum dspCV_ClockPresetMode dspCV_ClockPresetMode;
typedef struct _dspCV_AttributeList__seq_dspCV_Attribute _dspCV_AttributeList__seq_dspCV_Attribute;
typedef _dspCV_AttributeList__seq_dspCV_Attribute dspCV_AttributeList;
struct _dspCV_AttributeList__seq_dspCV_Attribute {
   dspCV_Attribute* data;
   int dataLen;
};
enum dspCV_ConcurrencyAttributeID {
   COMPUTE_RECOMMENDATION,
   CURRENT_DSP_MHZ_SETTING,
   NUM_TOTAL_HVX_UNITS,
   NUM_AVAILABLE_HVX_UNITS,
   EXISTING_CONCURRENCIES,
   _32BIT_PLACEHOLDER_dspCV_ConcurrencyAttributeID = 0x7fffffff
};
typedef enum dspCV_ConcurrencyAttributeID dspCV_ConcurrencyAttributeID;
typedef struct dspCV_ConcurrencyAttribute dspCV_ConcurrencyAttribute;
struct dspCV_ConcurrencyAttribute {
   dspCV_ConcurrencyAttributeID ID;
   int value;
};
#define dspCV_CONCURRENCY_ATTRIBUTE_UNSUPPORTED -1
enum dspCV_ComputeRecommendation {
   COMPUTE_RECOMMENDATION_OK,
   COMPUTE_RECOMMENDATION_NOT_OK,
   _32BIT_PLACEHOLDER_dspCV_ComputeRecommendation = 0x7fffffff
};
typedef enum dspCV_ComputeRecommendation dspCV_ComputeRecommendation;
#define dspCV_VOICE_CONCURRENCY_BITMASK 1
#define dspCV_AUDIO_CONCURRENCY_BITMASK 2
#define dspCV_COMPUTE_CONCURRENCY_BITMASK 4
#define dspCV_SINGLE_HVX_CAMERA_STREAMING_CONCURRENCY_BITMASK 8
#define dspCV_DUAL_HVX_CAMERA_STREAMING_CONCURRENCY_BITMASK 16
typedef struct _dspCV_ConcurrencyAttributeList__seq_dspCV_ConcurrencyAttribute _dspCV_ConcurrencyAttributeList__seq_dspCV_ConcurrencyAttribute;
typedef _dspCV_ConcurrencyAttributeList__seq_dspCV_ConcurrencyAttribute dspCV_ConcurrencyAttributeList;
struct _dspCV_ConcurrencyAttributeList__seq_dspCV_ConcurrencyAttribute {
   dspCV_ConcurrencyAttribute* data;
   int dataLen;
};
enum dspCV_DefaultHVXMode {
   HVX_MODE_DONT_CARE,
   HVX_MODE_64B,
   HVX_MODE_128B,
   _32BIT_PLACEHOLDER_dspCV_DefaultHVXMode = 0x7fffffff
};
typedef enum dspCV_DefaultHVXMode dspCV_DefaultHVXMode;
/// @brief
///   This function initializes the DSP for subsequent RPC function invocations
///   from the calling process. This function establishes a DSP thread pool 
///   for multi-threading, and votes for maximum supported DSP/bus clocks. 
///   It is recommended to instead use initQ6_with_attributes to allow more 
///   control options, but this function is kept for backward compatibility.  
/// @detailed
///    TBD.
__QAIC_HEADER_EXPORT AEEResult __QAIC_HEADER(dspCV_initQ6)(void) __QAIC_HEADER_ATTRIBUTE;
/// @brief
///   This function restores the dspCV related state for the calling process
///   to defaults. For example, it revokes clock votes. It is optional to 
///   call this function before closing the shared library. Closing the library
///   cleans up the resources. 
/// @detailed
///    TBD.
__QAIC_HEADER_EXPORT AEEResult __QAIC_HEADER(dspCV_deinitQ6)(void) __QAIC_HEADER_ATTRIBUTE;
/// @brief
///   This function initializes the DSP for subsequent RPC function invocations
///   from the calling process. This function establishes a DSP thread pool 
///   for multi-threading, and votes clocks according to the provided
///   attributes. It may be called multiple times, with the most recent call
///   overwriting any attributes that were set by an earlier call (from any thread
///   in the same process).
/// @detailed
///    TBD.
/// @param attrib
///   This is a list of attributes and their values for initializing the DSP.
__QAIC_HEADER_EXPORT AEEResult __QAIC_HEADER(dspCV_initQ6_with_attributes)(const dspCV_Attribute* attrib, int attribLen) __QAIC_HEADER_ATTRIBUTE;
/// @brief
///   This function queries the DSP for high-level concurrency information.  
/// @detailed
///    TBD.
/// @param attrib
///   This is a list of requested attribute/value pairs. Caller provides the
///   attribute ID's, and implementation fills in the measured values.
__QAIC_HEADER_EXPORT AEEResult __QAIC_HEADER(dspCV_getQ6_concurrency_attributes)(dspCV_ConcurrencyAttribute* attrib, int attribLen) __QAIC_HEADER_ATTRIBUTE;
#ifdef __cplusplus
}
#endif
#endif //_DSPCV_H
