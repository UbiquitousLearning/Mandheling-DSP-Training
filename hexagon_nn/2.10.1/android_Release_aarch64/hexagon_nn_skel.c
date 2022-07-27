#ifndef _HEXAGON_NN_SKEL_H
#define _HEXAGON_NN_SKEL_H
#include "hexagon_nn.h"
#ifndef _QAIC_ENV_H
#define _QAIC_ENV_H

#ifdef __GNUC__
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#else
#pragma GCC diagnostic ignored "-Wpragmas"
#endif
#pragma GCC diagnostic ignored "-Wuninitialized"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

#ifndef _ATTRIBUTE_UNUSED

#ifdef _WIN32
#define _ATTRIBUTE_UNUSED
#else
#define _ATTRIBUTE_UNUSED __attribute__ ((unused))
#endif

#endif // _ATTRIBUTE_UNUSED

#ifndef __QAIC_REMOTE
#define __QAIC_REMOTE(ff) ff
#endif //__QAIC_REMOTE

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

#ifndef __QAIC_STUB
#define __QAIC_STUB(ff) ff
#endif //__QAIC_STUB

#ifndef __QAIC_STUB_EXPORT
#define __QAIC_STUB_EXPORT
#endif // __QAIC_STUB_EXPORT

#ifndef __QAIC_STUB_ATTRIBUTE
#define __QAIC_STUB_ATTRIBUTE
#endif // __QAIC_STUB_ATTRIBUTE

#ifndef __QAIC_SKEL
#define __QAIC_SKEL(ff) ff
#endif //__QAIC_SKEL__

#ifndef __QAIC_SKEL_EXPORT
#define __QAIC_SKEL_EXPORT
#endif // __QAIC_SKEL_EXPORT

#ifndef __QAIC_SKEL_ATTRIBUTE
#define __QAIC_SKEL_ATTRIBUTE
#endif // __QAIC_SKEL_ATTRIBUTE

#ifdef __QAIC_DEBUG__
   #ifndef __QAIC_DBG_PRINTF__
   #include <stdio.h>
   #define __QAIC_DBG_PRINTF__( ee ) do { printf ee ; } while(0)
   #endif
#else
   #define __QAIC_DBG_PRINTF__( ee ) (void)0
#endif


#define _OFFSET(src, sof)  ((void*)(((char*)(src)) + (sof)))

#define _COPY(dst, dof, src, sof, sz)  \
   do {\
         struct __copy { \
            char ar[sz]; \
         };\
         *(struct __copy*)_OFFSET(dst, dof) = *(struct __copy*)_OFFSET(src, sof);\
   } while (0)

#define _COPYIF(dst, dof, src, sof, sz)  \
   do {\
      if(_OFFSET(dst, dof) != _OFFSET(src, sof)) {\
         _COPY(dst, dof, src, sof, sz); \
      } \
   } while (0)

_ATTRIBUTE_UNUSED
static __inline void _qaic_memmove(void* dst, void* src, int size) {
   int i;
   for(i = 0; i < size; ++i) {
      ((char*)dst)[i] = ((char*)src)[i];
   }
}

#define _MEMMOVEIF(dst, src, sz)  \
   do {\
      if(dst != src) {\
         _qaic_memmove(dst, src, sz);\
      } \
   } while (0)


#define _ASSIGN(dst, src, sof)  \
   do {\
      dst = OFFSET(src, sof); \
   } while (0)

#define _STD_STRLEN_IF(str) (str == 0 ? 0 : strlen(str))

#include "AEEStdErr.h"

#define _TRY(ee, func) \
   do { \
      if (AEE_SUCCESS != ((ee) = func)) {\
         __QAIC_DBG_PRINTF__((__FILE__ ":%d:error:%d:%s\n", __LINE__, (int)(ee),#func));\
         goto ee##bail;\
      } \
   } while (0)

#define _CATCH(exception) exception##bail: if (exception != AEE_SUCCESS)

#define _ASSERT(nErr, ff) _TRY(nErr, 0 == (ff) ? AEE_EBADPARM : AEE_SUCCESS)

#ifdef __QAIC_DEBUG__
#define _ALLOCATE(nErr, pal, size, alignment, pv) _TRY(nErr, _allocator_alloc(pal, __FILE_LINE__, size, alignment, (void**)&pv));\
                                                  _ASSERT(nErr,pv || !(size))   
#else
#define _ALLOCATE(nErr, pal, size, alignment, pv) _TRY(nErr, _allocator_alloc(pal, 0, size, alignment, (void**)&pv));\
                                                  _ASSERT(nErr,pv || !(size))
#endif


#endif // _QAIC_ENV_H

#include "remote.h"
#include <string.h>
#ifndef _ALLOCATOR_H
#define _ALLOCATOR_H

#include <stdlib.h>
#include <stdint.h>

typedef struct _heap _heap;
struct _heap {
   _heap* pPrev;
   const char* loc;
   uint64_t buf;
};

typedef struct _allocator {
   _heap* pheap;
   uint8_t* stack;
   uint8_t* stackEnd;
   int nSize;
} _allocator;

_ATTRIBUTE_UNUSED
static __inline int _heap_alloc(_heap** ppa, const char* loc, int size, void** ppbuf) {
   _heap* pn = 0;
   pn = malloc(size + sizeof(_heap) - sizeof(uint64_t));
   if(pn != 0) {
      pn->pPrev = *ppa;
      pn->loc = loc;
      *ppa = pn;
      *ppbuf = (void*)&(pn->buf);
      return 0;
   } else {
      return -1;
   }
}
#define _ALIGN_SIZE(x, y) (((x) + (y-1)) & ~(y-1))

_ATTRIBUTE_UNUSED
static __inline int _allocator_alloc(_allocator* me,
                                    const char* loc,
                                    int size,
                                    unsigned int al,
                                    void** ppbuf) {
   if(size < 0) {
      return -1;
   } else if (size == 0) {
      *ppbuf = 0;
      return 0;
   }
   if((_ALIGN_SIZE((uintptr_t)me->stackEnd, al) + size) < (uintptr_t)me->stack + me->nSize) {
      *ppbuf = (uint8_t*)_ALIGN_SIZE((uintptr_t)me->stackEnd, al);
      me->stackEnd = (uint8_t*)_ALIGN_SIZE((uintptr_t)me->stackEnd, al) + size;
      return 0;
   } else {
      return _heap_alloc(&me->pheap, loc, size, ppbuf);
   }
}

_ATTRIBUTE_UNUSED
static __inline void _allocator_deinit(_allocator* me) {
   _heap* pa = me->pheap;
   while(pa != 0) {
      _heap* pn = pa;
      const char* loc = pn->loc;
      (void)loc;
      pa = pn->pPrev;
      free(pn);
   }
}

_ATTRIBUTE_UNUSED
static __inline void _allocator_init(_allocator* me, uint8_t* stack, int stackSize) {
   me->stack =  stack;
   me->stackEnd =  stack + stackSize;
   me->nSize = stackSize;
   me->pheap = 0;
}


#endif // _ALLOCATOR_H

#ifndef SLIM_H
#define SLIM_H

#include <stdint.h>

//a C data structure for the idl types that can be used to implement
//static and dynamic language bindings fairly efficiently.
//
//the goal is to have a minimal ROM and RAM footprint and without
//doing too many allocations.  A good way to package these things seemed
//like the module boundary, so all the idls within  one module can share
//all the type references.


#define PARAMETER_IN       0x0
#define PARAMETER_OUT      0x1
#define PARAMETER_INOUT    0x2
#define PARAMETER_ROUT     0x3
#define PARAMETER_INROUT   0x4

//the types that we get from idl
#define TYPE_OBJECT             0x0
#define TYPE_INTERFACE          0x1
#define TYPE_PRIMITIVE          0x2
#define TYPE_ENUM               0x3
#define TYPE_STRING             0x4
#define TYPE_WSTRING            0x5
#define TYPE_STRUCTURE          0x6
#define TYPE_UNION              0x7
#define TYPE_ARRAY              0x8
#define TYPE_SEQUENCE           0x9

//these require the pack/unpack to recurse
//so it's a hint to those languages that can optimize in cases where
//recursion isn't necessary.
#define TYPE_COMPLEX_STRUCTURE  (0x10 | TYPE_STRUCTURE)
#define TYPE_COMPLEX_UNION      (0x10 | TYPE_UNION)
#define TYPE_COMPLEX_ARRAY      (0x10 | TYPE_ARRAY)
#define TYPE_COMPLEX_SEQUENCE   (0x10 | TYPE_SEQUENCE)


typedef struct Type Type;

#define INHERIT_TYPE\
   int32_t nativeSize;                /*in the simple case its the same as wire size and alignment*/\
   union {\
      struct {\
         const uintptr_t         p1;\
         const uintptr_t         p2;\
      } _cast;\
      struct {\
         uint32_t  iid;\
         uint32_t  bNotNil;\
      } object;\
      struct {\
         const Type  *arrayType;\
         int32_t      nItems;\
      } array;\
      struct {\
         const Type *seqType;\
         int32_t      nMaxLen;\
      } seqSimple; \
      struct {\
         uint32_t bFloating;\
         uint32_t bSigned;\
      } prim; \
      const SequenceType* seqComplex;\
      const UnionType  *unionType;\
      const StructType *structType;\
      int32_t         stringMaxLen;\
      uint8_t        bInterfaceNotNil;\
   } param;\
   uint8_t    type;\
   uint8_t    nativeAlignment\

typedef struct UnionType UnionType;
typedef struct StructType StructType;
typedef struct SequenceType SequenceType;
struct Type {
   INHERIT_TYPE;
};

struct SequenceType {
   const Type *         seqType;
   uint32_t               nMaxLen;
   uint32_t               inSize;
   uint32_t               routSizePrimIn;
   uint32_t               routSizePrimROut;
};

//byte offset from the start of the case values for
//this unions case value array.  it MUST be aligned
//at the alignment requrements for the descriptor
//
//if negative it means that the unions cases are
//simple enumerators, so the value read from the descriptor
//can be used directly to find the correct case
typedef union CaseValuePtr CaseValuePtr;
union CaseValuePtr {
   const uint8_t*   value8s;
   const uint16_t*  value16s;
   const uint32_t*  value32s;
   const uint64_t*  value64s;
};

//these are only used in complex cases
//so I pulled them out of the type definition as references to make
//the type smaller
struct UnionType {
   const Type           *descriptor;
   uint32_t               nCases;
   const CaseValuePtr   caseValues;
   const Type * const   *cases;
   int32_t               inSize;
   int32_t               routSizePrimIn;
   int32_t               routSizePrimROut;
   uint8_t                inAlignment;
   uint8_t                routAlignmentPrimIn;
   uint8_t                routAlignmentPrimROut;
   uint8_t                inCaseAlignment;
   uint8_t                routCaseAlignmentPrimIn;
   uint8_t                routCaseAlignmentPrimROut;
   uint8_t                nativeCaseAlignment;
   uint8_t              bDefaultCase;
};

struct StructType {
   uint32_t               nMembers;
   const Type * const   *members;
   int32_t               inSize;
   int32_t               routSizePrimIn;
   int32_t               routSizePrimROut;
   uint8_t                inAlignment;
   uint8_t                routAlignmentPrimIn;
   uint8_t                routAlignmentPrimROut;
};

typedef struct Parameter Parameter;
struct Parameter {
   INHERIT_TYPE;
   uint8_t    mode;
   uint8_t  bNotNil;
};

#define SLIM_IFPTR32(is32,is64) (sizeof(uintptr_t) == 4 ? (is32) : (is64))
#define SLIM_SCALARS_IS_DYNAMIC(u) (((u) & 0x00ffffff) == 0x00ffffff)

typedef struct Method Method;
struct Method {
   uint32_t                    uScalars;            //no method index
   int32_t                     primInSize;
   int32_t                     primROutSize;
   int                         maxArgs;
   int                         numParams;
   const Parameter * const     *params;
   uint8_t                       primInAlignment;
   uint8_t                       primROutAlignment;
};

typedef struct Interface Interface;

struct Interface {
   int                            nMethods;
   const Method  * const          *methodArray;
   int                            nIIds;
   const uint32_t                   *iids;
   const uint16_t*                  methodStringArray;
   const uint16_t*                  methodStrings;
   const char*                    strings;
};


#endif //SLIM_H


#ifndef _HEXAGON_NN_SLIM_H
#define _HEXAGON_NN_SLIM_H
#include "remote.h"
#include <stdint.h>

#ifndef __QAIC_SLIM
#define __QAIC_SLIM(ff) ff
#endif
#ifndef __QAIC_SLIM_EXPORT
#define __QAIC_SLIM_EXPORT
#endif

static const Type types[14];
static const Type* const typeArrays[14] = {&(types[1]),&(types[1]),&(types[1]),&(types[1]),&(types[13]),&(types[1]),&(types[1]),&(types[1]),&(types[8]),&(types[1]),&(types[9]),&(types[10]),&(types[1]),&(types[3])};
static const StructType structTypes[6] = {{0x1,&(typeArrays[10]),0x4,0x0,0x4,0x4,0x1,0x4},{0x2,&(typeArrays[0]),0x8,0x0,0x8,0x4,0x1,0x4},{0x2,&(typeArrays[12]),0x104,0x0,0x104,0x4,0x1,0x4},{0x5,&(typeArrays[7]),0x30,0x0,0x30,0x4,0x1,0x4},{0x4,&(typeArrays[0]),0x10,0x0,0x10,0x4,0x1,0x4},{0x7,&(typeArrays[0]),0x1c,0x4,0x18,0x4,0x4,0x4}};
static const SequenceType sequenceTypes[1] = {{&(types[12]),0x0,0x1c,0x4,0x18}};
static const Type types[14] = {{0x8,{{(const uintptr_t)&(structTypes[1]),0}}, 6,0x4},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4},{0x104,{{(const uintptr_t)&(structTypes[2]),0}}, 6,0x4},{0x100,{{(const uintptr_t)&(types[4]),(const uintptr_t)0x100}}, 8,0x1},{0x1,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x1},{0x1,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x1},{0x8,{{(const uintptr_t)&(structTypes[1]),0}}, 6,0x4},{0x30,{{(const uintptr_t)&(structTypes[3]),0}}, 6,0x4},{0x20,{{(const uintptr_t)&(types[1]),(const uintptr_t)0x8}}, 8,0x4},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4},{0x10,{{(const uintptr_t)&(structTypes[4]),0}}, 6,0x4},{SLIM_IFPTR32(0x20,0x28),{{(const uintptr_t)&(structTypes[5]),0}}, 22,SLIM_IFPTR32(0x4,0x8)},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[5]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8)}};
static const Parameter parameters[22] = {{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[0]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[2]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),0,0},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4,0,0},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4,3,0},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4,3,0},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4,0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[5]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),4,0},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4,0,0},{0x4,{{0,0}}, 3,0x4,0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[6]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[7]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[5]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[5]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),3,0},{0x4,{{0,0}}, 3,0x4,0,0},{0x4,{{0,0}}, 3,0x4,0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[11]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),3,0},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4,3,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)0x0,0}}, 4,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)0x0,0}}, 4,SLIM_IFPTR32(0x4,0x8),3,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(sequenceTypes[0]),0}}, 25,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(sequenceTypes[0]),0}}, 25,SLIM_IFPTR32(0x4,0x8),3,0},{0x4,{{(const uintptr_t)&(structTypes[0]),0}}, 6,0x4,0,0}};
static const Parameter* const parameterArrays[89] = {(&(parameters[2])),(&(parameters[7])),(&(parameters[7])),(&(parameters[7])),(&(parameters[7])),(&(parameters[11])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[12])),(&(parameters[3])),(&(parameters[2])),(&(parameters[7])),(&(parameters[5])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[12])),(&(parameters[3])),(&(parameters[2])),(&(parameters[7])),(&(parameters[5])),(&(parameters[7])),(&(parameters[7])),(&(parameters[7])),(&(parameters[7])),(&(parameters[11])),(&(parameters[2])),(&(parameters[7])),(&(parameters[7])),(&(parameters[7])),(&(parameters[7])),(&(parameters[7])),(&(parameters[7])),(&(parameters[2])),(&(parameters[7])),(&(parameters[7])),(&(parameters[7])),(&(parameters[7])),(&(parameters[7])),(&(parameters[11])),(&(parameters[2])),(&(parameters[7])),(&(parameters[7])),(&(parameters[8])),(&(parameters[9])),(&(parameters[10])),(&(parameters[2])),(&(parameters[7])),(&(parameters[5])),(&(parameters[11])),(&(parameters[2])),(&(parameters[7])),(&(parameters[11])),(&(parameters[7])),(&(parameters[2])),(&(parameters[17])),(&(parameters[5])),(&(parameters[2])),(&(parameters[2])),(&(parameters[3])),(&(parameters[2])),(&(parameters[19])),(&(parameters[20])),(&(parameters[2])),(&(parameters[3])),(&(parameters[3])),(&(parameters[2])),(&(parameters[15])),(&(parameters[3])),(&(parameters[13])),(&(parameters[14])),(&(parameters[7])),(&(parameters[2])),(&(parameters[0])),(&(parameters[1])),(&(parameters[4])),(&(parameters[21])),(&(parameters[7])),(&(parameters[18])),(&(parameters[17])),(&(parameters[3])),(&(parameters[2])),(&(parameters[6])),(&(parameters[2])),(&(parameters[5])),(&(parameters[16]))};
static const Method methods[31] = {{REMOTE_SCALARS_MAKEX(0,0,0x0,0x0,0x0,0x0),0x0,0x0,0,0,0,0x0,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x3,0x0,0x0,0x0),0x8,0x0,4,2,(&(parameterArrays[76])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x3,0x0,0x0,0x0),0xc,0x0,5,3,(&(parameterArrays[75])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x0,0x1,0x0,0x0),0x0,0x8,2,2,(&(parameterArrays[6])),0x1,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x0,0x1,0x0,0x0),0x0,0x4,1,1,(&(parameterArrays[78])),0x1,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x0,0x0,0x0),0x8,0x0,2,2,(&(parameterArrays[86])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x1,0x0,0x0),0xc,0x0,4,2,(&(parameterArrays[84])),0x4,0x1},{REMOTE_SCALARS_MAKEX(0,0,0x3,0x0,0x0,0x0),0x18,0x0,8,6,(&(parameterArrays[43])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x0),0x1c,0x0,8,7,(&(parameterArrays[36])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x0,0x0,0x0),0x1c,0x0,7,7,(&(parameterArrays[29])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x0),0x10,0x0,5,4,(&(parameterArrays[53])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x0,0x0,0x0),0x4,0x0,1,1,(&(parameterArrays[0])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x2,0x0,0x0),0x1c,0x14,15,12,(&(parameterArrays[0])),0x4,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x2,0x0,0x0),0x10,0x14,11,9,(&(parameterArrays[12])),0x4,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x0),0x20,0x0,9,8,(&(parameterArrays[21])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x0),0x10,0x0,5,4,(&(parameterArrays[49])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x0,0x0,0x0),0x4,0x0,1,1,(&(parameterArrays[1])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x0,0x0,0x0),0xc,0x0,3,3,(&(parameterArrays[72])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x2,0x0,0x0),0x8,0x4,5,3,(&(parameterArrays[69])),0x4,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x0,0x0,0x0),0x8,0x0,2,2,(&(parameterArrays[0])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x1,0x0,0x0),0x4,0x8,3,3,(&(parameterArrays[66])),0x4,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x0,0x1,0x0,0x0),0x0,0x4,1,1,(&(parameterArrays[88])),0x1,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x1,0x0,0x0),0x4,0x4,2,2,(&(parameterArrays[82])),0x4,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x1,0x0,0x0),0x8,0x0,4,2,(&(parameterArrays[80])),0x4,0x1},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x1,0x0,0x0),0x4,0x4,2,2,(&(parameterArrays[61])),0x4,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x0),0x4,0x0,2,1,(&(parameterArrays[5])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,255,255,15,15),0xc,0x0,6,3,(&(parameterArrays[63])),0x4,0x1},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x1,0x0,0x0),0x4,0x4,2,2,(&(parameterArrays[78])),0x4,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x1,0x0,0x0),0x8,0x4,3,3,(&(parameterArrays[60])),0x4,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x0,0x0,0x0),0x4,0x0,1,1,(&(parameterArrays[14])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x0),0xc,0x0,3,3,(&(parameterArrays[57])),0x4,0x0}};
static const Method* const methodArrays[36] = {&(methods[0]),&(methods[1]),&(methods[2]),&(methods[3]),&(methods[4]),&(methods[5]),&(methods[6]),&(methods[6]),&(methods[7]),&(methods[8]),&(methods[9]),&(methods[10]),&(methods[11]),&(methods[12]),&(methods[11]),&(methods[13]),&(methods[14]),&(methods[15]),&(methods[16]),&(methods[17]),&(methods[18]),&(methods[19]),&(methods[20]),&(methods[21]),&(methods[22]),&(methods[23]),&(methods[24]),&(methods[0]),&(methods[21]),&(methods[25]),&(methods[26]),&(methods[27]),&(methods[28]),&(methods[20]),&(methods[29]),&(methods[30])};
static const char strings[1049] = "GetHexagonBinaryVersion\0append_empty_const_node\0multi_execution_cycles\0get_num_nodes_in_graph\0last_execution_cycles\0set_powersave_details\0set_powersave_level\0variable_write_flat\0populate_const_node\0config_with_options\0fastrpc_shell_addr\0append_const_node\0set_graph_option\0set_debug_level\0libhexagon_addr\0init_with_info\0data_valid_len\0reset_perfinfo\0variable_write\0get_dsp_offset\0string_options\0op_id_to_name\0op_name_to_id\0variable_read\0target_offset\0get_nodetype\0disable_dcvs\0get_perfinfo\0output_index\0data_len_out\0graph_config\0uint_options\0execute_new\0batches_out\0zero_offset\0elementsize\0append_node\0string_data\0counter_hi\0counter_lo\0executions\0height_out\0batches_in\0output_idx\0uint_value\0get_power\0node_type\0num_nodes\0cycles_hi\0cycles_lo\0depth_out\0width_out\0height_in\0max_sizes\0operation\0option_id\0graph_id\0priority\0PrintLog\0info_out\0teardown\0data_out\0depth_in\0width_in\0stepsize\0snpprint\0version\0n_items\0latency\0data_in\0execute\0prepare\0batches\0outputs\0padding\0node_id\0unused\0corner\0height\0src_id\0inputs\0getlog\0event\0depth\0width\0rank\0init\0ver\0buf\0";
static const uint16_t methodStrings[179] = {541,419,998,938,984,1024,1018,608,319,970,946,938,984,1024,1018,608,319,970,589,419,962,780,954,998,991,668,946,1030,770,577,565,872,922,419,657,760,863,854,914,553,646,750,740,845,502,422,419,962,489,553,646,750,740,845,502,349,419,962,489,938,984,1024,1018,914,476,419,827,962,635,624,613,898,24,419,962,938,984,1024,1018,584,237,419,962,938,984,1024,1018,608,515,419,528,790,679,379,790,601,198,528,790,679,379,790,601,158,419,962,489,914,178,419,962,608,436,255,419,403,684,48,419,730,720,450,800,962,700,304,526,314,809,94,419,730,720,116,977,471,906,71,419,710,394,962,403,408,403,962,334,419,1012,1005,419,1044,881,419,1044,272,419,152,364,288,218,690,458,818,1044,0,1040,890,1040,138,152,836,419,930,419,1035,526,463,521};
static const uint16_t methodStringsArrays[36] = {178,96,88,158,175,155,152,149,18,80,72,108,173,32,171,45,55,103,169,133,64,146,129,167,143,140,137,177,165,163,0,125,121,117,161,113};
__QAIC_SLIM_EXPORT const Interface __QAIC_SLIM(hexagon_nn_slim) = {36,&(methodArrays[0]),0,0,&(methodStringsArrays [0]),methodStrings,strings};
#endif //_HEXAGON_NN_SLIM_H
extern int adsp_mmap_fd_getinfo(int, uint32_t *);
#ifdef __cplusplus
extern "C" {
#endif
static __inline int _skel_method(int (*_pfn)(hexagon_nn_nn_id, const char*, int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   uint32_t _in0[1];
   hexagon_nn_nn_id _in1[1];
   const char* _in2[1];
   int _in2Len[1];
   int _in3[1];
   uint32_t* _primIn;
   remote_arg* _praIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((2 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 16);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_in2Len, 0, _primIn, 8, 4);
   _praIn = (_pra + 1);
   _ASSERT(_nErr, ((_praIn[0].buf.nLen / 1)) >= (size_t)(_in2Len[0]));
   _in2[0] = _praIn[0].buf.pv;
   _ASSERT(_nErr, (_in2Len[0] > 0) && (_in2[0][(_in2Len[0] - 1)] == 0));
   _COPY(_in3, 0, _primIn, 12, 4);
   _TRY(_nErr, _pfn(*_in1, *_in2, *_in3));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_1(int (*_pfn)(int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   uint32_t _in0[1];
   int _in1[1];
   uint32_t* _primIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((1 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 8);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _TRY(_nErr, _pfn(*_in1));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_2(int (*_pfn)(hexagon_nn_nn_id, unsigned int*, unsigned int*), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   uint32_t _in0[1];
   hexagon_nn_nn_id _in1[1];
   unsigned int _rout2[1];
   unsigned int _rout3[1];
   uint32_t* _primIn;
   int _numIn[1];
   uint32_t* _primROut;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((1 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 8);
   _primIn = _pra[0].buf.pv;
   _ASSERT(_nErr, _pra[(_numIn[0] + 1)].buf.nLen >= 8);
   _primROut = _pra[(_numIn[0] + 1)].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _TRY(_nErr, _pfn(*_in1, _rout2, _rout3));
   _COPY(_primROut, 0, _rout2, 0, 4);
   _COPY(_primROut, 4, _rout3, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_3(int (*_pfn)(hexagon_nn_nn_id, hexagon_nn_nn_id, unsigned int*), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   uint32_t _in0[1];
   hexagon_nn_nn_id _in1[1];
   hexagon_nn_nn_id _in2[1];
   unsigned int _rout3[1];
   uint32_t* _primIn;
   int _numIn[1];
   uint32_t* _primROut;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((1 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 12);
   _primIn = _pra[0].buf.pv;
   _ASSERT(_nErr, _pra[(_numIn[0] + 1)].buf.nLen >= 4);
   _primROut = _pra[(_numIn[0] + 1)].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_in2, 0, _primIn, 8, 4);
   _TRY(_nErr, _pfn(*_in1, *_in2, _rout3));
   _COPY(_primROut, 0, _rout3, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_4(int (*_pfn)(hexagon_nn_nn_id*, const hexagon_nn_initinfo*), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   uint32_t _in0[1];
   hexagon_nn_nn_id _rout1[1];
   const hexagon_nn_initinfo _in2[1];
   uint32_t* _primIn;
   int _numIn[1];
   uint32_t* _primROut;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((1 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 8);
   _primIn = _pra[0].buf.pv;
   _ASSERT(_nErr, _pra[(_numIn[0] + 1)].buf.nLen >= 4);
   _primROut = _pra[(_numIn[0] + 1)].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in2, 0, _primIn, 4, 4);
   _TRY(_nErr, _pfn(_rout1, _in2));
   _COPY(_primROut, 0, _rout1, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_invoke(uint32_t _mid, uint32_t _sc, remote_arg* _pra) {
   switch(_mid)
   {
      case 31:
      return _skel_method_4((void*)__QAIC_IMPL(hexagon_nn_init_with_info), _sc, _pra);
      case 32:
      return _skel_method_3((void*)__QAIC_IMPL(hexagon_nn_get_nodetype), _sc, _pra);
      case 33:
      return _skel_method_2((void*)__QAIC_IMPL(hexagon_nn_multi_execution_cycles), _sc, _pra);
      case 34:
      return _skel_method_1((void*)__QAIC_IMPL(hexagon_nn_get_power), _sc, _pra);
      case 35:
      return _skel_method((void*)__QAIC_IMPL(hexagon_nn_set_graph_option), _sc, _pra);
   }
   return AEE_EUNSUPPORTED;
}
static __inline int _skel_pack(remote_arg* _praROutPost, remote_arg* _ppraROutPost[1], void* _primROut, unsigned int _rout0[1], unsigned int _rout1[1], unsigned int _rout2[1], unsigned int _rout3[1], unsigned char* _rout4[1], int _rout4Len[1], unsigned int _rout5[1], unsigned int _rout6[1]) {
   int _nErr = 0;
   remote_arg* _praROutPostStart = _praROutPost;
   remote_arg** _ppraROutPostStart = _ppraROutPost;
   _ppraROutPost = &_praROutPost;
   _COPY(_primROut, 0, _rout0, 0, 4);
   _COPY(_primROut, 4, _rout1, 0, 4);
   _COPY(_primROut, 8, _rout2, 0, 4);
   _COPY(_primROut, 12, _rout3, 0, 4);
   _COPY(_primROut, 16, _rout5, 0, 4);
   _COPY(_primROut, 20, _rout6, 0, 4);
   _ppraROutPostStart[0] += (_praROutPost - _praROutPostStart) +1;
   return _nErr;
}
static __inline int _skel_pack_1(remote_arg* _praROutPost, remote_arg* _ppraROutPost[1], void* _primROut, hexagon_nn_tensordef _rout0[SLIM_IFPTR32(8, 5)]) {
   int _nErr = 0;
   remote_arg* _praROutPostStart = _praROutPost;
   remote_arg** _ppraROutPostStart = _ppraROutPost;
   _ppraROutPost = &_praROutPost;
   _TRY(_nErr, _skel_pack((_praROutPost + 0), _ppraROutPost, ((char*)_primROut + 0), (unsigned int*)&(((uint32_t*)_rout0)[0]), (unsigned int*)&(((uint32_t*)_rout0)[1]), (unsigned int*)&(((uint32_t*)_rout0)[2]), (unsigned int*)&(((uint32_t*)_rout0)[3]), SLIM_IFPTR32((unsigned char**)&(((uint32_t*)_rout0)[4]), (unsigned char**)&(((uint64_t*)_rout0)[2])), SLIM_IFPTR32((int*)&(((uint32_t*)_rout0)[5]), (int*)&(((uint32_t*)_rout0)[6])), SLIM_IFPTR32((unsigned int*)&(((uint32_t*)_rout0)[6]), (unsigned int*)&(((uint32_t*)_rout0)[7])), SLIM_IFPTR32((unsigned int*)&(((uint32_t*)_rout0)[7]), (unsigned int*)&(((uint32_t*)_rout0)[8]))));
   _ppraROutPostStart[0] += (_praROutPost - _praROutPostStart) +0;
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_pack_2(remote_arg* _praROutPost, remote_arg* _ppraROutPost[1], void* _primROut, const hexagon_nn_tensordef _in0[SLIM_IFPTR32(8, 5)]) {
   int _nErr = 0;
   remote_arg* _praROutPostStart = _praROutPost;
   remote_arg** _ppraROutPostStart = _ppraROutPost;
   _ppraROutPost = &_praROutPost;
   _ppraROutPostStart[0] += (_praROutPost - _praROutPostStart) +0;
   return _nErr;
}
static __inline int _skel_unpack(_allocator* _al, remote_arg* _praIn, remote_arg* _ppraIn[1], remote_arg* _praROut, remote_arg* _ppraROut[1], remote_arg* _praHIn, remote_arg* _ppraHIn[1], remote_arg* _praHROut, remote_arg* _ppraHROut[1], void* _primIn, void* _primROut, unsigned int _rout0[1], unsigned int _rout1[1], unsigned int _rout2[1], unsigned int _rout3[1], unsigned char* _rout4[1], int _rout4Len[1], unsigned int _rout5[1], unsigned int _rout6[1]) {
   int _nErr = 0;
   remote_arg* _praInStart = _praIn;
   remote_arg** _ppraInStart = _ppraIn;
   remote_arg* _praROutStart = _praROut;
   remote_arg** _ppraROutStart = _ppraROut;
   _ppraIn = &_praIn;
   _ppraROut = &_praROut;
   _COPY(_rout4Len, 0, _primIn, 0, 4);
   _ASSERT(_nErr, ((_praROut[0].buf.nLen / 1)) >= (size_t)(_rout4Len[0]));
   _rout4[0] = _praROut[0].buf.pv;
   _ppraInStart[0] += (_praIn - _praInStart) + 0;
   _ppraROutStart[0] += (_praROut - _praROutStart) +1;
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_unpack_1(_allocator* _al, remote_arg* _praIn, remote_arg* _ppraIn[1], remote_arg* _praROut, remote_arg* _ppraROut[1], remote_arg* _praHIn, remote_arg* _ppraHIn[1], remote_arg* _praHROut, remote_arg* _ppraHROut[1], void* _primIn, void* _primROut, hexagon_nn_tensordef _rout0[SLIM_IFPTR32(8, 5)]) {
   int _nErr = 0;
   remote_arg* _praInStart = _praIn;
   remote_arg** _ppraInStart = _ppraIn;
   remote_arg* _praROutStart = _praROut;
   remote_arg** _ppraROutStart = _ppraROut;
   _ppraIn = &_praIn;
   _ppraROut = &_praROut;
   _TRY(_nErr, _skel_unpack(_al, (_praIn + 0), _ppraIn, (_praROut + 0), _ppraROut, _praHIn, _ppraHIn, _praHROut, _ppraHROut, ((char*)_primIn + 0), ((char*)_primROut + 0), (unsigned int*)&(((uint32_t*)_rout0)[0]), (unsigned int*)&(((uint32_t*)_rout0)[1]), (unsigned int*)&(((uint32_t*)_rout0)[2]), (unsigned int*)&(((uint32_t*)_rout0)[3]), SLIM_IFPTR32((unsigned char**)&(((uint32_t*)_rout0)[4]), (unsigned char**)&(((uint64_t*)_rout0)[2])), SLIM_IFPTR32((int*)&(((uint32_t*)_rout0)[5]), (int*)&(((uint32_t*)_rout0)[6])), SLIM_IFPTR32((unsigned int*)&(((uint32_t*)_rout0)[6]), (unsigned int*)&(((uint32_t*)_rout0)[7])), SLIM_IFPTR32((unsigned int*)&(((uint32_t*)_rout0)[7]), (unsigned int*)&(((uint32_t*)_rout0)[8]))));
   _ppraInStart[0] += (_praIn - _praInStart) + 0;
   _ppraROutStart[0] += (_praROut - _praROutStart) +0;
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_unpack_2(_allocator* _al, remote_arg* _praIn, remote_arg* _ppraIn[1], remote_arg* _praROut, remote_arg* _ppraROut[1], remote_arg* _praHIn, remote_arg* _ppraHIn[1], remote_arg* _praHROut, remote_arg* _ppraHROut[1], void* _primIn, void* _primROut, unsigned int _in0[1], unsigned int _in1[1], unsigned int _in2[1], unsigned int _in3[1], const unsigned char* _in4[1], int _in4Len[1], unsigned int _in5[1], unsigned int _in6[1]) {
   int _nErr = 0;
   remote_arg* _praInStart = _praIn;
   remote_arg** _ppraInStart = _ppraIn;
   remote_arg* _praROutStart = _praROut;
   remote_arg** _ppraROutStart = _ppraROut;
   _ppraIn = &_praIn;
   _ppraROut = &_praROut;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_in2, 0, _primIn, 8, 4);
   _COPY(_in3, 0, _primIn, 12, 4);
   _COPY(_in4Len, 0, _primIn, 16, 4);
   _ASSERT(_nErr, ((_praIn[0].buf.nLen / 1)) >= (size_t)(_in4Len[0]));
   _in4[0] = _praIn[0].buf.pv;
   _COPY(_in5, 0, _primIn, 20, 4);
   _COPY(_in6, 0, _primIn, 24, 4);
   _ppraInStart[0] += (_praIn - _praInStart) + 1;
   _ppraROutStart[0] += (_praROut - _praROutStart) +0;
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_unpack_3(_allocator* _al, remote_arg* _praIn, remote_arg* _ppraIn[1], remote_arg* _praROut, remote_arg* _ppraROut[1], remote_arg* _praHIn, remote_arg* _ppraHIn[1], remote_arg* _praHROut, remote_arg* _ppraHROut[1], void* _primIn, void* _primROut, const hexagon_nn_tensordef _in0[SLIM_IFPTR32(8, 5)]) {
   int _nErr = 0;
   remote_arg* _praInStart = _praIn;
   remote_arg** _ppraInStart = _ppraIn;
   remote_arg* _praROutStart = _praROut;
   remote_arg** _ppraROutStart = _ppraROut;
   _ppraIn = &_praIn;
   _ppraROut = &_praROut;
   _TRY(_nErr, _skel_unpack_2(_al, (_praIn + 0), _ppraIn, (_praROut + 0), _ppraROut, _praHIn, _ppraHIn, _praHROut, _ppraHROut, ((char*)_primIn + 0), 0, (unsigned int*)&(((uint32_t*)_in0)[0]), (unsigned int*)&(((uint32_t*)_in0)[1]), (unsigned int*)&(((uint32_t*)_in0)[2]), (unsigned int*)&(((uint32_t*)_in0)[3]), SLIM_IFPTR32((const unsigned char**)&(((uint32_t*)_in0)[4]), (const unsigned char**)&(((uint64_t*)_in0)[2])), SLIM_IFPTR32((int*)&(((uint32_t*)_in0)[5]), (int*)&(((uint32_t*)_in0)[6])), SLIM_IFPTR32((unsigned int*)&(((uint32_t*)_in0)[6]), (unsigned int*)&(((uint32_t*)_in0)[7])), SLIM_IFPTR32((unsigned int*)&(((uint32_t*)_in0)[7]), (unsigned int*)&(((uint32_t*)_in0)[8]))));
   _ppraInStart[0] += (_praIn - _praInStart) + 0;
   _ppraROutStart[0] += (_praROut - _praROutStart) +0;
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_5(int (*_pfn)(hexagon_nn_nn_id, const hexagon_nn_tensordef*, int, hexagon_nn_tensordef*, int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   const hexagon_nn_tensordef* _in1[1];
   int _in1Len[1];
   hexagon_nn_tensordef* _rout2[1];
   int _rout2Len[1];
   uint32_t* _primIn;
   int _numIn[1];
   int _numInH[1];
   int _numROut[1];
   remote_arg* _praIn;
   remote_arg* _praROut;
   remote_arg* _praROutPost;
   remote_arg** _ppraROutPost = &_praROutPost;
   _allocator _al[1] = {{0}};
   remote_arg** _ppraIn = &_praIn;
   remote_arg** _ppraROut = &_praROut;
   remote_arg* _praHIn = 0;
   remote_arg** _ppraHIn = &_praHIn;
   remote_arg* _praHROut = 0;
   remote_arg** _ppraHROut = &_praHROut;
   char* _seq_primIn1;
   char* _seq_nat1;
   int _ii;
   int _nErr = 0;
   char* _seq_primIn2;
   char* _seq_primROut2;
   char* _seq_nat2;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((3 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 12);
   _primIn = _pra[0].buf.pv;
   _numInH[0] = REMOTE_SCALARS_INHANDLES(_sc);
   _numROut[0] = REMOTE_SCALARS_OUTBUFS(_sc);
   _praIn = (_pra + 1);
   _praROut = (_praIn + _numIn[0] + 0);
   _praROutPost = _praROut;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1Len, 0, _primIn, 4, 4);
   _allocator_init(_al, 0, 0);
   if(_praHIn == 0)
   {
      _praHIn = ((_praROut + _numROut[0]) + 0);
   }
   if(_praHROut == 0)
      (_praHROut = _praHIn + _numInH[0] + 0);
   _ASSERT(_nErr, ((_praIn[0].buf.nLen / 28)) >= (size_t)(_in1Len[0]));
   _ALLOCATE(_nErr, _al, (_in1Len[0] * SLIM_IFPTR32(32, 40)), SLIM_IFPTR32(4, 8), _in1[0]);
   for(_ii = 0, _seq_primIn1 = (char*)_praIn[0].buf.pv, _seq_nat1 = (char*)_in1[0];_ii < (int)_in1Len[0];++_ii, _seq_primIn1 = (_seq_primIn1 + 28), _seq_nat1 = (_seq_nat1 + SLIM_IFPTR32(32, 40)))
   {
      _TRY(_nErr, _skel_unpack_3(_al, (_praIn + 1), _ppraIn, (_praROut + 0), _ppraROut, _praHIn, _ppraHIn, _praHROut, _ppraHROut, _seq_primIn1, 0, SLIM_IFPTR32((const hexagon_nn_tensordef*)&(((uint32_t*)_seq_nat1)[0]), (const hexagon_nn_tensordef*)&(((uint64_t*)_seq_nat1)[0]))));
   }
   _COPY(_rout2Len, 0, _primIn, 8, 4);
   _ASSERT(_nErr, ((_praIn[1].buf.nLen / 4)) >= (size_t)(_rout2Len[0]));
   _ASSERT(_nErr, ((_praROut[0].buf.nLen / 24)) >= (size_t)(_rout2Len[0]));
   _ALLOCATE(_nErr, _al, (_rout2Len[0] * SLIM_IFPTR32(32, 40)), SLIM_IFPTR32(4, 8), _rout2[0]);
   for(_ii = 0, _seq_primIn2 = (char*)_praIn[1].buf.pv, _seq_primROut2 = (char*)_praROut[0].buf.pv, _seq_nat2 = (char*)_rout2[0];_ii < (int)_rout2Len[0];++_ii, _seq_primIn2 = (_seq_primIn2 + 4), _seq_primROut2 = (_seq_primROut2 + 24), _seq_nat2 = (_seq_nat2 + SLIM_IFPTR32(32, 40)))
   {
      _TRY(_nErr, _skel_unpack_1(_al, (_praIn + 2), _ppraIn, (_praROut + 1), _ppraROut, _praHIn, _ppraHIn, _praHROut, _ppraHROut, _seq_primIn2, _seq_primROut2, SLIM_IFPTR32((hexagon_nn_tensordef*)&(((uint32_t*)_seq_nat2)[0]), (hexagon_nn_tensordef*)&(((uint64_t*)_seq_nat2)[0]))));
   }
   _TRY(_nErr, _pfn(*_in0, *_in1, *_in1Len, *_rout2, *_rout2Len));
   for(_ii = 0, _seq_nat1 = (char*)_in1[0];_ii < (int)_in1Len[0];++_ii, _seq_nat1 = (_seq_nat1 + SLIM_IFPTR32(32, 40)))
   {
      _TRY(_nErr, _skel_pack_2((_praROutPost + 0), _ppraROutPost, 0, SLIM_IFPTR32((const hexagon_nn_tensordef*)&(((uint32_t*)_seq_nat1)[0]), (const hexagon_nn_tensordef*)&(((uint64_t*)_seq_nat1)[0]))));
   }
   for(_ii = 0, _seq_primROut2 = (char*)_praROutPost[0].buf.pv, _seq_nat2 = (char*)_rout2[0];_ii < (int)_rout2Len[0];++_ii, _seq_primROut2 = (_seq_primROut2 + 24), _seq_nat2 = (_seq_nat2 + SLIM_IFPTR32(32, 40)))
   {
      _TRY(_nErr, _skel_pack_1((_praROutPost + 1), _ppraROutPost, _seq_primROut2, SLIM_IFPTR32((hexagon_nn_tensordef*)&(((uint32_t*)_seq_nat2)[0]), (hexagon_nn_tensordef*)&(((uint64_t*)_seq_nat2)[0]))));
   }
   _CATCH(_nErr) {}
   _allocator_deinit(_al);
   return _nErr;
}
static __inline int _skel_method_6(int (*_pfn)(const unsigned char*, int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   const unsigned char* _in0[1];
   int _in0Len[1];
   uint32_t* _primIn;
   remote_arg* _praIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((2 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 4);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0Len, 0, _primIn, 0, 4);
   _praIn = (_pra + 1);
   _ASSERT(_nErr, ((_praIn[0].buf.nLen / 1)) >= (size_t)(_in0Len[0]));
   _in0[0] = _praIn[0].buf.pv;
   _TRY(_nErr, _pfn(*_in0, *_in0Len));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_7(int (*_pfn)(int*), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   int _rout0[1];
   uint32_t* _primROut;
   int _numIn[1];
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((0 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 0);
   _ASSERT(_nErr, _pra[(_numIn[0] + 0)].buf.nLen >= 4);
   _primROut = _pra[(_numIn[0] + 0)].buf.pv;
   _TRY(_nErr, _pfn(_rout0));
   _COPY(_primROut, 0, _rout0, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_8(int (*_pfn)(void), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((0 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _TRY(_nErr, _pfn());
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_9(int (*_pfn)(hexagon_nn_nn_id, unsigned int*), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   unsigned int _rout1[1];
   uint32_t* _primIn;
   int _numIn[1];
   uint32_t* _primROut;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((1 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 4);
   _primIn = _pra[0].buf.pv;
   _ASSERT(_nErr, _pra[(_numIn[0] + 1)].buf.nLen >= 4);
   _primROut = _pra[(_numIn[0] + 1)].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _TRY(_nErr, _pfn(*_in0, _rout1));
   _COPY(_primROut, 0, _rout1, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_10(int (*_pfn)(unsigned int, char*, int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   unsigned int _in0[1];
   char* _rout1[1];
   int _rout1Len[1];
   uint32_t* _primIn;
   int _numIn[1];
   remote_arg* _praIn;
   remote_arg* _praROut;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((1 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 8);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_rout1Len, 0, _primIn, 4, 4);
   _praIn = (_pra + 1);
   _praROut = (_praIn + _numIn[0] + 0);
   _ASSERT(_nErr, ((_praROut[0].buf.nLen / 1)) >= (size_t)(_rout1Len[0]));
   _rout1[0] = _praROut[0].buf.pv;
   _TRY(_nErr, _pfn(*_in0, *_rout1, *_rout1Len));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_11(int (*_pfn)(const char*, unsigned int*), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   const char* _in0[1];
   int _in0Len[1];
   unsigned int _rout1[1];
   uint32_t* _primIn;
   int _numIn[1];
   uint32_t* _primROut;
   remote_arg* _praIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((2 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 4);
   _primIn = _pra[0].buf.pv;
   _ASSERT(_nErr, _pra[(_numIn[0] + 1)].buf.nLen >= 4);
   _primROut = _pra[(_numIn[0] + 1)].buf.pv;
   _COPY(_in0Len, 0, _primIn, 0, 4);
   _praIn = (_pra + 1);
   _ASSERT(_nErr, ((_praIn[0].buf.nLen / 1)) >= (size_t)(_in0Len[0]));
   _in0[0] = _praIn[0].buf.pv;
   _ASSERT(_nErr, (_in0Len[0] > 0) && (_in0[0][(_in0Len[0] - 1)] == 0));
   _TRY(_nErr, _pfn(*_in0, _rout1));
   _COPY(_primROut, 0, _rout1, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_12(int (*_pfn)(hexagon_nn_nn_id, unsigned int*, unsigned int*), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   unsigned int _rout1[1];
   unsigned int _rout2[1];
   uint32_t* _primIn;
   int _numIn[1];
   uint32_t* _primROut;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((1 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 4);
   _primIn = _pra[0].buf.pv;
   _ASSERT(_nErr, _pra[(_numIn[0] + 1)].buf.nLen >= 8);
   _primROut = _pra[(_numIn[0] + 1)].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _TRY(_nErr, _pfn(*_in0, _rout1, _rout2));
   _COPY(_primROut, 0, _rout1, 0, 4);
   _COPY(_primROut, 4, _rout2, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_13(int (*_pfn)(hexagon_nn_nn_id, unsigned int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   unsigned int _in1[1];
   uint32_t* _primIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((1 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 8);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _TRY(_nErr, _pfn(*_in0, *_in1));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_14(int (*_pfn)(hexagon_nn_nn_id, hexagon_nn_perfinfo*, int, unsigned int*), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   hexagon_nn_perfinfo* _rout1[1];
   int _rout1Len[1];
   unsigned int _rout2[1];
   uint32_t* _primIn;
   int _numIn[1];
   uint32_t* _primROut;
   remote_arg* _praIn;
   remote_arg* _praROut;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((1 + 2) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 8);
   _primIn = _pra[0].buf.pv;
   _ASSERT(_nErr, _pra[(_numIn[0] + 1)].buf.nLen >= 4);
   _primROut = _pra[(_numIn[0] + 1)].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_rout1Len, 0, _primIn, 4, 4);
   _praIn = (_pra + 1);
   _praROut = (_praIn + _numIn[0] + 1);
   _ASSERT(_nErr, ((_praROut[0].buf.nLen / 16)) >= (size_t)(_rout1Len[0]));
   _rout1[0] = _praROut[0].buf.pv;
   _TRY(_nErr, _pfn(*_in0, *_rout1, *_rout1Len, _rout2));
   _COPY(_primROut, 0, _rout2, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_15(int (*_pfn)(hexagon_nn_corner_type, hexagon_nn_dcvs_type, unsigned int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_corner_type _in0[1];
   hexagon_nn_dcvs_type _in1[1];
   unsigned int _in2[1];
   uint32_t* _primIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((1 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 12);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_in2, 0, _primIn, 8, 4);
   _TRY(_nErr, _pfn(*_in0, *_in1, *_in2));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_16(int (*_pfn)(unsigned int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   unsigned int _in0[1];
   uint32_t* _primIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((1 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 4);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _TRY(_nErr, _pfn(*_in0));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_17(int (*_pfn)(hexagon_nn_nn_id, unsigned int, int, const unsigned char*, int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   unsigned int _in1[1];
   int _in2[1];
   const unsigned char* _in3[1];
   int _in3Len[1];
   uint32_t* _primIn;
   remote_arg* _praIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((2 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 16);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_in2, 0, _primIn, 8, 4);
   _COPY(_in3Len, 0, _primIn, 12, 4);
   _praIn = (_pra + 1);
   _ASSERT(_nErr, ((_praIn[0].buf.nLen / 1)) >= (size_t)(_in3Len[0]));
   _in3[0] = _praIn[0].buf.pv;
   _TRY(_nErr, _pfn(*_in0, *_in1, *_in2, *_in3, *_in3Len));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_18(int (*_pfn)(hexagon_nn_nn_id, unsigned int, int, unsigned int, unsigned int, unsigned int, unsigned int, const unsigned char*, int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   unsigned int _in1[1];
   int _in2[1];
   unsigned int _in3[1];
   unsigned int _in4[1];
   unsigned int _in5[1];
   unsigned int _in6[1];
   const unsigned char* _in7[1];
   int _in7Len[1];
   uint32_t* _primIn;
   remote_arg* _praIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((2 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 32);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_in2, 0, _primIn, 8, 4);
   _COPY(_in3, 0, _primIn, 12, 4);
   _COPY(_in4, 0, _primIn, 16, 4);
   _COPY(_in5, 0, _primIn, 20, 4);
   _COPY(_in6, 0, _primIn, 24, 4);
   _COPY(_in7Len, 0, _primIn, 28, 4);
   _praIn = (_pra + 1);
   _ASSERT(_nErr, ((_praIn[0].buf.nLen / 1)) >= (size_t)(_in7Len[0]));
   _in7[0] = _praIn[0].buf.pv;
   _TRY(_nErr, _pfn(*_in0, *_in1, *_in2, *_in3, *_in4, *_in5, *_in6, *_in7, *_in7Len));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_19(int (*_pfn)(hexagon_nn_nn_id, unsigned int, int, unsigned int*, unsigned int*, unsigned int*, unsigned int*, unsigned char*, int, unsigned int*), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   unsigned int _in1[1];
   int _in2[1];
   unsigned int _rout3[1];
   unsigned int _rout4[1];
   unsigned int _rout5[1];
   unsigned int _rout6[1];
   unsigned char* _rout7[1];
   int _rout7Len[1];
   unsigned int _rout8[1];
   uint32_t* _primIn;
   int _numIn[1];
   uint32_t* _primROut;
   remote_arg* _praIn;
   remote_arg* _praROut;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((1 + 2) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 16);
   _primIn = _pra[0].buf.pv;
   _ASSERT(_nErr, _pra[(_numIn[0] + 1)].buf.nLen >= 20);
   _primROut = _pra[(_numIn[0] + 1)].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_in2, 0, _primIn, 8, 4);
   _COPY(_rout7Len, 0, _primIn, 12, 4);
   _praIn = (_pra + 1);
   _praROut = (_praIn + _numIn[0] + 1);
   _ASSERT(_nErr, ((_praROut[0].buf.nLen / 1)) >= (size_t)(_rout7Len[0]));
   _rout7[0] = _praROut[0].buf.pv;
   _TRY(_nErr, _pfn(*_in0, *_in1, *_in2, _rout3, _rout4, _rout5, _rout6, *_rout7, *_rout7Len, _rout8));
   _COPY(_primROut, 0, _rout3, 0, 4);
   _COPY(_primROut, 4, _rout4, 0, 4);
   _COPY(_primROut, 8, _rout5, 0, 4);
   _COPY(_primROut, 12, _rout6, 0, 4);
   _COPY(_primROut, 16, _rout8, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_20(int (*_pfn)(hexagon_nn_nn_id), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   uint32_t* _primIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((1 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 4);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _TRY(_nErr, _pfn(*_in0));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_21(int (*_pfn)(hexagon_nn_nn_id, unsigned int, unsigned int, unsigned int, unsigned int, const unsigned char*, int, unsigned int*, unsigned int*, unsigned int*, unsigned int*, unsigned char*, int, unsigned int*), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   unsigned int _in1[1];
   unsigned int _in2[1];
   unsigned int _in3[1];
   unsigned int _in4[1];
   const unsigned char* _in5[1];
   int _in5Len[1];
   unsigned int _rout6[1];
   unsigned int _rout7[1];
   unsigned int _rout8[1];
   unsigned int _rout9[1];
   unsigned char* _rout10[1];
   int _rout10Len[1];
   unsigned int _rout11[1];
   uint32_t* _primIn;
   int _numIn[1];
   uint32_t* _primROut;
   remote_arg* _praIn;
   remote_arg* _praROut;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((2 + 2) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 28);
   _primIn = _pra[0].buf.pv;
   _ASSERT(_nErr, _pra[(_numIn[0] + 1)].buf.nLen >= 20);
   _primROut = _pra[(_numIn[0] + 1)].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_in2, 0, _primIn, 8, 4);
   _COPY(_in3, 0, _primIn, 12, 4);
   _COPY(_in4, 0, _primIn, 16, 4);
   _COPY(_in5Len, 0, _primIn, 20, 4);
   _praIn = (_pra + 1);
   _ASSERT(_nErr, ((_praIn[0].buf.nLen / 1)) >= (size_t)(_in5Len[0]));
   _in5[0] = _praIn[0].buf.pv;
   _COPY(_rout10Len, 0, _primIn, 24, 4);
   _praROut = (_praIn + _numIn[0] + 1);
   _ASSERT(_nErr, ((_praROut[0].buf.nLen / 1)) >= (size_t)(_rout10Len[0]));
   _rout10[0] = _praROut[0].buf.pv;
   _TRY(_nErr, _pfn(*_in0, *_in1, *_in2, *_in3, *_in4, *_in5, *_in5Len, _rout6, _rout7, _rout8, _rout9, *_rout10, *_rout10Len, _rout11));
   _COPY(_primROut, 0, _rout6, 0, 4);
   _COPY(_primROut, 4, _rout7, 0, 4);
   _COPY(_primROut, 8, _rout8, 0, 4);
   _COPY(_primROut, 12, _rout9, 0, 4);
   _COPY(_primROut, 16, _rout11, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_22(int (*_pfn)(hexagon_nn_nn_id, unsigned int, const unsigned char*, int, unsigned int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   unsigned int _in1[1];
   const unsigned char* _in2[1];
   int _in2Len[1];
   unsigned int _in3[1];
   uint32_t* _primIn;
   remote_arg* _praIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((2 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 16);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_in2Len, 0, _primIn, 8, 4);
   _praIn = (_pra + 1);
   _ASSERT(_nErr, ((_praIn[0].buf.nLen / 1)) >= (size_t)(_in2Len[0]));
   _in2[0] = _praIn[0].buf.pv;
   _COPY(_in3, 0, _primIn, 12, 4);
   _TRY(_nErr, _pfn(*_in0, *_in1, *_in2, *_in2Len, *_in3));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_23(int (*_pfn)(hexagon_nn_nn_id, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   unsigned int _in1[1];
   unsigned int _in2[1];
   unsigned int _in3[1];
   unsigned int _in4[1];
   unsigned int _in5[1];
   unsigned int _in6[1];
   uint32_t* _primIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((1 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 28);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_in2, 0, _primIn, 8, 4);
   _COPY(_in3, 0, _primIn, 12, 4);
   _COPY(_in4, 0, _primIn, 16, 4);
   _COPY(_in5, 0, _primIn, 20, 4);
   _COPY(_in6, 0, _primIn, 24, 4);
   _TRY(_nErr, _pfn(*_in0, *_in1, *_in2, *_in3, *_in4, *_in5, *_in6));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_24(int (*_pfn)(hexagon_nn_nn_id, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, const unsigned char*, int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   unsigned int _in1[1];
   unsigned int _in2[1];
   unsigned int _in3[1];
   unsigned int _in4[1];
   unsigned int _in5[1];
   const unsigned char* _in6[1];
   int _in6Len[1];
   uint32_t* _primIn;
   remote_arg* _praIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((2 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 28);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_in2, 0, _primIn, 8, 4);
   _COPY(_in3, 0, _primIn, 12, 4);
   _COPY(_in4, 0, _primIn, 16, 4);
   _COPY(_in5, 0, _primIn, 20, 4);
   _COPY(_in6Len, 0, _primIn, 24, 4);
   _praIn = (_pra + 1);
   _ASSERT(_nErr, ((_praIn[0].buf.nLen / 1)) >= (size_t)(_in6Len[0]));
   _in6[0] = _praIn[0].buf.pv;
   _TRY(_nErr, _pfn(*_in0, *_in1, *_in2, *_in3, *_in4, *_in5, *_in6, *_in6Len));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_25(int (*_pfn)(hexagon_nn_nn_id, unsigned int, unsigned int, hexagon_nn_padding_type, const hexagon_nn_input*, int, const hexagon_nn_output*, int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   unsigned int _in1[1];
   unsigned int _in2[1];
   hexagon_nn_padding_type _in3[1];
   const hexagon_nn_input* _in4[1];
   int _in4Len[1];
   const hexagon_nn_output* _in5[1];
   int _in5Len[1];
   uint32_t* _primIn;
   remote_arg* _praIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((3 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 24);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_in2, 0, _primIn, 8, 4);
   _COPY(_in3, 0, _primIn, 12, 4);
   _COPY(_in4Len, 0, _primIn, 16, 4);
   _praIn = (_pra + 1);
   _ASSERT(_nErr, ((_praIn[0].buf.nLen / 8)) >= (size_t)(_in4Len[0]));
   _in4[0] = _praIn[0].buf.pv;
   _COPY(_in5Len, 0, _primIn, 20, 4);
   _ASSERT(_nErr, ((_praIn[1].buf.nLen / 48)) >= (size_t)(_in5Len[0]));
   _in5[0] = _praIn[1].buf.pv;
   _TRY(_nErr, _pfn(*_in0, *_in1, *_in2, *_in3, *_in4, *_in4Len, *_in5, *_in5Len));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_26(int (*_pfn)(hexagon_nn_nn_id, unsigned char*, int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   const unsigned char* _in1[1];
   int _in1Len[1];
   unsigned char* _rout1[1];
   int _rout1Len[1];
   uint32_t* _primIn;
   int _numIn[1];
   remote_arg* _praIn;
   remote_arg* _praROut;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((2 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 12);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1Len, 0, _primIn, 4, 4);
   _praIn = (_pra + 1);
   _ASSERT(_nErr, ((_praIn[0].buf.nLen / 1)) >= (size_t)(_in1Len[0]));
   _in1[0] = _praIn[0].buf.pv;
   _COPY(_rout1Len, 0, _primIn, 8, 4);
   _praROut = (_praIn + _numIn[0] + 0);
   _ASSERT(_nErr, ((_praROut[0].buf.nLen / 1)) >= (size_t)(_rout1Len[0]));
   _rout1[0] = _praROut[0].buf.pv;
   _ASSERT(_nErr, (_rout1Len[0]) >= (size_t)(_in1Len[0]));
   _MEMMOVEIF(_rout1[0], (void*) _in1[0], (_in1Len[0] * 1));
   _TRY(_nErr, _pfn(*_in0, *_rout1, *_rout1Len));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_27(int (*_pfn)(hexagon_nn_nn_id, int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   int _in1[1];
   uint32_t* _primIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((1 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 8);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _TRY(_nErr, _pfn(*_in0, *_in1));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_28(int (*_pfn)(hexagon_nn_nn_id*), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _rout0[1];
   uint32_t* _primROut;
   int _numIn[1];
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((0 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 0);
   _ASSERT(_nErr, _pra[(_numIn[0] + 0)].buf.nLen >= 4);
   _primROut = _pra[(_numIn[0] + 0)].buf.pv;
   _TRY(_nErr, _pfn(_rout0));
   _COPY(_primROut, 0, _rout0, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_29(int (*_pfn)(unsigned int*, unsigned int*), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   unsigned int _rout0[1];
   unsigned int _rout1[1];
   uint32_t* _primROut;
   int _numIn[1];
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((0 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 0);
   _ASSERT(_nErr, _pra[(_numIn[0] + 0)].buf.nLen >= 8);
   _primROut = _pra[(_numIn[0] + 0)].buf.pv;
   _TRY(_nErr, _pfn(_rout0, _rout1));
   _COPY(_primROut, 0, _rout0, 0, 4);
   _COPY(_primROut, 4, _rout1, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_30(int (*_pfn)(hexagon_nn_nn_id, const hexagon_nn_uint_option*, int, const hexagon_nn_string_option*, int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   hexagon_nn_nn_id _in0[1];
   const hexagon_nn_uint_option* _in1[1];
   int _in1Len[1];
   const hexagon_nn_string_option* _in2[1];
   int _in2Len[1];
   uint32_t* _primIn;
   remote_arg* _praIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((3 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 12);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1Len, 0, _primIn, 4, 4);
   _praIn = (_pra + 1);
   _ASSERT(_nErr, ((_praIn[0].buf.nLen / 8)) >= (size_t)(_in1Len[0]));
   _in1[0] = _praIn[0].buf.pv;
   _COPY(_in2Len, 0, _primIn, 8, 4);
   _ASSERT(_nErr, ((_praIn[1].buf.nLen / 260)) >= (size_t)(_in2Len[0]));
   _in2[0] = _praIn[1].buf.pv;
   _TRY(_nErr, _pfn(*_in0, *_in1, *_in1Len, *_in2, *_in2Len));
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_31(int (*_pfn)(const hexagon_nn_uint_option*, int, const hexagon_nn_string_option*, int), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd;
   const hexagon_nn_uint_option* _in0[1];
   int _in0Len[1];
   const hexagon_nn_string_option* _in1[1];
   int _in1Len[1];
   uint32_t* _primIn;
   remote_arg* _praIn;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _ASSERT(_nErr, (_pra + ((3 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _ASSERT(_nErr, _pra[0].buf.nLen >= 8);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0Len, 0, _primIn, 0, 4);
   _praIn = (_pra + 1);
   _ASSERT(_nErr, ((_praIn[0].buf.nLen / 8)) >= (size_t)(_in0Len[0]));
   _in0[0] = _praIn[0].buf.pv;
   _COPY(_in1Len, 0, _primIn, 4, 4);
   _ASSERT(_nErr, ((_praIn[1].buf.nLen / 260)) >= (size_t)(_in1Len[0]));
   _in1[0] = _praIn[1].buf.pv;
   _TRY(_nErr, _pfn(*_in0, *_in0Len, *_in1, *_in1Len));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_SKEL_EXPORT int __QAIC_SKEL(hexagon_nn_skel_invoke)(uint32_t _sc, remote_arg* _pra) __QAIC_SKEL_ATTRIBUTE {
   switch(REMOTE_SCALARS_METHOD(_sc))
   {
      case 0:
      return _skel_method_8((void*)__QAIC_IMPL(hexagon_nn_config), _sc, _pra);
      case 1:
      return _skel_method_31((void*)__QAIC_IMPL(hexagon_nn_config_with_options), _sc, _pra);
      case 2:
      return _skel_method_30((void*)__QAIC_IMPL(hexagon_nn_graph_config), _sc, _pra);
      case 3:
      return _skel_method_29((void*)__QAIC_IMPL(hexagon_nn_get_dsp_offset), _sc, _pra);
      case 4:
      return _skel_method_28((void*)__QAIC_IMPL(hexagon_nn_init), _sc, _pra);
      case 5:
      return _skel_method_27((void*)__QAIC_IMPL(hexagon_nn_set_debug_level), _sc, _pra);
      case 6:
      return _skel_method_26((void*)__QAIC_IMPL(hexagon_nn_snpprint), _sc, _pra);
      case 7:
      return _skel_method_26((void*)__QAIC_IMPL(hexagon_nn_getlog), _sc, _pra);
      case 8:
      return _skel_method_25((void*)__QAIC_IMPL(hexagon_nn_append_node), _sc, _pra);
      case 9:
      return _skel_method_24((void*)__QAIC_IMPL(hexagon_nn_append_const_node), _sc, _pra);
      case 10:
      return _skel_method_23((void*)__QAIC_IMPL(hexagon_nn_append_empty_const_node), _sc, _pra);
      case 11:
      return _skel_method_22((void*)__QAIC_IMPL(hexagon_nn_populate_const_node), _sc, _pra);
      case 12:
      return _skel_method_20((void*)__QAIC_IMPL(hexagon_nn_prepare), _sc, _pra);
      case 13:
      return _skel_method_21((void*)__QAIC_IMPL(hexagon_nn_execute), _sc, _pra);
      case 14:
      return _skel_method_20((void*)__QAIC_IMPL(hexagon_nn_teardown), _sc, _pra);
      case 15:
      return _skel_method_19((void*)__QAIC_IMPL(hexagon_nn_variable_read), _sc, _pra);
      case 16:
      return _skel_method_18((void*)__QAIC_IMPL(hexagon_nn_variable_write), _sc, _pra);
      case 17:
      return _skel_method_17((void*)__QAIC_IMPL(hexagon_nn_variable_write_flat), _sc, _pra);
      case 18:
      return _skel_method_16((void*)__QAIC_IMPL(hexagon_nn_set_powersave_level), _sc, _pra);
      case 19:
      return _skel_method_15((void*)__QAIC_IMPL(hexagon_nn_set_powersave_details), _sc, _pra);
      case 20:
      return _skel_method_14((void*)__QAIC_IMPL(hexagon_nn_get_perfinfo), _sc, _pra);
      case 21:
      return _skel_method_13((void*)__QAIC_IMPL(hexagon_nn_reset_perfinfo), _sc, _pra);
      case 22:
      return _skel_method_12((void*)__QAIC_IMPL(hexagon_nn_last_execution_cycles), _sc, _pra);
      case 23:
      return _skel_method_7((void*)__QAIC_IMPL(hexagon_nn_version), _sc, _pra);
      case 24:
      return _skel_method_11((void*)__QAIC_IMPL(hexagon_nn_op_name_to_id), _sc, _pra);
      case 25:
      return _skel_method_10((void*)__QAIC_IMPL(hexagon_nn_op_id_to_name), _sc, _pra);
      case 26:
      return _skel_method_9((void*)__QAIC_IMPL(hexagon_nn_get_num_nodes_in_graph), _sc, _pra);
      case 27:
      return _skel_method_8((void*)__QAIC_IMPL(hexagon_nn_disable_dcvs), _sc, _pra);
      case 28:
      return _skel_method_7((void*)__QAIC_IMPL(hexagon_nn_GetHexagonBinaryVersion), _sc, _pra);
      case 29:
      return _skel_method_6((void*)__QAIC_IMPL(hexagon_nn_PrintLog), _sc, _pra);
      case 30:
      return _skel_method_5((void*)__QAIC_IMPL(hexagon_nn_execute_new), _sc, _pra);
      case 31:
      {
         uint32_t* _mid;
         if(REMOTE_SCALARS_INBUFS(_sc) < 1 || _pra[0].buf.nLen < 4) { return AEE_EBADPARM; }
         _mid = (uint32_t*)_pra[0].buf.pv;
         return _skel_invoke(*_mid, _sc, _pra);
      }
   }
   return AEE_EUNSUPPORTED;
}
#ifdef __cplusplus
}
#endif
#endif //_HEXAGON_NN_SKEL_H
