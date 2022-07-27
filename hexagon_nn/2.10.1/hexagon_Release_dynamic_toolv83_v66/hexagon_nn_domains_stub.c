#ifndef _HEXAGON_NN_DOMAINS_STUB_H
#define _HEXAGON_NN_DOMAINS_STUB_H
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


#ifndef _HEXAGON_NN_DOMAINS_SLIM_H
#define _HEXAGON_NN_DOMAINS_SLIM_H
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
static const Parameter parameters[24] = {{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)0x0,0}}, 4,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x4,0x8),{{(const uintptr_t)0xdeadc0de,(const uintptr_t)0}}, 0,SLIM_IFPTR32(0x4,0x8),3,0},{SLIM_IFPTR32(0x4,0x8),{{(const uintptr_t)0xdeadc0de,(const uintptr_t)0}}, 0,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[0]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[2]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),0,0},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4,0,0},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4,3,0},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4,3,0},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4,0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[5]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),4,0},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4,0,0},{0x4,{{0,0}}, 3,0x4,0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[6]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[7]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[5]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[5]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),3,0},{0x4,{{0,0}}, 3,0x4,0,0},{0x4,{{0,0}}, 3,0x4,0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[11]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),3,0},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4,3,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)0x0,0}}, 4,SLIM_IFPTR32(0x4,0x8),3,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(sequenceTypes[0]),0}}, 25,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(sequenceTypes[0]),0}}, 25,SLIM_IFPTR32(0x4,0x8),3,0},{0x4,{{(const uintptr_t)&(structTypes[0]),0}}, 6,0x4,0,0}};
static const Parameter* const parameterArrays[92] = {(&(parameters[5])),(&(parameters[10])),(&(parameters[10])),(&(parameters[10])),(&(parameters[10])),(&(parameters[14])),(&(parameters[6])),(&(parameters[6])),(&(parameters[6])),(&(parameters[6])),(&(parameters[15])),(&(parameters[6])),(&(parameters[5])),(&(parameters[10])),(&(parameters[8])),(&(parameters[6])),(&(parameters[6])),(&(parameters[6])),(&(parameters[6])),(&(parameters[15])),(&(parameters[6])),(&(parameters[5])),(&(parameters[10])),(&(parameters[8])),(&(parameters[10])),(&(parameters[10])),(&(parameters[10])),(&(parameters[10])),(&(parameters[14])),(&(parameters[5])),(&(parameters[10])),(&(parameters[10])),(&(parameters[10])),(&(parameters[10])),(&(parameters[10])),(&(parameters[10])),(&(parameters[5])),(&(parameters[10])),(&(parameters[10])),(&(parameters[10])),(&(parameters[10])),(&(parameters[10])),(&(parameters[14])),(&(parameters[5])),(&(parameters[10])),(&(parameters[10])),(&(parameters[11])),(&(parameters[12])),(&(parameters[13])),(&(parameters[5])),(&(parameters[10])),(&(parameters[8])),(&(parameters[14])),(&(parameters[5])),(&(parameters[10])),(&(parameters[14])),(&(parameters[10])),(&(parameters[5])),(&(parameters[0])),(&(parameters[8])),(&(parameters[5])),(&(parameters[5])),(&(parameters[6])),(&(parameters[5])),(&(parameters[21])),(&(parameters[22])),(&(parameters[5])),(&(parameters[6])),(&(parameters[6])),(&(parameters[5])),(&(parameters[18])),(&(parameters[6])),(&(parameters[16])),(&(parameters[17])),(&(parameters[10])),(&(parameters[5])),(&(parameters[3])),(&(parameters[4])),(&(parameters[7])),(&(parameters[23])),(&(parameters[10])),(&(parameters[20])),(&(parameters[0])),(&(parameters[6])),(&(parameters[5])),(&(parameters[9])),(&(parameters[5])),(&(parameters[8])),(&(parameters[0])),(&(parameters[1])),(&(parameters[19])),(&(parameters[2]))};
static const Method methods[33] = {{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x1),0x4,0x0,2,2,(&(parameterArrays[88])),0x4,0x1},{REMOTE_SCALARS_MAKEX(0,0,0x0,0x0,0x1,0x0),0x0,0x0,1,1,(&(parameterArrays[91])),0x1,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x0,0x0,0x0,0x0),0x0,0x0,0,0,0,0x0,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x3,0x0,0x0,0x0),0x8,0x0,4,2,(&(parameterArrays[76])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x3,0x0,0x0,0x0),0xc,0x0,5,3,(&(parameterArrays[75])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x0,0x1,0x0,0x0),0x0,0x8,2,2,(&(parameterArrays[6])),0x1,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x0,0x1,0x0,0x0),0x0,0x4,1,1,(&(parameterArrays[78])),0x1,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x0,0x0,0x0),0x8,0x0,2,2,(&(parameterArrays[86])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x1,0x0,0x0),0xc,0x0,4,2,(&(parameterArrays[84])),0x4,0x1},{REMOTE_SCALARS_MAKEX(0,0,0x3,0x0,0x0,0x0),0x18,0x0,8,6,(&(parameterArrays[43])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x0),0x1c,0x0,8,7,(&(parameterArrays[36])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x0,0x0,0x0),0x1c,0x0,7,7,(&(parameterArrays[29])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x0),0x10,0x0,5,4,(&(parameterArrays[53])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x0,0x0,0x0),0x4,0x0,1,1,(&(parameterArrays[0])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x2,0x0,0x0),0x1c,0x14,15,12,(&(parameterArrays[0])),0x4,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x2,0x0,0x0),0x10,0x14,11,9,(&(parameterArrays[12])),0x4,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x0),0x20,0x0,9,8,(&(parameterArrays[21])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x0),0x10,0x0,5,4,(&(parameterArrays[49])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x0,0x0,0x0),0x4,0x0,1,1,(&(parameterArrays[1])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x0,0x0,0x0),0xc,0x0,3,3,(&(parameterArrays[72])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x2,0x0,0x0),0x8,0x4,5,3,(&(parameterArrays[69])),0x4,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x0,0x0,0x0),0x8,0x0,2,2,(&(parameterArrays[0])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x1,0x0,0x0),0x4,0x8,3,3,(&(parameterArrays[66])),0x4,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x0,0x1,0x0,0x0),0x0,0x4,1,1,(&(parameterArrays[90])),0x1,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x1,0x0,0x0),0x4,0x4,2,2,(&(parameterArrays[82])),0x4,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x1,0x0,0x0),0x8,0x0,4,2,(&(parameterArrays[80])),0x4,0x1},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x1,0x0,0x0),0x4,0x4,2,2,(&(parameterArrays[61])),0x4,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x0),0x4,0x0,2,1,(&(parameterArrays[5])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,255,255,15,15),0xc,0x0,6,3,(&(parameterArrays[63])),0x4,0x1},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x1,0x0,0x0),0x4,0x4,2,2,(&(parameterArrays[78])),0x4,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x1,0x0,0x0),0x8,0x4,3,3,(&(parameterArrays[60])),0x4,0x4},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x0,0x0,0x0),0x4,0x0,1,1,(&(parameterArrays[14])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x0),0xc,0x0,3,3,(&(parameterArrays[57])),0x4,0x0}};
static const Method* const methodArrays[38] = {&(methods[0]),&(methods[1]),&(methods[2]),&(methods[3]),&(methods[4]),&(methods[5]),&(methods[6]),&(methods[7]),&(methods[8]),&(methods[8]),&(methods[9]),&(methods[10]),&(methods[11]),&(methods[12]),&(methods[13]),&(methods[14]),&(methods[13]),&(methods[15]),&(methods[16]),&(methods[17]),&(methods[18]),&(methods[19]),&(methods[20]),&(methods[21]),&(methods[22]),&(methods[23]),&(methods[24]),&(methods[25]),&(methods[26]),&(methods[2]),&(methods[23]),&(methods[27]),&(methods[28]),&(methods[29]),&(methods[30]),&(methods[22]),&(methods[31]),&(methods[32])};
static const char strings[1064] = "GetHexagonBinaryVersion\0append_empty_const_node\0multi_execution_cycles\0get_num_nodes_in_graph\0last_execution_cycles\0set_powersave_details\0set_powersave_level\0variable_write_flat\0populate_const_node\0config_with_options\0fastrpc_shell_addr\0append_const_node\0set_graph_option\0set_debug_level\0libhexagon_addr\0init_with_info\0data_valid_len\0reset_perfinfo\0variable_write\0get_dsp_offset\0string_options\0op_id_to_name\0op_name_to_id\0variable_read\0target_offset\0get_nodetype\0disable_dcvs\0get_perfinfo\0output_index\0data_len_out\0graph_config\0uint_options\0execute_new\0batches_out\0zero_offset\0elementsize\0append_node\0string_data\0counter_hi\0counter_lo\0executions\0height_out\0batches_in\0output_idx\0uint_value\0get_power\0node_type\0num_nodes\0cycles_hi\0cycles_lo\0depth_out\0width_out\0height_in\0max_sizes\0operation\0option_id\0graph_id\0priority\0PrintLog\0info_out\0teardown\0data_out\0depth_in\0width_in\0stepsize\0snpprint\0version\0n_items\0latency\0data_in\0execute\0prepare\0batches\0outputs\0padding\0node_id\0unused\0corner\0height\0src_id\0inputs\0getlog\0event\0depth\0width\0close\0rank\0init\0open\0ver\0buf\0uri\0";
static const uint16_t methodStrings[184] = {541,419,998,938,984,1024,1018,608,319,970,946,938,984,1024,1018,608,319,970,589,419,962,780,954,998,991,668,946,1036,770,577,565,872,922,419,657,760,863,854,914,553,646,750,740,845,502,422,419,962,489,553,646,750,740,845,502,349,419,962,489,938,984,1024,1018,914,476,419,827,962,635,624,613,898,24,419,962,938,984,1024,1018,584,237,419,962,938,984,1024,1018,608,515,419,528,790,679,379,790,601,198,528,790,679,379,790,601,158,419,962,489,914,178,419,962,608,436,255,419,403,684,48,419,730,720,450,800,962,700,304,526,314,809,94,419,730,720,116,977,471,906,71,419,710,394,962,403,408,403,962,334,419,1012,1005,419,1055,881,419,1055,272,419,152,364,288,218,1046,1059,92,690,458,818,1055,0,1051,890,1051,138,152,836,419,930,419,1041,526,1030,92,463,521};
static const uint16_t methodStringsArrays[38] = {161,180,183,96,88,158,178,155,152,149,18,80,72,108,176,32,174,45,55,103,172,133,64,146,129,170,143,140,137,182,168,166,0,125,121,117,164,113};
__QAIC_SLIM_EXPORT const Interface __QAIC_SLIM(hexagon_nn_domains_slim) = {38,&(methodArrays[0]),0,0,&(methodStringsArrays [0]),methodStrings,strings};
#endif //_HEXAGON_NN_DOMAINS_SLIM_H
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_skel_handle_invoke)(remote_handle64 _h, uint32_t _sc, remote_arg* _pra) __QAIC_STUB_ATTRIBUTE {
   return __QAIC_REMOTE(remote_handle64_invoke)(_h, _sc, _pra);
}
#ifdef __cplusplus
extern "C" {
#endif
extern int remote_register_dma_handle(int, uint32_t);
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_open)(const char* uri, remote_handle64* h) __QAIC_STUB_ATTRIBUTE {
   return __QAIC_REMOTE(remote_handle64_open)(uri, h);
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_close)(remote_handle64 h) __QAIC_STUB_ATTRIBUTE {
   return __QAIC_REMOTE(remote_handle64_close)(h);
}
static __inline int _stub_method(remote_handle64 _handle, uint32_t _mid) {
   remote_arg* _pra = 0;
   int _nErr = 0;
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 0, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_config)(remote_handle64 _handle) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 2;
   return _stub_method(_handle, _mid);
}
static __inline int _stub_method_1(remote_handle64 _handle, uint32_t _mid, const hexagon_nn_uint_option* _in0[1], int _in0Len[1], const hexagon_nn_string_option* _in1[1], int _in1Len[1]) {
   remote_arg _pra[3];
   uint32_t _primIn[2];
   remote_arg* _praIn;
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in0[0];
   _praIn[0].buf.nLen = (8 * _in0Len[0]);
   _COPY(_primIn, 4, _in1Len, 0, 4);
   _praIn[1].buf.pv = (void*) _in1[0];
   _praIn[1].buf.nLen = (260 * _in1Len[0]);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 3, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_config_with_options)(remote_handle64 _handle, const hexagon_nn_uint_option* uint_options, int uint_optionsLen, const hexagon_nn_string_option* string_options, int string_optionsLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 3;
   return _stub_method_1(_handle, _mid, (const hexagon_nn_uint_option**)&uint_options, (int*)&uint_optionsLen, (const hexagon_nn_string_option**)&string_options, (int*)&string_optionsLen);
}
static __inline int _stub_method_2(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _in0[1], const hexagon_nn_uint_option* _in1[1], int _in1Len[1], const hexagon_nn_string_option* _in2[1], int _in2Len[1]) {
   remote_arg _pra[3];
   uint32_t _primIn[3];
   remote_arg* _praIn;
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in1[0];
   _praIn[0].buf.nLen = (8 * _in1Len[0]);
   _COPY(_primIn, 8, _in2Len, 0, 4);
   _praIn[1].buf.pv = (void*) _in2[0];
   _praIn[1].buf.nLen = (260 * _in2Len[0]);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 3, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_graph_config)(remote_handle64 _handle, hexagon_nn_nn_id id, const hexagon_nn_uint_option* uint_options, int uint_optionsLen, const hexagon_nn_string_option* string_options, int string_optionsLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 4;
   return _stub_method_2(_handle, _mid, (hexagon_nn_nn_id*)&id, (const hexagon_nn_uint_option**)&uint_options, (int*)&uint_optionsLen, (const hexagon_nn_string_option**)&string_options, (int*)&string_optionsLen);
}
static __inline int _stub_method_3(remote_handle64 _handle, uint32_t _mid, unsigned int _rout0[1], unsigned int _rout1[1]) {
   int _numIn[1];
   remote_arg _pra[1];
   uint32_t _primROut[2];
   int _nErr = 0;
   _numIn[0] = 0;
   _pra[(_numIn[0] + 0)].buf.pv = (void*)_primROut;
   _pra[(_numIn[0] + 0)].buf.nLen = sizeof(_primROut);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 0, 1, 0, 0), _pra));
   _COPY(_rout0, 0, _primROut, 0, 4);
   _COPY(_rout1, 0, _primROut, 4, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_get_dsp_offset)(remote_handle64 _handle, unsigned int* libhexagon_addr, unsigned int* fastrpc_shell_addr) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 5;
   return _stub_method_3(_handle, _mid, (unsigned int*)libhexagon_addr, (unsigned int*)fastrpc_shell_addr);
}
static __inline int _stub_method_4(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _rout0[1]) {
   int _numIn[1];
   remote_arg _pra[1];
   uint32_t _primROut[1];
   int _nErr = 0;
   _numIn[0] = 0;
   _pra[(_numIn[0] + 0)].buf.pv = (void*)_primROut;
   _pra[(_numIn[0] + 0)].buf.nLen = sizeof(_primROut);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 0, 1, 0, 0), _pra));
   _COPY(_rout0, 0, _primROut, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_init)(remote_handle64 _handle, hexagon_nn_nn_id* g) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 6;
   return _stub_method_4(_handle, _mid, (hexagon_nn_nn_id*)g);
}
static __inline int _stub_method_5(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _in0[1], int _in1[1]) {
   remote_arg _pra[1];
   uint32_t _primIn[2];
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_set_debug_level)(remote_handle64 _handle, hexagon_nn_nn_id id, int level) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 7;
   return _stub_method_5(_handle, _mid, (hexagon_nn_nn_id*)&id, (int*)&level);
}
static __inline int _stub_method_6(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _in0[1], const unsigned char* _in1[1], int _in1Len[1], unsigned char* _rout1[1], int _rout1Len[1]) {
   int _numIn[1];
   remote_arg _pra[3];
   uint32_t _primIn[3];
   remote_arg* _praIn;
   remote_arg* _praROut;
   int _nErr = 0;
   _numIn[0] = 1;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in1[0];
   _praIn[0].buf.nLen = (1 * _in1Len[0]);
   _COPY(_primIn, 8, _rout1Len, 0, 4);
   _praROut = (_praIn + _numIn[0] + 0);
   _praROut[0].buf.pv = _rout1[0];
   _praROut[0].buf.nLen = (1 * _rout1Len[0]);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 2, 1, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_snpprint)(remote_handle64 _handle, hexagon_nn_nn_id id, unsigned char* buf, int bufLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 8;
   return _stub_method_6(_handle, _mid, (hexagon_nn_nn_id*)&id, (const unsigned char**)&buf, (int*)&bufLen, (unsigned char**)&buf, (int*)&bufLen);
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_getlog)(remote_handle64 _handle, hexagon_nn_nn_id id, unsigned char* buf, int bufLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 9;
   return _stub_method_6(_handle, _mid, (hexagon_nn_nn_id*)&id, (const unsigned char**)&buf, (int*)&bufLen, (unsigned char**)&buf, (int*)&bufLen);
}
static __inline int _stub_method_7(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _in0[1], unsigned int _in1[1], unsigned int _in2[1], hexagon_nn_padding_type _in3[1], const hexagon_nn_input* _in4[1], int _in4Len[1], const hexagon_nn_output* _in5[1], int _in5Len[1]) {
   remote_arg _pra[3];
   uint32_t _primIn[6];
   remote_arg* _praIn;
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _in2, 0, 4);
   _COPY(_primIn, 12, _in3, 0, 4);
   _COPY(_primIn, 16, _in4Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in4[0];
   _praIn[0].buf.nLen = (8 * _in4Len[0]);
   _COPY(_primIn, 20, _in5Len, 0, 4);
   _praIn[1].buf.pv = (void*) _in5[0];
   _praIn[1].buf.nLen = (48 * _in5Len[0]);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 3, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_append_node)(remote_handle64 _handle, hexagon_nn_nn_id id, unsigned int node_id, unsigned int operation, hexagon_nn_padding_type padding, const hexagon_nn_input* inputs, int inputsLen, const hexagon_nn_output* outputs, int outputsLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 10;
   return _stub_method_7(_handle, _mid, (hexagon_nn_nn_id*)&id, (unsigned int*)&node_id, (unsigned int*)&operation, (hexagon_nn_padding_type*)&padding, (const hexagon_nn_input**)&inputs, (int*)&inputsLen, (const hexagon_nn_output**)&outputs, (int*)&outputsLen);
}
static __inline int _stub_method_8(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _in0[1], unsigned int _in1[1], unsigned int _in2[1], unsigned int _in3[1], unsigned int _in4[1], unsigned int _in5[1], const unsigned char* _in6[1], int _in6Len[1]) {
   remote_arg _pra[2];
   uint32_t _primIn[7];
   remote_arg* _praIn;
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _in2, 0, 4);
   _COPY(_primIn, 12, _in3, 0, 4);
   _COPY(_primIn, 16, _in4, 0, 4);
   _COPY(_primIn, 20, _in5, 0, 4);
   _COPY(_primIn, 24, _in6Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in6[0];
   _praIn[0].buf.nLen = (1 * _in6Len[0]);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 2, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_append_const_node)(remote_handle64 _handle, hexagon_nn_nn_id id, unsigned int node_id, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, const unsigned char* data, int dataLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 11;
   return _stub_method_8(_handle, _mid, (hexagon_nn_nn_id*)&id, (unsigned int*)&node_id, (unsigned int*)&batches, (unsigned int*)&height, (unsigned int*)&width, (unsigned int*)&depth, (const unsigned char**)&data, (int*)&dataLen);
}
static __inline int _stub_method_9(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _in0[1], unsigned int _in1[1], unsigned int _in2[1], unsigned int _in3[1], unsigned int _in4[1], unsigned int _in5[1], unsigned int _in6[1]) {
   remote_arg _pra[1];
   uint32_t _primIn[7];
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _in2, 0, 4);
   _COPY(_primIn, 12, _in3, 0, 4);
   _COPY(_primIn, 16, _in4, 0, 4);
   _COPY(_primIn, 20, _in5, 0, 4);
   _COPY(_primIn, 24, _in6, 0, 4);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_append_empty_const_node)(remote_handle64 _handle, hexagon_nn_nn_id id, unsigned int node_id, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, unsigned int size) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 12;
   return _stub_method_9(_handle, _mid, (hexagon_nn_nn_id*)&id, (unsigned int*)&node_id, (unsigned int*)&batches, (unsigned int*)&height, (unsigned int*)&width, (unsigned int*)&depth, (unsigned int*)&size);
}
static __inline int _stub_method_10(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _in0[1], unsigned int _in1[1], const unsigned char* _in2[1], int _in2Len[1], unsigned int _in3[1]) {
   remote_arg _pra[2];
   uint32_t _primIn[4];
   remote_arg* _praIn;
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _in2Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in2[0];
   _praIn[0].buf.nLen = (1 * _in2Len[0]);
   _COPY(_primIn, 12, _in3, 0, 4);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 2, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_populate_const_node)(remote_handle64 _handle, hexagon_nn_nn_id id, unsigned int node_id, const unsigned char* data, int dataLen, unsigned int target_offset) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 13;
   return _stub_method_10(_handle, _mid, (hexagon_nn_nn_id*)&id, (unsigned int*)&node_id, (const unsigned char**)&data, (int*)&dataLen, (unsigned int*)&target_offset);
}
static __inline int _stub_method_11(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _in0[1]) {
   remote_arg _pra[1];
   uint32_t _primIn[1];
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_prepare)(remote_handle64 _handle, hexagon_nn_nn_id id) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 14;
   return _stub_method_11(_handle, _mid, (hexagon_nn_nn_id*)&id);
}
static __inline int _stub_method_12(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _in0[1], unsigned int _in1[1], unsigned int _in2[1], unsigned int _in3[1], unsigned int _in4[1], const unsigned char* _in5[1], int _in5Len[1], unsigned int _rout6[1], unsigned int _rout7[1], unsigned int _rout8[1], unsigned int _rout9[1], unsigned char* _rout10[1], int _rout10Len[1], unsigned int _rout11[1]) {
   int _numIn[1];
   remote_arg _pra[4];
   uint32_t _primIn[7];
   uint32_t _primROut[5];
   remote_arg* _praIn;
   remote_arg* _praROut;
   int _nErr = 0;
   _numIn[0] = 1;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _pra[(_numIn[0] + 1)].buf.pv = (void*)_primROut;
   _pra[(_numIn[0] + 1)].buf.nLen = sizeof(_primROut);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _in2, 0, 4);
   _COPY(_primIn, 12, _in3, 0, 4);
   _COPY(_primIn, 16, _in4, 0, 4);
   _COPY(_primIn, 20, _in5Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in5[0];
   _praIn[0].buf.nLen = (1 * _in5Len[0]);
   _COPY(_primIn, 24, _rout10Len, 0, 4);
   _praROut = (_praIn + _numIn[0] + 1);
   _praROut[0].buf.pv = _rout10[0];
   _praROut[0].buf.nLen = (1 * _rout10Len[0]);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 2, 2, 0, 0), _pra));
   _COPY(_rout6, 0, _primROut, 0, 4);
   _COPY(_rout7, 0, _primROut, 4, 4);
   _COPY(_rout8, 0, _primROut, 8, 4);
   _COPY(_rout9, 0, _primROut, 12, 4);
   _COPY(_rout11, 0, _primROut, 16, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_execute)(remote_handle64 _handle, hexagon_nn_nn_id id, unsigned int batches_in, unsigned int height_in, unsigned int width_in, unsigned int depth_in, const unsigned char* data_in, int data_inLen, unsigned int* batches_out, unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out, unsigned char* data_out, int data_outLen, unsigned int* data_len_out) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 15;
   return _stub_method_12(_handle, _mid, (hexagon_nn_nn_id*)&id, (unsigned int*)&batches_in, (unsigned int*)&height_in, (unsigned int*)&width_in, (unsigned int*)&depth_in, (const unsigned char**)&data_in, (int*)&data_inLen, (unsigned int*)batches_out, (unsigned int*)height_out, (unsigned int*)width_out, (unsigned int*)depth_out, (unsigned char**)&data_out, (int*)&data_outLen, (unsigned int*)data_len_out);
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_teardown)(remote_handle64 _handle, hexagon_nn_nn_id id) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 16;
   return _stub_method_11(_handle, _mid, (hexagon_nn_nn_id*)&id);
}
static __inline int _stub_method_13(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _in0[1], unsigned int _in1[1], int _in2[1], unsigned int _rout3[1], unsigned int _rout4[1], unsigned int _rout5[1], unsigned int _rout6[1], unsigned char* _rout7[1], int _rout7Len[1], unsigned int _rout8[1]) {
   int _numIn[1];
   remote_arg _pra[3];
   uint32_t _primIn[4];
   uint32_t _primROut[5];
   remote_arg* _praIn;
   remote_arg* _praROut;
   int _nErr = 0;
   _numIn[0] = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _pra[(_numIn[0] + 1)].buf.pv = (void*)_primROut;
   _pra[(_numIn[0] + 1)].buf.nLen = sizeof(_primROut);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _in2, 0, 4);
   _COPY(_primIn, 12, _rout7Len, 0, 4);
   _praIn = (_pra + 1);
   _praROut = (_praIn + _numIn[0] + 1);
   _praROut[0].buf.pv = _rout7[0];
   _praROut[0].buf.nLen = (1 * _rout7Len[0]);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 2, 0, 0), _pra));
   _COPY(_rout3, 0, _primROut, 0, 4);
   _COPY(_rout4, 0, _primROut, 4, 4);
   _COPY(_rout5, 0, _primROut, 8, 4);
   _COPY(_rout6, 0, _primROut, 12, 4);
   _COPY(_rout8, 0, _primROut, 16, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_variable_read)(remote_handle64 _handle, hexagon_nn_nn_id id, unsigned int node_id, int output_index, unsigned int* batches_out, unsigned int* height_out, unsigned int* width_out, unsigned int* depth_out, unsigned char* data_out, int data_outLen, unsigned int* data_len_out) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 17;
   return _stub_method_13(_handle, _mid, (hexagon_nn_nn_id*)&id, (unsigned int*)&node_id, (int*)&output_index, (unsigned int*)batches_out, (unsigned int*)height_out, (unsigned int*)width_out, (unsigned int*)depth_out, (unsigned char**)&data_out, (int*)&data_outLen, (unsigned int*)data_len_out);
}
static __inline int _stub_method_14(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _in0[1], unsigned int _in1[1], int _in2[1], unsigned int _in3[1], unsigned int _in4[1], unsigned int _in5[1], unsigned int _in6[1], const unsigned char* _in7[1], int _in7Len[1]) {
   remote_arg _pra[2];
   uint32_t _primIn[8];
   remote_arg* _praIn;
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _in2, 0, 4);
   _COPY(_primIn, 12, _in3, 0, 4);
   _COPY(_primIn, 16, _in4, 0, 4);
   _COPY(_primIn, 20, _in5, 0, 4);
   _COPY(_primIn, 24, _in6, 0, 4);
   _COPY(_primIn, 28, _in7Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in7[0];
   _praIn[0].buf.nLen = (1 * _in7Len[0]);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 2, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_variable_write)(remote_handle64 _handle, hexagon_nn_nn_id id, unsigned int node_id, int output_index, unsigned int batches, unsigned int height, unsigned int width, unsigned int depth, const unsigned char* data_in, int data_inLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 18;
   return _stub_method_14(_handle, _mid, (hexagon_nn_nn_id*)&id, (unsigned int*)&node_id, (int*)&output_index, (unsigned int*)&batches, (unsigned int*)&height, (unsigned int*)&width, (unsigned int*)&depth, (const unsigned char**)&data_in, (int*)&data_inLen);
}
static __inline int _stub_method_15(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _in0[1], unsigned int _in1[1], int _in2[1], const unsigned char* _in3[1], int _in3Len[1]) {
   remote_arg _pra[2];
   uint32_t _primIn[4];
   remote_arg* _praIn;
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _in2, 0, 4);
   _COPY(_primIn, 12, _in3Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in3[0];
   _praIn[0].buf.nLen = (1 * _in3Len[0]);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 2, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_variable_write_flat)(remote_handle64 _handle, hexagon_nn_nn_id id, unsigned int node_id, int output_index, const unsigned char* data_in, int data_inLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 19;
   return _stub_method_15(_handle, _mid, (hexagon_nn_nn_id*)&id, (unsigned int*)&node_id, (int*)&output_index, (const unsigned char**)&data_in, (int*)&data_inLen);
}
static __inline int _stub_method_16(remote_handle64 _handle, uint32_t _mid, unsigned int _in0[1]) {
   remote_arg _pra[1];
   uint32_t _primIn[1];
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_set_powersave_level)(remote_handle64 _handle, unsigned int level) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 20;
   return _stub_method_16(_handle, _mid, (unsigned int*)&level);
}
static __inline int _stub_method_17(remote_handle64 _handle, uint32_t _mid, hexagon_nn_corner_type _in0[1], hexagon_nn_dcvs_type _in1[1], unsigned int _in2[1]) {
   remote_arg _pra[1];
   uint32_t _primIn[3];
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _in2, 0, 4);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_set_powersave_details)(remote_handle64 _handle, hexagon_nn_corner_type corner, hexagon_nn_dcvs_type dcvs, unsigned int latency) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 21;
   return _stub_method_17(_handle, _mid, (hexagon_nn_corner_type*)&corner, (hexagon_nn_dcvs_type*)&dcvs, (unsigned int*)&latency);
}
static __inline int _stub_method_18(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _in0[1], hexagon_nn_perfinfo* _rout1[1], int _rout1Len[1], unsigned int _rout2[1]) {
   int _numIn[1];
   remote_arg _pra[3];
   uint32_t _primIn[2];
   uint32_t _primROut[1];
   remote_arg* _praIn;
   remote_arg* _praROut;
   int _nErr = 0;
   _numIn[0] = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _pra[(_numIn[0] + 1)].buf.pv = (void*)_primROut;
   _pra[(_numIn[0] + 1)].buf.nLen = sizeof(_primROut);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _rout1Len, 0, 4);
   _praIn = (_pra + 1);
   _praROut = (_praIn + _numIn[0] + 1);
   _praROut[0].buf.pv = _rout1[0];
   _praROut[0].buf.nLen = (16 * _rout1Len[0]);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 2, 0, 0), _pra));
   _COPY(_rout2, 0, _primROut, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_get_perfinfo)(remote_handle64 _handle, hexagon_nn_nn_id id, hexagon_nn_perfinfo* info_out, int info_outLen, unsigned int* n_items) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 22;
   return _stub_method_18(_handle, _mid, (hexagon_nn_nn_id*)&id, (hexagon_nn_perfinfo**)&info_out, (int*)&info_outLen, (unsigned int*)n_items);
}
static __inline int _stub_method_19(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _in0[1], unsigned int _in1[1]) {
   remote_arg _pra[1];
   uint32_t _primIn[2];
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_reset_perfinfo)(remote_handle64 _handle, hexagon_nn_nn_id id, unsigned int event) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 23;
   return _stub_method_19(_handle, _mid, (hexagon_nn_nn_id*)&id, (unsigned int*)&event);
}
static __inline int _stub_method_20(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _in0[1], unsigned int _rout1[1], unsigned int _rout2[1]) {
   int _numIn[1];
   remote_arg _pra[2];
   uint32_t _primIn[1];
   uint32_t _primROut[2];
   int _nErr = 0;
   _numIn[0] = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _pra[(_numIn[0] + 1)].buf.pv = (void*)_primROut;
   _pra[(_numIn[0] + 1)].buf.nLen = sizeof(_primROut);
   _COPY(_primIn, 0, _in0, 0, 4);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 1, 0, 0), _pra));
   _COPY(_rout1, 0, _primROut, 0, 4);
   _COPY(_rout2, 0, _primROut, 4, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_last_execution_cycles)(remote_handle64 _handle, hexagon_nn_nn_id id, unsigned int* cycles_lo, unsigned int* cycles_hi) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 24;
   return _stub_method_20(_handle, _mid, (hexagon_nn_nn_id*)&id, (unsigned int*)cycles_lo, (unsigned int*)cycles_hi);
}
static __inline int _stub_method_21(remote_handle64 _handle, uint32_t _mid, int _rout0[1]) {
   int _numIn[1];
   remote_arg _pra[1];
   uint32_t _primROut[1];
   int _nErr = 0;
   _numIn[0] = 0;
   _pra[(_numIn[0] + 0)].buf.pv = (void*)_primROut;
   _pra[(_numIn[0] + 0)].buf.nLen = sizeof(_primROut);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 0, 1, 0, 0), _pra));
   _COPY(_rout0, 0, _primROut, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_version)(remote_handle64 _handle, int* ver) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 25;
   return _stub_method_21(_handle, _mid, (int*)ver);
}
static __inline int _stub_method_22(remote_handle64 _handle, uint32_t _mid, const char* _in0[1], unsigned int _rout1[1]) {
   int _in0Len[1];
   int _numIn[1];
   remote_arg _pra[3];
   uint32_t _primIn[1];
   uint32_t _primROut[1];
   remote_arg* _praIn;
   int _nErr = 0;
   _numIn[0] = 1;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _pra[(_numIn[0] + 1)].buf.pv = (void*)_primROut;
   _pra[(_numIn[0] + 1)].buf.nLen = sizeof(_primROut);
   _in0Len[0] = (1 + strlen(_in0[0]));
   _COPY(_primIn, 0, _in0Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in0[0];
   _praIn[0].buf.nLen = (1 * _in0Len[0]);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 2, 1, 0, 0), _pra));
   _COPY(_rout1, 0, _primROut, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_op_name_to_id)(remote_handle64 _handle, const char* name, unsigned int* node_id) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 26;
   return _stub_method_22(_handle, _mid, (const char**)&name, (unsigned int*)node_id);
}
static __inline int _stub_method_23(remote_handle64 _handle, uint32_t _mid, unsigned int _in0[1], char* _rout1[1], int _rout1Len[1]) {
   int _numIn[1];
   remote_arg _pra[2];
   uint32_t _primIn[2];
   remote_arg* _praIn;
   remote_arg* _praROut;
   int _nErr = 0;
   _numIn[0] = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _rout1Len, 0, 4);
   _praIn = (_pra + 1);
   _praROut = (_praIn + _numIn[0] + 0);
   _praROut[0].buf.pv = _rout1[0];
   _praROut[0].buf.nLen = (1 * _rout1Len[0]);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 1, 0, 0), _pra));
   if(_rout1Len[0] > 0)
   {
      _rout1[0][(_rout1Len[0] - 1)] = 0;
   }
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_op_id_to_name)(remote_handle64 _handle, unsigned int node_id, char* name, int nameLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 27;
   return _stub_method_23(_handle, _mid, (unsigned int*)&node_id, (char**)&name, (int*)&nameLen);
}
static __inline int _stub_method_24(remote_handle64 _handle, uint32_t _mid, hexagon_nn_nn_id _in0[1], unsigned int _rout1[1]) {
   int _numIn[1];
   remote_arg _pra[2];
   uint32_t _primIn[1];
   uint32_t _primROut[1];
   int _nErr = 0;
   _numIn[0] = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _pra[(_numIn[0] + 1)].buf.pv = (void*)_primROut;
   _pra[(_numIn[0] + 1)].buf.nLen = sizeof(_primROut);
   _COPY(_primIn, 0, _in0, 0, 4);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 1, 0, 0), _pra));
   _COPY(_rout1, 0, _primROut, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_get_num_nodes_in_graph)(remote_handle64 _handle, hexagon_nn_nn_id id, unsigned int* num_nodes) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 28;
   return _stub_method_24(_handle, _mid, (hexagon_nn_nn_id*)&id, (unsigned int*)num_nodes);
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_disable_dcvs)(remote_handle64 _handle) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 29;
   return _stub_method(_handle, _mid);
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_GetHexagonBinaryVersion)(remote_handle64 _handle, int* ver) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 30;
   return _stub_method_21(_handle, _mid, (int*)ver);
}
static __inline int _stub_method_25(remote_handle64 _handle, uint32_t _mid, uint32_t _in0[1], const unsigned char* _in1[1], int _in1Len[1]) {
   remote_arg _pra[2];
   uint32_t _primIn[2];
   remote_arg* _praIn;
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in1[0];
   _praIn[0].buf.nLen = (1 * _in1Len[0]);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 2, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_PrintLog)(remote_handle64 _handle, const unsigned char* buf, int bufLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 31;
   return _stub_method_25(_handle, 31, &_mid, (const unsigned char**)&buf, (int*)&bufLen);
}
static __inline int _stub_unpack(remote_arg* _praROutPost, remote_arg* _ppraROutPost[1], void* _primROut, unsigned int _rout0[1], unsigned int _rout1[1], unsigned int _rout2[1], unsigned int _rout3[1], unsigned char* _rout4[1], int _rout4Len[1], unsigned int _rout5[1], unsigned int _rout6[1]) {
   int _nErr = 0;
   remote_arg* _praROutPostStart = _praROutPost;
   remote_arg** _ppraROutPostStart = _ppraROutPost;
   _ppraROutPost = &_praROutPost;
   _COPY(_rout0, 0, _primROut, 0, 4);
   _COPY(_rout1, 0, _primROut, 4, 4);
   _COPY(_rout2, 0, _primROut, 8, 4);
   _COPY(_rout3, 0, _primROut, 12, 4);
   _COPY(_rout5, 0, _primROut, 16, 4);
   _COPY(_rout6, 0, _primROut, 20, 4);
   _ppraROutPostStart[0] += (_praROutPost - _praROutPostStart) +1;
   return _nErr;
}
static __inline int _stub_unpack_1(remote_arg* _praROutPost, remote_arg* _ppraROutPost[1], void* _primROut, hexagon_nn_tensordef _rout0[SLIM_IFPTR32(8, 5)]) {
   int _nErr = 0;
   _allocator _al[1] = {{0}};
   remote_arg* _praROutPostStart = _praROutPost;
   remote_arg** _ppraROutPostStart = _ppraROutPost;
   _ppraROutPost = &_praROutPost;
   _allocator_init(_al, 0, 0);
   _TRY(_nErr, _stub_unpack((_praROutPost + 0), _ppraROutPost, ((char*)_primROut + 0), (unsigned int*)&(((uint32_t*)_rout0)[0]), (unsigned int*)&(((uint32_t*)_rout0)[1]), (unsigned int*)&(((uint32_t*)_rout0)[2]), (unsigned int*)&(((uint32_t*)_rout0)[3]), SLIM_IFPTR32((unsigned char**)&(((uint32_t*)_rout0)[4]), (unsigned char**)&(((uint64_t*)_rout0)[2])), SLIM_IFPTR32((int*)&(((uint32_t*)_rout0)[5]), (int*)&(((uint32_t*)_rout0)[6])), SLIM_IFPTR32((unsigned int*)&(((uint32_t*)_rout0)[6]), (unsigned int*)&(((uint32_t*)_rout0)[7])), SLIM_IFPTR32((unsigned int*)&(((uint32_t*)_rout0)[7]), (unsigned int*)&(((uint32_t*)_rout0)[8]))));
   _ppraROutPostStart[0] += (_praROutPost - _praROutPostStart) +0;
   _CATCH(_nErr) {}
   _allocator_deinit(_al);
   return _nErr;
}
static __inline int _stub_unpack_2(remote_arg* _praROutPost, remote_arg* _ppraROutPost[1], void* _primROut, const hexagon_nn_tensordef _in0[SLIM_IFPTR32(8, 5)]) {
   int _nErr = 0;
   remote_arg* _praROutPostStart = _praROutPost;
   remote_arg** _ppraROutPostStart = _ppraROutPost;
   _ppraROutPost = &_praROutPost;
   _ppraROutPostStart[0] += (_praROutPost - _praROutPostStart) +0;
   return _nErr;
}
static __inline int _stub_pack(_allocator* _al, remote_arg* _praIn, remote_arg* _ppraIn[1], remote_arg* _praROut, remote_arg* _ppraROut[1], remote_arg* _praHIn, remote_arg* _ppraHIn[1], remote_arg* _praHROut, remote_arg* _ppraHROut[1], void* _primIn, void* _primROut, unsigned int _rout0[1], unsigned int _rout1[1], unsigned int _rout2[1], unsigned int _rout3[1], unsigned char* _rout4[1], int _rout4Len[1], unsigned int _rout5[1], unsigned int _rout6[1]) {
   int _nErr = 0;
   remote_arg* _praInStart = _praIn;
   remote_arg** _ppraInStart = _ppraIn;
   remote_arg* _praROutStart = _praROut;
   remote_arg** _ppraROutStart = _ppraROut;
   _ppraIn = &_praIn;
   _ppraROut = &_praROut;
   _COPY(_primIn, 0, _rout4Len, 0, 4);
   _praROut[0].buf.pv = _rout4[0];
   _praROut[0].buf.nLen = (1 * _rout4Len[0]);
   _ppraInStart[0] += (_praIn - _praInStart) + 0;
   _ppraROutStart[0] += (_praROut - _praROutStart) +1;
   return _nErr;
}
static __inline int _stub_pack_1(_allocator* _al, remote_arg* _praIn, remote_arg* _ppraIn[1], remote_arg* _praROut, remote_arg* _ppraROut[1], remote_arg* _praHIn, remote_arg* _ppraHIn[1], remote_arg* _praHROut, remote_arg* _ppraHROut[1], void* _primIn, void* _primROut, hexagon_nn_tensordef _rout0[SLIM_IFPTR32(8, 5)]) {
   int _nErr = 0;
   remote_arg* _praInStart = _praIn;
   remote_arg** _ppraInStart = _ppraIn;
   remote_arg* _praROutStart = _praROut;
   remote_arg** _ppraROutStart = _ppraROut;
   _ppraIn = &_praIn;
   _ppraROut = &_praROut;
   _TRY(_nErr, _stub_pack(_al, (_praIn + 0), _ppraIn, (_praROut + 0), _ppraROut, _praHIn, _ppraHIn, _praHROut, _ppraHROut, ((char*)_primIn + 0), ((char*)_primROut + 0), (unsigned int*)&(((uint32_t*)_rout0)[0]), (unsigned int*)&(((uint32_t*)_rout0)[1]), (unsigned int*)&(((uint32_t*)_rout0)[2]), (unsigned int*)&(((uint32_t*)_rout0)[3]), SLIM_IFPTR32((unsigned char**)&(((uint32_t*)_rout0)[4]), (unsigned char**)&(((uint64_t*)_rout0)[2])), SLIM_IFPTR32((int*)&(((uint32_t*)_rout0)[5]), (int*)&(((uint32_t*)_rout0)[6])), SLIM_IFPTR32((unsigned int*)&(((uint32_t*)_rout0)[6]), (unsigned int*)&(((uint32_t*)_rout0)[7])), SLIM_IFPTR32((unsigned int*)&(((uint32_t*)_rout0)[7]), (unsigned int*)&(((uint32_t*)_rout0)[8]))));
   _ppraInStart[0] += (_praIn - _praInStart) + 0;
   _ppraROutStart[0] += (_praROut - _praROutStart) +0;
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline int _stub_pack_2(_allocator* _al, remote_arg* _praIn, remote_arg* _ppraIn[1], remote_arg* _praROut, remote_arg* _ppraROut[1], remote_arg* _praHIn, remote_arg* _ppraHIn[1], remote_arg* _praHROut, remote_arg* _ppraHROut[1], void* _primIn, void* _primROut, unsigned int _in0[1], unsigned int _in1[1], unsigned int _in2[1], unsigned int _in3[1], const unsigned char* _in4[1], int _in4Len[1], unsigned int _in5[1], unsigned int _in6[1]) {
   int _nErr = 0;
   remote_arg* _praInStart = _praIn;
   remote_arg** _ppraInStart = _ppraIn;
   remote_arg* _praROutStart = _praROut;
   remote_arg** _ppraROutStart = _ppraROut;
   _ppraIn = &_praIn;
   _ppraROut = &_praROut;
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _in2, 0, 4);
   _COPY(_primIn, 12, _in3, 0, 4);
   _COPY(_primIn, 16, _in4Len, 0, 4);
   _praIn[0].buf.pv = (void*) _in4[0];
   _praIn[0].buf.nLen = (1 * _in4Len[0]);
   _COPY(_primIn, 20, _in5, 0, 4);
   _COPY(_primIn, 24, _in6, 0, 4);
   _ppraInStart[0] += (_praIn - _praInStart) + 1;
   _ppraROutStart[0] += (_praROut - _praROutStart) +0;
   return _nErr;
}
static __inline int _stub_pack_3(_allocator* _al, remote_arg* _praIn, remote_arg* _ppraIn[1], remote_arg* _praROut, remote_arg* _ppraROut[1], remote_arg* _praHIn, remote_arg* _ppraHIn[1], remote_arg* _praHROut, remote_arg* _ppraHROut[1], void* _primIn, void* _primROut, const hexagon_nn_tensordef _in0[SLIM_IFPTR32(8, 5)]) {
   int _nErr = 0;
   remote_arg* _praInStart = _praIn;
   remote_arg** _ppraInStart = _ppraIn;
   remote_arg* _praROutStart = _praROut;
   remote_arg** _ppraROutStart = _ppraROut;
   _ppraIn = &_praIn;
   _ppraROut = &_praROut;
   _TRY(_nErr, _stub_pack_2(_al, (_praIn + 0), _ppraIn, (_praROut + 0), _ppraROut, _praHIn, _ppraHIn, _praHROut, _ppraHROut, ((char*)_primIn + 0), 0, (unsigned int*)&(((uint32_t*)_in0)[0]), (unsigned int*)&(((uint32_t*)_in0)[1]), (unsigned int*)&(((uint32_t*)_in0)[2]), (unsigned int*)&(((uint32_t*)_in0)[3]), SLIM_IFPTR32((const unsigned char**)&(((uint32_t*)_in0)[4]), (const unsigned char**)&(((uint64_t*)_in0)[2])), SLIM_IFPTR32((int*)&(((uint32_t*)_in0)[5]), (int*)&(((uint32_t*)_in0)[6])), SLIM_IFPTR32((unsigned int*)&(((uint32_t*)_in0)[6]), (unsigned int*)&(((uint32_t*)_in0)[7])), SLIM_IFPTR32((unsigned int*)&(((uint32_t*)_in0)[7]), (unsigned int*)&(((uint32_t*)_in0)[8]))));
   _ppraInStart[0] += (_praIn - _praInStart) + 0;
   _ppraROutStart[0] += (_praROut - _praROutStart) +0;
   _CATCH(_nErr) {}
   return _nErr;
}
static __inline void _count(int _numIn[1], int _numROut[1], int _numInH[1], int _numROutH[1], unsigned int _rout0[1], unsigned int _rout1[1], unsigned int _rout2[1], unsigned int _rout3[1], unsigned char* _rout4[1], int _rout4Len[1], unsigned int _rout5[1], unsigned int _rout6[1]) {
   _numIn[0] += 0;
   _numROut[0] += 1;
   _numInH[0] += 0;
   _numROutH[0] += 0;
}
static __inline void _count_1(int _numIn[1], int _numROut[1], int _numInH[1], int _numROutH[1], hexagon_nn_tensordef _rout0[SLIM_IFPTR32(8, 5)]) {
   _numIn[0] += 0;
   _numROut[0] += 0;
   _numInH[0] += 0;
   _numROutH[0] += 0;
   _count(_numIn, _numROut, _numInH, _numROutH, (unsigned int*)&(((uint32_t*)_rout0)[0]), (unsigned int*)&(((uint32_t*)_rout0)[1]), (unsigned int*)&(((uint32_t*)_rout0)[2]), (unsigned int*)&(((uint32_t*)_rout0)[3]), SLIM_IFPTR32((unsigned char**)&(((uint32_t*)_rout0)[4]), (unsigned char**)&(((uint64_t*)_rout0)[2])), SLIM_IFPTR32((int*)&(((uint32_t*)_rout0)[5]), (int*)&(((uint32_t*)_rout0)[6])), SLIM_IFPTR32((unsigned int*)&(((uint32_t*)_rout0)[6]), (unsigned int*)&(((uint32_t*)_rout0)[7])), SLIM_IFPTR32((unsigned int*)&(((uint32_t*)_rout0)[7]), (unsigned int*)&(((uint32_t*)_rout0)[8])));
}
static __inline void _count_2(int _numIn[1], int _numROut[1], int _numInH[1], int _numROutH[1], unsigned int _in0[1], unsigned int _in1[1], unsigned int _in2[1], unsigned int _in3[1], const unsigned char* _in4[1], int _in4Len[1], unsigned int _in5[1], unsigned int _in6[1]) {
   _numIn[0] += 1;
   _numROut[0] += 0;
   _numInH[0] += 0;
   _numROutH[0] += 0;
}
static __inline void _count_3(int _numIn[1], int _numROut[1], int _numInH[1], int _numROutH[1], const hexagon_nn_tensordef _in0[SLIM_IFPTR32(8, 5)]) {
   _numIn[0] += 0;
   _numROut[0] += 0;
   _numInH[0] += 0;
   _numROutH[0] += 0;
   _count_2(_numIn, _numROut, _numInH, _numROutH, (unsigned int*)&(((uint32_t*)_in0)[0]), (unsigned int*)&(((uint32_t*)_in0)[1]), (unsigned int*)&(((uint32_t*)_in0)[2]), (unsigned int*)&(((uint32_t*)_in0)[3]), SLIM_IFPTR32((const unsigned char**)&(((uint32_t*)_in0)[4]), (const unsigned char**)&(((uint64_t*)_in0)[2])), SLIM_IFPTR32((int*)&(((uint32_t*)_in0)[5]), (int*)&(((uint32_t*)_in0)[6])), SLIM_IFPTR32((unsigned int*)&(((uint32_t*)_in0)[6]), (unsigned int*)&(((uint32_t*)_in0)[7])), SLIM_IFPTR32((unsigned int*)&(((uint32_t*)_in0)[7]), (unsigned int*)&(((uint32_t*)_in0)[8])));
}
static __inline int _stub_method_26(remote_handle64 _handle, uint32_t _mid, uint32_t _in0[1], hexagon_nn_nn_id _in1[1], const hexagon_nn_tensordef* _in2[1], int _in2Len[1], hexagon_nn_tensordef* _rout3[1], int _rout3Len[1]) {
   remote_arg* _pra;
   int _numIn[1];
   int _numROut[1];
   int _numInH[1];
   int _numROutH[1];
   char* _seq_nat2;
   int _ii;
   char* _seq_nat3;
   _allocator _al[1] = {{0}};
   uint32_t _primIn[4];
   remote_arg* _praIn;
   remote_arg* _praROut;
   remote_arg* _praROutPost;
   remote_arg** _ppraROutPost = &_praROutPost;
   remote_arg** _ppraIn = &_praIn;
   remote_arg** _ppraROut = &_praROut;
   remote_arg* _praHIn = 0;
   remote_arg** _ppraHIn = &_praHIn;
   remote_arg* _praHROut = 0;
   remote_arg** _ppraHROut = &_praHROut;
   char* _seq_primIn2;
   int _nErr = 0;
   char* _seq_primIn3;
   char* _seq_primROut3;
   _numIn[0] = 2;
   _numROut[0] = 1;
   _numInH[0] = 0;
   _numROutH[0] = 0;
   for(_ii = 0, _seq_nat2 = (char*)_in2[0];_ii < (int)_in2Len[0];++_ii, _seq_nat2 = (_seq_nat2 + SLIM_IFPTR32(32, 40)))
   {
      _count_3(_numIn, _numROut, _numInH, _numROutH, SLIM_IFPTR32((const hexagon_nn_tensordef*)&(((uint32_t*)_seq_nat2)[0]), (const hexagon_nn_tensordef*)&(((uint64_t*)_seq_nat2)[0])));
   }
   for(_ii = 0, _seq_nat3 = (char*)_rout3[0];_ii < (int)_rout3Len[0];++_ii, _seq_nat3 = (_seq_nat3 + SLIM_IFPTR32(32, 40)))
   {
      _count_1(_numIn, _numROut, _numInH, _numROutH, SLIM_IFPTR32((hexagon_nn_tensordef*)&(((uint32_t*)_seq_nat3)[0]), (hexagon_nn_tensordef*)&(((uint64_t*)_seq_nat3)[0])));
   }
   _allocator_init(_al, 0, 0);
   _ALLOCATE(_nErr, _al, ((((((((_numIn[0] + _numROut[0]) + _numInH[0]) + _numROutH[0]) + 1) + 0) + 0) + 0) * sizeof(_pra[0])), 4, _pra);
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _praIn = (_pra + 1);
   _praROut = (_praIn + _numIn[0] + 0);
   _praROutPost = _praROut;
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _in2Len, 0, 4);
   if(_praHIn == 0)
   {
      _praHIn = ((_praROut + _numROut[0]) + 0);
   }
   if(_praHROut == 0)
      (_praHROut = _praHIn + _numInH[0] + 0);
   _ALLOCATE(_nErr, _al, (_in2Len[0] * 28), 4, _praIn[0].buf.pv);
   _praIn[0].buf.nLen = (28 * _in2Len[0]);
   for(_ii = 0, _seq_primIn2 = (char*)_praIn[0].buf.pv, _seq_nat2 = (char*)_in2[0];_ii < (int)_in2Len[0];++_ii, _seq_primIn2 = (_seq_primIn2 + 28), _seq_nat2 = (_seq_nat2 + SLIM_IFPTR32(32, 40)))
   {
      _TRY(_nErr, _stub_pack_3(_al, (_praIn + 1), _ppraIn, (_praROut + 0), _ppraROut, _praHIn, _ppraHIn, _praHROut, _ppraHROut, _seq_primIn2, 0, SLIM_IFPTR32((const hexagon_nn_tensordef*)&(((uint32_t*)_seq_nat2)[0]), (const hexagon_nn_tensordef*)&(((uint64_t*)_seq_nat2)[0]))));
   }
   _COPY(_primIn, 12, _rout3Len, 0, 4);
   _ALLOCATE(_nErr, _al, (_rout3Len[0] * 4), 4, _praIn[1].buf.pv);
   _praIn[1].buf.nLen = (4 * _rout3Len[0]);
   _ALLOCATE(_nErr, _al, (_rout3Len[0] * 24), 4, _praROut[0].buf.pv);
   _praROut[0].buf.nLen = (24 * _rout3Len[0]);
   for(_ii = 0, _seq_primIn3 = (char*)_praIn[1].buf.pv, _seq_primROut3 = (char*)_praROut[0].buf.pv, _seq_nat3 = (char*)_rout3[0];_ii < (int)_rout3Len[0];++_ii, _seq_primIn3 = (_seq_primIn3 + 4), _seq_primROut3 = (_seq_primROut3 + 24), _seq_nat3 = (_seq_nat3 + SLIM_IFPTR32(32, 40)))
   {
      _TRY(_nErr, _stub_pack_1(_al, (_praIn + 2), _ppraIn, (_praROut + 1), _ppraROut, _praHIn, _ppraHIn, _praHROut, _ppraHROut, _seq_primIn3, _seq_primROut3, SLIM_IFPTR32((hexagon_nn_tensordef*)&(((uint32_t*)_seq_nat3)[0]), (hexagon_nn_tensordef*)&(((uint64_t*)_seq_nat3)[0]))));
   }
   _ASSERT(_nErr, (_numInH[0] + 0) <= 15);
   _ASSERT(_nErr, (_numROutH[0] + 0) <= 15);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, (_numIn[0] + 1), (_numROut[0] + 0), (_numInH[0] + 0), (_numROutH[0] + 0)), _pra));
   for(_ii = 0, _seq_nat2 = (char*)_in2[0];_ii < (int)_in2Len[0];++_ii, _seq_nat2 = (_seq_nat2 + SLIM_IFPTR32(32, 40)))
   {
      _TRY(_nErr, _stub_unpack_2((_praROutPost + 0), _ppraROutPost, 0, SLIM_IFPTR32((const hexagon_nn_tensordef*)&(((uint32_t*)_seq_nat2)[0]), (const hexagon_nn_tensordef*)&(((uint64_t*)_seq_nat2)[0]))));
   }
   for(_ii = 0, _seq_primROut3 = (char*)_praROutPost[0].buf.pv, _seq_nat3 = (char*)_rout3[0];_ii < (int)_rout3Len[0];++_ii, _seq_primROut3 = (_seq_primROut3 + 24), _seq_nat3 = (_seq_nat3 + SLIM_IFPTR32(32, 40)))
   {
      _TRY(_nErr, _stub_unpack_1((_praROutPost + 1), _ppraROutPost, _seq_primROut3, SLIM_IFPTR32((hexagon_nn_tensordef*)&(((uint32_t*)_seq_nat3)[0]), (hexagon_nn_tensordef*)&(((uint64_t*)_seq_nat3)[0]))));
   }
   _CATCH(_nErr) {}
   _allocator_deinit(_al);
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_execute_new)(remote_handle64 _handle, hexagon_nn_nn_id id, const hexagon_nn_tensordef* inputs, int inputsLen, hexagon_nn_tensordef* outputs, int outputsLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 32;
   return _stub_method_26(_handle, 31, &_mid, (hexagon_nn_nn_id*)&id, (const hexagon_nn_tensordef**)&inputs, (int*)&inputsLen, (hexagon_nn_tensordef**)&outputs, (int*)&outputsLen);
}
static __inline int _stub_method_27(remote_handle64 _handle, uint32_t _mid, uint32_t _in0[1], hexagon_nn_nn_id _rout1[1], const hexagon_nn_initinfo _in2[1]) {
   int _numIn[1];
   remote_arg _pra[2];
   uint32_t _primIn[2];
   uint32_t _primROut[1];
   int _nErr = 0;
   _numIn[0] = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _pra[(_numIn[0] + 1)].buf.pv = (void*)_primROut;
   _pra[(_numIn[0] + 1)].buf.nLen = sizeof(_primROut);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in2, 0, 4);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 1, 0, 0), _pra));
   _COPY(_rout1, 0, _primROut, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_init_with_info)(remote_handle64 _handle, hexagon_nn_nn_id* g, const hexagon_nn_initinfo* info) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 33;
   return _stub_method_27(_handle, 31, &_mid, (hexagon_nn_nn_id*)g, (const hexagon_nn_initinfo*)info);
}
static __inline int _stub_method_28(remote_handle64 _handle, uint32_t _mid, uint32_t _in0[1], hexagon_nn_nn_id _in1[1], hexagon_nn_nn_id _in2[1], unsigned int _rout3[1]) {
   int _numIn[1];
   remote_arg _pra[2];
   uint32_t _primIn[3];
   uint32_t _primROut[1];
   int _nErr = 0;
   _numIn[0] = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _pra[(_numIn[0] + 1)].buf.pv = (void*)_primROut;
   _pra[(_numIn[0] + 1)].buf.nLen = sizeof(_primROut);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _in2, 0, 4);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 1, 0, 0), _pra));
   _COPY(_rout3, 0, _primROut, 0, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_get_nodetype)(remote_handle64 _handle, hexagon_nn_nn_id graph_id, hexagon_nn_nn_id node_id, unsigned int* node_type) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 34;
   return _stub_method_28(_handle, 31, &_mid, (hexagon_nn_nn_id*)&graph_id, (hexagon_nn_nn_id*)&node_id, (unsigned int*)node_type);
}
static __inline int _stub_method_29(remote_handle64 _handle, uint32_t _mid, uint32_t _in0[1], hexagon_nn_nn_id _in1[1], unsigned int _rout2[1], unsigned int _rout3[1]) {
   int _numIn[1];
   remote_arg _pra[2];
   uint32_t _primIn[2];
   uint32_t _primROut[2];
   int _nErr = 0;
   _numIn[0] = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _pra[(_numIn[0] + 1)].buf.pv = (void*)_primROut;
   _pra[(_numIn[0] + 1)].buf.nLen = sizeof(_primROut);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 1, 0, 0), _pra));
   _COPY(_rout2, 0, _primROut, 0, 4);
   _COPY(_rout3, 0, _primROut, 4, 4);
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_multi_execution_cycles)(remote_handle64 _handle, hexagon_nn_nn_id id, unsigned int* cycles_lo, unsigned int* cycles_hi) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 35;
   return _stub_method_29(_handle, 31, &_mid, (hexagon_nn_nn_id*)&id, (unsigned int*)cycles_lo, (unsigned int*)cycles_hi);
}
static __inline int _stub_method_30(remote_handle64 _handle, uint32_t _mid, uint32_t _in0[1], int _in1[1]) {
   remote_arg _pra[1];
   uint32_t _primIn[2];
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_get_power)(remote_handle64 _handle, int type) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 36;
   return _stub_method_30(_handle, 31, &_mid, (int*)&type);
}
static __inline int _stub_method_31(remote_handle64 _handle, uint32_t _mid, uint32_t _in0[1], hexagon_nn_nn_id _in1[1], const char* _in2[1], int _in3[1]) {
   int _in2Len[1];
   remote_arg _pra[2];
   uint32_t _primIn[4];
   remote_arg* _praIn;
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _in2Len[0] = (1 + strlen(_in2[0]));
   _COPY(_primIn, 8, _in2Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in2[0];
   _praIn[0].buf.nLen = (1 * _in2Len[0]);
   _COPY(_primIn, 12, _in3, 0, 4);
   _TRY(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 2, 0, 0, 0), _pra));
   _CATCH(_nErr) {}
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(hexagon_nn_domains_set_graph_option)(remote_handle64 _handle, hexagon_nn_nn_id id, const char* name, int value) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 37;
   return _stub_method_31(_handle, 31, &_mid, (hexagon_nn_nn_id*)&id, (const char**)&name, (int*)&value);
}
#ifdef __cplusplus
}
#endif
#endif //_HEXAGON_NN_DOMAINS_STUB_H
