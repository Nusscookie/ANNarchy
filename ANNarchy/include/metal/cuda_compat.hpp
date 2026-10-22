/*
 *    cuda_compat.hpp
 *
 *    This file is part of ANNarchy.
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    ANNarchy is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once

/*
 * The Metal paradigm reuses the host code generated for CUDA. This header
 * implements the subset of the CUDA runtime API used by this code on top of
 * the Metal runtime (MetalRuntime.h).
 */
#include <cfloat>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "MetalRuntime.h"

// CUDA function qualifiers are meaningless for host code
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __restrict__
#define __restrict__ __restrict
#endif

// Markers of kernel arguments for the Metal translator (generator/Metal/Translator.py)
#define __annmtl_constant__
#define __annmtl_deref__

typedef int cudaError_t;
enum : int {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 2,
    cudaErrorUnknown = 999
};

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

// Metal uses one in-order queue, so streams are only placeholders
typedef void* cudaStream_t;

struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
};

// Must have the same layout as curandState in the Metal shader prelude
struct curandState {
    unsigned int d;
    unsigned int v[5];
    int boxmuller_flag;
    float boxmuller_extra;
};

inline cudaError_t cudaMalloc(void** ptr, size_t size) {
    *ptr = annmtl_malloc(size);
    return (*ptr != nullptr) ? cudaSuccess : cudaErrorMemoryAllocation;
}

template <typename T>
inline cudaError_t cudaMalloc(T** ptr, size_t size) {
    return cudaMalloc(reinterpret_cast<void**>(ptr), size);
}

inline cudaError_t cudaFree(void* ptr) {
    annmtl_free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind) {
    annmtl_memcpy(dst, src, count);
    return annmtl_peek_last_error() ? cudaErrorUnknown : cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t = nullptr) {
    return cudaMemcpy(dst, src, count, kind);
}

inline cudaError_t cudaMemset(void* ptr, int value, size_t count) {
    annmtl_memset(ptr, value, count);
    return cudaSuccess;
}

inline cudaError_t cudaGetLastError() {
    return annmtl_get_last_error() ? cudaErrorUnknown : cudaSuccess;
}

inline cudaError_t cudaPeekAtLastError() {
    return annmtl_peek_last_error() ? cudaErrorUnknown : cudaSuccess;
}

inline const char* cudaGetErrorString(cudaError_t err) {
    if (err == cudaSuccess)
        return "no error";
    if (err == cudaErrorMemoryAllocation)
        return "out of memory";
    return annmtl_error_string();
}

inline cudaError_t cudaDeviceSynchronize() {
    annmtl_sync();
    return annmtl_peek_last_error() ? cudaErrorUnknown : cudaSuccess;
}

inline cudaError_t cudaSetDevice(int device) {
    if (device != 0)
        fprintf(stderr, "ANNarchy (Metal): only the default device is supported, device %d is ignored.\n", device);
    return annmtl_init() == 0 ? cudaSuccess : cudaErrorUnknown;
}

inline cudaError_t cudaStreamCreate(cudaStream_t* stream) {
    *stream = nullptr;
    return cudaSuccess;
}

inline cudaError_t cudaStreamDestroy(cudaStream_t) {
    return cudaSuccess;
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t) {
    annmtl_sync();
    return cudaSuccess;
}

inline cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    size_t total = annmtl_recommended_working_set();
    size_t used = annmtl_allocated_bytes();
    *total_bytes = total;
    *free_bytes = (total > used) ? (total - used) : 0;
    return cudaSuccess;
}

// Host-side definition of the type traits used in the kernel declarations
#include "cuda_type_traits.cuh"
