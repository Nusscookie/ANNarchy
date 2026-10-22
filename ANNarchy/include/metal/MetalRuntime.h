/*
 *    MetalRuntime.h
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
 * Small runtime used by the Metal paradigm (macOS, Apple GPUs).
 *
 * All device memory is allocated as MTLBuffers in shared storage mode, i.e. on
 * Apple Silicon the returned pointer is directly accessible by the CPU. The
 * runtime keeps track of the allocations so that pointers (also pointers into
 * an allocation) can be bound to kernel arguments.
 *
 * Kernels are encoded into one compute command encoder and committed in
 * batches. Host-to-device and device-to-device copies are ordered with the
 * queued GPU work without waiting for it; only reads from device memory
 * (device-to-host copies) wait for the GPU to finish.
 *
 * The interface is plain C, so that the Objective-C++ implementation and the
 * generated C++ code do not depend on each other's compiler or standard library.
 */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialization (called implicitly by all other functions)
int annmtl_init(void);
const char* annmtl_device_name(void);

// Shader library: the precompiled library (metallib) is used if it exists,
// otherwise the source code is compiled at runtime.
void annmtl_register_library(const char* metallib_path, const char* source);
int annmtl_load_library(void);

// Memory management
void* annmtl_malloc(size_t bytes);
void annmtl_free(void* ptr);
int annmtl_is_device_ptr(const void* ptr);
size_t annmtl_allocated_bytes(void);
size_t annmtl_recommended_working_set(void);

// Memory transfers, ordered with respect to the queued kernels
void annmtl_memcpy(void* dst, const void* src, size_t bytes);
void annmtl_memset(void* dst, int value, size_t bytes);

// Kernel execution
int annmtl_pipeline(const char* name);
unsigned annmtl_max_threads_per_group(int pipeline);
void annmtl_dispatch(int pipeline,
                     unsigned groups_x, unsigned groups_y, unsigned groups_z,
                     unsigned threads_x, unsigned threads_y, unsigned threads_z,
                     const void* const* buffers, int num_buffers,
                     const void* args, size_t args_size,
                     size_t threadgroup_memory);
// Same, written[i] != 0 if the kernel may write into buffers[i] (all buffers
// are assumed to be written by annmtl_dispatch). Kernels that do not access
// the same buffers run concurrently on the GPU.
void annmtl_dispatch_rw(int pipeline,
                        unsigned groups_x, unsigned groups_y, unsigned groups_z,
                        unsigned threads_x, unsigned threads_y, unsigned threads_z,
                        const void* const* buffers, const unsigned char* written, int num_buffers,
                        const void* args, size_t args_size,
                        size_t threadgroup_memory);

// Synchronization
void annmtl_flush(void);
void annmtl_sync(void);

// Error handling: returns 0 if no error occurred since the last call, the
// message of the last error is available through annmtl_error_string().
int annmtl_get_last_error(void);
int annmtl_peek_last_error(void);
const char* annmtl_error_string(void);

#ifdef __cplusplus
}
#endif
