/*
 *    MetalRuntime.mm
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
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalRuntime.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#if !__has_feature(objc_arc)
#error "MetalRuntime.mm must be compiled with -fobjc-arc"
#endif

namespace {

// Number of kernels encoded into one command buffer before it is committed
// (without waiting), so that the GPU can start while the CPU encodes further work.
constexpr int kDispatchesPerCommandBuffer = 256;

// Kernel arguments larger than this are not passed with setBytes()
constexpr size_t kMaxInlineBytes = 4096;

struct Allocation {
    id<MTLBuffer> buffer;
    size_t size;
};

struct State {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    bool library_failed = false;

    std::string metallib_path;
    std::string source;

    std::map<uintptr_t, Allocation> allocations;
    size_t allocated_bytes = 0;

    std::vector<id<MTLComputePipelineState>> pipelines;
    std::vector<std::string> pipeline_names;
    std::map<std::string, int> pipeline_ids;

    id<MTLCommandBuffer> command_buffer = nil;   // currently recorded
    id<MTLComputeCommandEncoder> encoder = nil;
    id<MTLCommandBuffer> last_committed = nil;  // most recently committed
    int dispatches = 0;

    // built-in kernels for memory transfers
    int copy_u32 = -1, copy_u8 = -1, fill_u32 = -1, fill_u8 = -1;

    // The kernels of a compute encoder run concurrently, a memory barrier is
    // inserted before a kernel accessing a buffer written by a previous kernel
    // (or writing a buffer accessed by a previous kernel) since the last
    // barrier. ANNARCHY_METAL_SERIAL=1 executes the kernels one after another.
    bool concurrent = std::getenv("ANNARCHY_METAL_SERIAL") == nullptr;
    std::unordered_set<const void*> epoch_reads, epoch_writes;

    // Optional profiling (environment variable ANNARCHY_METAL_PROFILE=1): each
    // kernel is executed in its own command buffer and its GPU time is measured.
    bool profile = std::getenv("ANNARCHY_METAL_PROFILE") != nullptr;
    std::map<std::string, std::pair<double, long>> profile_data;  // name -> (seconds, calls)

    void print_profile() {
        if (profile_data.empty())
            return;
        std::vector<std::pair<std::string, std::pair<double, long>>> entries(profile_data.begin(), profile_data.end());
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
        double total = 0.0;
        for (auto& e : entries)
            total += e.second.first;
        fprintf(stderr, "\nANNarchy (Metal) kernel profile: %.3f s GPU time\n", total);
        fprintf(stderr, "%-40s %10s %10s %12s %7s\n", "kernel", "calls", "total [s]", "per call [us]", "share");
        for (auto& e : entries) {
            fprintf(stderr, "%-40s %10ld %10.4f %12.2f %6.1f%%\n", e.first.c_str(), e.second.second, e.second.first,
                    1e6 * e.second.first / e.second.second, 100.0 * e.second.first / total);
        }
    }

    ~State() {
        if (profile)
            print_profile();
        // Metal asserts if an encoder is released without endEncoding(), e.g. when
        // the process exits while kernels are queued.
        @autoreleasepool {
            if (encoder != nil)
                [encoder endEncoding];
            if (command_buffer != nil) {
                [command_buffer commit];
                [command_buffer waitUntilCompleted];
            }
            if (last_committed != nil)
                [last_committed waitUntilCompleted];
        }
    }
};

State& state() {
    static State s;
    return s;
}

// Errors may be reported from completion handlers (other threads)
std::mutex error_mutex;
std::string error_message;
std::atomic<int> error_flag{0};

void set_error(const std::string& msg) {
    {
        std::lock_guard<std::mutex> lock(error_mutex);
        error_message = msg;
    }
    error_flag = 1;
    fprintf(stderr, "ANNarchy (Metal): %s\n", msg.c_str());
}

bool ensure_device() {
    State& s = state();
    if (s.device != nil)
        return true;

    s.device = MTLCreateSystemDefaultDevice();
    if (s.device == nil) {
        set_error("no Metal device available.");
        return false;
    }
    s.queue = [s.device newCommandQueueWithMaxCommandBufferCount:256];
    return s.queue != nil;
}

bool ensure_library() {
    State& s = state();
    if (s.library != nil)
        return true;
    if (s.library_failed || !ensure_device())
        return false;

    NSError* error = nil;

    // precompiled library
    if (!s.metallib_path.empty()) {
        NSString* path = [NSString stringWithUTF8String:s.metallib_path.c_str()];
        if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
            s.library = [s.device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
            if (s.library == nil) {
                fprintf(stderr, "ANNarchy (Metal): could not load %s (%s), compiling the shader source instead.\n",
                        s.metallib_path.c_str(), error.localizedDescription.UTF8String);
            }
        }
    }

    // compile the shader source at runtime
    if (s.library == nil && !s.source.empty()) {
        MTLCompileOptions* options = [MTLCompileOptions new];
        // keep IEEE-754 semantics closer to the CPU implementation
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        error = nil;
        s.library = [s.device newLibraryWithSource:[NSString stringWithUTF8String:s.source.c_str()]
                                           options:options
                                             error:&error];
        if (s.library == nil) {
            set_error(std::string("shader compilation failed:\n") + error.localizedDescription.UTF8String);
        }
    }

    if (s.library == nil) {
        if (s.metallib_path.empty() && s.source.empty())
            set_error("no shader library was registered.");
        s.library_failed = true;
        return false;
    }
    return true;
}

// Resolve a pointer to the allocation containing it.
bool resolve(const void* ptr, id<MTLBuffer>* buffer, size_t* offset) {
    State& s = state();
    uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    auto it = s.allocations.upper_bound(p);
    if (it == s.allocations.begin())
        return false;
    --it;
    if (p >= it->first + it->second.size)
        return false;
    *buffer = it->second.buffer;
    *offset = p - it->first;
    return true;
}

bool has_pending_work() {
    State& s = state();
    if (s.dispatches > 0)
        return true;
    if (s.last_committed != nil) {
        MTLCommandBufferStatus status = s.last_committed.status;
        if (status != MTLCommandBufferStatusCompleted && status != MTLCommandBufferStatusError)
            return true;
    }
    return false;
}

id<MTLComputeCommandEncoder> current_encoder() {
    State& s = state();
    if (s.command_buffer == nil)
        s.command_buffer = [s.queue commandBuffer];
    if (s.encoder == nil) {
        s.encoder = s.concurrent ? [s.command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                                 : [s.command_buffer computeCommandEncoder];
        s.epoch_reads.clear();
        s.epoch_writes.clear();
    }
    return s.encoder;
}

// Inserts a memory barrier if the next kernel conflicts with the kernels
// encoded since the last barrier. uses: (buffer, written)
void track_hazards(id<MTLComputeCommandEncoder> enc, const std::pair<id<MTLBuffer>, bool>* uses, int num_uses) {
    State& s = state();
    if (!s.concurrent)
        return;
    bool conflict = false;
    for (int i = 0; i < num_uses && !conflict; i++) {
        const void* key = (__bridge const void*)uses[i].first;
        conflict = s.epoch_writes.count(key) > 0 || (uses[i].second && s.epoch_reads.count(key) > 0);
    }
    if (conflict) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        s.epoch_reads.clear();
        s.epoch_writes.clear();
    }
    for (int i = 0; i < num_uses; i++) {
        const void* key = (__bridge const void*)uses[i].first;
        if (uses[i].second)
            s.epoch_writes.insert(key);
        else
            s.epoch_reads.insert(key);
    }
}

void commit_current() {
    State& s = state();
    if (s.encoder != nil) {
        [s.encoder endEncoding];
        s.encoder = nil;
    }
    if (s.command_buffer != nil) {
        [s.command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            if (cb.status == MTLCommandBufferStatusError) {
                set_error(std::string("GPU execution failed: ") +
                          (cb.error ? cb.error.localizedDescription.UTF8String : "unknown error"));
            }
        }];
        [s.command_buffer commit];
        s.last_committed = s.command_buffer;
        s.command_buffer = nil;
    }
    s.dispatches = 0;
}

void after_dispatch() {
    State& s = state();
    s.dispatches++;
    if (s.dispatches >= kDispatchesPerCommandBuffer)
        commit_current();
}

int pipeline_by_name(const char* name) {
    State& s = state();
    auto it = s.pipeline_ids.find(name);
    if (it != s.pipeline_ids.end())
        return it->second;

    if (!ensure_library())
        return -1;

    id<MTLFunction> function = [s.library newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (function == nil) {
        set_error(std::string("kernel '") + name + "' not found in the shader library.");
        return -1;
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pso = [s.device newComputePipelineStateWithFunction:function error:&error];
    if (pso == nil) {
        set_error(std::string("could not create the pipeline for '") + name + "': " +
                  error.localizedDescription.UTF8String);
        return -1;
    }

    int id_ = static_cast<int>(s.pipelines.size());
    s.pipelines.push_back(pso);
    s.pipeline_names.push_back(name);
    s.pipeline_ids[name] = id_;
    return id_;
}

// Encode a copy between two buffers on the GPU
void encode_copy(id<MTLBuffer> dst, size_t dst_off, id<MTLBuffer> src, size_t src_off, size_t bytes) {
    State& s = state();
    bool aligned = (dst_off % 4 == 0) && (src_off % 4 == 0) && (bytes % 4 == 0);

    int& pso_id = aligned ? s.copy_u32 : s.copy_u8;
    if (pso_id < 0)
        pso_id = pipeline_by_name(aligned ? "annmtl_copy_u32" : "annmtl_copy_u8");
    if (pso_id < 0)
        return;

    unsigned int count = static_cast<unsigned int>(aligned ? bytes / 4 : bytes);
    const unsigned int tpb = 256;
    unsigned int groups = (count + tpb - 1) / tpb;

    id<MTLComputeCommandEncoder> enc = current_encoder();
    std::pair<id<MTLBuffer>, bool> uses[2] = {{dst, true}, {src, false}};
    track_hazards(enc, uses, 2);
    [enc setComputePipelineState:s.pipelines[pso_id]];
    [enc setBuffer:dst offset:dst_off atIndex:0];
    [enc setBuffer:src offset:src_off atIndex:1];
    [enc setBytes:&count length:sizeof(count) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpb, 1, 1)];
    after_dispatch();
}

// Encode a fill of a buffer region with a byte value on the GPU
void encode_fill(id<MTLBuffer> dst, size_t dst_off, unsigned char value, size_t bytes) {
    State& s = state();
    bool aligned = (dst_off % 4 == 0) && (bytes % 4 == 0);

    int& pso_id = aligned ? s.fill_u32 : s.fill_u8;
    if (pso_id < 0)
        pso_id = pipeline_by_name(aligned ? "annmtl_fill_u32" : "annmtl_fill_u8");
    if (pso_id < 0)
        return;

    unsigned int args[2];
    args[0] = static_cast<unsigned int>(aligned ? bytes / 4 : bytes);
    args[1] = aligned ? (0x01010101u * value) : value;
    const unsigned int tpb = 256;
    unsigned int groups = (args[0] + tpb - 1) / tpb;

    id<MTLComputeCommandEncoder> enc = current_encoder();
    std::pair<id<MTLBuffer>, bool> uses[1] = {{dst, true}};
    track_hazards(enc, uses, 1);
    [enc setComputePipelineState:s.pipelines[pso_id]];
    [enc setBuffer:dst offset:dst_off atIndex:0];
    [enc setBytes:args length:sizeof(args) atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpb, 1, 1)];
    after_dispatch();
}

}  // namespace

extern "C" {

int annmtl_init(void) {
    @autoreleasepool {
        return ensure_device() ? 0 : -1;
    }
}

const char* annmtl_device_name(void) {
    @autoreleasepool {
        static std::string name;
        if (!ensure_device())
            return "none";
        name = state().device.name.UTF8String;
        return name.c_str();
    }
}

void annmtl_register_library(const char* metallib_path, const char* source) {
    State& s = state();
    s.metallib_path = metallib_path ? metallib_path : "";
    s.source = source ? source : "";
}

int annmtl_load_library(void) {
    @autoreleasepool {
        return ensure_library() ? 0 : -1;
    }
}

void* annmtl_malloc(size_t bytes) {
    @autoreleasepool {
        if (!ensure_device())
            return nullptr;
        State& s = state();

        // zero-sized allocations are valid in CUDA, but not in Metal
        size_t length = bytes > 0 ? bytes : 16;
        id<MTLBuffer> buffer = [s.device newBufferWithLength:length options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            set_error("could not allocate " + std::to_string(bytes) + " bytes.");
            return nullptr;
        }
        void* ptr = buffer.contents;
        s.allocations[reinterpret_cast<uintptr_t>(ptr)] = Allocation{buffer, length};
        s.allocated_bytes += length;
        return ptr;
    }
}

void annmtl_free(void* ptr) {
    @autoreleasepool {
        if (ptr == nullptr)
            return;
        State& s = state();
        auto it = s.allocations.find(reinterpret_cast<uintptr_t>(ptr));
        if (it == s.allocations.end())
            return;
        // command buffers retain the buffers they use, so pending work stays valid
        s.allocated_bytes -= it->second.size;
        s.allocations.erase(it);
    }
}

int annmtl_is_device_ptr(const void* ptr) {
    id<MTLBuffer> buffer;
    size_t offset;
    return resolve(ptr, &buffer, &offset) ? 1 : 0;
}

size_t annmtl_allocated_bytes(void) {
    return state().allocated_bytes;
}

size_t annmtl_recommended_working_set(void) {
    @autoreleasepool {
        if (!ensure_device())
            return 0;
        return static_cast<size_t>(state().device.recommendedMaxWorkingSetSize);
    }
}

void annmtl_memcpy(void* dst, const void* src, size_t bytes) {
    @autoreleasepool {
        if (bytes == 0 || dst == src)
            return;

        id<MTLBuffer> dst_buf = nil, src_buf = nil;
        size_t dst_off = 0, src_off = 0;
        bool dst_dev = resolve(dst, &dst_buf, &dst_off);
        bool src_dev = resolve(src, &src_buf, &src_off);

        if (src_dev && !dst_dev) {
            // device to host: the data must be complete
            annmtl_sync();
            std::memcpy(dst, src, bytes);
            return;
        }

        if (!dst_dev || !has_pending_work()) {
            // host to host, or the GPU is idle
            std::memcpy(dst, src, bytes);
            return;
        }

        if (!src_dev) {
            // host to device while kernels are queued: copy the data now into a
            // staging buffer, the transfer into the target is queued on the GPU.
            src_buf = [state().device newBufferWithBytes:src length:bytes options:MTLResourceStorageModeShared];
            src_off = 0;
            if (src_buf == nil) {
                annmtl_sync();
                std::memcpy(dst, src, bytes);
                return;
            }
        }
        encode_copy(dst_buf, dst_off, src_buf, src_off, bytes);
    }
}

void annmtl_memset(void* dst, int value, size_t bytes) {
    @autoreleasepool {
        if (bytes == 0)
            return;
        id<MTLBuffer> dst_buf = nil;
        size_t dst_off = 0;
        if (!resolve(dst, &dst_buf, &dst_off) || !has_pending_work()) {
            std::memset(dst, value, bytes);
            return;
        }
        encode_fill(dst_buf, dst_off, static_cast<unsigned char>(value), bytes);
    }
}

int annmtl_pipeline(const char* name) {
    @autoreleasepool {
        return pipeline_by_name(name);
    }
}

unsigned annmtl_max_threads_per_group(int pipeline) {
    State& s = state();
    if (pipeline < 0 || pipeline >= static_cast<int>(s.pipelines.size()))
        return 0;
    return static_cast<unsigned>(s.pipelines[pipeline].maxTotalThreadsPerThreadgroup);
}

void annmtl_dispatch(int pipeline,
                     unsigned groups_x, unsigned groups_y, unsigned groups_z,
                     unsigned threads_x, unsigned threads_y, unsigned threads_z,
                     const void* const* buffers, int num_buffers,
                     const void* args, size_t args_size,
                     size_t threadgroup_memory) {
    annmtl_dispatch_rw(pipeline, groups_x, groups_y, groups_z, threads_x, threads_y, threads_z, buffers, nullptr,
                       num_buffers, args, args_size, threadgroup_memory);
}

void annmtl_dispatch_rw(int pipeline,
                        unsigned groups_x, unsigned groups_y, unsigned groups_z,
                        unsigned threads_x, unsigned threads_y, unsigned threads_z,
                        const void* const* buffers, const unsigned char* written, int num_buffers,
                        const void* args, size_t args_size,
                        size_t threadgroup_memory) {
    @autoreleasepool {
        State& s = state();
        if (pipeline < 0 || pipeline >= static_cast<int>(s.pipelines.size()))
            return;

        // empty grids are skipped (e.g. empty populations)
        if (groups_x * groups_y * groups_z == 0 || threads_x * threads_y * threads_z == 0)
            return;

        id<MTLComputePipelineState> pso = s.pipelines[pipeline];
        if (threads_x * threads_y * threads_z > pso.maxTotalThreadsPerThreadgroup) {
            set_error("kernel '" + s.pipeline_names[pipeline] + "' launched with " +
                      std::to_string(threads_x * threads_y * threads_z) + " threads per threadgroup, but supports at most " +
                      std::to_string(pso.maxTotalThreadsPerThreadgroup) + ".");
            return;
        }

        id<MTLComputeCommandEncoder> enc = current_encoder();

        // resolve the buffers first, a memory barrier must precede the dispatch
        std::vector<std::pair<id<MTLBuffer>, bool>> uses;
        std::vector<size_t> offsets(num_buffers, 0);
        uses.reserve(num_buffers);
        for (int i = 0; i < num_buffers; i++) {
            id<MTLBuffer> buffer = nil;
            if (buffers[i] != nullptr && !resolve(buffers[i], &buffer, &offsets[i])) {
                set_error("kernel '" + s.pipeline_names[pipeline] + "': argument " + std::to_string(i) +
                          " is not device memory.");
                return;
            }
            uses.emplace_back(buffer, written == nullptr || written[i] != 0);
        }
        std::vector<std::pair<id<MTLBuffer>, bool>> tracked;
        for (auto& u : uses)
            if (u.first != nil)
                tracked.push_back(u);
        track_hazards(enc, tracked.data(), static_cast<int>(tracked.size()));

        [enc setComputePipelineState:pso];
        for (int i = 0; i < num_buffers; i++)
            [enc setBuffer:uses[i].first offset:offsets[i] atIndex:i];

        if (args_size > 0) {
            if (args_size <= kMaxInlineBytes) {
                [enc setBytes:args length:args_size atIndex:num_buffers];
            } else {
                id<MTLBuffer> arg_buf = [s.device newBufferWithBytes:args length:args_size
                                                             options:MTLResourceStorageModeShared];
                [enc setBuffer:arg_buf offset:0 atIndex:num_buffers];
            }
        }

        if (threadgroup_memory > 0) {
            size_t length = (threadgroup_memory + 15) / 16 * 16;
            [enc setThreadgroupMemoryLength:length atIndex:0];
        }

        [enc dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, groups_z)
            threadsPerThreadgroup:MTLSizeMake(threads_x, threads_y, threads_z)];

        if (s.profile) {
            id<MTLCommandBuffer> cb = s.command_buffer;
            commit_current();
            [cb waitUntilCompleted];
            auto& entry = s.profile_data[s.pipeline_names[pipeline]];
            entry.first += cb.GPUEndTime - cb.GPUStartTime;
            entry.second += 1;
            return;
        }
        after_dispatch();
    }
}

void annmtl_flush(void) {
    @autoreleasepool {
        commit_current();
    }
}

void annmtl_sync(void) {
    @autoreleasepool {
        State& s = state();
        commit_current();
        if (s.last_committed != nil) {
            [s.last_committed waitUntilCompleted];
            s.last_committed = nil;
        }
    }
}

int annmtl_get_last_error(void) {
    return error_flag.exchange(0);
}

int annmtl_peek_last_error(void) {
    return error_flag.load();
}

const char* annmtl_error_string(void) {
    static std::string msg;
    std::lock_guard<std::mutex> lock(error_mutex);
    msg = error_message;
    return msg.c_str();
}

}  // extern "C"
