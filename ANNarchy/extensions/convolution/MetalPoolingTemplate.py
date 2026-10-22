"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.

Pooling kernel tuned for Apple GPUs (Metal paradigm), written in the CUDA
dialect understood by the Metal translator (generator/Metal/Translator.py).

Each output is computed by 1, 32 (one SIMD-group), 256 or 1024 threads (one
threadgroup), depending on the size of the pooling window. Consecutive threads
read consecutive elements of the window, the partial results are combined
with SIMD shuffles (and threadgroup memory for a threadgroup). The start index
of each window is precomputed on the host (pre_base). The kernel is used when
all windows lie within the pre-synaptic population (see Pooling._generate_metal()).
"""

metal_pool_declare = """
    // Metal: start index of each pooling window
    std::vector<int> pre_base;
    bool pre_base_dirty = false;
    int* gpu_pre_base = nullptr;
"""

metal_pool_transfer = """
        if (pre_base_dirty) {
            if (gpu_pre_base != nullptr)
                cudaFree(gpu_pre_base);
            cudaMalloc((void**)&gpu_pre_base, pre_base.size() * sizeof(int));
            cudaMemcpy(gpu_pre_base, pre_base.data(), pre_base.size() * sizeof(int), cudaMemcpyHostToDevice);
            pre_base_dirty = false;
        }
"""

metal_pool_clear = """
        cudaFree(gpu_pre_base);
        gpu_pre_base = nullptr;
"""

metal_pool_wrapper = """
        .def_rw("pre_base", &ProjStruct%(id_proj)s::pre_base)
        .def_rw("pre_base_dirty", &ProjStruct%(id_proj)s::pre_base_dirty)
"""

metal_pool = {
    "psp_body": """__global__ void cu_pooling_proj%(id_proj)s(%(float_prec)s* __restrict__ psp, const int* __restrict__ pre_base, const %(float_prec)s* __restrict__ pre_%(pre_var)s) {
    // %(threads)s thread(s) per output
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int out = gid / %(threads)s;
    const int lane = gid %% %(threads)s;
    const bool active = out < %(size_post)s;   // no early return: all threads take part in the shuffles
    const int base = pre_base[active ? out : 0];

    %(float_prec)s local_res = %(sum_default)s;
    for (int e = lane; e < %(window_size)s; e += %(threads)s) {
        int rest = e;
        int offset = 0;
%(offset_code)s
        %(float_prec)s local_r = pre_%(pre_var)s[base + offset];
%(apply_psp)s
        %(operation)s
    }
%(reduction_code)s
    if (active && lane == 0)
        %(final_result)s
}
""",
    "psp_invoke": """
void pooling_proj%(id_proj)s(RunConfig cfg, %(float_prec)s* psp, const int* pre_base, const %(float_prec)s* pre_%(pre_var)s) {
    cu_pooling_proj%(id_proj)s<<<cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream>>>(psp, pre_base, pre_%(pre_var)s);
}
""",
    "psp_header": """void pooling_proj%(id_proj)s(RunConfig cfg, %(float_prec)s* psp, const int* pre_base, const %(float_prec)s* pre_%(pre_var)s);""",
    "psp_call": """
    if (pop%(id_post)s->_active && proj%(id_proj)s->_transmission) {
        pooling_proj%(id_proj)s(
            RunConfig(%(num_blocks)s, %(block_size)s, 0, proj%(id_proj)s->stream),
            pop%(id_post)s->gpu__sum_%(target)s, proj%(id_proj)s->gpu_pre_base, %(pre_array)s
        );
    }
""",
}

# Combination of local_res with v, per operation
metal_pool_operation = {
    "sum": "local_res += %(v)s;",
    "mean": "local_res += %(v)s;",
    "max": "if (%(v)s > local_res) local_res = %(v)s;",
    "min": "if (%(v)s < local_res) local_res = %(v)s;",
}

# Reduction over the threads of one output
metal_pool_simd_reduction = """
    for (int lane_mask = 16; lane_mask > 0; lane_mask /= 2) {
        %(float_prec)s other = __shfl_xor_sync(0xffffffff, local_res, lane_mask);
        %(combine)s
    }
"""

metal_pool_threadgroup_reduction = """
    // one value per SIMD-group, then reduction by the first SIMD-group
    __shared__ %(float_prec)s partial[%(num_simd)s];
    if (lane %% 32 == 0)
        partial[lane / 32] = local_res;
    __syncthreads();
    if (lane < 32) {
        local_res = (lane < %(num_simd)s) ? partial[lane] : %(sum_default)s;
        for (int lane_mask = %(num_simd)s / 2; lane_mask > 0; lane_mask /= 2) {
            %(float_prec)s other = __shfl_xor_sync(0xffffffff, local_res, lane_mask);
            %(combine)s
        }
    }
"""
