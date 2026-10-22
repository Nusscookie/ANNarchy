"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.

Convolution kernels tuned for Apple GPUs (Metal paradigm).

The kernels are written in the CUDA dialect understood by the Metal translator
(generator/Metal/Translator.py). They are used when the psp has the form
f(pre) * w and the padding is 0.0 (see Convolution._metal_conv_plan()):

    1. cu_conv_prep_proj<id>: one thread per pre-synaptic neuron writes f(pre)
       into a zero-padded copy of the pre-synaptic population. Outside the
       population the copy stays 0, which contributes nothing to sum/mean and,
       as the result is initialized with 0, nothing to max/min either.

    2. cu_conv_proj<id>: each thread computes several positions x several
       filters of the bank, so every loaded pre-synaptic value and weight is
       used several times. The filters of a thread are the same for the whole
       threadgroup (blockIdx.y), so the weights are broadcast to all threads
       of a SIMD-group. If there are few outputs, consecutive threads split
       the outermost window dimension and combine their results with SIMD
       shuffles. The window is read from the padded copy without bounds
       checks; the start index of each window is precomputed on the host
       (pre_base), so any sub-sampling is supported.
"""

metal_conv_declare = """
    // Metal: start index of each window in the padded pre-synaptic copy
    std::vector<int> pre_base;
    bool pre_base_dirty = false;
    int* gpu_pre_base = nullptr;
    // Metal: zero-padded copy of f(pre)
    %(float_prec)s* gpu_padded = nullptr;
"""

metal_conv_transfer = """
        if (pre_base_dirty) {
            if (gpu_pre_base != nullptr)
                cudaFree(gpu_pre_base);
            cudaMalloc((void**)&gpu_pre_base, pre_base.size() * sizeof(int));
            cudaMemcpy(gpu_pre_base, pre_base.data(), pre_base.size() * sizeof(int), cudaMemcpyHostToDevice);
            pre_base_dirty = false;
        }
        if (gpu_padded == nullptr) {
            cudaMalloc((void**)&gpu_padded, %(padded_size)s * sizeof(%(float_prec)s));
            cudaMemset(gpu_padded, 0, %(padded_size)s * sizeof(%(float_prec)s));
        }
"""

metal_conv_clear = """
        cudaFree(gpu_pre_base);
        gpu_pre_base = nullptr;
        cudaFree(gpu_padded);
        gpu_padded = nullptr;
"""

metal_conv_wrapper = """
        .def_rw("pre_base", &ProjStruct%(id_proj)s::pre_base)
        .def_rw("pre_base_dirty", &ProjStruct%(id_proj)s::pre_base_dirty)
"""

metal_conv = {
    "body": """__global__ void cu_conv_prep_proj%(id_proj)s(%(float_prec)s* __restrict__ padded%(pre_variables_header)s) {
    // one thread per pre-synaptic neuron
    int rk_pre = blockIdx.x * blockDim.x + threadIdx.x;
    if (rk_pre >= %(pre_size)s) return;

    int idx = rk_pre;
    int padded_idx = 0;
%(padded_index_code)s
    padded[padded_idx] = %(psp_pre)s;
}

__global__ void cu_conv_proj%(id_proj)s(%(float_prec)s* __restrict__ psp, const int* __restrict__ pre_base, __annmtl_constant__ const %(float_prec)s* __restrict__ w, const %(float_prec)s* __restrict__ padded) {
    // the threadgroups of one grid row share the filters blockIdx.y * %(filters_per_thread)s, ...
    const int first_filter = blockIdx.y * %(filters_per_thread)s;
    auto w_group = &w[first_filter * %(filter_size)s];

    // %(window_parts)s consecutive threads of a SIMD-group split the window of the same outputs,
    // each thread computes %(outputs_per_thread)s positions x %(filters_per_thread)s filters
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int part = gid %% %(window_parts)s;
    const int tid = gid / %(window_parts)s;
    const bool active = tid < %(num_threads)s;   // no early return: all threads take part in the shuffles

    int base[%(outputs_per_thread)s];
    for (int r = 0; r < %(outputs_per_thread)s; r++) {
        int pos = tid + r * %(num_threads)s;
        base[r] = pre_base[pos < %(num_positions)s ? pos : %(num_positions)s - 1];
    }
    %(float_prec)s acc[%(num_acc)s];
    for (int a = 0; a < %(num_acc)s; a++)
        acc[a] = 0.0;

%(loop_code)s

%(reduction_code)s

    if (active && part == 0) {
        for (int r = 0; r < %(outputs_per_thread)s; r++) {
            int pos = tid + r * %(num_threads)s;
            if (pos >= %(num_positions)s)
                continue;
            for (int f = 0; f < %(filters_per_thread)s; f++)
                psp[pos * %(num_filters)s + first_filter + f] += %(sum_code)s;
        }
    }
}
""",
    "invoke": """
void conv_prep_proj%(id_proj)s(RunConfig cfg, %(float_prec)s* padded%(pre_variables_header)s) {
    cu_conv_prep_proj%(id_proj)s<<<cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream>>>(padded%(pre_variables_invoke)s);
}

void convolution_proj%(id_proj)s(RunConfig cfg, %(float_prec)s* psp, const int* pre_base, const %(float_prec)s* w, const %(float_prec)s* padded) {
    cu_conv_proj%(id_proj)s<<<cfg.nb, cfg.tpb, cfg.smem_size, cfg.stream>>>(psp, pre_base, w, padded);
}
""",
    "header": """void conv_prep_proj%(id_proj)s(RunConfig cfg, %(float_prec)s* padded%(pre_variables_header)s);
void convolution_proj%(id_proj)s(RunConfig cfg, %(float_prec)s* psp, const int* pre_base, const %(float_prec)s* w, const %(float_prec)s* padded);""",
    "call": """
    if (proj%(id_proj)s->_transmission && pop%(id_post)s->_active ) {
        conv_prep_proj%(id_proj)s(
            RunConfig((%(pre_size)s + 255) / 256, 256, 0, proj%(id_proj)s->stream),
            proj%(id_proj)s->gpu_padded%(pre_variables_call)s
        );

        RunConfig cfg((%(num_threads)s * %(window_parts)s + 127) / 128, 128, 0, proj%(id_proj)s->stream);
        cfg.nb = dim3((%(num_threads)s * %(window_parts)s + 127) / 128, %(num_filters)s / %(filters_per_thread)s, 1);
        convolution_proj%(id_proj)s(
            cfg, pop%(id_post)s->gpu__sum_%(target)s, proj%(id_proj)s->gpu_pre_base, proj%(id_proj)s->gpu_w, proj%(id_proj)s->gpu_padded
        );
    }
""",
}

# Accumulation of v into a, per operation
metal_conv_accumulate = {
    "sum": "%(a)s += %(v)s;",
    "mean": "%(a)s += %(v)s;",
    "max": "{ %(float_prec)s v = %(v)s; if (v > %(a)s) %(a)s = v; }",
    "min": "{ %(float_prec)s v = %(v)s; if (v < %(a)s) %(a)s = v; }",
}

# Same with v = dot product / element-wise product of 4 consecutive elements
metal_conv_accumulate4 = {
    "sum": "%(a)s += dot(%(x)s, %(w)s);",
    "mean": "%(a)s += dot(%(x)s, %(w)s);",
    "max": "{ float4 v = %(x)s * %(w)s; %(float_prec)s m = fmax(fmax(v.x, v.y), fmax(v.z, v.w)); if (m > %(a)s) %(a)s = m; }",
    "min": "{ float4 v = %(x)s * %(w)s; %(float_prec)s m = fmin(fmin(v.x, v.y), fmin(v.z, v.w)); if (m < %(a)s) %(a)s = m; }",
}
