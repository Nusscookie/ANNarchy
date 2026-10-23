"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.

FFT-based convolution for Apple GPUs (Metal paradigm).

Used instead of the direct kernels (MetalConvolveTemplate.py) for large filters,
see Convolution._metal_fft_plan(). The filters span the first two dimensions of
the pre-synaptic population, the remaining dimensions are channels which are
filtered independently (filter extent 1) or summed (the filter covers all of
them). The cross-correlation of each channel with each filter is computed on a
N0 x N1 grid (powers of two) holding the zero-padded pre-synaptic map, so that
there is no wrap-around:

    1. cu_fft_prep_proj<id>: f(pre) is written into the grid (the padding stays 0).
    2. cu_fft_rows_proj<id>: FFT of the rows holding data, one threadgroup per row.
    3. cu_fft_cols_proj<id>: FFT of each column of the summed channels,
       multiplication with the spectra of the filters (computed on the host,
       already conjugated and scaled by 1/(N0*N1)), sum over the channels and
       inverse FFT of the column. Only the rows of the outputs are kept.
    4. cu_fft_out_proj<id>: inverse FFT of these rows, the values are added to
       the post-synaptic neurons listed for the row (row_ptr/entries).

As the inputs and filters are real, two of them share one complex grid: two
channels as real and imaginary part (the results are the real and imaginary
part), or two filters for one channel (filter a + i * filter b).
"""

metal_fft_declare = """
    // Metal FFT convolution: filter spectra, twiddle factors and for each
    // output row the (post-synaptic rank, value index) pairs
    std::vector<float> fft_filters;
    std::vector<float> fft_twiddles;
    std::vector<int> fft_row_ptr;
    std::vector<int> fft_entries;
    bool fft_filters_dirty = false;
    bool fft_tables_dirty = false;
    int fft_clamp = 0;
    float* gpu_fft_filters = nullptr;
    float* gpu_fft_twiddles = nullptr;
    int* gpu_fft_row_ptr = nullptr;
    int* gpu_fft_entries = nullptr;
    // complex grids: input, row spectra, filtered rows of the outputs
    float* gpu_fft_x = nullptr;
    float* gpu_fft_spec = nullptr;
    float* gpu_fft_out = nullptr;
"""

metal_fft_transfer = """
        if (fft_filters_dirty) {
            if (gpu_fft_filters == nullptr)
                cudaMalloc((void**)&gpu_fft_filters, fft_filters.size() * sizeof(float));
            cudaMemcpy(gpu_fft_filters, fft_filters.data(), fft_filters.size() * sizeof(float), cudaMemcpyHostToDevice);
            fft_filters_dirty = false;
        }
        if (fft_tables_dirty) {
            cudaFree(gpu_fft_twiddles);
            cudaFree(gpu_fft_row_ptr);
            cudaFree(gpu_fft_entries);
            cudaMalloc((void**)&gpu_fft_twiddles, fft_twiddles.size() * sizeof(float));
            cudaMalloc((void**)&gpu_fft_row_ptr, fft_row_ptr.size() * sizeof(int));
            cudaMalloc((void**)&gpu_fft_entries, fft_entries.size() * sizeof(int));
            cudaMemcpy(gpu_fft_twiddles, fft_twiddles.data(), fft_twiddles.size() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_fft_row_ptr, fft_row_ptr.data(), fft_row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_fft_entries, fft_entries.data(), fft_entries.size() * sizeof(int), cudaMemcpyHostToDevice);
            fft_tables_dirty = false;
        }
        if (gpu_fft_x == nullptr) {
            cudaMalloc((void**)&gpu_fft_x, %(x_size)s * sizeof(float));
            cudaMemset(gpu_fft_x, 0, %(x_size)s * sizeof(float));
            cudaMalloc((void**)&gpu_fft_spec, %(x_size)s * sizeof(float));
            cudaMemset(gpu_fft_spec, 0, %(x_size)s * sizeof(float));
            cudaMalloc((void**)&gpu_fft_out, %(out_size)s * sizeof(float));
        }
"""

metal_fft_clear = """
        cudaFree(gpu_fft_filters);
        gpu_fft_filters = nullptr;
        cudaFree(gpu_fft_twiddles);
        gpu_fft_twiddles = nullptr;
        cudaFree(gpu_fft_row_ptr);
        gpu_fft_row_ptr = nullptr;
        cudaFree(gpu_fft_entries);
        gpu_fft_entries = nullptr;
        cudaFree(gpu_fft_x);
        gpu_fft_x = nullptr;
        cudaFree(gpu_fft_spec);
        gpu_fft_spec = nullptr;
        cudaFree(gpu_fft_out);
        gpu_fft_out = nullptr;
"""

metal_fft_wrapper = """
        .def_rw("fft_filters", &ProjStruct%(id_proj)s::fft_filters)
        .def_rw("fft_twiddles", &ProjStruct%(id_proj)s::fft_twiddles)
        .def_rw("fft_row_ptr", &ProjStruct%(id_proj)s::fft_row_ptr)
        .def_rw("fft_entries", &ProjStruct%(id_proj)s::fft_entries)
        .def_rw("fft_filters_dirty", &ProjStruct%(id_proj)s::fft_filters_dirty)
        .def_rw("fft_tables_dirty", &ProjStruct%(id_proj)s::fft_tables_dirty)
        .def_rw("fft_clamp", &ProjStruct%(id_proj)s::fft_clamp)
"""

metal_fft = {
    "body": """__global__ void cu_fft_prep_proj%(id_proj)s(float* __restrict__ x%(pre_variables_header)s) {
    // one thread per pre-synaptic neuron
    int rk_pre = blockIdx.x * blockDim.x + threadIdx.x;
    if (rk_pre >= %(pre_size)s) return;

    int annmtl_idx = rk_pre;
%(grid_index_code)s
    x[annmtl_x_idx] = (float)(%(psp_pre)s);
}

__global__ void cu_fft_rows_proj%(id_proj)s(const float* __restrict__ x, float* __restrict__ spec, const float* __restrict__ twiddles) {
    __shared__ float2 annmtl_buf[%(n1)s];
    {   // inner scope: the locals may shadow global constants
        // threadgroup (row, plane), %(n1_half)s threads
        auto tw = (const device float2*)twiddles;
        const uint tid = threadIdx.x;
        const uint offset = (blockIdx.y * %(n0)s + blockIdx.x + %(row_first)s) * %(n1)s;
        auto src = (const device float2*)x + offset;
        auto dst = (device float2*)spec + offset;

        annmtl_buf[annmtl_bitrev(tid, %(log_n1)s)] = src[tid];
        annmtl_buf[annmtl_bitrev(tid + %(n1_half)s, %(log_n1)s)] = src[tid + %(n1_half)s];
        annmtl_fft(annmtl_buf, tid, %(log_n1)s, %(log_n_max)s, tw, false);
        dst[tid] = annmtl_buf[tid];
        dst[tid + %(n1_half)s] = annmtl_buf[tid + %(n1_half)s];
    }
}

__global__ void cu_fft_cols_proj%(id_proj)s(const float* __restrict__ spec, const float* __restrict__ filters, float* __restrict__ out, const float* __restrict__ twiddles) {
    __shared__ float2 annmtl_col[%(num_reduce)s * %(n0)s];
    __shared__ float2 annmtl_prod[%(n0)s];
    {   // inner scope: the locals may shadow global constants
        // threadgroup (column, batch plane), %(n0_half)s threads
        auto tw = (const device float2*)twiddles;
        const uint tid = threadIdx.x;
        const uint j = blockIdx.x;
        const uint plane = blockIdx.y;

        // spectra of the %(num_reduce)s summed channel(s)
        for (int r = 0; r < %(num_reduce)s; r++) {
            auto src = (const device float2*)spec + (plane * %(num_reduce)s + r) * %(n0)s * %(n1)s + j;
            auto c = annmtl_col + r * %(n0)s;
            c[annmtl_bitrev(tid, %(log_n0)s)] = src[tid * %(n1)s];
            c[annmtl_bitrev(tid + %(n0_half)s, %(log_n0)s)] = src[(tid + %(n0_half)s) * %(n1)s];
            annmtl_fft(c, tid, %(log_n0)s, %(log_n_max)s, tw, false);
        }

        for (int s = 0; s < %(num_sets)s; s++) {
            auto g = (const device float2*)filters + s * %(num_reduce)s * %(n0)s * %(n1)s + j;
            for (uint i = tid; i < %(n0)s; i += %(n0_half)s) {
                float2 acc = float2(0.0f);
                for (int r = 0; r < %(num_reduce)s; r++)
                    acc += annmtl_cmul(annmtl_col[r * %(n0)s + i], g[(r * %(n0)s + i) * %(n1)s]);
                annmtl_prod[annmtl_bitrev(i, %(log_n0)s)] = acc;
            }
            annmtl_fft(annmtl_prod, tid, %(log_n0)s, %(log_n_max)s, tw, true);

            // rows %(row_lo)s ... %(row_lo)s + %(num_rows)s - 1 hold outputs
            auto dst = (device float2*)out + (plane * %(num_sets)s + s) * %(num_rows)s * %(n1)s + j;
            for (uint r = tid; r < %(n0)s; r += %(n0_half)s) {
                int row = (int)r - %(row_lo)s;
                if (row >= 0 && row < %(num_rows)s)
                    dst[row * %(n1)s] = annmtl_prod[r];
            }
            __syncthreads();
        }
    }
}

__global__ void cu_fft_out_proj%(id_proj)s(const float* __restrict__ out, %(float_prec)s* __restrict__ psp, const int* __restrict__ row_ptr, const int* __restrict__ entries, const float* __restrict__ twiddles, const int clamp_zero) {
    __shared__ float2 annmtl_buf[%(n1)s];
    {   // inner scope: the locals may shadow global constants
        // threadgroup (row, output plane), %(n1_half)s threads
        auto tw = (const device float2*)twiddles;
        const uint tid = threadIdx.x;
        const int g = blockIdx.y * %(num_rows)s + blockIdx.x;
        if (row_ptr[g] == row_ptr[g + 1]) return;   // uniform within the threadgroup
        auto src = (const device float2*)out + g * %(n1)s;

        annmtl_buf[annmtl_bitrev(tid, %(log_n1)s)] = src[tid];
        annmtl_buf[annmtl_bitrev(tid + %(n1_half)s, %(log_n1)s)] = src[tid + %(n1_half)s];
        annmtl_fft(annmtl_buf, tid, %(log_n1)s, %(log_n_max)s, tw, true);

        // value index = 2 * column + (0: real part, 1: imaginary part)
        auto values = (const threadgroup float*)annmtl_buf;
        for (int e = row_ptr[g] + tid; e < row_ptr[g + 1]; e += %(n1_half)s) {
            float v = values[entries[2 * e + 1]];
            if (clamp_zero)
                v = fmax(v, 0.0f);
            psp[entries[2 * e]] += %(sum_code)s;
        }
    }
}
""",
    "invoke": """
void fft_conv_proj%(id_proj)s(cudaStream_t stream, float* x, float* spec, float* out, const float* filters, const float* twiddles, %(float_prec)s* psp, const int* row_ptr, const int* entries, const int clamp_zero%(pre_variables_header)s) {
    cu_fft_prep_proj%(id_proj)s<<<(%(pre_size)s + 255) / 256, 256, 0, stream>>>(x%(pre_variables_invoke)s);
    cu_fft_rows_proj%(id_proj)s<<<dim3(%(num_data_rows)s, %(num_in_planes)s), %(n1_half)s, 0, stream>>>(x, spec, twiddles);
    cu_fft_cols_proj%(id_proj)s<<<dim3(%(n1)s, %(num_batch_planes)s), %(n0_half)s, 0, stream>>>(spec, filters, out, twiddles);
    cu_fft_out_proj%(id_proj)s<<<dim3(%(num_rows)s, %(num_out_planes)s), %(n1_half)s, 0, stream>>>(out, psp, row_ptr, entries, twiddles, clamp_zero);
}
""",
    "header": """void fft_conv_proj%(id_proj)s(cudaStream_t stream, float* x, float* spec, float* out, const float* filters, const float* twiddles, %(float_prec)s* psp, const int* row_ptr, const int* entries, const int clamp_zero%(pre_variables_header)s);""",
    "call": """
    if (proj%(id_proj)s->_transmission && pop%(id_post)s->_active ) {
        fft_conv_proj%(id_proj)s(
            proj%(id_proj)s->stream, proj%(id_proj)s->gpu_fft_x, proj%(id_proj)s->gpu_fft_spec, proj%(id_proj)s->gpu_fft_out,
            proj%(id_proj)s->gpu_fft_filters, proj%(id_proj)s->gpu_fft_twiddles, pop%(id_post)s->gpu__sum_%(target)s,
            proj%(id_proj)s->gpu_fft_row_ptr, proj%(id_proj)s->gpu_fft_entries, proj%(id_proj)s->fft_clamp%(pre_variables_call)s
        );
    }
""",
}
