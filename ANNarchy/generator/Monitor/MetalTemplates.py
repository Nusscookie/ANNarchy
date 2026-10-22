"""
:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.

Recording on Apple GPUs (Metal paradigm).

The CUDA recorders copy the recorded variables to the host at every recorded
step, which requires to wait for the GPU each time. Instead, the Metal
recorders copy the variables into device memory (the copy is executed by the
GPU in order with the kernels) and move the data into the containers at the
end of simulate() (finalize()), i.e. the GPU is synchronized only once.
"""
from copy import deepcopy

from ANNarchy.generator.Monitor.CUDATemplates import cuda_population

metal_population = deepcopy(cuda_population)

_record_targets = """    void record_targets() {
%(recording_target_code)s
    }
"""
assert _record_targets in cuda_population["template"]

metal_population["template"] = cuda_population["template"].replace(
    _record_targets,
    _record_targets
    + """
    // Move the recorded data from the device memory into the containers
    void finalize() {
%(finalize_code)s
    }
""",
)

metal_population["local"]["struct"] = """
    // Local variable %(name)s
    std::vector< std::vector< %(type)s > > %(name)s ;
    bool record_%(name)s ;
    // recorded steps not yet moved into %(name)s (device memory)
    std::vector< %(type)s* > _chunks_%(name)s ;
    size_t _num_pending_%(name)s = 0;
    size_t _chunk_slots_%(name)s() {
        // about 4 MB per chunk
        return std::max<size_t>(1, (size_t(1) << 22) / (pop%(id)s->size * sizeof(%(type)s)));
    }
    void finalize_%(name)s() {
        if (_num_pending_%(name)s == 0)
            return;

        annmtl_sync();
        size_t n = pop%(id)s->size;
        size_t slots = _chunk_slots_%(name)s();
        for (size_t k = 0; k < _num_pending_%(name)s; k++) {
            const %(type)s* data = _chunks_%(name)s[k / slots] + (k %% slots) * n;
            if (!this->partial) {
                this->%(name)s.push_back(std::vector< %(type)s >(data, data + n));
            } else {
                std::vector< %(type)s > tmp;
                tmp.reserve(this->ranks.size());
                for (auto rk : this->ranks)
                    tmp.push_back(data[rk]);
                this->%(name)s.push_back(tmp);
            }
        }
        for (auto chunk : _chunks_%(name)s)
            annmtl_free(chunk);
        _chunks_%(name)s.clear();
        _num_pending_%(name)s = 0;
    }"""

metal_population["local"]["recording"] = """
        if(this->record_%(name)s && ( (t - this->offset_) %% this->period_ == this->period_offset_ )){
            size_t n = pop%(id)s->size;
            size_t slots = _chunk_slots_%(name)s();
            size_t slot = _num_pending_%(name)s %% slots;
            if (slot == 0) {
                // limit the memory used for pending records (256 MB)
                if (_chunks_%(name)s.size() * slots * n * sizeof(%(type)s) >= (size_t(1) << 28))
                    finalize_%(name)s();
                _chunks_%(name)s.push_back(static_cast<%(type)s*>(annmtl_malloc(slots * n * sizeof(%(type)s))));
            }
            // copy on the GPU, ordered with respect to the kernels
            annmtl_memcpy(_chunks_%(name)s.back() + slot * n, pop%(id)s->gpu_%(name)s, n * sizeof(%(type)s));
            _num_pending_%(name)s++;
        }"""

metal_population["local"]["clear"] = """
    void clear_%(name)s() {
        for (auto chunk : _chunks_%(name)s)
            annmtl_free(chunk);
        _chunks_%(name)s.clear();
        _num_pending_%(name)s = 0;

        for(auto it = this->%(name)s.begin(); it != this->%(name)s.end(); it++) {
            it->clear();
            it->shrink_to_fit();
        }
        this->%(name)s.clear();
    }
"""

metal_population["local"]["finalize"] = """        this->finalize_%(name)s();
"""
