# ANNarchy 

[![PyPI version](https://badge.fury.io/py/ANNarchy.svg)](https://badge.fury.io/py/ANNarchy)
[![DOI](https://zenodo.org/badge/57382690.svg)](https://zenodo.org/badge/latestdoi/57382690)


ANNarchy (Artificial Neural Networks architect) is a parallel and hybrid simulator for distributed rate-coded or spiking neural networks. The core of the library is written in C++ and distributed using openMP or CUDA. It provides an interface in Python for the definition of the networks. It is released under the [GNU GPL v2 or later](http://www.gnu.org/licenses/gpl.html).

* Source code: [github.com/ANNarchy/ANNarchy](https://github.com/ANNarchy/ANNarchy)
* Documentation: [annarchy.github.io](https://annarchy.github.io)
* Forum: [google forum](https://groups.google.com/forum/#!forum/annarchy)
* Bug reports and feature requests: [Issue Tracker](https://github.com/ANNarchy/ANNarchy/issues).

### Citation

If you use ANNarchy for your research, we would appreciate if you cite the following paper:

> Vitay J, Dinkelbach HÜ and Hamker FH (2015). ANNarchy: a code generation approach to neural simulations on parallel hardware. *Frontiers in Neuroinformatics* 9:19. [doi:10.3389/fninf.2015.00019](http://dx.doi.org/10.3389/fninf.2015.00019)

### Authors

* Julien Vitay (julien.vitay@gmail.com).
* Helge Ülo Dinkelbach (helge-uelo.dinkelbach@informatik.tu-chemnitz.de).
* Oliver Maith (oliver.maith@informatik.tu-chemnitz.de)
* Fred Hamker (fred.hamker@informatik.tu-chemnitz.de).

## Installation

Using pip, you can install the latest stable release:

```bash
pip install ANNarchy
```

See <https://annarchy.github.io/Installation> for further instructions.

## Platforms

* GNU/Linux
* MacOS X
* Windows (inside WSL2)

### Apple GPUs (Metal)

On Apple Silicon Macs, rate-coded networks can be simulated on the GPU:

```python
import ANNarchy as ann

net = ann.Network()
net.config(paradigm="metal")   # implies dtype=float32, Metal GPUs have no double precision
```

The Metal backend reuses the CUDA code generators: the generated CUDA kernels
are translated into Metal shaders, the host code runs on top of a small CUDA
compatibility layer (`ANNarchy/include/metal`). It requires Apple's `clang++`
(Command Line Tools). The shaders are compiled at runtime, or ahead of time if
the Metal Toolchain of Xcode is installed (`xcrun -sdk macosx metal`).

Supported: rate-coded populations (including `TimedArray` and `InputArray`),
rate-coded projections (`csr`, `dense`, `ellr`, `sell` formats, uniform delays,
synaptic plasticity), `Convolution` and `Pooling`, monitors, random
variables and global operations. Not supported (yet): profiling and
non-uniform delays; spiking networks are experimental.

Some kernels are tuned for Apple GPUs instead of being translated:
convolutions with a psp of the form `f(pre) * w` and padding 0 (zero-padded
input, several filters and positions per thread, vector loads), pooling,
`csr` transmission, and global operations (`max(r)` etc. stay on the GPU).
Independent kernels of a time step run concurrently on the GPU.

Environment variables:

* `ANNARCHY_METAL_PROFILE=1` prints the GPU time of each kernel at the end of
  the process (the kernels are executed one by one, so the total is higher).
* `ANNARCHY_METAL_SERIAL=1` executes the kernels one after another.
* `tools/metal_bench/conv_micro.py` compares and benchmarks convolution and
  pooling kernels between paradigms.

## Dependencies

* `python` >= 3.10 (with the development files, e.g. `python-dev` or `python-devel`)
* `g++` >= 7.4 or `clang++` >= 3.4
* `cmake` >= 3.17
* `setuptools` >= 65.0
* `nanobind` >= 2.4.0
* `cython` >= 3.0
* `numpy` >= 1.21
* `sympy` >= 1.11
* `scipy` >= 1.9
* `matplotlib` >= 3.0
* `tqdm` >= 4.60

Recommended:

* `lxml` (to save the networks in `.xml` format).
* `h5py` (to export data in `.h5` format).
* `pandoc` (for `report()`).
* `tensorflow` (for the `ann_to_snn_conversion` extension)
* `tensorboardX` (for the `logging` extension).
