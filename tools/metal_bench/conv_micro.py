"""
Micro-benchmark and cross-check of the Convolution and Pooling kernels.

usage: python conv_micro.py <metal|openmp> <outfile.npz> [steps] [case ...]

Builds one network per case, checks the result after a few steps and measures
the time of simulate(steps). Comparing the output files of two paradigms:

    python conv_micro.py compare a.npz b.npz
"""
import os
import sys
import time
import tempfile
import zlib

import numpy as np


def cases():
    def rnd(*shape, signed=False):
        # the seed depends only on the shape, so adding cases does not change the others
        w = np.random.default_rng(zlib.crc32(str(shape).encode())).random(shape).astype(np.float32)
        return (w - 0.5) if signed else w

    gauss = np.exp(-(np.arange(-12, 13)[:, None] ** 2 + np.arange(-12, 13)[None, :] ** 2) / (2 * 4.0**2))
    return {
        # name: (pre geometry, post geometry, kind, weights, options)
        "bank4d_27x27": ((75, 75, 2, 8), (75, 75, 16), "filters", rnd(16, 27, 27, 2, 8), dict(psp="pow(pre.r, 2.0) * w")),
        "bank4d_17x17_f2": ((75, 75, 2, 8), (75, 75, 2), "filters", rnd(2, 17, 17, 2, 8), dict(psp="pow(pre.r, 2.0) * w")),
        "bank2d_51x51_f2": ((75, 75), (75, 75, 2), "filters", rnd(2, 51, 51), dict()),
        "single2d_25x25": ((188, 188), (188, 188), "filter", rnd(25, 25, signed=True), dict()),
        "gauss2d_25x25": ((188, 188), (188, 188), "filter", gauss, dict()),
        "keeplast_7x7": ((100, 100, 3), (100, 100, 3), "filter", rnd(7, 7), dict(psp="pow(pre.r, 4.0) * w", keep_last_dimension=True)),
        "bank3d_max": ((64, 64), (64, 64, 8), "filters", rnd(8, 9, 9, signed=True), dict(operation="max")),
        "bank4d_min": ((40, 40, 4), (40, 40, 6), "filters", rnd(6, 5, 5, 4, signed=True), dict(operation="min")),
        "subsample_mean": ((60, 60), (30, 30), "filter", rnd(5, 5), dict(operation="mean")),
        "delay_2d": ((50, 50), (50, 50), "filter", rnd(9, 9, signed=True), dict(delays=2.0)),
        "custom_psp": ((50, 50), (50, 50), "filter", rnd(5, 5), dict(psp="pos(w - pre.r)")),
        "pool_global_mean": ((188, 188), (1, 1), "pool", None, dict(operation="mean", extent=(188, 188))),
        "pool_global_max": ((75, 75, 2), (1, 1, 2), "pool", None, dict(operation="max", extent=(75, 75, 1))),
        "pool_2x2_max": ((100, 100), (50, 50), "pool", None, dict(operation="max", extent=(2, 2))),
    }


def run_case(paradigm, name, spec, steps):
    import ANNarchy as ann
    from ANNarchy.extensions.convolution import Convolution, Pooling

    pre_geo, post_geo, kind, weights, opts = spec
    opts = dict(opts)
    net = ann.Network()
    if paradigm == "metal":
        net.config(paradigm="metal")
    else:
        net.config(paradigm="openmp", num_threads=8, dtype=ann.float32)

    In = ann.Neuron(parameters="baseline = 0.0", equations="r = baseline")
    Out = ann.Neuron(equations="r = sum(exc)")
    pre = net.create(geometry=pre_geo, neuron=In)
    post = net.create(geometry=post_geo, neuron=Out)

    if kind == "pool":
        proj = net.connect(Pooling(pre, post, target="exc", operation=opts.pop("operation")))
        proj.connect_pooling(extent=opts.pop("extent"), delays=opts.pop("delays", 0.0))
    else:
        conv_opts = {k: opts.pop(k) for k in ["psp", "operation"] if k in opts}
        proj = net.connect(Convolution(pre, post, target="exc", **conv_opts))
        if kind == "filter":
            proj.connect_filter(weights=weights, **opts)
        else:
            proj.connect_filters(weights=weights, **opts)

    folder = os.path.join(tempfile.gettempdir(), "annarchy_conv_micro", f"{paradigm}_{name}")
    net.compile(directory=folder, silent=True)

    rng = np.random.default_rng(7)
    pre.baseline = rng.random(pre_geo).astype(np.float32)
    net.simulate(4)
    result = np.array(post.r, dtype=np.float32)

    net.simulate(10)  # warm-up
    t0 = time.time()
    net.simulate(steps)
    float(np.sum(post.r))  # waits for the GPU
    elapsed = (time.time() - t0) / steps
    return result, elapsed


def main():
    if sys.argv[1] == "compare":
        a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
        print(f"{'case':20s} {'max rel diff':>12s} {'time a [us]':>12s} {'time b [us]':>12s} {'speedup':>8s}")
        for k in sorted(f[:-2] for f in a.files if f.endswith("_r")):
            x, y = a[k + "_r"].astype(np.float64), b[k + "_r"].astype(np.float64)
            d = np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-12)
            ta, tb = float(a[k + "_t"]) * 1e6, float(b[k + "_t"]) * 1e6
            print(f"{k:20s} {d:12.2e} {ta:12.1f} {tb:12.1f} {tb / ta:8.2f}")
        return

    paradigm, outfile = sys.argv[1], sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    selected = sys.argv[4:]
    out = {}
    for name, spec in cases().items():
        if selected and name not in selected:
            continue
        r, t = run_case(paradigm, name, spec, steps)
        out[name + "_r"], out[name + "_t"] = r, t
        print(f"{name:20s} {t * 1e6:10.1f} us/step", flush=True)
    np.savez(outfile, **out)


if __name__ == "__main__":
    main()
