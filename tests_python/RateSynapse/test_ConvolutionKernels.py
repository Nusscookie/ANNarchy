"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

import os
import unittest
import numpy

from conftest import TARGET_FOLDER
from ANNarchy import Network, Neuron
from ANNarchy.extensions.convolution import Convolution, Pooling
from ANNarchy.intern.ConfigManagement import _check_paradigm


def convolution_reference(pre, weights, coords, operation, psp=lambda r, w: r * w, bank=False):
    """
    Cross-correlation as computed by ANNarchy: the window is centered on the
    coordinates of each post-synaptic neuron, values outside the pre-synaptic
    population are skipped and max/min start from 0.
    """
    result = numpy.zeros(len(coords))
    for rk, coord in enumerate(coords):
        w = weights[coord[-1]] if bank else weights
        acc = 0.0
        for idx in numpy.ndindex(*w.shape):
            pre_idx = list(coord[: pre.ndim])
            for d in range(w.ndim):
                center = w.shape[d] // 2 if w.shape[d] % 2 == 1 else w.shape[d] // 2 - 1
                pre_idx[d] += idx[d] - center
            if any(i < 0 or i >= s for i, s in zip(pre_idx, pre.shape)):
                continue
            v = psp(pre[tuple(pre_idx)], w[idx])
            if operation in ["sum", "mean"]:
                acc += v
            elif operation == "max":
                acc = max(acc, v)
            else:
                acc = min(acc, v)
        result[rk] = acc / w.size if operation == "mean" else acc
    return result


def pooling_reference(pre, coords, extent, operation):
    result = numpy.zeros(len(coords))
    for rk, coord in enumerate(coords):
        window = pre[tuple(slice(c, c + e) for c, e in zip(coord, extent))]
        result[rk] = {"max": numpy.max, "min": numpy.min, "mean": numpy.mean, "sum": numpy.sum}[operation](window)
    return result


class test_ConvolutionKernels(unittest.TestCase):
    """
    Compares convolution and pooling projections of different shapes,
    operations and psps with a numpy implementation. On Apple GPUs, these
    cases cover the Metal-specific kernels (vectorized filter banks, split
    windows, pooling with one or more threads per output) and the fallbacks.
    """

    @classmethod
    def setUpClass(cls):
        rng = numpy.random.default_rng(1)
        In = Neuron(parameters="baseline = 0.0", equations="r = baseline")
        Out = Neuron(equations="r = sum(exc)")

        cls.test_net = Network()
        cls.inputs = {}
        cls.cases = {}

        def add_input(name, geometry):
            pop = cls.test_net.create(geometry=geometry, neuron=In)
            cls.inputs[name] = (pop, rng.random(geometry))
            return pop

        def add_conv(name, pre, post_geometry, weights, bank=False, f=lambda r, w: r * w, **kwargs):
            post = cls.test_net.create(geometry=post_geometry, neuron=Out)
            conv_args = {k: kwargs.pop(k) for k in ["psp", "operation"] if k in kwargs}
            proj = cls.test_net.connect(Convolution(pre, post, target="exc", **conv_args))
            if bank:
                proj.connect_filters(weights=weights, **kwargs)
            else:
                proj.connect_filter(weights=weights, **kwargs)
            cls.cases[name] = ("conv", pre, post, proj, weights, bank, f, conv_args.get("operation", "sum"))

        def add_pool(name, pre, post_geometry, operation, extent):
            post = cls.test_net.create(geometry=post_geometry, neuron=Out)
            proj = cls.test_net.connect(Pooling(pre, post, target="exc", operation=operation))
            proj.connect_pooling(extent=extent)
            cls.cases[name] = ("pool", pre, post, proj, extent, operation)

        p4d = add_input("p4d", (12, 10, 2, 8))
        p2d = add_input("p2d", (20, 18))
        p3d = add_input("p3d", (16, 14, 3))
        p2d_large = add_input("p2d_large", (130, 130))

        # filter bank on a 4D population, innermost dimension of 8 (vector loads)
        add_conv("bank4d", p4d, (12, 10, 4), rng.random((4, 5, 5, 2, 8)) - 0.5, bank=True,
                 psp="pow(pre.r, 2.0) * w", f=lambda r, w: r**2 * w)
        # filter banks with max / min
        add_conv("bank3d_max", p2d, (20, 18, 6), rng.random((6, 5, 5)) - 0.5, bank=True, operation="max")
        add_conv("bank3d_min", p3d, (16, 14, 2), rng.random((2, 3, 3, 3)) - 0.5, bank=True, operation="min")
        # single filters: large window (split among threads), keep_last_dimension, sub-sampling + mean
        add_conv("single_large", p2d, (20, 18), rng.random((15, 13)) - 0.5)
        add_conv("keep_last", p3d, (16, 14, 3), rng.random((5, 5)), keep_last_dimension=True,
                 psp="w * pre.r * pre.r", f=lambda r, w: w * r * r)
        add_conv("subsampled_mean", p2d, (10, 9), rng.random((3, 3)), operation="mean")
        # psp not of the form f(pre) * w: generic kernel
        add_conv("custom_psp", p2d, (20, 18), rng.random((3, 3)), psp="pos(w - pre.r)", f=lambda r, w: max(w - r, 0.0))

        # pooling with 1, 32 and 256 / 1024 threads per output
        add_pool("pool_small", p2d, (10, 9), "max", (2, 2))
        add_pool("pool_medium", p3d, (2, 2, 3), "mean", (8, 7, 1))
        add_pool("pool_global_max", p2d_large, (1, 1), "max", (130, 130))
        add_pool("pool_global_mean", p2d_large, (1, 1), "mean", (130, 130))

        cls.test_net.compile(silent=True, directory=TARGET_FOLDER)

    @classmethod
    def tearDownClass(cls):
        del cls.test_net

    def setUp(self):
        self.test_net.reset()
        for pop, values in self.inputs.values():
            pop.baseline = values
        self.test_net.simulate(2)

    def check(self, name):
        case = self.cases[name]
        if case[0] == "conv":
            _, pre, post, proj, weights, bank, f, operation = case
            values = self.inputs[[k for k, v in self.inputs.items() if v[0] is pre][0]][1]
            expected = convolution_reference(values, weights, numpy.array(proj.pre_coordinates), operation, f, bank)
        else:
            _, pre, post, proj, extent, operation = case
            values = self.inputs[[k for k, v in self.inputs.items() if v[0] is pre][0]][1]
            expected = pooling_reference(values, numpy.array(proj.pre_coordinates), extent, operation)
        numpy.testing.assert_allclose(numpy.array(post.r).flatten(), expected, rtol=1e-5, atol=1e-6)

    def test_bank4d(self):
        self.check("bank4d")

    def test_bank3d_max(self):
        self.check("bank3d_max")

    def test_bank3d_min(self):
        self.check("bank3d_min")

    def test_single_large(self):
        self.check("single_large")

    def test_keep_last(self):
        self.check("keep_last")

    def test_subsampled_mean(self):
        self.check("subsampled_mean")

    def test_custom_psp(self):
        self.check("custom_psp")

    def test_pool_small(self):
        self.check("pool_small")

    def test_pool_medium(self):
        self.check("pool_medium")

    def test_pool_global_max(self):
        self.check("pool_global_max")

    def test_pool_global_mean(self):
        self.check("pool_global_mean")


class test_ConvolutionFFT(unittest.TestCase):
    """
    Same comparison for the FFT-based convolution of the Metal paradigm, which
    is forced for all applicable projections (ANNARCHY_METAL_CONV_FFT=1). The
    FFT spreads rounding errors over the whole map, so the tolerance is
    relative to the largest output. On other paradigms, the direct kernels are
    tested once more.
    """

    @classmethod
    def setUpClass(cls):
        rng = numpy.random.default_rng(2)
        In = Neuron(parameters="baseline = 0.0", equations="r = baseline")
        Out = Neuron(equations="r = sum(exc)")

        cls.test_net = Network()
        cls.inputs = {}
        cls.cases = {}

        def add_input(name, geometry):
            pop = cls.test_net.create(geometry=geometry, neuron=In)
            cls.inputs[name] = (pop, rng.random(geometry))
            return pop

        def add_conv(name, pre, post_geometry, weights, bank=False, f=lambda r, w: r * w, **kwargs):
            post = cls.test_net.create(geometry=post_geometry, neuron=Out)
            conv_args = {k: kwargs.pop(k) for k in ["psp", "operation"] if k in kwargs}
            proj = cls.test_net.connect(Convolution(pre, post, target="exc", **conv_args))
            if bank:
                proj.connect_filters(weights=weights, **kwargs)
            else:
                proj.connect_filter(weights=weights, **kwargs)
            cls.cases[name] = [pre, post, proj, weights, bank, f, conv_args.get("operation", "sum")]

        p2d = add_input("p2d", (20, 18))
        p3d = add_input("p3d", (16, 14, 3))
        p4d = add_input("p4d", (12, 10, 2, 4))

        # filters covering all channels (the channels are summed)
        add_conv("bank_reduce", p4d, (12, 10, 3), rng.random((3, 5, 5, 2, 4)) - 0.5, bank=True,
                 psp="pow(pre.r, 2.0) * w", f=lambda r, w: r**2 * w)
        add_conv("single_reduce", p3d, (16, 14), rng.random((7, 7, 3)) - 0.5)
        # odd and even filter sizes, larger than half of the population
        add_conv("single_odd", p2d, (20, 18), rng.random((15, 13)) - 0.5)
        add_conv("single_even", p2d, (20, 18), rng.random((6, 8)) - 0.5)
        # bank of an odd number of filters (two filters share one grid)
        add_conv("bank", p2d, (20, 18, 3), rng.random((3, 7, 9)) - 0.5, bank=True)
        # channels filtered independently (two channels share one grid), outputs clamped to >= 0
        add_conv("channels", p3d, (16, 14, 3), rng.random((9, 9, 1)),
                 psp="pow(pre.r, 2.0) * w", f=lambda r, w: r**2 * w)
        add_conv("keep_last", p3d, (16, 14, 3), rng.random((5, 5)) - 0.5, keep_last_dimension=True)
        # sub-sampling with mean, up-sampling with explicit coordinates, delays
        add_conv("subsampled_mean", p2d, (10, 9), rng.random((7, 7)), operation="mean")
        add_conv("upsampled", p2d, (40, 36), rng.random((5, 5)) - 0.5,
                 subsampling=[[i // 2, j // 2] for i in range(40) for j in range(36)])
        add_conv("delayed", p2d, (20, 18), rng.random((9, 9)) - 0.5, delays=2.0)
        # filter changed after compilation
        add_conv("update", p2d, (20, 18), rng.random((11, 11)) - 0.5)

        previous = os.environ.get("ANNARCHY_METAL_CONV_FFT")
        os.environ["ANNARCHY_METAL_CONV_FFT"] = "1"
        try:
            cls.test_net.compile(silent=True, directory=TARGET_FOLDER)
        finally:
            if previous is None:
                del os.environ["ANNARCHY_METAL_CONV_FFT"]
            else:
                os.environ["ANNARCHY_METAL_CONV_FFT"] = previous

    @classmethod
    def tearDownClass(cls):
        del cls.test_net

    def setUp(self):
        self.test_net.reset()
        for pop, values in self.inputs.values():
            pop.baseline = values
        self.test_net.simulate(4)

    def check(self, name):
        pre, post, proj, weights, bank, f, operation = self.cases[name]
        values = self.inputs[[k for k, v in self.inputs.items() if v[0] is pre][0]][1]
        expected = convolution_reference(values, weights, numpy.array(proj.pre_coordinates), operation, f, bank)
        numpy.testing.assert_allclose(numpy.array(post.r).flatten(), expected, rtol=1e-5,
                                      atol=1e-5 * numpy.max(numpy.abs(expected)))

    def test_fft_used(self):
        if _check_paradigm("metal", self.test_net.id):
            for name, case in self.cases.items():
                self.assertIsNotNone(case[2]._metal_fft, name)

    def test_bank_reduce(self):
        self.check("bank_reduce")

    def test_single_reduce(self):
        self.check("single_reduce")

    def test_single_odd(self):
        self.check("single_odd")

    def test_single_even(self):
        self.check("single_even")

    def test_bank(self):
        self.check("bank")

    def test_channels(self):
        self.check("channels")

    def test_keep_last(self):
        self.check("keep_last")

    def test_subsampled_mean(self):
        self.check("subsampled_mean")

    def test_upsampled(self):
        self.check("upsampled")

    def test_delayed(self):
        self.check("delayed")

    def test_update(self):
        case = self.cases["update"]
        weights = case[3]
        case[3] = numpy.flip(weights) * 2.0
        try:
            case[2].weights = case[3]
            self.test_net.simulate(1)
            self.check("update")
        finally:
            case[3] = weights
            case[2].weights = weights
