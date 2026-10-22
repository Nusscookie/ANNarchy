"""
This file is part of ANNarchy.

:copyright: Copyright 2013 - now, see AUTHORS.
:license: GPLv2, see LICENSE for details.
"""

# Equals the ANNarchy default configuration, i.e. single-thread simulation.
NUM_OMP_THREADS = 1
USED_PARADIGM = "openmp"
TARGET_FOLDER = "annarchy_st"


def pytest_addoption(parser):
    """
    Extend the default arguments of pytest.
    """
    parser.addoption("--openmp", action="store_true", default=False, help="TODO")
    parser.addoption("--cuda", action="store_true", default=False, help="TODO")
    parser.addoption("--metal", action="store_true", default=False, help="run the tests on Apple GPUs (Metal)")


def _use_single_precision_tolerances():
    """
    The Metal paradigm computes in single precision, the default tolerance of
    numpy.testing.assert_allclose (rtol=1e-7) assumes double precision.
    """
    import numpy.testing

    original = numpy.testing.assert_allclose

    def assert_allclose(actual, desired, rtol=1e-7, atol=0, *args, **kwargs):
        return original(actual, desired, max(rtol, 1e-5), max(atol, 1e-6), *args, **kwargs)

    numpy.testing.assert_allclose = assert_allclose


def pytest_configure(config):
    """
    Updates the runtime configuration if necessary. Automatically called by pytest.
    """
    global NUM_OMP_THREADS, USED_PARADIGM, TARGET_FOLDER

    if config.getoption("--metal"):
        USED_PARADIGM = "metal"
        NUM_OMP_THREADS = 1
        TARGET_FOLDER = "annarchy_metal"
        _use_single_precision_tolerances()
    elif config.getoption("--cuda"):
        USED_PARADIGM = "cuda"
        NUM_OMP_THREADS = 1
        TARGET_FOLDER = "annarchy_cuda"
    elif config.getoption("--openmp"):
        USED_PARADIGM = "openmp"
        NUM_OMP_THREADS = 3
        TARGET_FOLDER = "annarchy_openmp"

    print(
        f"[TEST CONFIGURATION] paradigm={USED_PARADIGM}, num_threads={NUM_OMP_THREADS}"
    )
