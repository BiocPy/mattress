import numpy as np
import delayedarray as da
from mattress import tatamize

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_delayed_noop():
    y = np.random.rand(1000, 100)
    x = da.DelayedArray(y)
    ptr = tatamize(x)
    assert all(ptr.row(0) == y[0, :])
    assert all(ptr.column(1) == y[:, 1])


def test_delayed_unary_isometric_simple():
    y = np.random.rand(1000, 100)
    x = da.DelayedArray(y)
    z = np.log1p(x)
    assert isinstance(z.seed, da.UnaryIsometricOpSimple)

    ptr = tatamize(z)
    assert all(ptr.row(0) == np.log1p(y[0, :]))
    assert all(ptr.column(1) == np.log1p(y[:, 1]))
