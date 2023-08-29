import numpy as np
from mattress import tatamize

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


def test_dense():
    y = np.random.rand(1000, 100)
    ptr = tatamize(y)
    assert all(ptr.row(0) == y[0, :])
    assert all(ptr.column(1) == y[:, 1])

    assert np.allclose(ptr.row_sums(), y.sum(axis=1))
    assert np.allclose(ptr.column_sums(), y.sum(axis=0))
    assert np.allclose(ptr.row_variances(), y.var(axis=1, ddof=1))
    assert np.allclose(ptr.column_variances(), y.var(axis=0, ddof=1))

    assert ptr.row_medians().shape == (1000,)
    assert ptr.column_medians().shape == (100,)

    assert (ptr.row_mins() == y.min(axis=1)).all()
    assert (ptr.column_mins() == y.min(axis=0)).all()
    assert (ptr.row_maxs() == y.max(axis=1)).all()
    assert (ptr.column_maxs() == y.max(axis=0)).all()

    mn, mx = ptr.row_ranges()
    assert (mn == y.min(axis=1)).all()
    assert (mx == y.max(axis=1)).all()
    mn, mx = ptr.column_ranges()
    assert (mn == y.min(axis=0)).all()
    assert (mx == y.max(axis=0)).all()


def test_numpy_with_dtype():
    y = (np.random.rand(50, 12) * 100).astype("i8")
    ptr = tatamize(y)
    assert all(ptr.row(0) == y[0, :])
    assert all(ptr.column(1) == y[:, 1])


def test_dense_column_major():
    y = np.ndarray((1000, 100), order="F")
    y[:, :] = np.random.rand(1000, 100)
    assert y.flags["F_CONTIGUOUS"]
    ptr = tatamize(y)
    assert all(ptr.row(0) == y[0, :])
    assert all(ptr.column(1) == y[:, 1])
