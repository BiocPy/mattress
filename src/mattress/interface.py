from functools import singledispatch
from typing import Any

import numpy as np
import scipy.sparse as sp

from .TatamiNumericPointer import TatamiNumericPointer

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@singledispatch
def tatamize(x: Any) -> TatamiNumericPointer:
    """Converts python matrix representations to tatami.

    Args:
        x (Any): Any matrix-like object.

    Raises:
        NotImplementedError: if x is not supported.

    Returns:
        TatamiNumericPointer: a pointer to tatami object.
    """
    raise NotImplementedError(
        f"tatamize is not supported for objects of class: {type(x)}"
    )


@tatamize.register
def _tatamize_numpy(x: np.ndarray) -> TatamiNumericPointer:
    """Converts numpy representations to tatami.

    Args:
        x (np.ndarray): A numpy nd-array object.

    Raises:
        NotImplementedError: If x is not supported.

    Returns:
        TatamiNumericPointer: Pointer to tatami object.
    """
    return TatamiNumericPointer.from_dense_matrix(x)


@tatamize.register
def _tatamize_sparse_csr_array(x: sp.csr_array) -> TatamiNumericPointer:
    """Converts scipy's CSR representations to tatami.

    Args:
        x (sp.csr_array): A scipy csr array.

    Raises:
        NotImplementedError: If x is not supported.

    Returns:
        TatamiNumericPointer: Pointer to tatami object.
    """
    return TatamiNumericPointer.from_csr_array(x)


@tatamize.register
def _tatamize_sparse_csr_matrix(x: sp.csr_matrix) -> TatamiNumericPointer:
    """Converts scipy's CSR representations to tatami.

    Args:
        x (sp.csr_matrix): A scipy csr array.

    Raises:
        NotImplementedError: If x is not supported.

    Returns:
        TatamiNumericPointer: Pointer to tatami object.
    """
    return TatamiNumericPointer.from_csr_array(x)


@tatamize.register
def _tatamize_sparse_csc_array(x: sp.csc_array) -> TatamiNumericPointer:
    """Converts scipy's CSC representations to tatami.

    Args:
        x (sp.csc_array): A scipy csc array.

    Raises:
        NotImplementedError: If x is not supported.

    Returns:
        TatamiNumericPointer: Pointer to tatami object.
    """
    return TatamiNumericPointer.from_csc_array(x)


@tatamize.register
def _tatamize_sparse_csc_matrix(x: sp.csc_matrix) -> TatamiNumericPointer:
    """Converts scipy's CSC representations to tatami.

    Args:
        x (sp.csc_matrix): A scipy csc array.

    Raises:
        NotImplementedError: If x is not supported.

    Returns:
        TatamiNumericPointer: Pointer to tatami object.
    """
    return TatamiNumericPointer.from_csc_array(x)
