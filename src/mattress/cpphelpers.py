import ctypes as ct
import os
from typing import Sequence

__author__ = "ltla, jkanche"
__copyright__ = "ltla, jkanche"
__license__ = "MIT"


# TODO: surely there's a better way than whatever this is.
def includes() -> Sequence[str]:
    """Provides access to C++ headers (including tatami) for downstream packages.

    Returns:
        Sequence[str]: list of paths to the header files.
    """
    dirname = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(dirname, "extern", "tatami", "include"),
        os.path.join(dirname, "include"),
    ]


def load_dll() -> ct.CDLL:
    """load the shared library.

    usually starts with core.<platform>.<so or dll>.

    Returns:
        (ct.CDLL, optional): shared object.
    """
    dirname = os.path.dirname(os.path.abspath(__file__))
    contents = os.listdir(dirname)
    for x in contents:
        if x.startswith("core") and not x.endswith("py"):
            return ct.CDLL(os.path.join(dirname, x))

    raise Exception(
        "Cannot find the shared object file! Report this issue to dev on github."
    )


lib = load_dll()

lib.py_free_mat.argtypes = [ct.c_void_p]
lib.py_extract_nrow.restype = ct.c_int
lib.py_extract_nrow.argtypes = [ct.c_void_p]
lib.py_extract_ncol.restype = ct.c_int
lib.py_extract_ncol.argtypes = [ct.c_void_p]
lib.py_extract_sparse.restype = ct.c_int
lib.py_extract_sparse.argtypes = [ct.c_void_p]
lib.py_extract_row.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]
lib.py_extract_column.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p]

lib.py_initialize_dense_matrix.restype = ct.c_void_p
lib.py_initialize_dense_matrix.argtypes = [
    ct.c_int,
    ct.c_int,
    ct.c_char_p,
    ct.c_void_p,
    ct.c_uint8,
]

lib.py_initialize_compressed_sparse_matrix.restype = ct.c_void_p
lib.py_initialize_compressed_sparse_matrix.argtypes = [
    ct.c_int,
    ct.c_int,
    ct.c_uint64,
    ct.c_char_p,
    ct.c_void_p,
    ct.c_char_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_uint8,
]
