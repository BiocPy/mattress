# DO NOT MODIFY: this is automatically generated by the cpptypes

import os
import ctypes as ct


def catch_errors(f):
    def wrapper(*args):
        errcode = ct.c_int32(0)
        errmsg = ct.c_char_p(0)
        output = f(*args, ct.byref(errcode), ct.byref(errmsg))
        if errcode.value != 0:
            msg = errmsg.value.decode("ascii")
            lib.free_error_message(errmsg)
            raise RuntimeError(msg)
        return output

    return wrapper


# TODO: surely there's a better way than whatever this is.
dirname = os.path.dirname(os.path.abspath(__file__))
contents = os.listdir(dirname)
lib = None
for x in contents:
    if x.startswith("core") and not x.endswith("py"):
        lib = ct.CDLL(os.path.join(dirname, x))
        break

if lib is None:
    raise ImportError("failed to find the core.* module")

lib.free_error_message.argtypes = [ct.POINTER(ct.c_char_p)]

import numpy as np


def np2ct(x, expected, contiguous=True):
    if not isinstance(x, np.ndarray):
        raise ValueError("expected a NumPy array")
    if x.dtype != expected:
        raise ValueError(
            "expected a NumPy array of type " + str(expected) + ", got " + str(x.dtype)
        )
    if contiguous:
        if not x.flags.c_contiguous and not x.flags.f_contiguous:
            raise ValueError("only contiguous NumPy arrays are supported")
    return x.ctypes.data


lib.py_extract_column.restype = None
lib.py_extract_column.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p),
]

lib.py_extract_ncol.restype = ct.c_int
lib.py_extract_ncol.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p),
]

lib.py_extract_nrow.restype = ct.c_int
lib.py_extract_nrow.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p),
]

lib.py_extract_row.restype = None
lib.py_extract_row.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p),
]

lib.py_extract_sparse.restype = ct.c_int
lib.py_extract_sparse.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p),
]

lib.py_free_mat.restype = None
lib.py_free_mat.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p),
]

lib.py_initialize_compressed_sparse_matrix.restype = ct.c_void_p
lib.py_initialize_compressed_sparse_matrix.argtypes = [
    ct.c_int32,
    ct.c_int32,
    ct.c_uint64,
    ct.c_char_p,
    ct.c_void_p,
    ct.c_char_p,
    ct.c_void_p,
    ct.c_void_p,
    ct.c_uint8,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p),
]

lib.py_initialize_delayed_binary_isometric_op.restype = ct.c_void_p
lib.py_initialize_delayed_binary_isometric_op.argtypes = [
    ct.c_void_p,
    ct.c_void_p,
    ct.c_char_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p),
]

lib.py_initialize_delayed_combine.restype = ct.c_void_p
lib.py_initialize_delayed_combine.argtypes = [
    ct.c_int32,
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p),
]

lib.py_initialize_delayed_subset.restype = ct.c_void_p
lib.py_initialize_delayed_subset.argtypes = [
    ct.c_void_p,
    ct.c_int32,
    ct.c_void_p,
    ct.c_int32,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p),
]

lib.py_initialize_delayed_transpose.restype = ct.c_void_p
lib.py_initialize_delayed_transpose.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p),
]

lib.py_initialize_delayed_unary_isometric_op_simple.restype = ct.c_void_p
lib.py_initialize_delayed_unary_isometric_op_simple.argtypes = [
    ct.c_void_p,
    ct.c_char_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p),
]

lib.py_initialize_delayed_unary_isometric_op_with_scalar.restype = ct.c_void_p
lib.py_initialize_delayed_unary_isometric_op_with_scalar.argtypes = [
    ct.c_void_p,
    ct.c_char_p,
    ct.c_bool,
    ct.c_double,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p),
]

lib.py_initialize_delayed_unary_isometric_op_with_vector.restype = ct.c_void_p
lib.py_initialize_delayed_unary_isometric_op_with_vector.argtypes = [
    ct.c_void_p,
    ct.c_char_p,
    ct.c_uint8,
    ct.c_int32,
    ct.c_void_p,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p),
]

lib.py_initialize_dense_matrix.restype = ct.c_void_p
lib.py_initialize_dense_matrix.argtypes = [
    ct.c_int32,
    ct.c_int32,
    ct.c_char_p,
    ct.c_void_p,
    ct.c_uint8,
    ct.POINTER(ct.c_int32),
    ct.POINTER(ct.c_char_p),
]


def extract_column(rawmat, c, output):
    return catch_errors(lib.py_extract_column)(rawmat, c, output)


def extract_ncol(mat):
    return catch_errors(lib.py_extract_ncol)(mat)


def extract_nrow(mat):
    return catch_errors(lib.py_extract_nrow)(mat)


def extract_row(rawmat, r, output):
    return catch_errors(lib.py_extract_row)(rawmat, r, output)


def extract_sparse(mat):
    return catch_errors(lib.py_extract_sparse)(mat)


def free_mat(mat):
    return catch_errors(lib.py_free_mat)(mat)


def initialize_compressed_sparse_matrix(
    nr, nc, nz, dtype, dptr, itype, iptr, indptr, byrow
):
    return catch_errors(lib.py_initialize_compressed_sparse_matrix)(
        nr, nc, nz, dtype, dptr, itype, iptr, indptr, byrow
    )


def initialize_delayed_binary_isometric_op(left, right, op):
    return catch_errors(lib.py_initialize_delayed_binary_isometric_op)(left, right, op)


def initialize_delayed_combine(n, ptrs, dim):
    return catch_errors(lib.py_initialize_delayed_combine)(n, ptrs, dim)


def initialize_delayed_subset(ptr, dim, subset, len):
    return catch_errors(lib.py_initialize_delayed_subset)(
        ptr, dim, np2ct(subset, np.uint32), len
    )


def initialize_delayed_transpose(ptr):
    return catch_errors(lib.py_initialize_delayed_transpose)(ptr)


def initialize_delayed_unary_isometric_op_simple(ptr, op):
    return catch_errors(lib.py_initialize_delayed_unary_isometric_op_simple)(ptr, op)


def initialize_delayed_unary_isometric_op_with_scalar(ptr, op, right, arg):
    return catch_errors(lib.py_initialize_delayed_unary_isometric_op_with_scalar)(
        ptr, op, right, arg
    )


def initialize_delayed_unary_isometric_op_with_vector(ptr, op, right, along, args):
    return catch_errors(lib.py_initialize_delayed_unary_isometric_op_with_vector)(
        ptr, op, right, along, np2ct(args, np.float64)
    )


def initialize_dense_matrix(nr, nc, type, ptr, byrow):
    return catch_errors(lib.py_initialize_dense_matrix)(nr, nc, type, ptr, byrow)
