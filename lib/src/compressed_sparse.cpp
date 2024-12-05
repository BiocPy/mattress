#include "def.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <string>
#include <stdexcept>
#include <cstdint>

template<typename Data_, typename Index_>
MatrixPointer initialize_compressed_sparse_matrix_raw(MatrixIndex nr, MatrixValue nc, const Data_* dptr, const Index_* iptr, const pybind11::array_t<uint64_t>& indptr, bool byrow) {
    size_t nz = indptr.at(indptr.size() - 1);
    tatami::ArrayView<Data_> dview(dptr, nz);
    tatami::ArrayView<Index_> iview(iptr, nz);
    tatami::ArrayView<uint64_t> pview(static_cast<const uint64_t*>(indptr.request().ptr), indptr.size());
    return MatrixPointer(new tatami::CompressedSparseMatrix<MatrixValue, MatrixIndex, decltype(dview), decltype(iview), decltype(pview)>(nr, nc, std::move(dview), std::move(iview), std::move(pview), byrow));
}

template<typename Data_>
MatrixPointer initialize_compressed_sparse_matrix_itype(MatrixIndex nr, MatrixValue nc, const Data_* dptr, const pybind11::array& index, const pybind11::array_t<uint64_t>& indptr, bool byrow) {
    auto dtype = index.dtype();
    auto iptr = index.request().ptr;

    if (dtype.is(pybind11::dtype::of<int64_t>())) {
        return initialize_compressed_sparse_matrix_raw(nr, nc, dptr, static_cast<const int64_t*>(iptr), indptr, byrow);
    } else if (dtype.is(pybind11::dtype::of<int32_t>())) {
        return initialize_compressed_sparse_matrix_raw(nr, nc, dptr, static_cast<const int32_t*>(iptr), indptr, byrow);
    } else if (dtype.is(pybind11::dtype::of<int16_t>())) {
        return initialize_compressed_sparse_matrix_raw(nr, nc, dptr, static_cast<const int16_t*>(iptr), indptr, byrow);
    } else if (dtype.is(pybind11::dtype::of<int8_t>())) {
        return initialize_compressed_sparse_matrix_raw(nr, nc, dptr, static_cast<const int8_t*>(iptr), indptr, byrow);
    } else if (dtype.is(pybind11::dtype::of<uint64_t>())) {
        return initialize_compressed_sparse_matrix_raw(nr, nc, dptr, static_cast<const uint64_t*>(iptr), indptr, byrow);
    } else if (dtype.is(pybind11::dtype::of<uint32_t>())) {
        return initialize_compressed_sparse_matrix_raw(nr, nc, dptr, static_cast<const uint32_t*>(iptr), indptr, byrow);
    } else if (dtype.is(pybind11::dtype::of<uint16_t>())) {
        return initialize_compressed_sparse_matrix_raw(nr, nc, dptr, static_cast<const uint16_t*>(iptr), indptr, byrow);
    } else if (dtype.is(pybind11::dtype::of<uint8_t>())) {
        return initialize_compressed_sparse_matrix_raw(nr, nc, dptr, static_cast<const uint8_t*>(iptr), indptr, byrow);
    }

    throw std::runtime_error("unrecognized index type '" + std::string(dtype.kind(), 1) + std::to_string(dtype.itemsize()) + "' for compressed sparse matrix initialization");
    return MatrixPointer();
}

MatrixPointer initialize_compressed_sparse_matrix(MatrixIndex nr, MatrixValue nc, const pybind11::array& data, const pybind11::array& index, const pybind11::array_t<uint64_t>& indptr, bool byrow) {
    auto dtype = data.dtype();
    auto dptr = data.request().ptr;

    if (dtype.is(pybind11::dtype::of<double>())) {
        return initialize_compressed_sparse_matrix_itype(nr, nc, reinterpret_cast<const double*>(dptr), index, indptr, byrow);
    } else if (dtype.is(pybind11::dtype::of<float>())) {
        return initialize_compressed_sparse_matrix_itype(nr, nc, reinterpret_cast<const float*>(dptr), index, indptr, byrow);
    } else if (dtype.is(pybind11::dtype::of<int64_t>())) {
        return initialize_compressed_sparse_matrix_itype(nr, nc, reinterpret_cast<const int64_t*>(dptr), index, indptr, byrow);
    } else if (dtype.is(pybind11::dtype::of<int32_t>())) {
        return initialize_compressed_sparse_matrix_itype(nr, nc, reinterpret_cast<const int32_t*>(dptr), index, indptr, byrow);
    } else if (dtype.is(pybind11::dtype::of<int16_t>())) {
        return initialize_compressed_sparse_matrix_itype(nr, nc, reinterpret_cast<const int16_t*>(dptr), index, indptr, byrow);
    } else if (dtype.is(pybind11::dtype::of<int8_t>())) {
        return initialize_compressed_sparse_matrix_itype(nr, nc, reinterpret_cast<const int8_t*>(dptr), index, indptr, byrow);
    } else if (dtype.is(pybind11::dtype::of<uint64_t>())) {
        return initialize_compressed_sparse_matrix_itype(nr, nc, reinterpret_cast<const uint64_t*>(dptr), index, indptr, byrow);
    } else if (dtype.is(pybind11::dtype::of<uint32_t>())) {
        return initialize_compressed_sparse_matrix_itype(nr, nc, reinterpret_cast<const uint32_t*>(dptr), index, indptr, byrow);
    } else if (dtype.is(pybind11::dtype::of<uint16_t>())) {
        return initialize_compressed_sparse_matrix_itype(nr, nc, reinterpret_cast<const uint16_t*>(dptr), index, indptr, byrow);
    } else if (dtype.is(pybind11::dtype::of<uint8_t>())) {
        return initialize_compressed_sparse_matrix_itype(nr, nc, reinterpret_cast<const uint8_t*>(dptr), index, indptr, byrow);
    }

    throw std::runtime_error("unrecognized data type '" + std::string(dtype.kind(), 1) + std::to_string(dtype.itemsize()) + "' for compressed sparse matrix initialization");
    return MatrixPointer();
}

void init_compressed_sparse(pybind11::module& m) {
    m.def("initialize_compressed_sparse_matrix", &initialize_compressed_sparse_matrix);
}
