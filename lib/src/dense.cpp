#include "def.h"
#include "tatami/tatami.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <stdexcept>
#include <string>

template<typename Type_>
MatrixPointer initialize_dense_matrix_internal(MatrixIndex nr, MatrixIndex nc, const Type_* ptr, bool byrow) {
    tatami::ArrayView<Type_> view(ptr, static_cast<size_t>(nr) * static_cast<size_t>(nc));
    return MatrixPointer(new tatami::DenseMatrix<MatrixValue, MatrixIndex, decltype(view)>(nr, nc, std::move(view), byrow));
}

MatrixPointer initialize_dense_matrix(MatrixIndex nr, MatrixIndex nc, const pybind11::array& buffer, bool byrow) {
    // Don't make any kind of copy of buffer to coerce the type or storage
    // order, as this should be handled by the caller; we don't provide any
    // GC protection here.
    auto dtype = buffer.dtype();

    if (dtype.is(pybind11::dtype::of<double>())) {
        return initialize_dense_matrix_internal(nr, nc, reinterpret_cast<const double*>(buffer.request().ptr), byrow);
    } else if (dtype.is(pybind11::dtype::of<float>())) {
        return initialize_dense_matrix_internal(nr, nc, reinterpret_cast<const float*>(buffer.request().ptr), byrow);
    } else if (dtype.is(pybind11::dtype::of<int64_t>())) {
        return initialize_dense_matrix_internal(nr, nc, reinterpret_cast<const int64_t*>(buffer.request().ptr), byrow);
    } else if (dtype.is(pybind11::dtype::of<int32_t>())) {
        return initialize_dense_matrix_internal(nr, nc, reinterpret_cast<const int32_t*>(buffer.request().ptr), byrow);
    } else if (dtype.is(pybind11::dtype::of<int16_t>())) {
        return initialize_dense_matrix_internal(nr, nc, reinterpret_cast<const int16_t*>(buffer.request().ptr), byrow);
    } else if (dtype.is(pybind11::dtype::of<int8_t>())) {
        return initialize_dense_matrix_internal(nr, nc, reinterpret_cast<const int8_t*>(buffer.request().ptr), byrow);
    } else if (dtype.is(pybind11::dtype::of<uint64_t>())) {
        return initialize_dense_matrix_internal(nr, nc, reinterpret_cast<const uint64_t*>(buffer.request().ptr), byrow);
    } else if (dtype.is(pybind11::dtype::of<MatrixIndex>())) {
        return initialize_dense_matrix_internal(nr, nc, reinterpret_cast<const MatrixIndex*>(buffer.request().ptr), byrow);
    } else if (dtype.is(pybind11::dtype::of<uint16_t>())) {
        return initialize_dense_matrix_internal(nr, nc, reinterpret_cast<const uint16_t*>(buffer.request().ptr), byrow);
    } else if (dtype.is(pybind11::dtype::of<uint8_t>())) {
        return initialize_dense_matrix_internal(nr, nc, reinterpret_cast<const uint8_t*>(buffer.request().ptr), byrow);
    }

    throw std::runtime_error("unrecognized array type '" + std::string(dtype.kind(), 1) + std::to_string(dtype.itemsize()) + "' for dense matrix initialization");
    return MatrixPointer();
}

void init_dense(pybind11::module& m) {
    m.def("initialize_dense_matrix", &initialize_dense_matrix);
}
