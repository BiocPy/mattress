#include "def.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <string>
#include <cstdint>

MatrixPointer initialize_delayed_subset(MatrixPointer mat, const pybind11::array_t<MatrixIndex>& subset, bool byrow) {
    return tatami::make_DelayedSubset(std::move(mat), tatami::ArrayView<MatrixIndex>(static_cast<const MatrixIndex*>(subset.request().ptr), subset.size()), byrow);
}

void init_delayed_subset(pybind11::module& m) {
    m.def("initialize_delayed_subset", &initialize_delayed_subset);
}
