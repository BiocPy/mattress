#include "pybind11/pybind11.h"

void init_common(pybind11::module&);
void init_dense(pybind11::module&);
void init_compressed_sparse(pybind11::module&);
void init_delayed_binary_isometric_operation(pybind11::module&);

PYBIND11_MODULE(lib_mattress, m) {
    init_common(m);
    init_dense(m);
    init_compressed_sparse(m);
    init_delayed_binary_isometric_operation(m);
}
