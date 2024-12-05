#include "pybind11/pybind11.h"

void init_common(pybind11::module&);
void init_dense(pybind11::module&);

PYBIND11_MODULE(lib_mattress, m) {
    init_common(m);
    init_dense(m);
}
