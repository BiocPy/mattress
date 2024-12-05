#include "pybind11/pybind11.h"

void init_common(pybind11::module&);

PYBIND11_MODULE(mattress, m) {
    init_common(m);
}
