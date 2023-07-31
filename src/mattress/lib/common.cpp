#include "Mattress.h"

//[[export]]
int extract_nrow(const void* mat) {
    return reinterpret_cast<const Mattress*>(mat)->ptr->nrow();
}

//[[export]]
int extract_ncol(const void* mat) {
    return reinterpret_cast<const Mattress*>(mat)->ptr->ncol();
}

//[[export]]
int extract_sparse(const void* mat) {
    return reinterpret_cast<const Mattress*>(mat)->ptr->sparse();
}

//[[export]]
void extract_row(void* rawmat, int r, double* output) {
    auto mat = reinterpret_cast<Mattress*>(rawmat);
    if (!mat->byrow) {
        mat->byrow = mat->ptr->dense_row();
    }
    mat->byrow->fetch_copy(r, output);
}

//[[export]]
void extract_column(void* rawmat, int c, double* output) {
    auto mat = reinterpret_cast<Mattress*>(rawmat);
    if (!mat->bycol) {
        mat->bycol = mat->ptr->dense_column();
    }
    mat->bycol->fetch_copy(c, output);
}

//[[export]]
void free_mat(void* mat) {
    delete reinterpret_cast<Mattress*>(mat);
}
