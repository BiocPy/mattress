#include "Mattress.h"
#include <cstdint>
#include <algorithm>

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
void extract_row(void* rawmat, int32_t r, double* output /** void_p */) {
    auto mat = reinterpret_cast<Mattress*>(rawmat);
    if (!mat->byrow) {
        mat->byrow = mat->ptr->dense_row();
    }
    mat->byrow->fetch_copy(r, output);
}

//[[export]]
void extract_column(void* rawmat, int32_t c, double* output /** void_p */) {
    auto mat = reinterpret_cast<Mattress*>(rawmat);
    if (!mat->bycol) {
        mat->bycol = mat->ptr->dense_column();
    }
    mat->bycol->fetch_copy(c, output);
}

//[[export]]
void compute_column_sums(void* rawmat, double* output /** void_p */, int32_t num_threads) {
    auto mat = reinterpret_cast<Mattress*>(rawmat);
    auto res = tatami::column_sums(mat->ptr.get(), num_threads);
    std::copy(res.begin(), res.end(), output);
}

//[[export]]
void compute_row_sums(void* rawmat, double* output /** void_p */, int32_t num_threads) {
    auto mat = reinterpret_cast<Mattress*>(rawmat);
    auto res = tatami::row_sums(mat->ptr.get(), num_threads);
    std::copy(res.begin(), res.end(), output);
}

//[[export]]
void free_mat(void* mat) {
    delete reinterpret_cast<Mattress*>(mat);
}

