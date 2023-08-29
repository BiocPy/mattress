#include "Mattress.h"
#include "tatami/stats/ranges.hpp" // oops.
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

/** Extraction **/

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

/** Stats **/

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
void compute_column_variances(void* rawmat, double* output /** void_p */, int32_t num_threads) {
    auto mat = reinterpret_cast<Mattress*>(rawmat);
    auto res = tatami::column_variances(mat->ptr.get(), num_threads);
    std::copy(res.begin(), res.end(), output);
}

//[[export]]
void compute_row_variances(void* rawmat, double* output /** void_p */, int32_t num_threads) {
    auto mat = reinterpret_cast<Mattress*>(rawmat);
    auto res = tatami::row_variances(mat->ptr.get(), num_threads);
    std::copy(res.begin(), res.end(), output);
}

//[[export]]
void compute_column_medians(void* rawmat, double* output /** void_p */, int32_t num_threads) {
    auto mat = reinterpret_cast<Mattress*>(rawmat);
    auto res = tatami::column_medians(mat->ptr.get(), num_threads);
    std::copy(res.begin(), res.end(), output);
}

//[[export]]
void compute_row_medians(void* rawmat, double* output /** void_p */, int32_t num_threads) {
    auto mat = reinterpret_cast<Mattress*>(rawmat);
    auto res = tatami::row_medians(mat->ptr.get(), num_threads);
    std::copy(res.begin(), res.end(), output);
}

//[[export]]
void compute_column_mins(void* rawmat, double* output /** void_p */, int32_t num_threads) {
    auto mat = reinterpret_cast<Mattress*>(rawmat);
    auto res = tatami::column_mins(mat->ptr.get(), num_threads);
    std::copy(res.begin(), res.end(), output);
}

//[[export]]
void compute_row_mins(void* rawmat, double* output /** void_p */, int32_t num_threads) {
    auto mat = reinterpret_cast<Mattress*>(rawmat);
    auto res = tatami::row_mins(mat->ptr.get(), num_threads);
    std::copy(res.begin(), res.end(), output);
}

//[[export]]
void compute_column_maxs(void* rawmat, double* output /** void_p */, int32_t num_threads) {
    auto mat = reinterpret_cast<Mattress*>(rawmat);
    auto res = tatami::column_maxs(mat->ptr.get(), num_threads);
    std::copy(res.begin(), res.end(), output);
}

//[[export]]
void compute_row_maxs(void* rawmat, double* output /** void_p */, int32_t num_threads) {
    auto mat = reinterpret_cast<Mattress*>(rawmat);
    auto res = tatami::row_maxs(mat->ptr.get(), num_threads);
    std::copy(res.begin(), res.end(), output);
}

//[[export]]
void compute_column_ranges(void* rawmat, double* min_output /** void_p */, double* max_output /** void_p */, int32_t num_threads) {
    auto mat = reinterpret_cast<Mattress*>(rawmat);
    auto res = tatami::column_ranges(mat->ptr.get(), num_threads);
    std::copy(res.first.begin(), res.first.end(), min_output);
    std::copy(res.second.begin(), res.second.end(), max_output);
}

//[[export]]
void compute_row_ranges(void* rawmat, double* min_output /** void_p */, double* max_output /** void_p */, int32_t num_threads) {
    auto mat = reinterpret_cast<Mattress*>(rawmat);
    auto res = tatami::row_ranges(mat->ptr.get(), num_threads);
    std::copy(res.first.begin(), res.first.end(), min_output);
    std::copy(res.second.begin(), res.second.end(), max_output);
}

/** Freeing **/

//[[export]]
void free_mat(void* mat) {
    delete reinterpret_cast<Mattress*>(mat);
}
