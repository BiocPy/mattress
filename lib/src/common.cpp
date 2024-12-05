#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

int extract_nrow(const std::shared_ptr<tatami::NumericMatrix>& mat) {
    return mat->nrow();
}

int extract_ncol(const std::shared_ptr<tatami::NumericMatrix>& mat) {
    return mat->ncol();
}

bool extract_sparse(const std::shared_ptr<tatami::NumericMatrix>& mat) {
    return mat->is_sparse();
}

pybind11::array_t<double> extract_row(const std::shared_ptr<tatami::NumericMatrix>& mat, int r) {
    pybind11::array_t<double> output(mat->ncol());
    auto optr = static_cast<double*>(output.request().ptr);
    auto ext = tatami::consecutive_extractor<false>(mat.get(), true, r, 1);
    auto out = ext->fetch(optr);
    tatami::copy_n(out, output.size(), optr);
    return output;
}

pybind11::array_t<double> extract_column(const std::shared_ptr<tatami::NumericMatrix>& mat, int c) {
    pybind11::array_t<double> output(mat->nrow());
    auto optr = static_cast<double*>(output.request().ptr);
    auto ext = tatami::consecutive_extractor<false>(mat.get(), false, c, 1);
    auto out = ext->fetch(optr);
    tatami::copy_n(out, output.size(), optr);
    return output;
}

/** Stats **/

pybind11::array_t<double> compute_column_sums(const std::shared_ptr<tatami::NumericMatrix>& mat, int num_threads) {
    pybind11::array_t<double> output(mat->ncol());
    auto optr = static_cast<double*>(output.request().ptr);
    tatami_stats::sums::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::sums::apply(false, mat.get(), optr, opt);
    return output;
}

pybind11::array_t<double> compute_row_sums(const std::shared_ptr<tatami::NumericMatrix>& mat, int num_threads) {
    pybind11::array_t<double> output(mat->nrow());
    auto optr = static_cast<double*>(output.request().ptr);
    tatami_stats::sums::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::sums::apply(true, mat.get(), optr, opt);
    return output;
}

pybind11::array_t<double> compute_column_variances(const std::shared_ptr<tatami::NumericMatrix>& mat, int num_threads) {
    pybind11::array_t<double> output(mat->ncol());
    auto optr = static_cast<double*>(output.request().ptr);
    tatami_stats::variances::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::variances::apply(false, mat.get(), optr, opt);
    return output;
}

pybind11::array_t<double> compute_row_variances(const std::shared_ptr<tatami::NumericMatrix>& mat, int num_threads) {
    pybind11::array_t<double> output(mat->nrow());
    auto optr = static_cast<double*>(output.request().ptr);
    tatami_stats::variances::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::variances::apply(true, mat.get(), optr, opt);
    return output;
}

pybind11::array_t<double> compute_column_medians(const std::shared_ptr<tatami::NumericMatrix>& mat, int num_threads) {
    pybind11::array_t<double> output(mat->ncol());
    auto optr = static_cast<double*>(output.request().ptr);
    tatami_stats::medians::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::medians::apply(false, mat.get(), optr, opt);
    return output;
}

pybind11::array_t<double> compute_row_medians(const std::shared_ptr<tatami::NumericMatrix>& mat, int num_threads) {
    pybind11::array_t<double> output(mat->nrow());
    auto optr = static_cast<double*>(output.request().ptr);
    tatami_stats::medians::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::medians::apply(true, mat.get(), optr, opt);
    return output;
}

pybind11::array_t<double> compute_column_mins(const std::shared_ptr<tatami::NumericMatrix>& mat, int num_threads) {
    pybind11::array_t<double> output(mat->ncol());
    auto optr = static_cast<double*>(output.request().ptr);
    tatami_stats::ranges::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::ranges::apply(false, mat.get(), optr, static_cast<double*>(NULL), opt);
    return output;
}

pybind11::array_t<double> compute_row_mins(const std::shared_ptr<tatami::NumericMatrix>& mat, int num_threads) {
    pybind11::array_t<double> output(mat->nrow());
    auto optr = static_cast<double*>(output.request().ptr);
    tatami_stats::ranges::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::ranges::apply(true, mat.get(), optr, static_cast<double*>(NULL), opt);
    return output;
}

pybind11::array_t<double> compute_column_maxs(const std::shared_ptr<tatami::NumericMatrix>& mat, int num_threads) {
    pybind11::array_t<double> output(mat->ncol());
    auto optr = static_cast<double*>(output.request().ptr);
    tatami_stats::ranges::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::ranges::apply(false, mat.get(), optr, static_cast<double*>(NULL), opt);
    return output;
}

pybind11::array_t<double> compute_row_maxs(const std::shared_ptr<tatami::NumericMatrix>& mat, int num_threads) {
    pybind11::array_t<double> output(mat->nrow());
    auto optr = static_cast<double*>(output.request().ptr);
    tatami_stats::ranges::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::ranges::apply(true, mat.get(), static_cast<double*>(NULL), optr, opt);
    return output;
}

pybind11::list compute_row_ranges(const std::shared_ptr<tatami::NumericMatrix>& mat, int num_threads) {
    pybind11::array_t<double> mnout(mat->nrow()), mxout(mat->nrow());
    auto mnptr = static_cast<double*>(mnout.request().ptr);
    auto mxptr = static_cast<double*>(mxout.request().ptr);
    tatami_stats::ranges::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::ranges::apply(true, mat.get(), mnptr, mxptr, opt);

    pybind11::list output(2);
    output[0] = mnout;
    output[1] = mxout;
    return output;
}

pybind11::list compute_column_ranges(const std::shared_ptr<tatami::NumericMatrix>& mat, int num_threads) {
    pybind11::array_t<double> mnout(mat->ncol()), mxout(mat->ncol());
    auto mnptr = static_cast<double*>(mnout.request().ptr);
    auto mxptr = static_cast<double*>(mxout.request().ptr);
    tatami_stats::ranges::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::ranges::apply(false, mat.get(), mnptr, mxptr, opt);

    pybind11::list output(2);
    output[0] = mnout;
    output[1] = mxout;
    return output;
}

pybind11::array_t<int32_t> compute_row_nan_counts(const std::shared_ptr<tatami::NumericMatrix>& mat, int num_threads) {
    pybind11::array_t<int32_t> output(mat->nrow());
    auto optr = static_cast<int32_t*>(output.request().ptr);
    tatami_stats::counts::nan::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::counts::nan::apply(true, mat.get(), optr, opt);
    return output;
}

pybind11::array_t<int32_t> compute_column_nan_counts(const std::shared_ptr<tatami::NumericMatrix>& mat, int num_threads) {
    pybind11::array_t<int32_t> output(mat->ncol());
    auto optr = static_cast<int32_t*>(output.request().ptr);
    tatami_stats::counts::nan::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::counts::nan::apply(false, mat.get(), optr, opt);
    return output;
}

/** Grouped stats **/

pybind11::array_t<double> compute_row_sums_by_group(const std::shared_ptr<tatami::NumericMatrix>& mat, const pybind11::array_t<int32_t>& grouping, int num_threads) {
    auto gptr = static_cast<const int32_t*>(grouping.request().ptr);
    size_t ngroups = tatami_stats::total_groups(gptr, mat->ncol());
    size_t nrow = mat->nrow();
    pybind11::array_t<double, pybind11::array::f_style> output({ nrow, ngroups });

    auto optr = static_cast<double*>(output.request().ptr);
    std::vector<double*> ptrs(ngroups);
    for (size_t g = 0; g < ngroups; ++g) {
        ptrs[g] = optr + g * nrow;
    }

    tatami_stats::grouped_sums::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::grouped_sums::apply(true, mat.get(), gptr, ngroups, ptrs.data(), opt);
    return output;
}

pybind11::array_t<double> compute_column_sums_by_group(const std::shared_ptr<tatami::NumericMatrix>& mat, const pybind11::array_t<int32_t>& grouping, int num_threads) {
    auto gptr = static_cast<const int32_t*>(grouping.request().ptr);
    size_t ngroups = tatami_stats::total_groups(gptr, mat->nrow());
    size_t ncol = mat->ncol();
    pybind11::array_t<double, pybind11::array::f_style> output({ ncol, ngroups });

    auto optr = static_cast<double*>(output.request().ptr);
    std::vector<double*> ptrs(ngroups);
    for (size_t g = 0; g < ngroups; ++g) {
        ptrs[g] = optr + g * ncol;
    }

    tatami_stats::grouped_sums::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::grouped_sums::apply(false, mat.get(), gptr, ngroups, ptrs.data(), opt);
    return output;
}

pybind11::array_t<double> compute_row_variances_by_group(const std::shared_ptr<tatami::NumericMatrix>& mat, const pybind11::array_t<int32_t>& grouping, int num_threads) {
    auto gptr = static_cast<const int32_t*>(grouping.request().ptr);
    auto group_sizes = tatami_stats::tabulate_groups(gptr, mat->ncol());
    size_t ngroups = group_sizes.size();
    size_t nrow = mat->nrow();
    pybind11::array_t<double, pybind11::array::f_style> output({ nrow, ngroups });

    auto optr = static_cast<double*>(output.request().ptr);
    std::vector<double*> ptrs(ngroups);
    for (size_t g = 0; g < ngroups; ++g) {
        ptrs[g] = optr + g * nrow;
    }

    tatami_stats::grouped_variances::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::grouped_variances::apply(true, mat.get(), gptr, ngroups, group_sizes.data(), ptrs.data(), opt);
    return output;
}

pybind11::array_t<double> compute_column_variances_by_group(const std::shared_ptr<tatami::NumericMatrix>& mat, const pybind11::array_t<int32_t>& grouping, int num_threads) {
    auto gptr = static_cast<const int32_t*>(grouping.request().ptr);
    auto group_sizes = tatami_stats::tabulate_groups(gptr, mat->ncol());
    size_t ngroups = group_sizes.size();
    size_t ncol = mat->ncol();
    pybind11::array_t<double, pybind11::array::f_style> output({ ncol, ngroups });

    auto optr = static_cast<double*>(output.request().ptr);
    std::vector<double*> ptrs(ngroups);
    for (size_t g = 0; g < ngroups; ++g) {
        ptrs[g] = optr + g * ncol;
    }

    tatami_stats::grouped_variances::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::grouped_variances::apply(false, mat.get(), grouping.data(), ngroups, group_sizes.data(), ptrs.data(), opt);
    return output;
}

pybind11::array_t<double> compute_row_medians_by_group(const std::shared_ptr<tatami::NumericMatrix>& mat, const pybind11::array_t<int32_t>& grouping, int num_threads) {
    auto gptr = static_cast<const int32_t*>(grouping.request().ptr);
    auto group_sizes = tatami_stats::tabulate_groups(gptr, mat->ncol());
    size_t ngroups = group_sizes.size();
    size_t nrow = mat->nrow();
    pybind11::array_t<double, pybind11::array::f_style> output({ nrow, ngroups });

    auto optr = static_cast<double*>(output.request().ptr);
    std::vector<double*> ptrs(ngroups);
    for (size_t g = 0; g < ngroups; ++g) {
        ptrs[g] = optr + g * nrow;
    }

    tatami_stats::grouped_medians::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::grouped_medians::apply(true, mat.get(), gptr, group_sizes, ptrs.data(), opt);
    return output;
}

pybind11::array_t<double> compute_column_medians_by_group(const std::shared_ptr<tatami::NumericMatrix>& mat, const pybind11::array_t<int32_t>& grouping, int num_threads) {
    auto gptr = static_cast<const int32_t*>(grouping.request().ptr);
    auto group_sizes = tatami_stats::tabulate_groups(gptr, mat->ncol());
    size_t ngroups = group_sizes.size();
    size_t ncol = mat->ncol();
    pybind11::array_t<double, pybind11::array::f_style> output({ ncol, ngroups });

    auto optr = static_cast<double*>(output.request().ptr);
    std::vector<double*> ptrs(ngroups);
    for (size_t g = 0; g < ngroups; ++g) {
        ptrs[g] = optr + g * ncol;
    }

    tatami_stats::grouped_medians::Options opt;
    opt.num_threads = num_threads;
    tatami_stats::grouped_medians::apply(false, mat.get(), gptr, group_sizes, ptrs.data(), opt);
    return output;
}

/** Extraction **/

pybind11::array_t<double> extract_dense_subset(std::shared_ptr<tatami::NumericMatrix> mat,
    bool row_noop, const pybind11::array_t<int32_t>& row_sub,
    bool col_noop, const pybind11::array_t<int32_t>& col_sub,
    int num_threads) 
{
    if (!row_noop) {
        auto tmp = tatami::make_DelayedSubset<0>(std::move(mat), tatami::ArrayView<int32_t>(row_sub.data(), row_sub.size()));
        mat.swap(tmp);
    }
    if (!col_noop) {
        auto tmp = tatami::make_DelayedSubset<1>(std::move(mat), tatami::ArrayView<int32_t>(col_sub.data(), col_sub.size()));
        mat.swap(tmp);
    }

    size_t NR = mat->nrow(), NC = mat->ncol();
    pybind11::array_t<double, pybind11::array::f_style> output({ NR, NC });
    auto optr = static_cast<double*>(output.request().ptr);
    tatami::convert_to_dense(mat.get(), false, optr, num_threads);
    return output;
}

pybind11::array_t<double> extract_sparse_subset(std::shared_ptr<tatami::NumericMatrix> mat,
    bool row_noop, const pybind11::array_t<int32_t>& row_sub,
    bool col_noop, const pybind11::array_t<int32_t>& col_sub,
    int num_threads)
{
    if (!row_noop) {
        auto tmp = tatami::make_DelayedSubset<0>(std::move(mat), tatami::ArrayView<int32_t>(row_sub.data(), row_sub.size()));
        mat.swap(tmp);
    }
    if (!col_noop) {
        auto tmp = tatami::make_DelayedSubset<1>(std::move(mat), tatami::ArrayView<int32_t>(col_sub.data(), col_sub.size()));
        mat.swap(tmp);
    }

    int NC = mat->ncol();
    int NR = mat->nrow();
    pybind11::list content(NC);

    if (mat->prefer_rows()) {
        std::vector<std::vector<double> > vcollection(NC);
        std::vector<std::vector<int32_t> > icollection(NC);

        auto ext = tatami::consecutive_extractor<true>(mat.get(), true, 0, NR);
        std::vector<double> vbuffer(NC);
        std::vector<int> ibuffer(NC);

        for (int r = 0; r < NR; ++r) {
            auto info = ext->fetch(vbuffer.data(), ibuffer.data());
            for (int i = 0; i < info.number; ++i) {
                auto c = info.index[i];
                vcollection[c].push_back(info.value[i]);
                icollection[c].push_back(r);
            }
        }

        for (int c = 0; c < NC; ++c) {
            pybind11::list tmp(2);
            tmp[0] = pybind11::array_t<double>(vcollection[c].size(), vcollection[c].data());
            tmp[1] = pybind11::array_t<int32_t>(icollection[c].size(), icollection[c].data());
            content[c] = std::move(tmp);
        }

    } else {
        auto ext = tatami::consecutive_extractor<true>(mat.get(), false, 0, NC);
        std::vector<double> vbuffer(NC);
        std::vector<int> ibuffer(NC);

        for (int c = 0; c < NC; ++c) {
            auto info = ext->fetch(vbuffer.data(), ibuffer.data());
            pybind11::list tmp(2);
            tmp[0] = pybind11::array_t<double>(info.number, info.value);
            tmp[1] = pybind11::array_t<int32_t>(info.number, info.index);
            content[c] = std::move(tmp);
        }
    }

    pybind11::tuple shape(2);
    shape[0] = NR;
    shape[1] = NC;
    pybind11::module bu = pybind11::module::import("delayedarray");
    return bu.attr("SparseNdarray")(shape, content, pybind11::dtype("float64"), pybind11::dtype("int32"), false, false);
}
