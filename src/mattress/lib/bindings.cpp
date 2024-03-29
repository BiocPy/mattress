/* DO NOT MODIFY: this is automatically generated by the cpptypes */

#include <cstring>
#include <stdexcept>
#include <cstdint>

#ifdef _WIN32
#define PYAPI __declspec(dllexport)
#else
#define PYAPI
#endif

static char* copy_error_message(const char* original) {
    auto n = std::strlen(original);
    auto copy = new char[n + 1];
    std::strcpy(copy, original);
    return copy;
}

void compute_column_maxs(void*, double*, int32_t);

void compute_column_medians(void*, double*, int32_t);

void compute_column_medians_by_group(void*, const int32_t*, double*, int32_t);

void compute_column_mins(void*, double*, int32_t);

void compute_column_nan_counts(void*, int32_t*, int32_t);

void compute_column_ranges(void*, double*, double*, int32_t);

void compute_column_sums(void*, double*, int32_t);

void compute_column_sums_by_group(void*, const int32_t*, double*, int32_t);

void compute_column_variances(void*, double*, int32_t);

void compute_row_maxs(void*, double*, int32_t);

void compute_row_medians(void*, double*, int32_t);

void compute_row_medians_by_group(void*, const int32_t*, double*, int32_t);

void compute_row_mins(void*, double*, int32_t);

void compute_row_nan_counts(void*, int32_t*, int32_t);

void compute_row_ranges(void*, double*, double*, int32_t);

void compute_row_sums(void*, double*, int32_t);

void compute_row_sums_by_group(void*, const int32_t*, double*, int32_t);

void compute_row_variances(void*, double*, int32_t);

void extract_column(void*, int32_t, double*);

void extract_dense_full(void*, double*);

void extract_dense_subset(void*, uint8_t, const int32_t*, int32_t, uint8_t, const int32_t*, int32_t, double*);

int extract_ncol(const void*);

int extract_nrow(const void*);

void extract_row(void*, int32_t, double*);

int extract_sparse(const void*);

void extract_sparse_subset(void*, uint8_t, const int32_t*, int32_t, uint8_t, const int32_t*, int32_t, int32_t*, int32_t*, double*);

void free_mat(void*);

void* initialize_compressed_sparse_matrix(int32_t, int32_t, uint64_t, const char*, void*, const char*, void*, void*, uint8_t);

void* initialize_delayed_binary_isometric_op(void*, void*, const char*);

void* initialize_delayed_combine(int32_t, uintptr_t*, int32_t);

void* initialize_delayed_subset(void*, int32_t, const int32_t*, int32_t);

void* initialize_delayed_transpose(void*);

void* initialize_delayed_unary_isometric_op_simple(void*, const char*);

void* initialize_delayed_unary_isometric_op_with_scalar(void*, const char*, bool, double);

void* initialize_delayed_unary_isometric_op_with_vector(void*, const char*, uint8_t, int32_t, const double*);

void* initialize_dense_matrix(int32_t, int32_t, const char*, void*, uint8_t);

extern "C" {

PYAPI void free_error_message(char** msg) {
    delete [] *msg;
}

PYAPI void py_compute_column_maxs(void* rawmat, double* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_column_maxs(rawmat, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_column_medians(void* rawmat, double* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_column_medians(rawmat, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_column_medians_by_group(void* rawmat, const int32_t* grouping, double* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_column_medians_by_group(rawmat, grouping, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_column_mins(void* rawmat, double* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_column_mins(rawmat, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_column_nan_counts(void* rawmat, int32_t* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_column_nan_counts(rawmat, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_column_ranges(void* rawmat, double* min_output, double* max_output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_column_ranges(rawmat, min_output, max_output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_column_sums(void* rawmat, double* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_column_sums(rawmat, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_column_sums_by_group(void* rawmat, const int32_t* grouping, double* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_column_sums_by_group(rawmat, grouping, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_column_variances(void* rawmat, double* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_column_variances(rawmat, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_row_maxs(void* rawmat, double* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_row_maxs(rawmat, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_row_medians(void* rawmat, double* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_row_medians(rawmat, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_row_medians_by_group(void* rawmat, const int32_t* grouping, double* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_row_medians_by_group(rawmat, grouping, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_row_mins(void* rawmat, double* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_row_mins(rawmat, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_row_nan_counts(void* rawmat, int32_t* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_row_nan_counts(rawmat, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_row_ranges(void* rawmat, double* min_output, double* max_output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_row_ranges(rawmat, min_output, max_output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_row_sums(void* rawmat, double* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_row_sums(rawmat, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_row_sums_by_group(void* rawmat, const int32_t* grouping, double* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_row_sums_by_group(rawmat, grouping, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_compute_row_variances(void* rawmat, double* output, int32_t num_threads, int32_t* errcode, char** errmsg) {
    try {
        compute_row_variances(rawmat, output, num_threads);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_extract_column(void* rawmat, int32_t c, double* output, int32_t* errcode, char** errmsg) {
    try {
        extract_column(rawmat, c, output);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_extract_dense_full(void* rawmat, double* output, int32_t* errcode, char** errmsg) {
    try {
        extract_dense_full(rawmat, output);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_extract_dense_subset(void* rawmat, uint8_t row_noop, const int32_t* row_sub, int32_t row_len, uint8_t col_noop, const int32_t* col_sub, int32_t col_len, double* output, int32_t* errcode, char** errmsg) {
    try {
        extract_dense_subset(rawmat, row_noop, row_sub, row_len, col_noop, col_sub, col_len, output);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI int py_extract_ncol(const void* mat, int32_t* errcode, char** errmsg) {
    int output = 0;
    try {
        output = extract_ncol(mat);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI int py_extract_nrow(const void* mat, int32_t* errcode, char** errmsg) {
    int output = 0;
    try {
        output = extract_nrow(mat);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void py_extract_row(void* rawmat, int32_t r, double* output, int32_t* errcode, char** errmsg) {
    try {
        extract_row(rawmat, r, output);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI int py_extract_sparse(const void* mat, int32_t* errcode, char** errmsg) {
    int output = 0;
    try {
        output = extract_sparse(mat);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void py_extract_sparse_subset(void* rawmat, uint8_t row_noop, const int32_t* row_sub, int32_t row_len, uint8_t col_noop, const int32_t* col_sub, int32_t col_len, int32_t* output_count, int32_t* output_indices, double* output_values, int32_t* errcode, char** errmsg) {
    try {
        extract_sparse_subset(rawmat, row_noop, row_sub, row_len, col_noop, col_sub, col_len, output_count, output_indices, output_values);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void py_free_mat(void* mat, int32_t* errcode, char** errmsg) {
    try {
        free_mat(mat);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void* py_initialize_compressed_sparse_matrix(int32_t nr, int32_t nc, uint64_t nz, const char* dtype, void* dptr, const char* itype, void* iptr, void* indptr, uint8_t byrow, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_compressed_sparse_matrix(nr, nc, nz, dtype, dptr, itype, iptr, indptr, byrow);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_delayed_binary_isometric_op(void* left, void* right, const char* op, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_delayed_binary_isometric_op(left, right, op);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_delayed_combine(int32_t n, uintptr_t* ptrs, int32_t dim, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_delayed_combine(n, ptrs, dim);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_delayed_subset(void* ptr, int32_t dim, const int32_t* subset, int32_t len, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_delayed_subset(ptr, dim, subset, len);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_delayed_transpose(void* ptr, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_delayed_transpose(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_delayed_unary_isometric_op_simple(void* ptr, const char* op, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_delayed_unary_isometric_op_simple(ptr, op);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_delayed_unary_isometric_op_with_scalar(void* ptr, const char* op, bool right, double arg, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_delayed_unary_isometric_op_with_scalar(ptr, op, right, arg);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_delayed_unary_isometric_op_with_vector(void* ptr, const char* op, uint8_t right, int32_t along, const double* args, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_delayed_unary_isometric_op_with_vector(ptr, op, right, along, args);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_dense_matrix(int32_t nr, int32_t nc, const char* type, void* ptr, uint8_t byrow, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_dense_matrix(nr, nc, type, ptr, byrow);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

}
