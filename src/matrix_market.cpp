#include "matrix_market.h"

#include <Eigen/Sparse>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include "mmio.h"
}

namespace {

[[noreturn]] void fail_with_file(FILE* f, const std::string& message) {
    if (f != nullptr) {
        fclose(f);
    }
    throw std::runtime_error(message);
}

} // namespace

Eigen::SparseMatrix<double> load_matrix_market(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (f == nullptr) {
        throw std::runtime_error("could not open file: " + path);
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        fail_with_file(f, "could not read Matrix Market banner: " + path);
    }

    if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode)) {
        fail_with_file(f, "only sparse Matrix Market matrices are supported: " +
                              std::string(mm_typecode_to_str(matcode)));
    }

    if (mm_is_complex(matcode)) {
        fail_with_file(f, "complex Matrix Market matrices are not supported");
    }

    int rows = 0;
    int cols = 0;
    int entries = 0;
    if (mm_read_mtx_crd_size(f, &rows, &cols, &entries) != 0) {
        fail_with_file(f, "could not read Matrix Market coordinate size");
    }

    const bool is_symmetric = mm_is_symmetric(matcode) || mm_is_skew(matcode);
    const bool is_pattern = mm_is_pattern(matcode);
    const bool is_integer = mm_is_integer(matcode);

    std::cout << "Loading " << path << ": " << rows << "x" << cols << ", "
              << entries << " entries";
    if (is_symmetric) {
        std::cout << " (symmetric)";
    }
    if (is_pattern) {
        std::cout << " (pattern)";
    }
    std::cout << '\n';

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(is_symmetric ? static_cast<size_t>(entries) * 2
                                  : static_cast<size_t>(entries));

    for (int idx = 0; idx < entries; ++idx) {
        int row = 0;
        int col = 0;
        double value = 1.0;

        if (is_pattern) {
            if (fscanf(f, "%d %d", &row, &col) != 2) {
                fail_with_file(f, "premature EOF while reading pattern entry");
            }
        }
        else if (is_integer) {
            int int_value = 0;
            if (fscanf(f, "%d %d %d", &row, &col, &int_value) != 3) {
                fail_with_file(f, "premature EOF while reading integer entry");
            }
            value = static_cast<double>(int_value);
        }
        else {
            if (fscanf(f, "%d %d %lg", &row, &col, &value) != 3) {
                fail_with_file(f, "premature EOF while reading real entry");
            }
        }

        --row;
        --col;

        triplets.emplace_back(row, col, value);

        if (is_symmetric && row != col) {
            const double mirrored_value = mm_is_skew(matcode) ? -value : value;
            triplets.emplace_back(col, row, mirrored_value);
        }
    }

    fclose(f);

    Eigen::SparseMatrix<double> matrix(rows, cols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();

    std::cout << "Loaded: " << matrix.rows() << "x" << matrix.cols() << ", "
              << matrix.nonZeros() << " nonzeros after expansion\n";

    return matrix;
}
