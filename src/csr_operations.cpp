#include "csr_operations.hpp"
#include <iostream>
#include <unordered_map>
#include <random>
#include <omp.h>

// ==================================================
// Multiply two CSR matrices: C = A * B
// - Only non-zero values are computed
// - Output is in CSR format
// ==================================================

CSRMatrix multiply_csr(const CSRMatrix& A, const CSRMatrix& B) {
    CSRMatrix C;
    C.rows = A.rows;
    C.cols = B.cols;

    std::vector<std::vector<int>> row_cols(A.rows);
    std::vector<std::vector<double>> row_vals(A.rows);

   #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < A.rows; ++i) {
        std::vector<double> dense_row(B.cols, 0.0);
        std::vector<bool> filled(B.cols, false);

        for (int aj = A.row_ptr[i]; aj < A.row_ptr[i + 1]; ++aj) {
            int a_col = A.col_indices[aj];
            double a_val = A.values[aj];

            for (int bj = B.row_ptr[a_col]; bj < B.row_ptr[a_col + 1]; ++bj) {
                int b_col = B.col_indices[bj];
                double b_val = B.values[bj];
                if (!filled[b_col]) {
                    filled[b_col] = true;
                    row_cols[i].push_back(b_col);
                }
                dense_row[b_col] += a_val * b_val;
            }
        }

        for (int col : row_cols[i]) {
            row_vals[i].push_back(dense_row[col]);
        }
    }

    C.row_ptr.push_back(0);
    for (int i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < row_cols[i].size(); ++j) {
            C.col_indices.push_back(row_cols[i][j]);
            C.values.push_back(row_vals[i][j]);
        }
        C.row_ptr.push_back(static_cast<int>(C.values.size()));
    }

    return C;
}



// ==================================================
// Print matrix contents in CSR form
// Useful for debugging or inspecting sparse structure
// ==================================================
void print_matrix(const CSRMatrix& mat) {
    std::cout << "CSR Matrix Representation:\n";
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = mat.row_ptr[i]; j < mat.row_ptr[i + 1]; ++j) {
            std::cout << "Row " << i << " Col " << mat.col_indices[j]
                      << " Val " << mat.values[j] << "\n";
        }
    }
}
