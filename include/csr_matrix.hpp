#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include <vector>

// ==================================================
// CSRMatrix struct for storing sparse matrices
// in Compressed Sparse Row format:
// - values: non-zero values
// - col_indices: corresponding column indices
// - row_ptr: pointers to row starts in values/col_indices
// ==================================================
struct CSRMatrix {
    std::vector<double> values;       // Non-zero entries
    std::vector<int> col_indices;     // Column indices of values
    std::vector<int> row_ptr;         // Row start pointers (size = rows + 1)
    int rows = 0;                     // Number of rows
    int cols = 0;                     // Number of columns
};

// ==================================================
// Function Prototypes
// ==================================================

// Multiply two CSR matrices
CSRMatrix multiply_csr(const CSRMatrix& A, const CSRMatrix& B);

// Print the matrix in human-readable sparse format
void print_matrix(const CSRMatrix& mat);

#endif // CSR_MATRIX_HPP
