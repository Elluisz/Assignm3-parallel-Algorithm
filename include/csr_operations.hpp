#ifndef CSR_OPERATIONS_HPP
#define CSR_OPERATIONS_HPP

#include "csr_matrix.hpp"

// Multiplies two CSR matrices
CSRMatrix multiply_csr(const CSRMatrix& A, const CSRMatrix& B);

// Prints a CSR matrix
void print_matrix(const CSRMatrix& mat);

#endif // CSR_OPERATIONS_HPP
