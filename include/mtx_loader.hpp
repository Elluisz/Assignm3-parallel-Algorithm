#ifndef MTX_LOADER_HPP
#define MTX_LOADER_HPP

#include "csr_matrix.hpp"
#include <string>

CSRMatrix load_matrix_market(const std::string& filename);

#endif // MTX_LOADER_HPP
