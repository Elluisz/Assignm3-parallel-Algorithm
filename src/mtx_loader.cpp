#include "mtx_loader.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <iostream>
#include <algorithm>
#include <unordered_map>

CSRMatrix load_matrix_market(const std::string& filename) {
    std::ifstream file(filename);
    if (!file)
        throw std::runtime_error("Failed to open file: " + filename);

    std::string line;
    // Skip comments
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Read dimensions and non-zeros
    std::istringstream header(line);
    int rows, cols, nnz;
    header >> rows >> cols >> nnz;

    std::vector<std::tuple<int, int, double>> entries;

    // Read triplets
    int r, c;
    double v;
    while (file >> r >> c >> v) {
        entries.emplace_back(r - 1, c - 1, v); // Convert to 0-based index
        if (r != c) // For symmetric matrix, mirror the entry
            entries.emplace_back(c - 1, r - 1, v);
    }

    // Sort entries by row then column
    std::sort(entries.begin(), entries.end());

    // Convert to CSR
    CSRMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.row_ptr.resize(rows + 1, 0);

    for (const auto& [row, col, val] : entries) {
        m.row_ptr[row + 1]++;
        m.values.push_back(val);
        m.col_indices.push_back(col);
    }

    for (int i = 0; i < rows; ++i)
        m.row_ptr[i + 1] += m.row_ptr[i];

    return m;
}
