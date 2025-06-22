#include "csr_matrix.hpp"
#include "csr_operations.hpp"
#include <iostream>
#include <chrono>
#include <omp.h>
#include <string>
#include <fstream> 

// forward-declare loader
CSRMatrix load_matrix_market(const std::string& path);

int main() {
    std::string file = "../1138_bus.mtx";  // relative path from build/
    std::cout << "Loading matrix from: " << file << "\n";

    CSRMatrix A = load_matrix_market(file);
    CSRMatrix B = A; // symmetric matrix

    std::cout << "Dimensions: " << A.rows << " × " << A.cols
              << ", nnz=" << A.values.size() << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    CSRMatrix C = multiply_csr(A, B);
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Multiply time: " << elapsed << " s\n";
    std::cout << "Result nnz: " << C.values.size() << "\n";

    // At the end of main()
    std::ofstream fout("timing_results_sequential.csv", std::ios::app); // append mode
    fout << "sequential," << elapsed << "\n"; 
    fout.close();

    return 0;
}
