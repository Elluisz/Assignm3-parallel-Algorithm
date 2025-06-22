#include "csr_matrix.hpp"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>


// Forward-declare loader
CSRMatrix load_matrix_market(const std::string& path);

// ————— Main MPI program —————
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    CSRMatrix A, B;

    double io_start = MPI_Wtime();
    if (rank == 0) {
        std::string file = "1138_bus.mtx";
        std::cout << "[Rank 0] Loading matrix from: " << file << "\n";
        A = load_matrix_market(file);
        B = A;  // Assume symmetric
    }
    double io_end = MPI_Wtime();

    // Broadcast metadata
    std::vector<int> meta(3);
    if (rank == 0) meta = {A.rows, A.cols, (int)A.values.size()};
    MPI_Bcast(meta.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> buf_vals(meta[2]);
    std::vector<int> buf_cols(meta[2]), buf_ptr(meta[0] + 1);
    if (rank == 0) {
        buf_vals = A.values;
        buf_cols = A.col_indices;
        buf_ptr = A.row_ptr;
    }
    MPI_Bcast(buf_vals.data(), meta[2], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(buf_cols.data(), meta[2], MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(buf_ptr.data(), meta[0] + 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        A.rows = B.rows = meta[0];
        A.cols = B.cols = meta[1];
        A.values = B.values = buf_vals;
        A.col_indices = B.col_indices = buf_cols;
        A.row_ptr = B.row_ptr = buf_ptr;
    }

    // Partition rows
    int rows_per = A.rows / size;
    int extra = A.rows % size;
    int start = rank * rows_per + std::min(rank, extra);
    int end = start + rows_per + (rank < extra);

    CSRMatrix localA;
    localA.rows = end - start;
    localA.cols = A.cols;
    localA.row_ptr.resize(localA.rows + 1);
    for (int i = 0; i <= localA.rows; ++i)
        localA.row_ptr[i] = A.row_ptr[start + i] - A.row_ptr[start];
    for (int i = A.row_ptr[start]; i < A.row_ptr[end]; ++i) {
        localA.values.push_back(A.values[i]);
        localA.col_indices.push_back(A.col_indices[i]);
    }

    // — Multiplication timing —
    double mul_start = MPI_Wtime();
    CSRMatrix localC = multiply_csr(localA, B);
    double mul_end = MPI_Wtime();
    double local_mul = mul_end - mul_start;

    double max_mul;
    MPI_Reduce(&local_mul, &max_mul, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // — Gathering stage —
    double gather_start = MPI_Wtime();

    std::vector<int> metaC = {localC.rows, localC.cols, (int)localC.values.size()};
    std::vector<int> all_meta(3 * size);
    MPI_Gather(metaC.data(), 3, MPI_INT, all_meta.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> nnz(size), rows(size), displs(size), ptr_counts(size), ptr_displs(size);
    int total_rows = 0, total_nnz = 0;
    if (rank == 0) {
        displs[0] = 0;
        ptr_displs[0] = 0;
        for (int i = 0; i < size; ++i) {
            rows[i] = all_meta[3 * i];
            nnz[i] = all_meta[3 * i + 2];
            ptr_counts[i] = rows[i];  // NOTE: exclude +1 for first rank
            if (i > 0) {
                displs[i] = displs[i - 1] + nnz[i - 1];
                ptr_displs[i] = ptr_displs[i - 1] + ptr_counts[i - 1];
            }
            total_rows += rows[i];
            total_nnz += nnz[i];
        }
    }

    std::vector<double> all_vals(total_nnz);
    std::vector<int> all_cols(total_nnz), all_ptr(total_rows + 1);

    // Gather values and column indices
    MPI_Gatherv(localC.values.data(), localC.values.size(), MPI_DOUBLE,
                all_vals.data(), nnz.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localC.col_indices.data(), localC.col_indices.size(), MPI_INT,
                all_cols.data(), nnz.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    // Gather row_ptrs excluding first "0" from each rank (except rank 0)
    std::vector<int> local_ptr_delta(localC.row_ptr.begin() + 1, localC.row_ptr.end());
    MPI_Gatherv(local_ptr_delta.data(), local_ptr_delta.size(), MPI_INT,
                all_ptr.data() + 1, ptr_counts.data(), ptr_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        all_ptr[0] = 0; // first global row starts at 0
        for (int i = 1; i < size; ++i) {
            int offset = displs[i];     // cumulative nnz
            int base = ptr_displs[i];   // index where this rank's row_ptr starts (excluding 0)
            for (int j = 0; j < ptr_counts[i]; ++j) {
                all_ptr[base + j + 1] += offset;
            }
        }
    }

    double gather_end = MPI_Wtime();
    double gather_time = gather_end - gather_start;

    // — Final reporting —
    if (rank == 0) {
        CSRMatrix C;
        C.rows = total_rows;
        C.cols = B.cols;
        C.values = all_vals;
        C.col_indices = all_cols;
        C.row_ptr = all_ptr;

        std::cout << "\n=== Timing Summary ===\n";
        std::cout << "Load time      : " << (io_end - io_start) << " s\n";
        std::cout << "Multiply time  : " << max_mul << " s (worst rank)\n";
        std::cout << "Gather time    : " << gather_time << " s\n";

        // Verify result
        CSRMatrix C_seq = multiply_csr(A, B);
        auto almost_equal = [](double a, double b, double eps = 1e-6) {
            return std::abs(a - b) < eps;
        };

        bool match = true;

        if (C.row_ptr != C_seq.row_ptr) {
            std::cerr << "Row pointers differ!\n";
            size_t min_sz = std::min(C.row_ptr.size(), C_seq.row_ptr.size());
            for (size_t i = 0; i < min_sz; ++i) {
                if (C.row_ptr[i] != C_seq.row_ptr[i]) {
                    std::cerr << "  At row_ptr[" << i << "]: MPI=" 
                              << C.row_ptr[i] << ", SEQ=" << C_seq.row_ptr[i] << "\n";
                    match = false;
                }
            }
        }

        if (C.col_indices.size() != C_seq.col_indices.size() ||
            C.values.size() != C_seq.values.size()) {
            std::cerr << "Sizes differ: MPI nnz=" << C.values.size()
                      << ", SEQ nnz=" << C_seq.values.size() << "\n";
            match = false;
        } else {
            for (size_t k = 0; k < C.values.size(); ++k) {
                if (C.col_indices[k] != C_seq.col_indices[k]) {
                    std::cerr << "Col index mismatch at k=" << k 
                              << ": MPI=" << C.col_indices[k] 
                              << ", SEQ=" << C_seq.col_indices[k] << "\n";
                    match = false;
                }
                if (!almost_equal(C.values[k], C_seq.values[k])) {
                    std::cerr << "Value mismatch at k=" << k 
                              << ": MPI=" << C.values[k] 
                              << ", SEQ=" << C_seq.values[k] << "\n";
                    match = false;
                }
            }
        }

        if (match)
            std::cout << "Verification PASS: MPI result matches sequential\n";
        else
            std::cout << "Verification FAIL: differences detected\n";
    }

    if (rank == 0) {
        std::ofstream fout("timing_results_mpi.csv", std::ios::app); // append mode
        fout << "mpi," << max_mul << "\n";
        fout.close();
    }



    MPI_Finalize();
    return 0;
}
