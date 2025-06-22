# Sparse Matrix Multiplication (CSR) with OpenMP and MPI

This project implements efficient sparse matrix multiplication using the **Compressed Sparse Row (CSR)** format. It includes two parallel variants:

- **OpenMP** (Shared-memory parallelism)
- **MPI** (Distributed-memory parallelism)

A real-world matrix (`1138_bus.mtx`) is used to benchmark performance across different execution models.

---

## Project Structure

```
├── CMakeLists.txt
├── include/
│   ├── csr_matrix.hpp
│   ├── csr_operations.hpp
│   └── mtx_loader.hpp
├── src/
│   ├── main.cpp               # OpenMP + Sequential variant
│   ├── mpi_sparse_mult.cpp   # MPI variant
│   ├── csr_operations.cpp
│   └── mtx_loader.cpp
└── 1138_bus.mtx              # Matrix Market input file
```

---

## 🛠️ Prerequisites

- **CMake ≥ 3.25**
- **GCC with OpenMP support**
- **MPI compiler (e.g., `mpic++` or `g++` with `-lmpi`)**
- **Matrix Market format input file**

---

## Building and Running the OpenMP Version

1. **Configure and build using CMake:**

   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

2. **Run the executable:**

   ```bash
   ./sparse_mult
   ```

This executes the OpenMP-enabled multiplication on the matrix `1138_bus.mtx`, outputting performance metrics to `timing_results_sequential.csv` you can check the inside the build file.

If you want to make sequential testing without OpenMP enabled just comment this line of code (21) on csr_operaiont.cpp `#pragma omp parallel for schedule(dynamic)`

---

## Building and Running the MPI Version

1. **Compile the MPI program using a general command:**

   ```bash
   g++ -std=c++17 -fopenmp -Iinclude \
       src/mpi_sparse_mult.cpp src/csr_operations.cpp src/mtx_loader.cpp \
       -lmpi -o mpi_sparse_mult
   ```

   > **Note:** On macOS with Homebrew GCC, you may need to use the full path to `g++-15` and explicitly set include/library directories. For example:
   > ```bash
   > /opt/homebrew/Cellar/gcc/15.1.0/bin/g++-15 \
   >     -I/opt/homebrew/include -L/opt/homebrew/lib \
   >     -std=c++17 -fopenmp -Iinclude \
   >     src/mpi_sparse_mult.cpp src/csr_operations.cpp src/mtx_loader.cpp \
   >     -lmpi -o mpi_sparse_mult
   > ```

2. **Run the program with multiple processes:**

   ```bash
   mpirun -np 4 ./mpi_sparse_mult
   ```

The final result is verified against a sequential baseline. Timing results are appended to `timing_results_mpi.csv`.

---

## Output Files

- `timing_results_sequential.csv` – Execution times for OpenMP/sequential runs
- `timing_results_mpi.csv` – Execution times for MPI runs

These can be used for plotting performance charts or comparing speedups.

---

## Tested On

- macOS (Apple Silicon) M1 using GCC 15.1.0 via Homebrew
- CMake 3.28+
- Matrix: `1138_bus.mtx` from the SuiteSparse Matrix Collection

---

## Results

- results.py file has the python code for generating the plots and figures using the execution times saved in .csv files previously.
- You might find as well on the results folder the generated figures and plots.
