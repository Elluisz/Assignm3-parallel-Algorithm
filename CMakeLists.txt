cmake_minimum_required(VERSION 3.25)
project(sparse_mult)

# Use GCC 15 explicitly
set(CMAKE_C_COMPILER /opt/homebrew/Cellar/gcc/15.1.0/bin/gcc-15)
set(CMAKE_CXX_COMPILER /opt/homebrew/Cellar/gcc/15.1.0/bin/g++-15)

# Enable C++17 and OpenMP
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add include directory
include_directories(include)

# Source files
set(SOURCES
    src/main.cpp
    src/csr_operations.cpp
    src/mtx_loader.cpp
)

# Add executable
add_executable(sparse_mult ${SOURCES})

# Link OpenMP if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(sparse_mult PUBLIC OpenMP::OpenMP_CXX)
endif()
