cmake_minimum_required (VERSION 2.6)
project (MfemTest)

# Set CMake modules path:
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

# add the binary tree to the search path for include files:
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/lib")
# MFEM_DIR
include_directories("$ENV{MFEM_DIR}")

# HYPRE_DIR has been set as an environment variable
include_directories("$ENV{HYPRE_DIR}/src/hypre/include")
# METIS_DIR includes:
include_directories("$ENV{METIS_DIR}/include")

# Need MPI:
#set(MPI_C_COMPILER mpic++)
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})
message(${MPI_CXX_INCLUDE_PATH})
message(${MPI_CXX_LIBRARIES})

# add HYPRE library:
add_library(HYPRE STATIC IMPORTED)
set_target_properties(HYPRE PROPERTIES
    IMPORTED_LOCATION $ENV{HYPRE_DIR}/src/hypre/lib/libHYPRE.a)

# add METIS library:
add_library(METIS STATIC IMPORTED)
set_target_properties(METIS PROPERTIES
    IMPORTED_LOCATION $ENV{METIS_DIR}/build/Linux-x86_64/libmetis/libmetis.a)

# add MFEM library:
add_library(MFEM STATIC IMPORTED)
set_target_properties(MFEM PROPERTIES
    IMPORTED_LOCATION $ENV{MFEM_DIR}/libmfem.a)

# Create executable for Poisson problem:
add_executable(poisson src/poisson.cpp)
target_link_libraries (poisson MFEM HYPRE METIS ${MPI_CXX_LIBRARIES})

# Create executable for Parallelized Poisson with Hypre AMG solver:
add_executable(poissonAMG src/poisson_amg.cpp)
target_link_libraries (poissonAMG MFEM HYPRE METIS ${MPI_CXX_LIBRARIES})
