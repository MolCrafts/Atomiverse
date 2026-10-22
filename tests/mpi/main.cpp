/// @file tests/mpi/main.cpp
/// @brief Entry point of atv_tests_mpi: MPI init around the gtest run.
#include <gtest/gtest.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}
