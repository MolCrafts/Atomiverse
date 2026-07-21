/// @file app/atomiverse/main.cpp
/// @brief The `atomiverse` MPI launcher: every rank starts native; rank 0
///        embeds CPython and runs the user script; ranks 1..N-1 block in
///        the task-protocol WorkerLoop until SHUTDOWN.
///
/// This main owns the MPI_Init_thread(MPI_THREAD_MULTIPLE)/MPI_Finalize
/// bracket — the library itself only adopts an initialized MPI (the
/// Backend probes MPI_Initialized). SHUTDOWN is broadcast on rank 0 on
/// EVERY path out of the Python host — normal return, script exception,
/// and C++ throw alike — so workers are always released; the script's
/// exit code is propagated to mpirun.

#include <mpi.h>

#include <cstdio>
#include <exception>

#include "atomiverse/cpu/task/task_protocol.h"
#include "launcher_args.h"
#include "python_host.h"

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int exit_code = 0;
    try {
        // Deterministically identical argv on every rank: a usage error
        // throws everywhere, and every rank reaches MPI_Finalize below.
        const ATV::app::LauncherArgs args(argc, argv);

        if (rank == 0) {
            ATV::cpu::task::TaskDispatcher dispatcher(MPI_COMM_WORLD);
            // SHUTDOWN must fire on EVERY path out of the host — a C++
            // throw from PythonHost would otherwise leave the workers
            // blocked in their command loop forever.
            try {
                ATV::app::PythonHost host(args.script(), argc - 1, argv + 1);
                exit_code = host.run(); // script runs; drivers destruct
            } catch (...) {
                dispatcher.shutdown();
                throw;
            }
            dispatcher.shutdown(); // worker release, normal path
        } else {
            ATV::cpu::task::WorkerLoop worker(MPI_COMM_WORLD);
            worker.run(); // blocks until SHUTDOWN
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[atomiverse rank %d] %s\n", rank, e.what());
        exit_code = 1;
    }

    MPI_Finalize();
    return exit_code;
}
