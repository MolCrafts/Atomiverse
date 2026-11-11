/// @file app/atomiverse/main.cpp
/// @brief `atomiverse` driver: native `-in` script when built with
///        ATV_BUILD_MD, MPI Python host when built with ATV_BUILD_LAUNCHER.

#include "cli_args.h"

#include <cstdio>
#include <exception>
#include <iostream>
#include <stdexcept>

#ifdef ATV_HAS_MD
#include "input_script.h"
#endif

#ifdef ATV_HAS_LAUNCHER
#include <mpi.h>

#include "atomiverse/cpu/task/task_protocol.h"
#include "python_host.h"
#endif

int main(int argc, char** argv) {
    int exit_code = 0;
    try {
        const ATV::app::CliArgs args = ATV::app::parse_cli_args(argc, argv);
        if (args.help) {
            std::fputs(ATV::app::cli_help(), stdout);
            return 0;
        }

        if (!args.in_path.empty()) {
#ifdef ATV_HAS_MD
            return ATV::app::run_input_script(args.in_path, args.log_path);
#else
            throw std::runtime_error(
                "native input scripts run classical MD and need "
                "ATV_BUILD_MD=ON. Rebuild with -DATV_BUILD_MD=ON.");
#endif
        }

        if (!args.py_script.empty()) {
#ifdef ATV_HAS_LAUNCHER
            int provided = 0;
            MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            try {
                if (rank == 0) {
                    ATV::cpu::task::TaskDispatcher dispatcher(MPI_COMM_WORLD);
                    try {
                        ATV::app::PythonHost host(args.py_script, args.py_argc,
                                                  args.py_argv);
                        exit_code = host.run();
                    } catch (...) {
                        dispatcher.shutdown();
                        throw;
                    }
                    dispatcher.shutdown();
                } else {
                    ATV::cpu::task::WorkerLoop worker(MPI_COMM_WORLD);
                    worker.run();
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[atomiverse rank %d] %s\n", rank,
                             e.what());
                exit_code = 1;
            }
            MPI_Finalize();
            return exit_code;
#else
            throw std::runtime_error(
                "Python scripts require ATV_BUILD_LAUNCHER=ON "
                "(MPI + embedded CPython). For a native input use "
                "`atomiverse -in in.lj`. Or run the script with "
                "`python -m atomiverse script.py`.");
#endif
        }

        std::fputs(ATV::app::cli_help(), stderr);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "atomiverse: %s\n", e.what());
        return 1;
    }
}
