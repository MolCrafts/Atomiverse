/// @file cpu/task/launcher_gate.cpp
/// @brief PID-qualified launcher-hosting probe (contract in launcher_gate.h).

#include "atomiverse/cpu/task/launcher_gate.h"

// POSIX getpid: the launcher process model is Linux-only by charter.
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace task {

namespace {
// fprintf, not ATV_WARN: this ctor is noexcept and spdlog may allocate.
std::once_flag g_inconsistent_host_warning;
} // namespace

LauncherGate::LauncherGate() noexcept {
    const char* marker = std::getenv("ATV_LAUNCHER");
    if (marker == nullptr)
        return; // plain `python atv.py` — one getenv, nothing else.

    long pid = -1;
    try {
        pid = std::stol(marker);
    } catch (...) {
        return; // malformed marker: not ours, silently inactive
    }
    if (pid != static_cast<long>(::getpid()))
        return; // inherited by a child process — silently inactive

#ifdef ATV_HAS_MPI
    int inited = 0;
    MPI_Initialized(&inited);
    if (!inited) {
        std::call_once(g_inconsistent_host_warning, [] {
            std::fprintf(stderr,
                         "[atomiverse] warning: ATV_LAUNCHER marker matches "
                         "this process but MPI is not initialized — "
                         "dispatch disabled, running serially\n");
        });
        return;
    }
    int world = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    active_ = world > 1;
#else
    // Non-MPI build: the launcher cannot exist; own-PID marker is the
    // same inconsistent-host shape.
    std::call_once(g_inconsistent_host_warning, [] {
        std::fprintf(stderr,
                     "[atomiverse] warning: ATV_LAUNCHER marker set but "
                     "this build has no MPI — dispatch disabled\n");
    });
#endif
}

} // namespace task
} // namespace cpu
} // namespace ATV
