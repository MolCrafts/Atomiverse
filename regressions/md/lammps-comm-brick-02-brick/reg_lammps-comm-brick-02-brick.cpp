/// @file reg_lammps-comm-brick-02-brick.cpp
/// @brief Pins `cpu::Comm*` brick goldens: n_ghost 0/8 and reverse sum fx.
///
/// Never constructs CommBrick; never MD.run. Obtains Comm* from Runtime.
/// Literature: Plimpton JCP 117 (1995); Thompson CPC 271 (2022).
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/comm.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/frame/frame.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef ATV_HAS_MPI
#include <mpi.h>
#include <unistd.h>
#endif

using namespace ATV;

namespace {

cpu::Frame make_8_atom_cube() {
    cpu::Frame frame(8);
    int idx = 0;
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy)
            for (int iz = 0; iz < 2; ++iz) {
                frame.span(kAtom_x)[idx] =
                    (static_cast<double>(ix) + 0.5) * 10.0;
                frame.span(kAtom_y)[idx] =
                    (static_cast<double>(iy) + 0.5) * 10.0;
                frame.span(kAtom_z)[idx] =
                    (static_cast<double>(iz) + 0.5) * 10.0;
                frame.span(kAtom_mass)[idx] = 1.0;
                ++idx;
            }
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

int run_serial() {
    auto frame = make_8_atom_cube();
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    cpu::Comm* comm = backend.context().md.comm;
    if (comm == nullptr) {
        std::fprintf(stderr, "FAIL: ctx.md.comm is null\n");
        return EXIT_FAILURE;
    }
    comm->set_ghost_cutoff(6.0);
    cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
    comm->setup(lctx);
    comm->borders(lctx);
    const int n_ghost = backend.context().md.dd->n_ghost;
    backend.shutdown();
    if (n_ghost != 0) {
        std::fprintf(stderr, "FAIL: serial n_ghost=%d want 0\n", n_ghost);
        return EXIT_FAILURE;
    }
    std::printf("n_ghost_n1 %d\n", n_ghost);
    return EXIT_SUCCESS;
}

#ifdef ATV_HAS_MPI

int run_child(const char* out_path) {
    int provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    ::setenv("ATV_FORCE_DIMS", "2,1,1", 1);
    auto frame = make_8_atom_cube();
    cpu::Runtime backend;
    backend.init();
    backend.set_frame(frame);
    cpu::Comm* comm = backend.context().md.comm;
    auto* dd = backend.context().md.dd;
    int rc = EXIT_SUCCESS;
    int n_ghost = -1;
    double sum_fx = 0.0;
    int n_owners = 0;
    if (comm == nullptr || dd == nullptr) {
        rc = EXIT_FAILURE;
    } else {
        comm->set_ghost_cutoff(6.0);
        cpu::LaunchContext lctx{backend.context(), frame, 0, 0};
        comm->setup(lctx);
        comm->borders(lctx);
        n_ghost = dd->n_ghost;
        for (int i = 0; i < dd->n_local; ++i)
            frame.span(kAtom_fx)[i] = 0.0;
        for (int g = 0; g < dd->n_ghost; ++g)
            frame.span(kAtom_fx)[dd->n_local + g] = 1.0;
        comm->reverse_comm(lctx);
        for (int i = 0; i < dd->n_local; ++i)
            sum_fx += static_cast<double>(frame.span(kAtom_fx)[i]);
        comm->exchange(lctx);
        int n_local = dd->n_local;
        MPI_Allreduce(&n_local, &n_owners, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (n_ghost != 8 || std::fabs(sum_fx - 8.0) > 1e-12 || n_owners != 8)
            rc = EXIT_FAILURE;
    }
    backend.shutdown();
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::FILE* out = std::fopen(out_path, "w");
        if (!out)
            rc = EXIT_FAILURE;
        else {
            std::fprintf(out, "%d %.17e %d\n", n_ghost, sum_fx, n_owners);
            std::fclose(out);
        }
    }
    MPI_Finalize();
    return rc;
}

#ifndef ATV_MPIEXEC
#error "ATV_MPIEXEC must be set to the configured MPIEXEC_EXECUTABLE"
#endif

bool run_n2(const std::string& self, int& n_ghost, double& sum_fx,
            int& n_owners) {
    const std::string path = "/tmp/atv-lammps-comm-brick-02-"
                             + std::to_string(static_cast<long>(::getpid()))
                             + ".txt";
    const std::string cmd = std::string(ATV_MPIEXEC) + " -n 2 \"" + self
                            + "\" --child \"" + path + "\"";
    if (std::system(cmd.c_str()) != 0)
        return false;
    std::FILE* in = std::fopen(path.c_str(), "r");
    if (!in)
        return false;
    const int got = std::fscanf(in, "%d %lf %d", &n_ghost, &sum_fx, &n_owners);
    std::fclose(in);
    std::remove(path.c_str());
    return got == 3;
}

#endif // ATV_HAS_MPI

} // namespace

int main(int argc, char** argv) {
#ifdef ATV_HAS_MPI
    if (argc >= 3 && std::strcmp(argv[1], "--child") == 0)
        return run_child(argv[2]);
#endif
    if (run_serial() != EXIT_SUCCESS)
        return EXIT_FAILURE;

#ifdef ATV_HAS_MPI
    char self_buf[4096];
    const ssize_t n =
        ::readlink("/proc/self/exe", self_buf, sizeof(self_buf) - 1);
    if (n <= 0) {
        std::fprintf(stderr, "FAIL: cannot resolve /proc/self/exe\n");
        return EXIT_FAILURE;
    }
    self_buf[n] = '\0';
    int n_ghost = -1, n_owners = -1;
    double sum_fx = 0.0;
    if (!run_n2(self_buf, n_ghost, sum_fx, n_owners)) {
        std::fprintf(stderr, "FAIL: n2 child\n");
        return EXIT_FAILURE;
    }
    if (n_ghost != 8 || std::fabs(sum_fx - 8.0) > 1e-12 || n_owners != 8) {
        std::fprintf(stderr, "FAIL: n_ghost=%d sum_fx=%.3e owners=%d\n",
                     n_ghost, sum_fx, n_owners);
        return EXIT_FAILURE;
    }
    std::printf("n_ghost_n2 %d\n", n_ghost);
    std::printf("sum_fx %.1f\n", sum_fx);
#endif
    return EXIT_SUCCESS;
}
