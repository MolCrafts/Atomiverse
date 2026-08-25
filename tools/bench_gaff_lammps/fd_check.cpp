/// @file tools/bench_gaff_lammps/fd_check.cpp
/// @brief Finite-difference force check for the fused lj/cut/coul/long +
///        PPPM composition (the GAFF production pair path).
///
/// Two opposite unit charges in a cubic box; compares the analytic force on
/// atom 0 from the composed evaluators against -dE/dx by central differences.
/// Run serially: mpirun -np 1 fd_check
#include <mpi.h>

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/pppm.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_coul_long.h"

using namespace ATV;

namespace {

cpu::Frame make_frame(double x0) {
    cpu::Frame f(2);
    f.box.set_cell([] {
        Mat3 c;
        c(0, 0) = 20.0;
        c(1, 1) = 20.0;
        c(2, 2) = 20.0;
        return c;
    }());
    StorageReal* x = f.span(kAtom_x);
    StorageReal* y = f.span(kAtom_y);
    StorageReal* z = f.span(kAtom_z);
    StorageReal* q = f.span(kAtom_charge);
    StorageReal* m = f.span(kAtom_mass);
    int* t = f.span(kAtom_type);
    x[0] = x0;
    y[0] = 5.0;
    z[0] = 5.0;
    x[1] = 8.0;
    y[1] = 5.0;
    z[1] = 5.0;
    q[0] = 1.0;
    q[1] = -1.0;
    m[0] = m[1] = 1.0;
    t[0] = t[1] = 0;
    return f;
}

double energy_at(double x0, double eps) {
    const double alpha = cpu::pppm_alpha_from_accuracy(1.0e-4, 9.0);
    cpu::Frame f = make_frame(x0);
    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{9.0, 1.0});
    eval.set_evaluator(cpu::PairLJCutCoulLong{{{0, 0, eps, 1.0}}, alpha, 9.0}
                           .set_special_bonds(1, 1, 1, 1, 1, 1));
    eval.set_evaluator(cpu::KSpacePPPM::from_accuracy(1.0e-4, 9.0, 2.0));
    eval.build();
    eval.run(f);
    return f.meta.get(kEnergy);
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int failures = 0;

    for (double eps : {0.0, 0.01}) {
        // Analytic force via one evaluation.
        const double alpha = cpu::pppm_alpha_from_accuracy(1.0e-4, 9.0);
        cpu::Frame f = make_frame(5.0);
        cpu::Evaluate eval;
        eval.set_neighbor(cpu::md::NeighborList<>{9.0, 1.0});
        eval.set_evaluator(
            cpu::PairLJCutCoulLong{{{0, 0, eps, 1.0}}, alpha, 9.0}
                .set_special_bonds(1, 1, 1, 1, 1, 1));
        eval.set_evaluator(cpu::KSpacePPPM::from_accuracy(1.0e-4, 9.0, 2.0));
        eval.build();
        eval.run(f);
        const AccReal* fptr = f.span(kAtom_fx);
        const double fx = static_cast<double>(fptr[0]);

        const double h = 5.0e-4;
        const double ep = energy_at(5.0 + h, eps);
        const double em = energy_at(5.0 - h, eps);
        const double fd = -(ep - em) / (2.0 * h);

        const double rel = std::fabs(fx - fd) / std::max(1e-12, std::fabs(fd));
        std::printf("eps=%.3f  fx=%+.8f  fd=%+.8f  rel_err=%.3e  %s\n", eps, fx,
                    fd, rel, rel < 5e-3 ? "OK" : "FAIL");
        if (!(rel < 5e-3))
            ++failures;
    }

    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
