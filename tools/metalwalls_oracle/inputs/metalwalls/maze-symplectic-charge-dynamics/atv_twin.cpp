/// @file tools/metalwalls_oracle/inputs/metalwalls/maze-symplectic-charge-dynamics/atv_twin.cpp
/// @brief Atomiverse side of the MetalWalls twin for
///        regressions/md/maze-symplectic-charge-dynamics.
///
/// Same 10-atom capacitor as the regression, but with the cell and pair
/// cutoff on the command line so the *dilute* (isolated-cluster) limit can be
/// compared against MetalWalls' Ewald result at growing box size — the one
/// configuration in which the two codes evaluate the same electrostatic
/// model.  Prints per-site electrode charges [e], plate totals [e], the
/// pair energy [eV] and the ion forces [eV/A].
///
/// Build (out-of-library consumer, flags copied from the regression target):
///   c++ -O3 -std=gnu++20 -fopenmp -I<repo>/include ... atv_twin.cpp \
///       build/libatomiverse.a ... -o atv_twin
/// Run: ./atv_twin <Lx> <Ly> <Lz> <cutoff>      (all in Angstrom)

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/dynamics/const_pot_maze.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pair_gaussian_coul.h"

using namespace ATV;

int main(int argc, char** argv) {
    const double Lx = argc > 1 ? std::atof(argv[1]) : 6.0;
    const double Ly = argc > 2 ? std::atof(argv[2]) : 6.0;
    const double Lz = argc > 3 ? std::atof(argv[3]) : 20.0;
    const double cut = argc > 4 ? std::atof(argv[4]) : 12.0;
    const double eta = 2.0; // [1/A]

    cpu::Frame f(10);
    int a = 0;
    for (int wall = 0; wall < 2; ++wall) {
        const double z = (wall == 0) ? 4.0 : 16.0;
        const int typ = (wall == 0) ? 1 : 2;
        for (int iy = 0; iy < 2; ++iy)
            for (int ix = 0; ix < 2; ++ix) {
                f.span(kAtom_x)[a] = (ix + 0.5) * 3.0;
                f.span(kAtom_y)[a] = (iy + 0.5) * 3.0;
                f.span(kAtom_z)[a] = z;
                f.span(kAtom_type)[a] = typ;
                f.span(kAtom_charge)[a] = 0.0;
                f.span(kAtom_mass)[a] = 196.97;
                f.span(kAtom_tag)[a] = a + 1;
                ++a;
            }
    }
    const double ix2[2] = {2.1, 4.2};
    const double iz2[2] = {8.5, 11.5};
    const double iq2[2] = {0.5, -0.5};
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_x)[a] = ix2[i];
        f.span(kAtom_y)[a] = ix2[i];
        f.span(kAtom_z)[a] = iz2[i];
        f.span(kAtom_type)[a] = 3;
        f.span(kAtom_charge)[a] = iq2[i];
        f.span(kAtom_mass)[a] = 40.0;
        f.span(kAtom_tag)[a] = a + 1;
        ++a;
    }
    f.box = cpu::Box::orthorhombic(Lx, Ly, Lz);

    cpu::md::ConstPotMaZe maze(eta);
    maze.add_electrode(1, 0.0);
    maze.add_electrode(2, 1.0);
    maze.symplectic = true;

    std::vector<double> etas = {0.0, eta, eta, 0.0};
    cpu::PairGaussianCoul pg(cut, etas, /*include_self=*/true);

    cpu::Context ctx;
    maze.build({ctx, f});
    pg.build({ctx, f});
    for (int i = 0; i < f.nlocal(); ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0.0;
    f.meta.get(kEnergy) = 0.0;
    maze.launch({ctx, f, 0, 0});
    pg.launch({ctx, f, 0, 0});

    std::printf("# ATV twin  box %.4f %.4f %.4f A  cutoff %.4f A\n", Lx, Ly, Lz,
                cut);
    double qbot = 0.0, qtop = 0.0;
    // Print in MetalWalls' electrode ordering: E1 (bottom) rows 0-3, then E2.
    for (int i = 0; i < 8; ++i) {
        const double q = f.span(kAtom_charge)[i];
        (i < 4 ? qbot : qtop) += q;
        std::printf("q[%d] = %+.15E\n", i, q);
    }
    std::printf("Q_bot   = %+.15E e\n", qbot);
    std::printf("Q_top   = %+.15E e\n", qtop);
    std::printf("sum_Q   = %+.3E e\n", qbot + qtop);
    std::printf("resid   = %.3E V\n", maze.constraint_residual());
    std::printf("E_pair  = %+.15E eV\n", f.meta.get(kEnergy));
    for (int i = 8; i < 10; ++i)
        std::printf("F[%d]    = %+.12E %+.12E %+.12E eV/A\n", i,
                    f.span(kAtom_fx)[i], f.span(kAtom_fy)[i],
                    f.span(kAtom_fz)[i]);
    return 0;
}
