/// @file reg_lammps-comm-brick-01-halo.cpp
/// @brief Serial pins for the MPI-free `comm::Halo` swap table.
///
/// Checks uniform periodic hop counts and lattice-image flags.
/// Algorithm reference: LAMMPS src/comm_brick.cpp at
/// a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b.
/// Copyright (2003) Sandia Corporation. GPL-2.0-only.
/// Literature: Plimpton JCP 117 (1995); Thompson CPC 271 (2022).
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/halo.h"
#include "atomiverse/cpu/frame/box.h"

#include <array>
#include <cstdio>
#include <cstdlib>

using namespace ATV;

namespace {

using DomainDecomposition = cpu::md::mpi::DomainDecomposition;

constexpr double kBoxL = 15.78; // [Angstrom]
constexpr double kShell = 6.6;  // [Angstrom]

cpu::Box cubic_box(double L) { return cpu::Box::cubic(L); }

DomainDecomposition uniform_dd(int dims_x, int coords_x, double L = kBoxL) {
    DomainDecomposition dd;
    dd.dims[0] = dims_x;
    dd.dims[1] = 1;
    dd.dims[2] = 1;
    dd.coords[0] = coords_x;
    dd.coords[1] = 0;
    dd.coords[2] = 0;

    dd.compute_sublo_subhi(cubic_box(L));
    const int lo = (coords_x - 1 + dims_x) % dims_x;
    const int hi = (coords_x + 1) % dims_x;
    dd.face_neighbors =
        std::array<int, 6>{lo, hi, coords_x, coords_x, coords_x, coords_x};
    return dd;
}

} // namespace

int main() {
    int hops[3] = {-1, -1, -1};
    const int dims[3] = {2, 4, 8};
    const int expect[3] = {1, 2, 4};
    for (int i = 0; i < 3; ++i) {
        cpu::comm::Halo table;
        table.setup(uniform_dd(dims[i], /*coords_x=*/1), cubic_box(kBoxL),
                    kShell, /*my_rank=*/1);
        hops[i] = table.max_hops(0);
        if (hops[i] != expect[i]) {
            std::fprintf(stderr, "hops mismatch: dims=%d got %d want %d\n",
                         dims[i], hops[i], expect[i]);
            return EXIT_FAILURE;
        }
    }
    std::printf("hops %d/%d/%d\n", hops[0], hops[1], hops[2]);

    {
        const int dims_x = 4;
        const double w = kBoxL / static_cast<double>(dims_x);
        cpu::comm::Halo table;
        table.setup(uniform_dd(dims_x, /*coords_x=*/1), cubic_box(kBoxL), w,
                    /*my_rank=*/1);
        const int edge_hops = table.max_hops(0);
        if (edge_hops != 2) {
            std::fprintf(stderr, "exact-slab hop=%d want 2\n", edge_hops);
            return EXIT_FAILURE;
        }
        std::printf("exact-slab %d\n", edge_hops);
    }

    {
        cpu::comm::Halo table;
        table.setup(uniform_dd(/*dims_x=*/1, /*coords_x=*/0), cubic_box(kBoxL),
                    kShell, /*my_rank=*/0);
        for (int i = 0; i < 6; ++i) {
            const auto& swap = table[i];
            const int axis = i / 2;
            const int image = i % 2 == 0 ? 1 : -1;
            double shift[3];
            swap.pbc_shift(cubic_box(kBoxL), shift);
            if (!swap.sendself || swap.is_silent() || swap.pbc[axis] != image
                || shift[axis] != image * kBoxL) {
                std::fprintf(stderr, "periodic self-image swap %d failed\n", i);
                return EXIT_FAILURE;
            }
        }
        std::printf("self-images 6\n");
    }

    {
        // The periodic chain includes this rank's images and may traverse
        // complete rings. No minimum-image restriction belongs in transport.
        cpu::comm::Halo table;
        table.setup(uniform_dd(/*dims_x=*/4, /*coords_x=*/1), cubic_box(kBoxL),
                    12.0, /*my_rank=*/1);
        if (table.max_hops(0) != 4) {
            std::fprintf(stderr, "full-ring hops=%d want 4\n",
                         table.max_hops(0));
            return EXIT_FAILURE;
        }
        table.setup(uniform_dd(/*dims_x=*/4, /*coords_x=*/1), cubic_box(kBoxL),
                    20.0, /*my_rank=*/1);
        if (table.max_hops(0) != 6) {
            std::fprintf(stderr, "beyond-ring hops=%d want 6\n",
                         table.max_hops(0));
            return EXIT_FAILURE;
        }
        std::printf("periodic-ring-hops 4/6\n");
    }

    return EXIT_SUCCESS;
}
