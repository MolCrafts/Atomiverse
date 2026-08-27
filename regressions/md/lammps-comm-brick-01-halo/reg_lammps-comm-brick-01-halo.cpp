/// @file reg_lammps-comm-brick-01-halo.cpp
/// @brief Serial pins for the MPI-free `comm::Halo` swap table.
///
/// Prints hard-coded hop and adapter literals. No third-party process.
/// Literature: Plimpton JCP 117 (1995); Thompson CPC 271 (2022).
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/halo.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/frame/box.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

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
    dd.periods[0] = 1;
    dd.periods[1] = 1;
    dd.periods[2] = 1;
    const double w = L / static_cast<double>(dims_x);
    dd.sublo[0] = w * static_cast<double>(coords_x);
    dd.subhi[0] = dd.sublo[0] + w;
    dd.sublo[1] = 0.0;
    dd.subhi[1] = L;
    dd.sublo[2] = 0.0;
    dd.subhi[2] = L;
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
        const int ceil_slab = table.max_hops(0);
        if (ceil_slab != 1) {
            std::fprintf(stderr, "ceil-slab hop=%d want 1\n", ceil_slab);
            return EXIT_FAILURE;
        }
        std::printf("ceil-slab %d\n", ceil_slab);
    }

    {
        cpu::comm::Halo table;
        table.setup(uniform_dd(/*dims_x=*/2, /*coords_x=*/0), cubic_box(kBoxL),
                    kShell, /*my_rank=*/0);
        int sendself = 1;
        for (int i = 2; i < 6; ++i) {
            if (!table[i].sendself || table[i].nrecv != 0)
                sendself = 0;
        }
        if (!sendself) {
            std::fprintf(stderr, "sendself skip failed\n");
            return EXIT_FAILURE;
        }
        std::printf("sendself %d\n", sendself);
    }

    {
        cpu::comm::Halo table;
        int mic_refuse = 0;
        try {
            // 12 A shell on a 15.78 A box split 4 ways: hops = ceil(12/3.945)
            // = 4 > dims-1 = 3.
            table.setup(uniform_dd(/*dims_x=*/4, /*coords_x=*/1),
                        cubic_box(kBoxL), 12.0, /*my_rank=*/1);
        } catch (const std::runtime_error&) {
            mic_refuse = 1;
        }
        if (!mic_refuse) {
            std::fprintf(stderr, "mic-refuse did not throw\n");
            return EXIT_FAILURE;
        }
        std::printf("mic-refuse %d\n", mic_refuse);
    }

    const int halo_is_component =
        std::is_base_of_v<cpu::Component, cpu::comm::Halo> ? 1 : 0;
    if (halo_is_component != 0) {
        std::fprintf(stderr, "comm::Halo must not inherit Component\n");
        return EXIT_FAILURE;
    }
    std::printf("halo-is-component %d\n", halo_is_component);
    return EXIT_SUCCESS;
}
