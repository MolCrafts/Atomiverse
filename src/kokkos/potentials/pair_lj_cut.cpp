/// @file kokkos/potentials/pair_lj_cut.cpp
/// @brief Serial execution-space port of LAMMPS `lj/cut/kk` pair kernel.

#include "atomiverse/kokkos/potentials/pair_lj_cut.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ATV {
namespace kokkos {
namespace {

/// LAMMPS `PairLJCutKokkos::compute_fpair` (STACKPARAMS arithmetic).
KOKKOS_INLINE_FUNCTION double compute_fpair(double rsq, double lj1,
                                            double lj2) {
    const double r2inv = 1.0 / rsq;
    const double r6inv = r2inv * r2inv * r2inv;
    return r6inv * Kokkos::fma(lj1, r6inv, -lj2) * r2inv;
}

/// LAMMPS `PairLJCutKokkos::compute_evdwl`.
KOKKOS_INLINE_FUNCTION double compute_evdwl(double rsq, double lj3, double lj4,
                                            double offset) {
    const double r2inv = 1.0 / rsq;
    const double r6inv = r2inv * r2inv * r2inv;
    return Kokkos::fma(r6inv, Kokkos::fma(lj3, r6inv, -lj4), -offset);
}

void ensure_kokkos() {
    static std::once_flag once;
    std::call_once(once, [] {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
            std::atexit([] {
                if (Kokkos::is_initialized())
                    Kokkos::finalize();
            });
        }
    });
}

[[noreturn]] void bad(const std::string& how) {
    throw std::invalid_argument("kokkos::PairLJCut: " + how);
}

} // namespace

PairLJCut::PairLJCut(double epsilon, double sigma, double cutoff) {
    if (!(sigma > 0.0) || !(cutoff > 0.0)) {
        bad("sigma and cutoff must be > 0 Å. Construct with the LJ size "
            "and the cutoff used for the shift V(rc); there is no default.");
    }
    // pair_lj_cut.cpp init_one: lj1 = 48 ε σ^12, lj2 = 24 ε σ^6,
    // lj3 = 4 ε σ^12, lj4 = 4 ε σ^6, offset = 4 ε [(σ/rc)^12 − (σ/rc)^6].
    // Multiplies, not std::pow, so the products match the CPU pair's shift.
    double s2 = sigma * sigma;
    double s4 = s2 * s2;
    double s6 = s4 * s2;
    double s12 = s6 * s6;
    lj1_ = 48.0 * epsilon * s12;
    lj2_ = 24.0 * epsilon * s6;
    lj3_ = 4.0 * epsilon * s12;
    lj4_ = 4.0 * epsilon * s6;
    const double ratio = sigma / cutoff;
    double r2 = ratio * ratio;
    double r4 = r2 * r2;
    double r6 = r4 * r2;
    offset_ = 4.0 * epsilon * (r6 * r6 - r6);
    cutsq_ = cutoff * cutoff;
}

double PairLJCut::evaluate(std::span<const double> x, std::span<const double> y,
                           std::span<const double> z, HalfNeighborList neigh,
                           std::span<double> fx, std::span<double> fy,
                           std::span<double> fz) const {
    const std::size_t n = x.size();
    if (y.size() != n || z.size() != n || fx.size() != n || fy.size() != n
        || fz.size() != n || neigh.counts.size() != n
        || neigh.offsets.size() != n) {
        bad("x, y, z, fx, fy, fz, counts, and offsets must all have length n "
            "(one entry per atom). Pass the half list beside the coordinate "
            "columns; this kernel does not read a Frame.");
    }
    if (n > static_cast<std::size_t>(0x7fffffff)) {
        bad("n exceeds the LAMMPS int neighbor index. This foothold launches "
            "parallel_reduce(inum) with inum as int.");
    }
    const int n_int = static_cast<int>(n);
    for (int i = 0; i < n_int; ++i) {
        const int jnum = neigh.counts[static_cast<std::size_t>(i)];
        const int off = neigh.offsets[static_cast<std::size_t>(i)];
        if (jnum < 0 || off < 0) {
            bad("counts[i] and offsets[i] must be >= 0. Atom i owns counts[i] "
                "neighbors starting at offsets[i] in the neighbors array.");
        }
        const long end = static_cast<long>(off) + static_cast<long>(jnum);
        if (end > static_cast<long>(neigh.neighbors.size())) {
            bad("offsets[i] + counts[i] runs past neighbors. Size the "
                "neighbor array to cover every atom's block.");
        }
        for (int jj = 0; jj < jnum; ++jj) {
            const int j = neigh.neighbors[static_cast<std::size_t>(off + jj)];
            if (j < 0 || j >= n_int || j == i) {
                bad("neighbor index out of range or a self pair. Store each "
                    "pair once, on one endpoint, with the other atom's index.");
            }
        }
    }

    ensure_kokkos();

    // Caller-side clear, same role as LAMMPS force clear before pair_compute.
    // The HALF kernel then accumulates; it does not zero inside the atom loop.
    std::fill(fx.begin(), fx.end(), 0.0);
    std::fill(fy.begin(), fy.end(), 0.0);
    std::fill(fz.begin(), fz.end(), 0.0);
    if (n_int == 0 || neigh.neighbors.empty())
        return 0.0;

    using Host = Kokkos::HostSpace;
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    Kokkos::View<const double*, Host, Unmanaged> vx(x.data(), n);
    Kokkos::View<const double*, Host, Unmanaged> vy(y.data(), n);
    Kokkos::View<const double*, Host, Unmanaged> vz(z.data(), n);
    Kokkos::View<double*, Host, Unmanaged> vfx(fx.data(), n);
    Kokkos::View<double*, Host, Unmanaged> vfy(fy.data(), n);
    Kokkos::View<double*, Host, Unmanaged> vfz(fz.data(), n);
    Kokkos::View<const int*, Host, Unmanaged> counts(neigh.counts.data(), n);
    Kokkos::View<const int*, Host, Unmanaged> offsets(neigh.offsets.data(), n);
    Kokkos::View<const int*, Host, Unmanaged> neighbors(neigh.neighbors.data(),
                                                        neigh.neighbors.size());

    const double lj1 = lj1_;
    const double lj2 = lj2_;
    const double lj3 = lj3_;
    const double lj4 = lj4_;
    const double offset = offset_;
    const double cutsq = cutsq_;

    // Host path of pair_compute: parallel_reduce over local atoms, not the
    // GPU TeamPolicy. Kokkos::Serial is the CPU execution space of this
    // foothold (HALF newton scatter is not atomic).
    double energy = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::Serial>(0, n_int),
        KOKKOS_LAMBDA(const int i, double& evdwl_acc) {
            const double xtmp = vx(i);
            const double ytmp = vy(i);
            const double ztmp = vz(i);
            double fxtmp = 0.0;
            double fytmp = 0.0;
            double fztmp = 0.0;
            const int jnum = counts(i);
            const int off = offsets(i);
            for (int jj = 0; jj < jnum; ++jj) {
                const int j = neighbors(off + jj);
                const double delx = xtmp - vx(j);
                const double dely = ytmp - vy(j);
                const double delz = ztmp - vz(j);
                const double rsq = delx * delx + dely * dely + delz * delz;
                if (rsq < cutsq) {
                    const double fpair = compute_fpair(rsq, lj1, lj2);
                    const double fxij = delx * fpair;
                    const double fyij = dely * fpair;
                    const double fzij = delz * fpair;
                    fxtmp += fxij;
                    fytmp += fyij;
                    fztmp += fzij;
                    vfx(j) -= fxij;
                    vfy(j) -= fyij;
                    vfz(j) -= fzij;
                    evdwl_acc += compute_evdwl(rsq, lj3, lj4, offset);
                }
            }
            vfx(i) += fxtmp;
            vfy(i) += fytmp;
            vfz(i) += fztmp;
        },
        energy);
    return energy;
}

} // namespace kokkos
} // namespace ATV
