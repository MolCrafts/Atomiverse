/// @file kokkos/potentials/pair_lj_cut.h
/// @brief Shifted LJ 12-6 on the Kokkos Serial (CPU) execution space.
///
/// Algorithm and launch transplanted from LAMMPS
/// `src/KOKKOS/pair_lj_cut_kokkos.cpp` (`compute_fpair`, `compute_evdwl`,
/// coefficient pack in `init_one` via `src/pair_lj_cut.cpp`) and the host
/// path of `pair_compute` in `src/KOKKOS/pair_kokkos.h`
/// (`Kokkos::parallel_reduce(inum, ...)`, `NEIGHFLAG=HALF`, newton on).
/// Checkout `e8c4c278` (2026-09-22), read at
/// `/nobackup/proj/disk/teoroo/personal/jicli594/work/lammps/src/src/KOKKOS/`.
/// Not a verbatim dump: one half-list kernel, no ghost exchange, no device
/// execution space. Public API is ATV-native.
///
/// Copyright (2003) Sandia Corporation.  Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software.  This software is distributed under
/// the GNU General Public License (GPL-2.0-only).
///
/// This is not a `Component` and it is not wired into `cpu::md::MD`.
/// Positions are used as given: the LAMMPS pair kernel does not apply
/// the minimum image; that is the neighbor list's job, and this foothold
/// does not build one.
#pragma once

#include <span>

namespace ATV {
namespace kokkos {

/// @brief Half neighbor list: each interacting pair stored once.
///
/// Atom `i` owns `counts[i]` neighbors in `neighbors`, starting at
/// `offsets[i]`. The kernel applies Newton's third law to both ends.
/// A self index is rejected. Duplicate pairs are the caller's error
/// and are counted twice.
struct HalfNeighborList {
    /// @brief Per-atom neighbor counts. Length = number of atoms.
    std::span<const int> counts;
    /// @brief Start of atom `i`'s block in `neighbors`. Length = number of atoms.
    std::span<const int> offsets;
    /// @brief Neighbor atom indices.
    std::span<const int> neighbors;
};

/// @brief Lennard-Jones 12-6 with the LAMMPS shifted cutoff, Kokkos Serial.
///
/// V(r) = 4ε[(σ/r)¹² − (σ/r)⁶] − V(rc)
/// The force multiplier matches `PairLJCutKokkos::compute_fpair`:
/// `fpair = r6inv * (lj1 * r6inv − lj2) * r2inv` with
/// `lj1 = 48 ε σ¹²`, `lj2 = 24 ε σ⁶`.
///
/// Execution space is `Kokkos::Serial` only. The `j` scatter is the
/// LAMMPS `HALF` newton path and is not atomic; Serial is what makes
/// that scatter race-free.
class PairLJCut {
  public:
    /// @brief Pack lj1..lj4, the energy offset, and `cutsq`.
    /// @param epsilon Well depth [eV].
    /// @param sigma Size parameter [Å]. Must be > 0.
    /// @param cutoff Cutoff [Å]. Must be > 0. The shift is V(cutoff).
    /// @throws std::invalid_argument if `sigma` or `cutoff` is not positive.
    PairLJCut(double epsilon, double sigma, double cutoff);

    /// @brief Half-list energy and forces on `Kokkos::Serial`.
    ///
    /// Overwrites `fx`/`fy`/`fz`. Pairs with `rsq >= cutoff²` contribute
    /// nothing. One pair stored once contributes its full energy (LAMMPS
    /// HALF + newton scale of 1).
    ///
    /// @param x Positions x [Å], length n.
    /// @param y Positions y [Å], length n.
    /// @param z Positions z [Å], length n.
    /// @param neigh Half neighbor list. `counts` and `offsets` have length n.
    /// @param fx Forces x [eV/Å], length n. Overwritten.
    /// @param fy Forces y [eV/Å], length n. Overwritten.
    /// @param fz Forces z [eV/Å], length n. Overwritten.
    /// @return Shifted pair energy [eV].
    /// @throws std::invalid_argument if lengths disagree, a count or offset
    ///         is negative, a neighbor index is out of range, or a neighbor
    ///         is a self pair.
    double evaluate(std::span<const double> x, std::span<const double> y,
                    std::span<const double> z, HalfNeighborList neigh,
                    std::span<double> fx, std::span<double> fy,
                    std::span<double> fz) const;

  private:
    double lj1_;
    double lj2_;
    double lj3_;
    double lj4_;
    double offset_;
    double cutsq_;
};

} // namespace kokkos
} // namespace ATV
