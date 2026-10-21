/// @file cpu/multiscale/internal/qm_mm_vdw.h
/// @brief QM–MM 12-6 LJ (shifted), Lorentz–Berthelot mixing (esmm-qmmm-03).
#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/lj_type_coeff.h" // LJTypeCoeff

namespace ATV {
namespace cpu {
namespace multiscale {
namespace internal {

/// @brief Per-type-pair QM–MM van der Waals (only cross pairs).
///
/// Diagonals in `cross_lj` seed Lorentz–Berthelot mixing for missing
/// off-diagonals. A type with no diagonal has ε = 0 (no LJ site).
class QmMmVdw {
  public:
    /// @param cross_lj Explicit type-pair rows and/or diagonals.
    /// @param cutoff Pair cutoff [Å] (shifted potential).
    QmMmVdw(std::vector<LJTypeCoeff> cross_lj, double cutoff);

    /// @brief Energy and optional forces for QM–MM pairs only.
    /// @param frame User frame (positions, types, forces).
    /// @param qm_rows User-frame rows of real QM atoms.
    /// @param forces_requested When true, accumulate forces [eV/Å].
    /// @return Pair energy [eV].
    double launch(Frame& frame, const std::vector<int>& qm_rows,
                  bool forces_requested) const;

    double cutoff() const { return cutoff_; }

  private:
    struct PairParam {
        double eps = 0;
        double sigma = 0;
    };
    /// Key: packed (min(ti,tj) << 32) | max(ti,tj)
    std::unordered_map<std::uint64_t, PairParam> table_;
    double cutoff_ = 0;
    double cutoff2_ = 0;

    static std::uint64_t key_(int ti, int tj) {
        const int a = ti < tj ? ti : tj;
        const int b = ti < tj ? tj : ti;
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32)
               | static_cast<std::uint32_t>(b);
    }

    PairParam lookup_(int ti, int tj) const;
};

} // namespace internal
} // namespace multiscale
} // namespace cpu
} // namespace ATV
