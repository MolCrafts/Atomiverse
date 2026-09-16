/// @file cpu/multiscale/internal/classical_nonbonded.h
/// @brief All-pairs classical LJ (+ optional Coulomb) for ONIOM low level.
#pragma once

#include <unordered_map>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

namespace ATV {
namespace cpu {
namespace multiscale {
namespace internal {

/// @brief Shifted 12-6 LJ over all pairs, or pairs within a row subset.
///
/// Used as ONIOM \(E_{\mathrm{low}}\). Empty `rows` ⇒ all `nlocal` atoms.
/// Coulomb (if enabled) uses `kAtom_charge` and the same pair selection.
class ClassicalNonbonded {
  public:
    /// @param lj Type-pair / diagonal LJ table (Lorentz–Berthelot fill).
    /// @param cutoff Nonbonded cutoff [Å].
    /// @param with_coulomb When true, add \(k q_i q_j / r\) among the same pairs.
    ClassicalNonbonded(std::vector<LJTypeCoeff> lj, double cutoff,
                       bool with_coulomb = false);

    /// @brief Energy [eV]; optional force accumulation scaled by `force_sign`
    ///        (+1 real, −1 model subtraction).
    /// @param rows If non-empty, only pairs with both atoms in `rows`.
    double launch(Frame& frame, const std::vector<int>& rows,
                  bool forces_requested, double force_sign = 1.0) const;

    double cutoff() const { return cutoff_; }

  private:
    struct PairParam {
        double eps = 0;
        double sigma = 0;
    };
    std::unordered_map<std::uint64_t, PairParam> table_;
    double cutoff_ = 0;
    double cutoff2_ = 0;
    bool with_coulomb_ = false;

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
