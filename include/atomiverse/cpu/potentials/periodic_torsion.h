/// @file cpu/potentials/periodic_torsion.h
/// @brief `PeriodicTorsionSet` — the shared kernel of every periodic 4-body
///        style: V = Σ_m K_m (1 + cos(n_m φ − δ_m)).
///
/// Three LAMMPS styles are this one sum with different parameter shapes:
///
/// | Style | Terms per type | Topology |
/// |---|---|---|
/// | `dihedral_style charmm` (weight 0) | one `{K, n, δ}` | dihedrals |
/// | `dihedral_style fourier` | `m` × `{K, n, δ}` | dihedrals |
/// | `improper_style cvff` | one `{K, n, δ ∈ {0, π}}` (`d = ±1`) | impropers |
///
/// Each concrete component owns one set as a member; the set owns the
/// resolved-tuple cache, thread force scratch and drop tally, so the three
/// styles share one loop without an intermediate base class (Kind depth
/// stays `Concrete → Dihedral/Improper → Component`).
#pragma once

#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/potentials/bonded_cache.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

namespace ATV {
namespace cpu {

/// @brief One periodic term K (1 + cos(n φ − δ)).
struct TorsionTerm {
    double K = 0.0;     ///< Amplitude [eV].
    int n = 1;          ///< Multiplicity, 0 ≤ n ≤ 6 (n = 0 is a constant).
    double delta = 0.0; ///< Phase [rad].
};

namespace md {

/// @brief Per-type periodic torsion terms + the bonded loop that sums them.
class PeriodicTorsionSet {
  public:
    /// @brief Which per-atom topology columns the set walks.
    enum class Topology { Dihedral, Improper };

    /// @param terms Terms per type (index = 0-based type id); a type with no
    ///        terms, or only K = 0 terms, contributes exactly nothing.
    /// @param topology Dihedral or improper columns (owner = atom 2).
    /// @param style Name used in error messages and drop reports.
    /// @throws std::invalid_argument on an empty table, a non-finite K or δ,
    ///         or a multiplicity outside [0, 6].
    PeriodicTorsionSet(std::vector<std::vector<TorsionTerm>> terms,
                       Topology topology, const char* style);

    /// @brief Skip the virial when nothing consumes stress.
    /// @param on true to accumulate the 4-body virial when requested.
    void set_compute_stress(bool on) { compute_stress_ = on; }

    /// @brief Allocate per-thread force scratch for the Frame's capacity.
    /// @param ctx Build context.
    void build(const BuildContext& ctx);

    /// @brief Add energy [eV] (to `kEnergy` and the style's key), forces
    ///        [eV/Å] and, when requested, the virial [eV] to the Frame.
    /// @param ctx Launch context.
    void compute(const LaunchContext& ctx);

    /// @brief Report dropped tuples and release scratch.
    void destroy();

  private:
    void rebuild_tuples_(const LaunchContext& ctx);

    // CSR over types: terms of type t are [offset_[t], offset_[t+1]).
    std::vector<int> offset_;
    std::vector<double> K_;
    std::vector<int> n_;
    std::vector<double> cos_delta_;
    std::vector<double> sin_delta_;
    std::vector<char> active_; ///< type has at least one K ≠ 0 term

    Topology topology_;
    const char* style_;
    bool compute_stress_ = true;

    DropTally drops_;
    BondedTupleCache cache_;
};

} // namespace md
} // namespace cpu
} // namespace ATV
