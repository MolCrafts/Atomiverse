/// @file cpu/potentials/pair_gaussian_coul.h
/// @brief Real-space Coulomb with Gaussian electrode charges (CPU).
///
/// Evaluates the Reed / MetalWalls Gaussian charge energy and forces over
/// the published pair list (min-image) with a hard cutoff. Per-type width
/// \(\eta\) [Å⁻¹]: \(\eta>0\) ⇒ Gaussian electrode site; \(\eta\le 0\) ⇒ point
/// charge (electrolyte). Optional Gaussian self-energy (default on).
///
/// A `NeighborConsumer` through `Pair`: the driver requires a `NeighborList`
/// rather than letting the style drop, unannounced, onto the O(N²) sweep it
/// keeps for the no-list case.
#pragma once

#include <array>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/ghost_contract.h"
#include "atomiverse/cpu/potentials/pair.h"

namespace ATV {
namespace cpu {

/// @brief Gaussian / point Coulomb pair potential for electrode MD footholds.
///
/// Parameters are constructor arguments — no public `*Config` bag.
class PairGaussianCoul : public Pair {
  public:
    /// @brief Construct with a hard cutoff and per-type Gaussian widths.
    ///
    /// @param cutoff      Pair cutoff [Å]; must be > 0.
    /// @param eta_by_type Width \(\eta\) [Å⁻¹] indexed by `kAtom_type`.
    ///                    Entry \(\le 0\) means point charge for that type.
    ///                    Empty vector ⇒ every atom is a point charge. A
    ///                    non-empty table must cover **every** type present in
    ///                    the frame — a type past the end throws rather than
    ///                    quietly becoming a point charge.
    /// @param include_self If true (default), add Gaussian self-energy
    ///                     \(\frac12\texttt{coulomb}\,q_i^2 G_{ii}\) for
    ///                     Gaussian types.
    PairGaussianCoul(double cutoff, std::vector<double> eta_by_type,
                     bool include_self = true);

    /// @brief The Gaussian Coulomb kernel reads the NEIGHBOUR charge, and a
    /// neighbour is a ghost across every cut.
    ///
    /// `stable`, matching `CoulLong` / `PairLJCutCoulLong`: a fixed-charge
    /// model never changes it, and a fluctuating-charge owner (QEq,
    /// constant-potential electrodes) refreshes ghosts itself through
    /// `ChargePacker` on the self-initiated `ForwardPacker` path.
    /// @return This style's ghost declarations.
    static constexpr std::array kGhosts = {
        GhostField::read_stable(kAtom_charge, "PairGaussianCoul"),
    };
    std::span<const GhostField> declare_ghosts() const override {
        return kGhosts;
    }

    std::string name() const override { return "gaussian/coul"; }

    // NeighborConsumer via Pair — production depth uses ctx.md.pair_list when
    // published; falls back to O(N²) if no list (small textbook cells).

    /// @brief Whether Gaussian self-energy is included.
    bool include_self() const { return include_self_; }

    /// @brief Width table (copy) [Å⁻¹].
    const std::vector<double>& eta_by_type() const { return eta_by_type_; }

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    double cutoff_sq_ = 0;
    std::vector<double> eta_by_type_;
    bool include_self_ = true;

    /// @brief Width for one atom type [Å⁻¹].
    ///
    /// An **empty** table is the documented "every atom is a point charge"
    /// mode, so it answers 0 for every type. A **non-empty** table that does
    /// not cover the type is a specification error and throws: the fallback it
    /// used to return silently reclassified an electrode site as a point
    /// charge, which changes the energy, the force and the induced charge
    /// without any sign that a parameter was missing.
    /// @throws std::out_of_range if the table is non-empty and @p type is
    ///         outside it.
    double eta_of(int type) const {
        if (eta_by_type_.empty())
            return 0.0;
        if (type < 0 || static_cast<std::size_t>(type) >= eta_by_type_.size())
            throw std::out_of_range(
                "PairGaussianCoul: atom type " + std::to_string(type)
                + " has no Gaussian width. The table passed to the constructor "
                  "covers types 0.."
                + std::to_string(eta_by_type_.size() - 1)
                + "; list every type (use eta <= 0 for a point charge), or "
                  "pass an empty table to make every atom a point charge.");
        return eta_by_type_[static_cast<std::size_t>(type)];
    }
};

} // namespace cpu
} // namespace ATV
