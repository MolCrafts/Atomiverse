/// @file cpu/potentials/sw.h
/// @brief cpu::StillingerWeber — the engine's first three-body force style.
#pragma once

#include <cmath>
#include <string_view>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/neighbor_protocol.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/molrs.hpp"

namespace ATV {
namespace cpu {

/// @brief Stillinger-Weber parameterization for one element (lookup, no parsing).
///
/// @see sw_data/sw_tables.h for the functional form and the published values.
class SWPotential {
  public:
    /// @brief Look up a Stillinger-Weber parameterization.
    /// @param element Chemical element (molrs periodic-table enum).
    /// @param name Parameterization name, e.g. `"stillinger-weber"`.
    /// @throws std::out_of_range if the (element, name) pair is not tabulated.
    SWPotential(molrs::Element element, std::string_view name);

    /// @brief Cutoff `a * sigma` [Angstrom]. Both terms vanish identically here.
    double cutoff() const { return a_ * sigma_; }

    /// @brief Two-body energy phi2(r) [eV]. Zero at and beyond the cutoff.
    /// @param r Separation [Angstrom].
    double phi2(double r) const {
        const double rc = cutoff();
        if (r >= rc)
            return 0.0;
        return A_ * eps_
               * (B_ * std::pow(sigma_ / r, p_) - std::pow(sigma_ / r, q_))
               * std::exp(sigma_ / (r - rc));
    }

    /// @brief d(phi2)/dr [eV/Angstrom]. Zero at and beyond the cutoff.
    /// @param r Separation [Angstrom].
    double dphi2(double r) const {
        const double rc = cutoff();
        if (r >= rc)
            return 0.0;
        const double e = std::exp(sigma_ / (r - rc));
        const double bracket =
            B_ * std::pow(sigma_ / r, p_) - std::pow(sigma_ / r, q_);
        const double dbracket = (-p_ * B_ * std::pow(sigma_ / r, p_)
                                 + q_ * std::pow(sigma_ / r, q_))
                                / r;
        const double de = e * (-sigma_ / ((r - rc) * (r - rc)));
        return A_ * eps_ * (dbracket * e + bracket * de);
    }

    /// @brief Three-body radial factor exp(gamma*sigma/(r - a*sigma)).
    /// @param r Leg length [Angstrom].
    double g(double r) const {
        const double rc = cutoff();
        if (r >= rc)
            return 0.0;
        return std::exp(gamma_ * sigma_ / (r - rc));
    }

    /// @brief d(g)/dr [1/Angstrom].
    /// @param r Leg length [Angstrom].
    double dg(double r) const {
        const double rc = cutoff();
        if (r >= rc)
            return 0.0;
        return g(r) * (-gamma_ * sigma_ / ((r - rc) * (r - rc)));
    }

    /// @brief Three-body strength lambda * eps [eV].
    double lambda_eps() const { return lambda_ * eps_; }
    /// @brief The tetrahedral reference cosine, -1/3.
    double cos_theta0() const { return cos_theta0_; }

  private:
    double eps_ = 0, sigma_ = 0, A_ = 0, B_ = 0, p_ = 0, q_ = 0, a_ = 0;
    double lambda_ = 0, gamma_ = 0, cos_theta0_ = 0;
};

/// @brief Stillinger-Weber three-body force style (single element).
///
/// Probe #1's second half. EAM proved that a per-atom intermediate (`rho_i`) can
/// be accumulated and consumed inside one `launch()`. SW proves the other half:
/// that the **full neighbor list** carries enough structure for a genuine
/// three-body term, whose energy depends on the ANGLE at a vertex atom and
/// therefore cannot be written as any function of pair separations alone.
///
/// For each atom `i` the angular term enumerates every unordered pair `(j,k)` of
/// `i`'s neighbours. That enumeration is only possible over a **full** list: a half
/// list gives `i` only the half of its neighbours that happened to land in its own
/// CSR row, so most triplets at `i` would simply be invisible. The component sets
/// `ctx.md.full_list_requested` in `build()`.
///
/// Like `cpu::EAM`, this is not a `cpu::Pair` (that models `fpair(r^2)`) — it is a
/// `Component + NeighborConsumer` bound with `.on(md::Phase::force).add(...)`.
/// Single-rank only for the same reason EAM is, and it throws under MPI.
class StillingerWeber : public cpu::Potential, public cpu::NeighborConsumer {
  public:
    /// @param potential Parameterization (constructed by lookup).
    explicit StillingerWeber(SWPotential potential);

    std::string name() const override { return "sw"; }
    /* role via Potential/kind base */

    /// @brief Cutoff the neighbor list must cover [Angstrom] — `a * sigma`.
    AccReal neighbor_cutoff() const override { return pot_.cutoff(); }
    const cpu::NeighborConsumer* as_neighbor_consumer() const override {
        return this;
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

  private:
    SWPotential pot_;
};

} // namespace cpu
} // namespace ATV
