/// @file cpu/potentials/sw.h
/// @brief cpu::StillingerWeber — the engine's first three-body force style.
///
/// Thread-owned force buffers and short-neighbor traversal ported from LAMMPS
/// `src/OPENMP/pair_sw_omp.cpp` and fused radial/angular coefficients from
/// `src/MANYBODY/pair_sw.cpp`, checkout
/// `a0e2bd6efb3d67a29c0c4e4de3b19556ffdb9d5b`.
/// Copyright (2003) Sandia Corporation. Under the terms of Contract
/// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
/// certain rights in this software. Distributed under the GNU General
/// Public License (GPL-2.0-only).
#pragma once

#include <array>
#include <cmath>
#include <string_view>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/neighbor_protocol.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/potentials/force_scratch.h"
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
    double phi2(double r) const { return two_body(r, nullptr); }

    /// @brief d(phi2)/dr [eV/Angstrom]. Zero at and beyond the cutoff.
    /// @param r Separation [Angstrom].
    double dphi2(double r) const {
        double derivative;
        two_body(r, &derivative);
        return derivative;
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
    friend class StillingerWeber;

    // Energy and its optional derivative share one exponential and power pair.
    // Keep the fused kernel private; the public operations retain their units.
    double two_body(double r, double* derivative) const {
        const double rc = cutoff();
        if (r >= rc) {
            if (derivative)
                *derivative = 0.0;
            return 0.0;
        }
        const double rinv = 1.0 / r;
        const double sr = sigma_ * rinv, sr2 = sr * sr;
        const double sp = p_ == 4.0 ? sr2 * sr2 : std::pow(sr, p_);
        const double sq = q_ == 0.0 ? 1.0 : std::pow(sr, q_);
        const double rainv = 1.0 / (r - rc);
        const double prefactor = A_ * eps_ * std::exp(sigma_ * rainv);
        const double bracket = B_ * sp - sq;
        if (derivative)
            *derivative = prefactor
                          * ((-p_ * B_ * sp + q_ * sq) * rinv
                             - bracket * sigma_ * rainv * rainv);
        return prefactor * bracket;
    }

    double eps_ = 0, sigma_ = 0, A_ = 0, B_ = 0, p_ = 0, q_ = 0, a_ = 0;
    double lambda_ = 0, gamma_ = 0, cos_theta0_ = 0;
};

/// @brief Stillinger-Weber three-body force style (single element).
///
/// For each atom `i` the angular term enumerates every unordered pair `(j,k)` of
/// `i`'s neighbours. That enumeration is only possible over a **full** list: a half
/// list gives `i` only the half of its neighbours that happened to land in its own
/// CSR row, so most triplets at `i` would be invisible. The consumer declares
/// `NeighborListMode::Full`; its two-body term selects each pair once by tag.
///
/// Like `cpu::EAM`, this is a `Potential + NeighborConsumer` bound with
/// `.on(md::Phase::force).add(...)`. Owned centres use full neighbour rows;
/// ghost force contributions reverse to owners.
class StillingerWeber : public cpu::Potential, public cpu::NeighborConsumer {
  public:
    /// @param potential Parameterization (constructed by lookup).
    explicit StillingerWeber(SWPotential potential);

    std::string name() const override { return "sw"; }
    /* role via Potential/kind base */

    /// @brief Cutoff the neighbor list must cover [Angstrom] — `a * sigma`.
    AccReal neighbor_cutoff() const override { return pot_.cutoff(); }
    NeighborListMode neighbor_list_mode() const override {
        return NeighborListMode::Full;
    }
    const cpu::NeighborConsumer* as_neighbor_consumer() const override {
        return this;
    }

    static constexpr std::array kGhosts = {
        GhostField::read_stable(kAtom_tag, "StillingerWeber"),
    };
    std::span<const GhostField> declare_ghosts() const override {
        return kGhosts;
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;
    void destroy(const cpu::DestroyContext& ctx) override;
    std::size_t memory_bytes() const override {
        std::size_t bytes = forces_.memory_bytes();
        for (const auto& neighbors : short_neighbors_)
            bytes += neighbors.capacity() * sizeof(int);
        return bytes;
    }

  private:
    SWPotential pot_;
    md::BondedForceScratch forces_;
    std::vector<std::vector<int>> short_neighbors_;
};

} // namespace cpu
} // namespace ATV
