/// @file cpu/observe/widom.h
/// @brief Widom test-particle insertion estimator for μ_ex (classical NVT).
///
/// Accumulates the Boltzmann factor of a ghost particle that never enters
/// the host Markov chain:
///
///   μ_ex = −k_B T ln ⟨exp(−β ΔU)⟩
///
/// where the average is over (host configurations × uniform insertion sites)
/// and ΔU is the interaction of the test particle with the host only.
///
/// v1: single species, cut-and-shifted LJ or hard-sphere ΔU, single-rank.
/// Snapshot / conditional estimator — average Boltzmann factors, never
/// average per-insertion μ values (Jensen bias).
///
/// Refs: Widom, J. Chem. Phys. 39, 2808 (1963), DOI 10.1063/1.1734110;
///       Frenkel & Smit, Understanding Molecular Simulation, §7.2.1.
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/observer.h"

namespace ATV {
namespace cpu {

/// @brief Test-particle interaction model for Widom ΔU.
enum class WidomPotential {
    /// Cut-and-shifted LJ 12-6 (same V_shift convention as PairLJCut).
    lj_cut,
    /// Hard sphere: ΔU = ∞ on overlap (r < σ), else 0.
    hard_sphere,
};

/// @brief Widom insertion Observer — Frame-immutable free-energy estimator.
///
/// Bind at MD or MC `end_of_step` (default), or call `sample(frame)` once on a
/// fixed host. Results: `mu_ex()` [eV], `boltzmann_avg()` = ⟨e^{−βΔU}⟩.
class Widom : public Observer {
  public:
    double temperature = 300.0; ///< Host temperature [K] (must be > 0).
    int n_trials = 1000;    ///< Ghost insertions per sample call (must be > 0).
    std::uint64_t seed = 1; ///< RNG seed for insertion positions.
    WidomPotential potential = WidomPotential::lj_cut;
    double epsilon = 0.0104; ///< LJ well depth [eV] (lj_cut only).
    double sigma = 3.4;      ///< LJ σ or HS diameter [Å] (must be > 0).
    double cutoff = 8.5;     ///< LJ cutoff [Å]; ignored for hard_sphere.

    /// @brief Construct with ASE-ish LJ defaults; set members before use.
    Widom() = default;

    /// @brief Construct a cut-and-shifted LJ Widom estimator.
    /// @param temperature_K Host temperature [K].
    /// @param epsilon_ev LJ ε [eV].
    /// @param sigma_a LJ σ [Å].
    /// @param cutoff_a Cutoff [Å].
    /// @param n_trials Ghost insertions per launch/sample.
    /// @param seed RNG seed.
    Widom(double temperature_K, double epsilon_ev, double sigma_a,
          double cutoff_a, int n_trials = 1000, std::uint64_t seed = 1);

    /// @brief Construct a hard-sphere Widom estimator.
    /// @param temperature_K Host temperature [K].
    /// @param diameter_a Hard-sphere diameter σ [Å].
    /// @param n_trials Ghost insertions per launch/sample.
    /// @param seed RNG seed.
    Widom(double temperature_K, double diameter_a, int n_trials,
          std::uint64_t seed, WidomPotential /*tag*/);

    std::string name() const override { return "widom"; }

    std::vector<DefaultBinding> default_bindings() const override {
        // Prefer explicit bind for dual MD/MC; defaults cover MD end_of_step.
        // Sampling: users also `on(sampling::Phase::end_of_step).add_ref(*this)`.
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +50)};
    }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    /// @brief Sample insertions on a host Frame without a full Driver.
    /// @param frame Host configuration (positions + box); not modified.
    void sample(const Frame& frame);

    /// @brief Reset running averages (keeps parameters and RNG seed reseeded).
    void reset();

    /// @brief Excess chemical potential μ_ex [eV].
    /// @return −kT ln⟨e^{−βΔU}⟩; NaN if no successful Boltzmann mass
    ///         (⟨e⟩ = 0, e.g. all HS insertions overlap).
    double mu_ex() const;

    /// @brief Running average ⟨exp(−β ΔU)⟩ (dimensionless).
    double boltzmann_avg() const;

    /// @brief Number of ghost insertions accumulated so far.
    long long n_insertions() const { return n_insertions_; }

  private:
    void validate_params_() const;
    double delta_u_(const Frame& frame, double x, double y, double z) const;
    void accumulate_frame_(const Frame& frame);

    double v_shift_ = 0; ///< Hard LJ at cutoff [eV].
    std::mt19937_64 rng_{1};
    long long n_insertions_ = 0;
    double sum_boltz_ = 0; ///< Σ exp(−β ΔU)
};

} // namespace cpu
} // namespace ATV
