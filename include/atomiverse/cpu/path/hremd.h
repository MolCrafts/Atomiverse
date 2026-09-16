/// @file cpu/path/hremd.h
/// @brief Hamiltonian REMD — replica exchange on a λ ladder at fixed T.
#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/replica/replica_expansion.h"
#include "atomiverse/cpu/replica/replica_set.h"

namespace ATV {
namespace cpu {
namespace path {

/// @brief Hamiltonian REMD: outer exchange loop + per-replica MD programs.
///
/// Exchange uses the fixed-T Hamiltonian criterion with dual energy
/// evaluation U_i(x_j), U_j(x_i) on adjacent λ pairs. Sibling of `Remd`
/// (temperature ladder); does not replace it. No `HRemdConfig` bag.
///
/// ## Every replica is thermostatted, and that is not optional
///
/// The exchange test
/// `P = min(1, exp[−β((U_i(x_j) + U_j(x_i)) − (U_i(x_i) + U_j(x_j)))])`
/// is the Metropolis rule for swapping configurations between two systems that
/// are each sampling the **canonical** ensemble at the same β with different
/// Hamiltonians. Replicas integrated at constant energy sample nothing of the
/// sort: each holds whatever energy its initial velocity draw gave it, β is a
/// constant rather than a property of a distribution, and the acceptance ratio
/// measures nothing. This is the same defect `Remd` carried — see its class
/// comment — and it is worth stating twice because the two ladders fail
/// identically and silently, with a plausible-looking acceptance ratio.
///
/// So `HRemd` attaches a CSVR thermostat (Bussi–Donadio–Parrinello, *J. Chem.
/// Phys.* **126**, 014101 (2007)) at `temperature` to every replica;
/// `thermostat_tau` is its coupling time.
class HRemd {
  public:
    /// @brief Distinct λ values, size ≥ 2.
    std::vector<double> lambdas;
    /// @brief Fixed T [K] for all replicas.
    double temperature = 300.0;
    /// @brief Nested MD timestep [fs].
    double dt = 1.0;
    /// @brief Nested MD steps between exchange rounds.
    int steps_per_exchange = 10;
    /// @brief Master seed.
    std::uint64_t seed = 1;
    /// @brief CSVR coupling time [fs] for the per-replica thermostat.
    ///        Must be > 0; the canonical ensemble every replica has to sample
    ///        for the exchange rule to mean anything depends on it.
    double thermostat_tau = 100.0;

    /// @param lambdas Distinct λ values, size ≥ 2.
    /// @param temperature Fixed T [K] (> 0).
    /// @param dt Nested MD timestep [fs].
    /// @param steps_per_exchange Nested MD steps between exchanges (≥ 1).
    /// @param seed Master RNG seed.
    explicit HRemd(std::vector<double> lambdas, double temperature = 300.0,
                   double dt = 1.0, int steps_per_exchange = 10,
                   std::uint64_t seed = 1);

    /// @brief Program builder: register forces for one λ (e.g. PairLJSoft).
    using ProgramBuilder = void (*)(cpu::md::MD& md, double lambda, void* user);
    void set_program_builder(ProgramBuilder builder, void* user = nullptr);

    /// @brief Optional energy evaluator for dual U(λ) on a foreign frame.
    using EnergyFn = double (*)(const cpu::Frame& frame, double lambda,
                                void* user);
    void set_energy_fn(EnergyFn fn, void* user = nullptr);

    void run(cpu::Frame& frame, int n_exchanges);

    int exchange_attempts() const { return attempts_; }
    int exchange_accepts() const { return accepts_; }

  private:
    void ensure_built_(cpu::Frame& frame);
    void propagate_();
    void exchange_round_(bool even_edges);
    double energy_at_(const cpu::Frame& frame, double lambda);

    ProgramBuilder builder_ = nullptr;
    void* builder_user_ = nullptr;
    EnergyFn energy_fn_ = nullptr;
    void* energy_user_ = nullptr;
    std::unique_ptr<IdentityReplication> expansion_;
    std::unique_ptr<ReplicaSet> band_;
    std::vector<std::unique_ptr<cpu::md::MD>> programs_;
    std::mt19937_64 exchange_rng_;
    bool built_ = false;
    int attempts_ = 0;
    int accepts_ = 0;
};

} // namespace path
} // namespace cpu
} // namespace ATV
