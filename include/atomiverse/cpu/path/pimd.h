/// @file cpu/path/pimd.h
/// @brief PIMD skeleton — ring-polymer MD over IdentityReplication.
///
/// Budget-terminating preset (like Remd): expands one Frame into P identical
/// beads, runs P owned `cpu::md::MD` trees with ring-polymer springs, collapses
/// to the **centroid**. Second real consumer of IdentityReplication after REMD.
///
/// Free-particle path: program builder may install nothing (springs only).
/// Optional external isotropic harmonic trap k_ext (eV/Å²) about the origin.
///
/// Out of scope (v1 skeleton): RPMD estimators, PILE thermostat, ring-polymer
/// contraction, multi-rank bead parallel, Trotter-factor tables.
///
/// Spring constant (Tuckerman / Ceriotti form):
///   k = (m / β² ℏ²) · P    [converted to eV/Å² via mass_v2_to_energy]
/// with ℏ = h/(2π), β = 1/(k_B T).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/path/ring_polymer_spring.h"
#include "atomiverse/cpu/replica/replica_expansion.h"
#include "atomiverse/cpu/replica/replica_set.h"

namespace ATV {
namespace cpu {
namespace path {

/// @brief Ring-polymer MD preset over IdentityReplication (bead indices).
///
/// Not a new Driver category — owns N `MD` programs + `ReplicaSet`, like Remd.
/// No `PimdConfig` bag — parameters are constructor args / public members.
class Pimd {
  public:
    /// @brief Trotter number P (≥ 2).
    int n_beads = 4;
    /// @brief Physical temperature [K] (> 0).
    double temperature = 300.0;
    /// @brief Nested MD timestep [fs] (> 0).
    double dt = 0.5;
    /// @brief MD steps between outer bookkeeping.
    int steps_per_block = 10;
    /// @brief InitVelocity seeds.
    std::uint64_t seed = 1;
    /// @brief Optional external harmonic trap k |r|²/2 [eV/Å²]; 0 = free particle.
    double k_external = 0.0;

    /// @param n_beads Trotter number P (≥ 2).
    /// @param temperature Physical T [K] (> 0).
    /// @param dt Nested MD timestep [fs] (> 0).
    /// @param steps_per_block MD steps between outer blocks.
    /// @param seed InitVelocity seed.
    /// @param k_external External trap [eV/Å²]; 0 = free particle.
    explicit Pimd(int n_beads = 4, double temperature = 300.0, double dt = 0.5,
                  int steps_per_block = 10, std::uint64_t seed = 1,
                  double k_external = 0.0);

    using ProgramBuilder = void (*)(cpu::md::MD& md, int bead_index,
                                    void* user);
    void set_program_builder(ProgramBuilder builder, void* user = nullptr);

    /// @brief Expand, build P MD programs, run `n_blocks` blocks, collapse centroid.
    void run(cpu::Frame& frame, int n_blocks);

    int n_beads_count() const { return n_beads; }
    double spring_constant() const { return k_spring_; }
    double mean_bead_spread() const { return mean_spread_; }
    double radius_of_gyration_sq() const { return rg2_; }
    double free_particle_rg2_analytic() const;

  private:
    void ensure_built_(cpu::Frame& frame);
    void propagate_();
    void collapse_centroid_(cpu::Frame& out);
    static double compute_k_spring_(double temperature_K, int n_beads,
                                    double mass_amu);

    ProgramBuilder builder_ = nullptr;
    void* builder_user_ = nullptr;
    std::unique_ptr<IdentityReplication> expansion_;
    std::unique_ptr<ReplicaSet> band_;
    std::vector<std::unique_ptr<RingPolymerSpring>> springs_;
    std::vector<std::unique_ptr<cpu::md::MD>> programs_;
    double k_spring_ = 0;
    double mean_spread_ = 0;
    double rg2_ = 0;
    double mass_amu_ = 1.0;
    bool built_ = false;
};

} // namespace path
} // namespace cpu
} // namespace ATV
