/// @file cpu/longrange/ewald.h
/// @brief Reciprocal-space Ewald summation (LAMMPS `kspace_style ewald`) — CPU.
///
/// Thin Frame/MPI wrapper over `ATV::longrange::EwaldSum`. Adds, on top of the
/// real-space `CoulLong` contribution:
///
///   E_recip = units::coulomb · (2π/V) Σ_{k≠0} exp(−k²/4α²)/k² · |S(k)|²
///   E_self  = −units::coulomb · (α/√π) Σ_i q_i²
///   E_corr  = −units::coulomb · π/(2 V α²) · (Σ_i q_i)²      (net-charge term)
///
/// **Distributed (MPI) operation.** Each rank forms the partial structure
/// factor over its owned atoms, the complex S(k) is `MPI_Allreduce`-summed
/// across ranks, then every rank evaluates reciprocal forces on its own
/// atoms from the global S(k). The reciprocal energy / stress and the
/// net-charge term are added on rank 0 only (the `Reduce` component sums
/// `scalars.energy` across ranks); the self-energy partitions naturally and
/// is added per rank over owned atoms.
///
/// Citation: P. P. Ewald, *Ann. Phys.* **369**, 253 (1921).
/// DOI: 10.1002/andp.19213690304.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/longrange/kspace.h"
#include "atomiverse/facade/component_role.h"
#include "atomiverse/facade/stage.h"
#include "atomiverse/longrange/ewald_sum.h"

namespace ATV {
namespace cpu {

/// @brief Reciprocal-space Ewald summation for periodic electrostatics.
class KSpaceEwald : public cpu::KSpace {
  public:
    /// @param alpha    Ewald splitting parameter α [1/Å]. Must match the
    ///                 real-space `CoulLong` α.
    /// @param accuracy Relative accuracy target; sets the reciprocal-vector
    ///                 cutoff. Default 1e-5.
    explicit KSpaceEwald(double alpha, double accuracy = 1e-5);

    /// @brief Recommended α for a real-space cutoff and accuracy (LAMMPS
    /// heuristic g_ewald = (1.35 − 0.15·ln(accuracy)) / cutoff).
    static double recommend_alpha(double cutoff, double accuracy = 1e-5);

    std::string name() const override { return "ewald"; }
    LongRangeContract long_range_contract() const override {
        return {LongRangeRole::ReciprocalSpace, sum_.alpha(), AccReal(0),
                sum_.accuracy()};
    }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::force, 100)};
    }

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;
    std::size_t memory_bytes() const override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    /// @brief Rebuild the reciprocal-vector tables if the cell changed.
    void ensure_kvectors(const cpu::Frame& frame, bool initial_build);

    longrange::EwaldSum sum_;

    // Canonical Box mutation revision used to detect NPT/deform changes.
    const cpu::Box* box_identity_ = nullptr;
    std::uint64_t cell_revision_ = 0;
    bool have_kvectors_ = false;

    // Scratch structure-factor accumulators (sized to #k vectors).
    std::vector<double> sr_, si_;
};

} // namespace cpu
} // namespace ATV
