/// @file cuda/component/hf.cuh
/// @brief cuda::HF — pure polymorphic tag for Gaussian-basis mean-field methods.
///
/// Mirrors cpu::HF: a non-Component energy/force-source tag. `cuda::scf::RHF`
/// is a `cuda::Driver<scf::Phase>` plus this tag. `on(phase).add` copies
/// `provides_energy` / `provides_force` onto the slot Component. Synthetic
/// `SCF` still inherits `Component` and overrides those predicates itself.
#pragma once

namespace ATV {
namespace cuda {

/// @brief Gaussian-basis electronic structure (HF and hybrid KS) kind tag.
class HF {
  public:
    virtual ~HF() = default;

    /// @brief True when this method contributes a system energy term.
    /// @return Dimensionless predicate. Default true.
    virtual bool provides_energy() const { return true; }
    /// @brief True when this method contributes a force / gradient term.
    /// @return Dimensionless predicate. Default true.
    virtual bool provides_force() const { return true; }
};

} // namespace cuda
} // namespace ATV
