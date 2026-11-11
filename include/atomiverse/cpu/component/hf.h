/// @file cpu/component/hf.h
/// @brief cpu::HF — pure polymorphic tag for Gaussian-basis mean-field methods.
///
/// Not a Component. Concrete methods (`RHF` / `UHF` / `RKS` / `UKS`) derive
/// `Driver` + `HF` and bind as nested driver nodes. `on(phase).add` copies
/// `provides_energy` / `provides_force` onto the slot Component.
#pragma once

namespace ATV {
namespace cpu {

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

} // namespace cpu
} // namespace ATV
