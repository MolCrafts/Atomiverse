/// @file cpu/component/hf.h
/// @brief cpu::HF — pure polymorphic tag for Gaussian-basis mean-field methods.
///
/// Not a Component. Concrete methods (`RHF` / `UHF` / `RKS` / `UKS`) derive
/// `CPUDriver` + `HF` and bind as nested driver nodes. Outer drivers recognise
/// energy/force sources via `dynamic_cast<const HF*>(driver_ptr)`.
#pragma once

namespace ATV {
namespace cpu {

/// @brief Gaussian-basis electronic structure (HF and hybrid KS) kind tag.
class HF {
  public:
    virtual ~HF() = default;
};

} // namespace cpu
} // namespace ATV
