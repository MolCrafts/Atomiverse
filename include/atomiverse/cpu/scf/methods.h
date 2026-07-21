/// @file cpu/scf/methods.h
/// @brief Umbrella header for the concrete Gaussian mean-field methods.
///
/// Includes the shared CRTP base `MeanFieldMethod<D>` (`mean_field_method.h`)
/// and the six concrete methods `RHF` / `ROHF` / `UHF` / `GHF` / `RKS` /
/// `UKS` (`rhf.h` / `rohf.h` / `uhf.h` / `ghf.h` / `rks.h` / `uks.h`), and
/// keeps the historical `cpu::scf::SCF` alias. Existing includers of
/// `methods.h` compile unchanged.
///
/// There is no bag named `SCFSolver` and no user-facing `SCFConfig` /
/// `IntegralPack` step. Each method is a concrete class; parameters are
/// **members**; the only I/O currency is `Frame`. SCF self-consistency is a
/// **Driver loop** — see `mean_field_method.h`.
///
/// @code
///   cpu::scf::RHF rhf;
///   rhf.basis = BasisSet::STO3G;
///   rhf.tol = 1e-10;
///   rhf.run(frame);                    // frame.meta.get(kEnergy) [eV];
///                                      // rhf.energy() [Hartree];
///                                      // wavefunction on rhf.determinant()
/// @endcode
#pragma once

#include "atomiverse/cpu/scf/counterpoise.h"
#include "atomiverse/cpu/scf/ghf.h"
#include "atomiverse/cpu/scf/mean_field_method.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/rks.h"
#include "atomiverse/cpu/scf/rohf.h"
#include "atomiverse/cpu/scf/uhf.h"
#include "atomiverse/cpu/scf/uks.h"

namespace ATV {
namespace cpu {
namespace scf {

// ── Compatibility aliases during migration (component name) ───────────
// Historical `cpu::scf::SCF` was a method-agnostic bag. Prefer RHF/UHF/RKS/UKS.
using SCF = RKS; // production default: closed-shell KS/PBE

} // namespace scf
} // namespace cpu
} // namespace ATV
