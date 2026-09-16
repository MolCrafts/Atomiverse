/// @file cpu/scf/methods.h
/// @brief Umbrella header for the concrete Gaussian mean-field methods.
///
/// Includes the shared CRTP base `MeanFieldMethod<D>` (`mean_field_method.h`)
/// and the seven concrete methods `RHF` / `ROHF` / `UHF` / `GHF` / `RKS` /
/// `UKS` / `ROKS` (`rhf.h` / `rohf.h` / `uhf.h` / `ghf.h` / `rks.h` /
/// `uks.h` / `roks.h`). There is no method-agnostic name here: pick the
/// concrete method whose ansatz you mean.
///
/// There is no bag named `SCFSolver` and no user-facing `SCFConfig` /
/// public integral bag. Each method is a concrete class; parameters are
/// **constructor arguments**; the only I/O currency is `Frame`. SCF
/// self-consistency is a **Driver loop** — see `mean_field_method.h`.
///
/// @code
///   cpu::scf::RHF rhf{BasisSet::STO3G, /*charge=*/0, /*tol=*/1e-10};
///   rhf.run(frame);                    // frame.meta.get(kEnergy) [eV];
///                                      // rhf.energy() [Hartree];
///                                      // wavefunction on rhf.determinant()
/// @endcode
#pragma once

#include "atomiverse/cpu/scf/counterpoise.h"
#include "atomiverse/cpu/scf/dftd3.h"
#include "atomiverse/cpu/scf/ghf.h"
#include "atomiverse/cpu/scf/mean_field_method.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/rks.h"
#include "atomiverse/cpu/scf/rohf.h"
#include "atomiverse/cpu/scf/roks.h"
#include "atomiverse/cpu/scf/uhf.h"
#include "atomiverse/cpu/scf/uks.h"

// This header declares no symbols of its own — it is the umbrella over the
// concrete methods above. Name the method you mean: `RHF` / `ROHF` / `UHF` /
// `GHF` for Hartree–Fock, `RKS` / `UKS` / `ROKS` for Kohn–Sham DFT.
