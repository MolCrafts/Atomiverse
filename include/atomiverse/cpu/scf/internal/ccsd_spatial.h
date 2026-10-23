/// @file cpu/scf/internal/ccsd_spatial.h
/// @brief Closed-shell spatial-orbital CCSD, one amplitude iteration.
///
/// The product loop is `cpu::scf::CCSD` (`Driver<cc::Phase>`). This header is
/// the engine that loop calls: `update` is PySCF `pyscf/cc/rccsd.py`
/// `update_amps` (Hirata et al., J. Chem. Phys. 120, 2581 (2004)), `mix` is
/// the accept step of `pyscf/cc/ccsd.py` `kernel` / `CCSD.run_diis`, and
/// `energy` is `rccsd.energy`. Nothing here sets a converged flag.
///
/// Reference: Scuseria, Janssen, Schaefer, J. Chem. Phys. 89, 7382 (1988);
/// Hirata et al., J. Chem. Phys. 120, 2581 (2004).
#pragma once

#include <memory>

#include "atomiverse/cpu/scf/ccsd.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {

struct CcsdSpatialCycle;

/// @brief Deletes a `CcsdSpatialCycle` defined in the .cpp.
struct CcsdSpatialDeleter {
    /// @brief Destroy a cycle allocated by `ccsd_spatial_open`.
    /// @param p Cycle, or null.
    void operator()(CcsdSpatialCycle* p) const;
};

/// @brief Owning handle to one closed-shell spatial CCSD amplitude state.
using CcsdSpatialPtr = std::unique_ptr<CcsdSpatialCycle, CcsdSpatialDeleter>;

/// @brief Build denominators, MP2 T₂ guess, and the DIIS buffers.
///
/// @param ints Integral pack with the dense ERI tensor materialized.
/// @param det Converged closed-shell RHF determinant.
/// @param e_scf_ha SCF total energy [Hartree].
/// @param use_diis When true, `ccsd_spatial_mix` extrapolates; when false it
///        copies the Jacobi amplitudes (same fixed point).
/// @return Cycle. No virtuals is a trivial zero-correlation state.
/// @throws std::invalid_argument on open-shell / missing ERI / non-aufbau
///         denominators (no silent fallback).
CcsdSpatialPtr ccsd_spatial_open(const internal::IntegralView& ints,
                                 const SlaterDeterminant& det, double e_scf_ha,
                                 bool use_diis);

/// @brief One `rccsd.update_amps` step. Stores the Jacobi increment.
/// @param cycle Open cycle.
void ccsd_spatial_update(CcsdSpatialCycle& cycle);

/// @brief DIIS extrapolant, or the raw Jacobi amplitudes when DIIS is off.
/// @param cycle Cycle after `ccsd_spatial_update`.
void ccsd_spatial_mix(CcsdSpatialCycle& cycle);

/// @brief `rccsd.energy` of the current amplitudes [Hartree]. Cached.
/// @param cycle Cycle after mix.
/// @return Correlation energy [Hartree].
double ccsd_spatial_energy(CcsdSpatialCycle& cycle);

/// @brief Cached correlation energy from the last `ccsd_spatial_energy`.
/// @param cycle Cycle.
/// @return Correlation energy [Hartree].
double ccsd_spatial_e_corr(const CcsdSpatialCycle& cycle);

/// @brief SCF total energy stored at open [Hartree].
/// @param cycle Cycle.
/// @return E_HF [Hartree].
double ccsd_spatial_e_scf(const CcsdSpatialCycle& cycle);

/// @brief max|Δt₁|, max|Δt₂| from the last update [dimensionless].
/// @param cycle Cycle.
/// @return Amplitude residual.
double ccsd_spatial_residual(const CcsdSpatialCycle& cycle);

/// @brief True when the reference has no virtual orbitals (zero correlation).
/// @param cycle Cycle.
/// @return Whether the amplitude equations are empty.
bool ccsd_spatial_trivial(const CcsdSpatialCycle& cycle);

} // namespace scf
} // namespace cpu
} // namespace ATV
