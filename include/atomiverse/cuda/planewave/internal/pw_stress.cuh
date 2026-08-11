/// @file cuda/planewave/internal/pw_stress.cuh
/// @brief Device Nielsen–Martin analytic cell stress for plane-wave DFT.
///
/// CUDA twin of `cpu::pw::internal::pw_stress_analytic_impl`: one device SCF
/// then Hellmann–Feynman strain-FD of the non-self-consistent total / free
/// energy under remapped density. Sign matches CPU / GPW / CP2K mechanical
/// \(\sigma=-(1/\Omega)\partial E/\partial\varepsilon\). Restricted metallic
/// smear uses the Mermin free energy \(F=E_{\mathrm{KS}}-TS\).
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/internal/smearing.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include <array>
#include <string>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

/// @brief Nielsen–Martin analytic stress σ [eV/Å³], Voigt xx,yy,zz,xy,xz,yz.
///
/// Runs a full device SCF at the reference geometry, then 12 device NSCF
/// evaluations under ±h symmetric strain (density remapped Ω₀/Ω). Frame is
/// not mutated. Restricted metallic smear (when `smearing != none`) uses
/// fractional occupations and the Mermin free energy on both the reference
/// SCF and the NSCF strain samples.
///
/// @param frame          Geometry [Å], types (Z), cell.
/// @param upf_path       UPF path (empty if gth_name is set).
/// @param gth_name       GTH name (empty if upf_path is set).
/// @param ecut_ry        Wavefunction cutoff [Ry].
/// @param k_mesh         Monkhorst–Pack mesh per dimension.
/// @param tol            SCF density residual tolerance.
/// @param max_iter       Max SCF iterations.
/// @param mix_beta       Density mixing parameter.
/// @param pulay_depth    Pulay history depth.
/// @param xc             XC functional (`XCType::LSDA` or `XCType::PBE`).
/// @param smearing       Metallic smear kind (`none` = fixed closed shell).
/// @param degauss        Smear width [Ry] (required > 0 when smearing).
/// @param mp_order       Methfessel–Paxton order (ignored for FD/Gaussian).
/// @param extra_bands    Empty bands beyond ceil(n_e/2) when smearing.
/// @param energy_ry_out  Optional: reference SCF free/total energy [Ry].
/// @return Intensive stress [eV/Å³] Voigt.
std::array<double, 6> pw_stress_analytic_impl(
    const cpu::Frame& frame, const std::string& upf_path,
    const std::string& gth_name, double ecut_ry, int k_mesh, double tol,
    int max_iter, double mix_beta, int pulay_depth, cpu::scf::XCType xc,
    cpu::pw::Smearing smearing, double degauss, int mp_order, int extra_bands,
    double* energy_ry_out = nullptr);

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
