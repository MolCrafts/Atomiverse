/// @file cpu/pw/internal/pw_setup.h
/// @brief Host-side plane-wave SCF problem assembly and density mixing,
///        shared by the CPU (`cpu::pw::PW`) and CUDA
///        (`cuda::pw::PW`) methods.
#pragma once

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/internal/g_vector.h"
#include "atomiverse/cpu/planewave/internal/kpoint.h"
#include "atomiverse/cpu/planewave/internal/pw_workspace.h"
#include "atomiverse/cpu/planewave/internal/upf_reader.h"
#include <string>
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Fully-assembled plane-wave SCF problem: workspace, G-spheres,
///        k-mesh and the density-independent Ewald energy.
///
/// Non-copyable: `ws.gvec` points at `gv0`, so the object must stay put
/// after `setup_pw_problem()` fills it.
struct PWProblem {
    PWWorkspace ws;
    GVectorList gv0;               ///< Γ-centred G-sphere at ecut.
    std::vector<GVectorList> gv_k; ///< Per-k-point G-spheres.
    KPointGrid kg;
    AccReal ew_total = 0; ///< Ewald energy incl. QE G=0 term [Hartree].
    AccReal vol_bohr = 0; ///< Cell volume [Bohr³].
    int n_elec = 0;
    int n_bands = 0;

    /// Per-k-point nonlocal projector / atomic-wfc tables, evaluated on
    /// q = k+G of each sphere (size and values are k-dependent). The SCF
    /// loop copies entry ik into ws.kb_proj / ws.atom_wfc before solving
    /// that k-point. Empty for the toy potential.
    std::vector<std::vector<std::vector<PWWorkspace::KBProjG>>> kb_proj_k;
    std::vector<std::vector<std::vector<PWWorkspace::AtomWfc>>> atom_wfc_k;

    PWProblem() = default;
    PWProblem(const PWProblem&) = delete;
    PWProblem& operator=(const PWProblem&) = delete;
};

/// @brief Number of valence electrons assumed for nuclear charge `z`
///        (H..Ar) when no UPF pseudopotential supplies it.
int valence_from_z(int z);

/// @brief Radial → G-space form factor v(G) of the UPF local potential
///        [Hartree], one value per G of `gv` (QE `vloc_mod.f90` route:
///        erf-screened quadrature + analytic screened-Coulomb transform;
///        the G = 0 entry carries the finite "αZ" part only).
///
/// Per-atom quantity (no structure factor). Shared by problem assembly
/// (`setup_pw_problem`) and the DFPT perturbation builder — dividing the
/// assembled `V_loc_G` by the structure factor is ill-defined on shells
/// where S(G) = 0 (e.g. diamond).
///
/// @param upf   Parsed UPF data.
/// @param gv    G-vector list to evaluate on.
/// @param omega Cell volume [Bohr³].
std::vector<AccReal> upf_vloc_to_gspace(const UPFData& upf,
                                        const GVectorList& gv, AccReal omega);

/// @brief Assemble the complete SCF problem from a Frame: Ewald energy,
///        G-vectors, k-mesh, FFT grids, V_loc (UPF or toy), KB projectors,
///        atomic wavefunctions and the initial density.
///
/// This is the entire pre-loop section of the original monolithic solver; both
/// backends call it and then run their own SCF loop.
///
/// @param frame    Input structure (positions in Å, type = Z).
/// @param upf_path UPF pseudopotential file ("" = toy V_loc).
/// @param ecut_ry  Plane-wave cutoff [Ry].
/// @param k_mesh   Monkhorst-Pack mesh size per dimension.
/// @param out      Problem to fill (must outlive the SCF loop).
void setup_pw_problem(const cpu::Frame& frame, const std::string& upf_path,
                      double ecut_ry, int k_mesh, PWProblem& out);

/// @brief Replace only the k-mesh tables on an assembled problem.
///
/// Keeps density-side fields on `out.ws` (ρ, \(V_\mathrm{eff}\), \(V_\mathrm{loc}\),
/// FFT) frozen for NSCF band / DOS diagonalization. Rebuilds
/// `out.kg`, `out.gv_k`, `out.kb_proj_k`, `out.atom_wfc_k` for the given
/// fractional k-list.
///
/// @param out      Problem from a prior `setup_pw_problem` (density frozen).
/// @param frame    Geometry (positions [Å], types); used for KB structure factors.
/// @param upf_path UPF path ("" = toy — no KB tables).
/// @param ecut_ry  Wavefunction cutoff [Ry] (same as SCF).
/// @param kfrac    Fractional (crystal) k-coordinates.
void rebuild_k_tables(PWProblem& out, const cpu::Frame& frame,
                      const std::string& upf_path, double ecut_ry,
                      const std::vector<Vec3>& kfrac);

/// @brief Solve the Pulay/DIIS constrained least-squares coefficients.
///
/// Builds and solves the augmented Lagrangian system
/// \([B\;\mathbf{1};\;\mathbf{1}^T\;0]\,[c;\lambda] = [0;1]\) by Gaussian
/// elimination with partial pivoting, so \(\sum_i c_i = 1\). Singular pivots
/// (|a_kk| < 1e-14) are skipped, leaving the corresponding coefficient at 0 —
/// the same guarded behaviour both backends must reproduce exactly.
///
/// Shared by `pulay_mix` (CPU) and `cuda::pw::internal::DeviceDensity::mix`
/// (CUDA), which builds the same Gram matrix with one `cublasZgemm` and then
/// calls this host solve on the ≤ 8×8 result.
///
/// @param B     Row-major m×m Gram matrix
///              \(B_{ij}=\mathrm{Re}\langle F_i|F_j\rangle\).
/// @param m     History length (≥ 1).
/// @param c_out Output coefficients, m entries (dimensionless).
void solve_pulay_coefficients(const AccReal* B, int m, AccReal* c_out);

/// @brief Pulay/DIIS density mixing on the G-sphere.
///
/// Expects `ws.rho_G` = output density of the current iteration and
/// `ws.rho_G_old` = input density; replaces `ws.rho_G` with the mixed
/// density and updates the Pulay history.
void pulay_mix(PWWorkspace& ws);

/// @brief Per-iteration SCF energy decomposition [Hartree].
///
/// Parseval contractions use the full Re[conj(rho_G)*V_G] product (both
/// real and imaginary parts of rho contribute for cells without
/// inversion symmetry about the origin).
struct PWEnergyTerms {
    AccReal e_one = 0;    ///< One-electron energy (band - 2*E_H - rho*Vxc).
    AccReal e_h = 0;      ///< Hartree energy.
    AccReal e_xc_int = 0; ///< Integral of rho * V_xc (decomposition term).
    AccReal e_loc = 0;    ///< Integral of rho * V_loc (diagnostic only).
    AccReal total = 0;    ///< Total energy incl. E_xc, Ewald, augmentation.
};

/// @brief Combine already-integrated SCF energy pieces into the decomposition.
///
/// The one place `e_one` and `total` are formed. `e_one` is the
/// catastrophic-cancellation site (band sum minus the double-counted
/// Hartree and XC potentials), so its term order is part of the contract
/// and must not be reassociated by any caller.
///
/// Shared by `iteration_energy` (CPU, after its Parseval G-loop) and
/// `cuda::pw::PW::run`, which supplies the same integrals reduced on the
/// device — there is exactly one definition of the decomposition.
///
/// @param e_band    Occupation-weighted band-energy sum Σ f_i ε_i [Hartree].
/// @param e_h       Hartree energy ½Ω Σ_G Re[ρ*V_H] [Hartree].
/// @param e_xc_int  Integral ∫ρ V_xc = Ω Σ_G Re[ρ*V_xc] [Hartree].
/// @param e_loc     Integral ∫ρ V_loc [Hartree] (diagnostic only).
/// @param e_xc      XC energy ∫ε_xc[ρ]ρ d³r [Hartree] (`ws.E_xc`).
/// @param ew_total  Ewald energy incl. the QE G=0 term [Hartree].
/// @param e_aug     Augmentation-charge energy [Hartree] (`ws.E_aug`).
/// @return Filled decomposition [Hartree].
PWEnergyTerms combine_energy_terms(AccReal e_band, AccReal e_h,
                                   AccReal e_xc_int, AccReal e_loc,
                                   AccReal e_xc, AccReal ew_total,
                                   AccReal e_aug);

/// @brief Compute the per-iteration energy decomposition (no I/O).
PWEnergyTerms iteration_energy(const PWWorkspace& ws, AccReal e_band,
                               AccReal ew_total, AccReal vol_bohr);

/// @brief Emit the per-iteration SCF energy summary as a rank-0-only
///        ATV_DEBUG line — the single formatter path shared by the CPU
///        (`PW::run`) and CUDA (`cuda::pw::PW::run`) SCF loops.
///
/// Host-compiled on purpose: `core/log.h` is CUDA-free, so `.cu` callers
/// delegate their logging here instead of including spdlog directly.
///
/// @param terms    Energy decomposition from `iteration_energy()` [Hartree].
/// @param ws       Workspace supplying the XC energy `E_xc` [Hartree].
/// @param ew_total Ewald energy incl. the QE G=0 term [Hartree].
/// @param iter     Zero-based SCF iteration index (logged as `iter + 1`).
void log_iteration_energy(const PWEnergyTerms& terms, const PWWorkspace& ws,
                          AccReal ew_total, int iter);

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
