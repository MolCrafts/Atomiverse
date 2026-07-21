/// @file cpu/pw/internal/pw_setup.h
/// @brief Host-side plane-wave SCF problem assembly and density mixing,
///        shared by the CPU (`cpu::pw::PWSolver`) and CUDA
///        (`cuda::pw::pw_scf`) drivers.
#pragma once

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/pw/internal/g_vector.h"
#include "atomiverse/cpu/pw/internal/kpoint.h"
#include "atomiverse/cpu/pw/internal/pw_workspace.h"
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

/// @brief Compute the per-iteration energy decomposition (no I/O).
PWEnergyTerms iteration_energy(const PWWorkspace& ws, AccReal e_band,
                               AccReal ew_total, AccReal vol_bohr);

/// @brief Emit the per-iteration SCF energy summary as a rank-0-only
///        ATV_DEBUG line — the single formatter path shared by the CPU
///        (`PWSolver::run`) and CUDA (`cuda::pw::pw_scf`) SCF loops.
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
