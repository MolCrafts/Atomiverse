/// @file cuda/planewave/internal/pw_forces.cuh
/// @brief Device analytic ionic forces for plane-wave DFT (local HF + KB).
///
/// After one CUDA SCF, F_I = F_I^loc + F_I^nl + F_I^Ewald. Local Hellmann–
/// Feynman and Kleinman–Bylander nonlocal terms run as device kernels;
/// Ewald is the shared host classical sum. See `cpu/planewave/internal/pw_forces`.
#pragma once

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/smearing.h"
#include "atomiverse/cuda/planewave/internal/device_density.cuh"
#include "atomiverse/cuda/planewave/internal/device_solver.cuh"
#include <array>
#include <string>
#include <vector>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

/// @brief Local Hellmann–Feynman forces on device: ρ(G) × v(G) × phase.
///
/// Kernel twin of CPU `accumulate_local_forces`. Accumulates into
/// `d_forces` [Hartree/Bohr] with atomicAdd. Caller zeros the buffer.
/// @param stream CUDA stream for the launch (pass `DeviceDensity::stream()`
///               so force kernels share the SCF stream).
void accumulate_local_forces_device(const cufftDoubleComplex* d_rho_G,
                                    const double* d_gx, const double* d_gy,
                                    const double* d_gz, const double* d_vg,
                                    const double* d_pos, int ng, int n_atoms,
                                    double omega, double* d_forces,
                                    cudaStream_t stream = nullptr);

/// @brief Analytic forces after a completed device SCF [eV/Å].
///
/// Local HF + KB nonlocal on device; Ewald on host. Re-solves bands at the
/// final V_eff for each k (CPU force-path twin). Restricted fixed-occ: one
/// V_eff and closed-shell f=2. Restricted metallic smear (`use_smear`): one
/// V_eff and per-band \(f_b\in[0,2]\) from a shared \(\mu\). Collinear spin
/// (`n_alpha ≥ 0`): local HF from total ρ; NL re-solves each spin channel
/// with f=1 and N_σ occupations (spin∧smear rejected upstream).
///
/// @param frame     Geometry [Å]; not mutated.
/// @param upf_path  UPF path (empty if gth_name is set).
/// @param gth_name  GTH name (empty if upf_path is set).
/// @param ecut_ry   Wavefunction cutoff [Ry] (Ewald G cut).
/// @param den       Device density after SCF mix (owns ρ(G)).
/// @param solver    Band solver bound to den grids, k-tables built.
/// @param prob      Assembled problem (same run as SCF).
/// @param ws        Workspace used for band solves (eigenvalues sink).
/// @param n_alpha   Occupied ↑ bands (−1 = restricted).
/// @param n_beta    Occupied ↓ bands (spin path only).
/// @param use_smear Restricted metallic-smear fractional NL path.
/// @param n_elec    Valence electron count (smear μ search; ignored if not smear).
std::vector<std::array<double, 3>>
pw_forces_analytic_impl(const cpu::Frame& frame, const std::string& upf_path,
                        const std::string& gth_name, double ecut_ry,
                        DeviceDensity& den, DeviceBandSolver& solver,
                        cpu::pw::internal::PWProblem& prob,
                        cpu::pw::internal::PWWorkspace& ws,
                        int n_alpha = -1, int n_beta = -1,
                        bool use_smear = false, int n_elec = 0,
                        cpu::pw::internal::SmearKind smear_kind =
                            cpu::pw::internal::SmearKind::fermi_dirac,
                        double degauss_ha = 0.0, int mp_order = 1);

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
