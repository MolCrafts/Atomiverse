/// @file cuda/scf/device_integrals.cuh
/// @brief GPU-native molecular integral kernels for s-type Gaussians.
///
/// True CUDA implementations — NOT a port of CPU code.
/// Key GPU design principles:
///   - 1 thread per output element (shell pair for 1e, shell quartet for 2e)
///   - Fused S+T+V computation to share exp() calls
///   - exp() hoisted out of ERI inner loop
///   - Boys function via hardware erf() intrinsic
///   - 8-fold permutation symmetry for ERI
///
/// Validated against CPU native integrals (ground truth: libcint).
#pragma once

namespace ATV {
namespace cuda {
namespace scf {

/// @brief Launch fused 1e integral kernel (S + T + V simultaneously).
///
/// One thread per unique shell pair (i >= j). Fuses overlap, kinetic, and
/// nuclear attraction to share gamma, mu, K_AB, and Gaussian product center.
/// Writes both (i,j) and (j,i) via symmetry.
///
/// @param S Output overlap matrix [n_shells × n_shells, col-major, device].
/// @param T Output kinetic matrix [device].
/// @param V Output nuclear attraction matrix [device].
/// @param shell_alpha Exponents [n_shells × n_prim, device].
/// @param shell_coeff Normalized coefficients [n_shells × n_prim, device].
/// @param shell_centers Shell centers [n_shells × 3, device] (Bohr).
/// @param atom_Z Nuclear charges [natm, device].
/// @param atom_pos Atom positions [natm × 3, device] (Bohr).
/// @param n_shells Number of shells.
/// @param n_prim Primitives per shell.
/// @param natm Number of atoms.
/// @param stream CUDA stream.
void launch_1e_integrals(double* S, double* T, double* V,
                         const double* shell_alpha, const double* shell_coeff,
                         const double* shell_centers, const int* atom_Z,
                         const double* atom_pos, int n_shells, int n_prim,
                         int natm, void* stream);

/// @brief Launch ERI kernel (full n^4 tensor for small systems).
///
/// One thread per unique shell quartet exploiting 8-fold permutation symmetry.
/// Hoists exp(-mu*R^2) out of the inner primitive loop.
/// Uses hardware erf() for Boys function.
///
/// @param ERI Output tensor [n_shells^4, col-major, device].
/// @param shell_alpha Exponents [n_shells × n_prim, device].
/// @param shell_coeff Normalized coefficients [n_shells × n_prim, device].
/// @param shell_centers Shell centers [n_shells × 3, device] (Bohr).
/// @param n_shells Number of shells.
/// @param n_prim Primitives per shell.
/// @param stream CUDA stream.
void launch_eri(double* ERI, const double* shell_alpha,
                const double* shell_coeff, const double* shell_centers,
                int n_shells, int n_prim, void* stream);

/// @brief Device Boys function F_0(x) = 0.5 * sqrt(pi/x) * erf(sqrt(x)).
///
/// Exposed for testing. Uses CUDA hardware erf() intrinsic.
/// Host wrapper that launches a single-element kernel for validation.
void launch_boys_f0_test(const double* d_x, double* d_out, int n, void* stream);

} // namespace scf
} // namespace cuda
} // namespace ATV
