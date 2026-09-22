/// @file cuda/scf/device_integrals.cuh
/// @brief GPU-native molecular integral kernels for s- and p-type Gaussians.
///
/// Wave 0–3: pure-s launchers (`launch_1e_integrals` / `launch_eri`) — one AO
/// per shell, F0 Boys only.
/// Wave 4: `launch_1e_integrals_sp` / `launch_eri_sp` — Cartesian s+p (l≤1)
/// via McMurchie–Davidson / Obara–Saika on device; matrices indexed by AO.
///
/// Validated against CPU native integrals (ground truth: libcint / PySCF).
#pragma once

namespace ATV {
namespace cuda {
namespace scf {

/// @brief Launch fused 1e integral kernel for pure-s shells (1 AO / shell).
///
/// One thread per unique shell pair (i >= j). Fuses overlap, kinetic, and
/// nuclear attraction. Writes both (i,j) and (j,i) via symmetry.
///
/// @param S Output overlap [n_shells × n_shells, col-major, device].
/// @param T Output kinetic [device].
/// @param V Output nuclear attraction [device].
/// @param shell_alpha Exponents [n_shells × n_prim, device].
/// @param shell_coeff Normalized coefficients [n_shells × n_prim, device].
/// @param shell_centers Shell centers [n_shells × 3, device] (Bohr).
/// @param atom_Z Nuclear charges [natm, device].
/// @param atom_pos Atom positions [natm × 3, device] (Bohr).
/// @param n_shells Number of s shells (= n_AO).
/// @param n_prim Primitives per shell.
/// @param natm Number of atoms.
/// @param stream CUDA stream.
void launch_1e_integrals(double* S, double* T, double* V,
                         const double* shell_alpha, const double* shell_coeff,
                         const double* shell_centers, const int* atom_Z,
                         const double* atom_pos, int n_shells, int n_prim,
                         int natm, void* stream);

/// @brief Launch pure-s ERI kernel (full n_shells^4 tensor).
///
/// @param ERI Output tensor [n_shells^4, col-major, device].
/// @param shell_alpha Exponents [n_shells × n_prim, device].
/// @param shell_coeff Normalized coefficients [n_shells × n_prim, device].
/// @param shell_centers Shell centers [n_shells × 3, device] (Bohr).
/// @param n_shells Number of s shells.
/// @param n_prim Primitives per shell.
/// @param stream CUDA stream.
void launch_eri(double* ERI, const double* shell_alpha,
                const double* shell_coeff, const double* shell_centers,
                int n_shells, int n_prim, void* stream);

/// @brief Fused 1e integrals for s+p shells (l ≤ 1), AO-indexed matrices.
///
/// One thread per unique AO pair (μ ≥ ν). Shell angular momentum and AO
/// mapping come from `shell_L` / `shell_ao_start` / `ao_shell` / `ao_cart`.
///
/// @param S,T,V Output [n_ao × n_ao, col-major, device].
/// @param shell_alpha [n_shells × n_prim], phys-normed coeffs in shell_coeff.
/// @param shell_centers [n_shells × 3] Bohr.
/// @param shell_L Angular momentum per shell (0 or 1) [n_shells].
/// @param shell_ao_start First AO index of each shell [n_shells].
/// @param ao_shell Shell index for each AO [n_ao].
/// @param ao_cart Cartesian component within shell (0 for s; 0=x,1=y,2=z for p).
/// @param atom_Z [natm], atom_pos [natm×3] Bohr.
/// @param n_shells Shell count, n_ao AO count, n_prim primitives/shell, natm.
/// @param stream CUDA stream.
void launch_1e_integrals_sp(double* S, double* T, double* V,
                            const double* shell_alpha,
                            const double* shell_coeff,
                            const double* shell_centers, const int* shell_L,
                            const int* shell_ao_start, const int* ao_shell,
                            const int* ao_cart, const int* atom_Z,
                            const double* atom_pos, int n_shells, int n_ao,
                            int n_prim, int natm, void* stream);

/// @brief ERI for s+p shells (l ≤ 1), AO-indexed n_ao^4 tensor.
///
/// One thread per AO quartet (μν|λσ). Uses McMurchie–Davidson for l ≤ 1
/// (hand E-tables, Boys F_m up to m=4 for (pp|pp)).
///
/// @param ERI [n_ao^4, col-major]: ERI[μ + ν*n + λ*n² + σ*n³].
/// @param stream CUDA stream.
void launch_eri_sp(double* ERI, const double* shell_alpha,
                   const double* shell_coeff, const double* shell_centers,
                   const int* shell_L, const int* shell_ao_start,
                   const int* ao_shell, const int* ao_cart, int n_shells,
                   int n_ao, int n_prim, void* stream);

/// @brief Device Boys function F_0(x) host-side validation wrapper.
void launch_boys_f0_test(const double* d_x, double* d_out, int n, void* stream);

/// @brief Pure-s RHF analytic nuclear forces on device (Wave 5 foothold).
///
/// Assembles
///   ∂E/∂R = Tr[P ∂Hcore/∂R] − Tr[W ∂S/∂R] + ∂E₂ₑ/∂R + ∂V_nuc/∂R
/// with angular-momentum raising on s primitives (raises to p; uses the same
/// MD/OS device primitives as Wave 4). Output forces are **−∂E/∂R** in eV/Å.
///
/// Scope: every shell must be s (l = 0); `n_ao == n_shells`. Prefer
/// `launch_rhf_forces_sp` for mixed s/p (and pure-s when AO maps are live).
///
/// @param fx,fy,fz Force columns [natm], device, eV/Å (overwritten).
/// @param P Density [n_ao×n_ao, col-major], device (RHF factor-2 total).
/// @param W Energy-weighted density [n_ao×n_ao], device:
///        W = 2 Σ_{i=0}^{n_occ−1} ε_i C_{μi} C_{νi}.
/// @param shell_alpha,shell_coeff Phys-normed contraction [n_shells×n_prim].
/// @param shell_centers [n_shells×3] Bohr.
/// @param shell_atom Atom index owning each shell [n_shells].
/// @param atom_Z [natm], atom_pos [natm×3] Bohr.
/// @param n_ao AO count (= n_shells for pure-s).
/// @param n_prim Primitives per shell.
/// @param natm Atom count.
/// @param stream CUDA stream.
void launch_rhf_forces_pure_s(
    double* fx, double* fy, double* fz, const double* P, const double* W,
    const double* shell_alpha, const double* shell_coeff,
    const double* shell_centers, const int* shell_atom, const int* atom_Z,
    const double* atom_pos, int n_ao, int n_prim, int natm, void* stream);

/// @brief s+p (l ≤ 1) RHF analytic nuclear forces on device (Wave 6).
///
/// Same energy assembly as pure-s, with full raise/lower on Cartesian
/// components (p → d raise). Ends the host force fallback for STO-3G H…O.
/// Output forces are **−∂E/∂R** in eV/Å.
///
/// @param shell_L Angular momentum per shell (0 or 1) [n_shells].
/// @param ao_shell Shell index for each AO [n_ao].
/// @param ao_cart Cartesian component within shell [n_ao].
void launch_rhf_forces_sp(double* fx, double* fy, double* fz, const double* P,
                          const double* W, const double* shell_alpha,
                          const double* shell_coeff,
                          const double* shell_centers, const int* shell_atom,
                          const int* shell_L, const int* ao_shell,
                          const int* ao_cart, const int* atom_Z,
                          const double* atom_pos, int n_ao, int n_prim,
                          int natm, void* stream);

/// @brief Build W = 2 Σ_i ε_i C[:,i] C[:,i]ᵀ for i < n_occ (col-major).
void launch_build_W_rhf(double* W, const double* C, const double* eig, int n_ao,
                        int n_occ, void* stream);

} // namespace scf
} // namespace cuda
} // namespace ATV
