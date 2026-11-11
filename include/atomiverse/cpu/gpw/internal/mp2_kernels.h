/// @file cpu/gpw/internal/mp2_kernels.h
/// @brief Private GPW MP2 kernels: MO values on the density grid, the
///        occupied–virtual MO repulsion tensor via per-pair Poisson solves,
///        the canonical closed-shell (RHF) and unrestricted (UHF) doubles
///        sums, cross-spin (ia|jb), and multi-k (complex Bloch)
///        momentum-conserving RI-GPW MP2 (closed-shell and unrestricted).
///
/// Private workspace behind `GpwDft::mp2` — never on the public include
/// path (unit tests may include it). The `(ia|jb)` route is CP2K's
/// Quickstep wavefunction-correlation integral scheme: collocate
/// \(\rho_{ia}=\psi_i\psi_a\), solve Poisson, contract with
/// \(\rho_{jb}\) by grid quadrature. The same tensor + denominators feed
/// the periodic RPA follow-on (`direct_rpa_rhf_energy`).
#pragma once

#include <array>
#include <complex>
#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h" // CompactBasisGrid, UniformGrid
#include "atomiverse/cpu/gpw/internal/kpoints.h" // KpointBasisGrid, CplxMatrix
#include "atomiverse/cpu/linalg/matrix.h"        // RealMatrix, CplxMatrix

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {

using Cplx = std::complex<double>;

/// @brief Evaluate MO values ψ_p(r_g) on the grid from the compact
///        (lattice-summed, CSR-by-point) AO representation.
/// @param cbg Compact AO grid from `build_compact_basis_grid` (periodic
///        Γ-image-summed or isolated, matching the SCF).
/// @param C  MO coefficients (nbasis × n_mo, columns = orbitals).
/// @param p0 First MO column to evaluate.
/// @param np Number of MO columns.
/// @return Row-major (np × npts): value of MO `p0+p` at grid point g is
///         `out[p*npts + g]`.
std::vector<double> mo_on_grid(const CompactBasisGrid& cbg,
                               const linalg::RealMatrix& C, int p0, int np);

/// @brief Occupied–virtual MO repulsion tensor (ia|jb) via per-pair Poisson.
///
/// For each pair density ρ_ia(r)=ψ_i(r)ψ_a(r) one Poisson solve gives
/// V_ia(r); then (ia|jb) = Σ_g w V_ia(g) ρ_jb(g). Uses the (ia|jb)=(jb|ia)
/// symmetry — n_occ·n_virt Poisson solves total. G=0 is dropped by the
/// periodic solver (neutral pair densities, S_ia = 0). Canonical foothold
/// path (`mp2_ri = false`); product path uses `mo_eri_ovov_ri`.
/// @param grid     Uniform density grid (as in the SCF).
/// @param psi_occ  Occupied MO values, row-major (n_occ × npts) [mo_on_grid].
/// @param n_occ    Occupied count.
/// @param psi_virt Virtual MO values, row-major (n_virt × npts).
/// @param n_virt   Virtual count.
/// @param nonperiodic true → truncated-Coulomb (MT-class) isolated solver.
/// @return Row-major ((n_occ·n_virt) × (n_occ·n_virt)) tensor,
///         `(ia|jb) = out[(i*n_virt+a)*n_occ*n_virt + j*n_virt+b]` [Hartree].
std::vector<double> mo_eri_ovov(const UniformGrid& grid,
                                const std::vector<double>& psi_occ, int n_occ,
                                const std::vector<double>& psi_virt, int n_virt,
                                bool nonperiodic);

/// @brief Periodic RI-GPW (ia|jb) via plane-wave Coulomb intermediates.
///
/// Del Ben / Quickstep product path: for each pair density ρ_ia form the
/// G-space Coulomb potential (4π/|G|², G=0 dropped) via the shared FFT
/// Poisson kernel, store the three-index real-space intermediate V_ia
/// (memory \(\mathcal O(n_{\mathrm{ov}} N_g)\)), then contract
/// \((ia|jb)=\sum_g w\, V_{ia}(g)\,\rho_{jb}(g)\). Numerically identical to
/// `mo_eri_ovov(..., nonperiodic=false)` on the same grid — the PW
/// resolution of the identity for the pair density. Periodic only (isolated
/// cells keep the MT Poisson foothold via `mo_eri_ovov`).
/// @param grid     Uniform density grid (as in the SCF).
/// @param psi_occ  Occupied MO values, row-major (n_occ × npts).
/// @param n_occ    Occupied count.
/// @param psi_virt Virtual MO values, row-major (n_virt × npts).
/// @param n_virt   Virtual count.
/// @return Same layout as `mo_eri_ovov` [Hartree].
std::vector<double>
mo_eri_ovov_ri(const UniformGrid& grid, const std::vector<double>& psi_occ,
               int n_occ, const std::vector<double>& psi_virt, int n_virt);

/// @brief Canonical closed-shell MP2 doubles sum (real orbitals).
///
/// E_OS = Σ (ia|jb)² / (ε_i+ε_j−ε_a−ε_b),
/// E_SS = Σ (ia|jb)[(ia|jb)−(ib|ja)] / (ε_i+ε_j−ε_a−ε_b); both ≤ 0 for a
/// gapped spectrum. E_corr(unscaled) = E_OS + E_SS.
/// @param v_ovov   (ia|jb) tensor from `mo_eri_ovov` [Hartree].
/// @param eps_occ  Occupied orbital energies (length n_occ) [Hartree].
/// @param eps_virt Virtual orbital energies (length n_virt) [Hartree].
/// @param e_os Out: unscaled opposite-spin part [Hartree].
/// @param e_ss Out: unscaled same-spin part [Hartree].
void mp2_rhf_energy(const std::vector<double>& v_ovov,
                    const std::vector<double>& eps_occ,
                    const std::vector<double>& eps_virt, double& e_os,
                    double& e_ss);

/// @brief Cross-spin occupied–virtual MO repulsion tensor
///        \((i_\alpha a_\alpha|j_\beta b_\beta)\) via per-pair Poisson.
///
/// Layout row-major
/// `(nov_a × nov_b)`:
/// `(ia|jb) = out[(i·n_virt_a+a)·nov_b + j·n_virt_b+b]`. No
/// \((ia|jb)=(jb|ia)\) fill — the two spin channels may differ in size.
/// @param nonperiodic true → MT isolated Poisson; false → periodic FFT.
std::vector<double>
mo_eri_ovov_ab(const UniformGrid& grid, const std::vector<double>& psi_occ_a,
               int n_occ_a, const std::vector<double>& psi_virt_a, int n_virt_a,
               const std::vector<double>& psi_occ_b, int n_occ_b,
               const std::vector<double>& psi_virt_b, int n_virt_b,
               bool nonperiodic);

/// @brief Periodic RI-GPW cross-spin \((i_\alpha a_\alpha|j_\beta b_\beta)\).
///
/// Same layout as `mo_eri_ovov_ab`; numerically matches the Poisson path
/// on a periodic cell (Parseval / PW resolution of the pair density).
std::vector<double>
mo_eri_ovov_ab_ri(const UniformGrid& grid, const std::vector<double>& psi_occ_a,
                  int n_occ_a, const std::vector<double>& psi_virt_a,
                  int n_virt_a, const std::vector<double>& psi_occ_b,
                  int n_occ_b, const std::vector<double>& psi_virt_b,
                  int n_virt_b);

/// @brief Canonical unrestricted MP2 doubles sum (real orbitals; Γ).
///
/// Same-spin αα / ββ:
/// \(E_{\mathrm{SS}}^\sigma = \tfrac14\sum_{ijab}
///   [(ia|jb)-(ib|ja)]^2 / \Delta_{ij}^{ab}\)
/// (molecular `ump2_energy` / PySCF UMP2 convention). Opposite-spin αβ:
/// \(E_{\mathrm{OS}} = \sum_{ijab} (i_\alpha a_\alpha|j_\beta b_\beta)^2
///   / \Delta\) (no exchange). Empty spin channels (n_occ or n_virt = 0)
/// contribute zero; same-spin needs ≥2 occupied and ≥2 virtual.
/// @param v_aa Same-spin α (ia|jb) [Ha], layout as `mo_eri_ovov` (may be empty).
/// @param v_bb Same-spin β (ia|jb) [Ha] (may be empty).
/// @param v_ab Cross-spin (iα aα|jβ bβ) [Ha], layout as `mo_eri_ovov_ab`.
/// @param eps_occ_a / eps_virt_a α orbital energies [Ha].
/// @param eps_occ_b / eps_virt_b β orbital energies [Ha].
/// @param e_os Out: unscaled opposite-spin [Hartree].
/// @param e_ss Out: unscaled same-spin (αα+ββ) [Hartree].
void mp2_uhf_energy(
    const std::vector<double>& v_aa, const std::vector<double>& v_bb,
    const std::vector<double>& v_ab, const std::vector<double>& eps_occ_a,
    const std::vector<double>& eps_virt_a, const std::vector<double>& eps_occ_b,
    const std::vector<double>& eps_virt_b, double& e_os, double& e_ss);

/// @brief Complex Bloch MO values on the density grid from pre-summed
///        \(\chi_\mu^k(g)\).
/// @param kbg K-point basis grid. @param ik k index.
/// @param C   MO coefficients (nb × n_mo). @param p0 first column. @param np count.
/// @return Row-major (np × npts) complex MO values.
std::vector<Cplx> mo_on_grid_k(const KpointBasisGrid& kbg, int ik,
                               const linalg::CplxMatrix& C, int p0, int np);

/// @brief Multi-k closed-shell RI-GPW MP2 (momentum-conserving).
///
/// Equal-weight formula \(E = N_k^{-1}\sum_{k_i k_j k_a}\sum_{ijab}
/// (\cdots)/\Delta\) with \(k_b \equiv k_i+k_j-k_a\) (fractional wrap into the
/// sampled set). Complex pair densities use linear real Poisson on Re/Im.
/// OS/SS reduce to the real Γ formulas when all orbitals are real.
/// @param grid      Uniform density grid (as in the SCF).
/// @param psi_occ_k Per-k occupied MO grids [nk][n_occ·npts].
/// @param psi_virt_k Per-k virtual MO grids [nk][n_virt·npts].
/// @param eps_occ_k Per-k occupied eigenvalues [nk][n_occ].
/// @param eps_virt_k Per-k virtual eigenvalues [nk][n_virt].
/// @param k_frac    Fractional k coordinates (length nk).
/// @param k_weights Quadrature weights (Σ = 1; equal-weight foothold).
/// @param e_os Out: unscaled opposite-spin [Hartree].
/// @param e_ss Out: unscaled same-spin [Hartree].
void mp2_rhf_energy_kpoints(const UniformGrid& grid,
                            const std::vector<std::vector<Cplx>>& psi_occ_k,
                            const std::vector<std::vector<Cplx>>& psi_virt_k,
                            const std::vector<std::vector<double>>& eps_occ_k,
                            const std::vector<std::vector<double>>& eps_virt_k,
                            const std::vector<std::array<double, 3>>& k_frac,
                            const std::vector<double>& k_weights, double& e_os,
                            double& e_ss);

/// @brief Multi-k unrestricted RI-GPW MP2 (momentum-conserving; complex
///        Bloch α/β).
///
/// Same-spin αα / ββ use the molecular UMP2 convention
/// \(E_{\mathrm{SS}}^\sigma=\tfrac14\sum |v_{iajb}-v_{ibja}|^2/\Delta\);
/// opposite-spin αβ uses \(\sum |v_{i_\alpha a_\alpha j_\beta b_\beta}|^2
/// /\Delta\) (no exchange). Momentum map \(k_b\equiv k_i+k_j-k_a\) is shared
/// with the closed-shell multi-k path. When \(C^\alpha\equiv C^\beta\) and
/// occupations match, OS/SS recover multi-k RMP2.
/// @param grid         Uniform density grid (as in the SCF).
/// @param psi_occ_a_k  Per-k α occupied MO grids [nk][n_oa·npts].
/// @param psi_virt_a_k Per-k α virtual MO grids [nk][n_va·npts].
/// @param psi_occ_b_k  Per-k β occupied MO grids [nk][n_ob·npts].
/// @param psi_virt_b_k Per-k β virtual MO grids [nk][n_vb·npts].
/// @param eps_occ_a_k / eps_virt_a_k α eigenvalues [Ha].
/// @param eps_occ_b_k / eps_virt_b_k β eigenvalues [Ha].
/// @param k_frac       Fractional k coordinates (length nk).
/// @param k_weights    Quadrature weights (Σ = 1).
/// @param e_os Out: unscaled opposite-spin [Hartree].
/// @param e_ss Out: unscaled same-spin (αα+ββ) [Hartree].
void mp2_uhf_energy_kpoints(
    const UniformGrid& grid, const std::vector<std::vector<Cplx>>& psi_occ_a_k,
    const std::vector<std::vector<Cplx>>& psi_virt_a_k,
    const std::vector<std::vector<Cplx>>& psi_occ_b_k,
    const std::vector<std::vector<Cplx>>& psi_virt_b_k,
    const std::vector<std::vector<double>>& eps_occ_a_k,
    const std::vector<std::vector<double>>& eps_virt_a_k,
    const std::vector<std::vector<double>>& eps_occ_b_k,
    const std::vector<std::vector<double>>& eps_virt_b_k,
    const std::vector<std::array<double, 3>>& k_frac,
    const std::vector<double>& k_weights, double& e_os, double& e_ss);

/// @brief Closed-shell direct RPA (dRPA) correlation via the Furche
///        plasmon formula on the Coulomb-only (\(\alpha_{\mathrm{hf}}=0\))
///        singlet Casida spectrum.
///
/// Builds
/// \(A_{ia,jb}=\delta_{ij}\delta_{ab}(\varepsilon_a-\varepsilon_i)+2(ia|jb)\),
/// \(B_{ia,jb}=2(ia|jb)\) (real orbitals; composite index
/// \(p = i\cdot n_{\mathrm{virt}}+a\) matching `mo_eri_ovov` layout),
/// reduces \((A-B)^{1/2}(A+B)(A-B)^{1/2}\), and returns
/// \(E_c=\tfrac12(\sum_s\Omega_s-\mathrm{Tr}\,A)\). Same science as
/// molecular `OrbitalHessian::direct_rpa(singlet, /*alpha_hf=*/0)`.
///
/// @param v_ovov   (ia|jb) tensor from `mo_eri_ovov` / `mo_eri_ovov_ri` [Ha].
/// @param eps_occ  Occupied orbital energies (length n_occ) [Hartree].
/// @param eps_virt Virtual orbital energies (length n_virt) [Hartree].
/// @param e_corr   Out: dRPA correlation energy [Hartree].
/// @param tr_A     Out: \(\mathrm{Tr}\,A\) [Hartree].
/// @param omega    Out: positive \(\Omega_s\) ascending [Hartree], length
///                 \(n_{\mathrm{occ}}\cdot n_{\mathrm{virt}}\).
/// @throws std::runtime_error when (A−B) is not PD or any \(\Omega^2\le 0\).
void direct_rpa_rhf_energy(const std::vector<double>& v_ovov,
                           const std::vector<double>& eps_occ,
                           const std::vector<double>& eps_virt, double& e_corr,
                           double& tr_A, std::vector<double>& omega);

/// @brief Multi-k closed-shell RI-GPW direct RPA (dRPA) via **finite-\(q\)
///        inter-\(k\)** Furche plasmon formula on complex Bloch orbitals.
///
/// Product multi-k path: unique transfers \(q=\mathrm{wrap}(k_a-k_i)\) over
/// the sampled mesh (Γ-avoiding MP is not a group under addition). For each
/// \(q\), builds the Coulomb-only (\(\alpha_{\mathrm{hf}}=0\)) Casida spectrum
/// from transitions \(i_k\to a_{k+q}\) for every \(k\) with \(k+q\) on the
/// mesh, with pair densities \(\rho_{ia}^{k,q}=\psi_{ik}^*\psi_{a,k+q}\) and
/// RI-GPW Poisson at transfer \(q\) (\(G=0\) dropped only at \(q=0\)). Reduces
/// \((A-B)^{1/2}(A+B)(A-B)^{1/2}\) with \(A-B=\mathrm{diag}(\Delta\varepsilon)\)
/// and averages
/// \(E_c=\sum_q w_q\,\tfrac12\bigl(\sum_s\Omega_{q,s}-\mathrm{Tr}\,A_q\bigr)\)
/// with \(w_q=n_{\mathrm{src}}(q)/N_k^2\) (Σ_q w_q = 1). Recovers the real Γ
/// formula when \(N_k=1\) and orbitals are real.
///
/// @param grid       Uniform density grid (as in the SCF).
/// @param psi_occ_k  Per-k occupied MO grids [nk][n_occ·npts].
/// @param psi_virt_k Per-k virtual MO grids [nk][n_virt·npts].
/// @param eps_occ_k  Per-k occupied eigenvalues [nk][n_occ].
/// @param eps_virt_k Per-k virtual eigenvalues [nk][n_virt].
/// @param k_frac     Fractional k coordinates (length nk).
/// @param k_weights  Quadrature weights (Σ = 1; shape check / future use).
/// @param e_corr Out: dRPA correlation energy [Hartree].
/// @param tr_A   Out: \(\sum_q w_q\,\mathrm{Tr}\,A_q\) [Hartree].
/// @param omega  Out: positive \(\Omega_{q,s}\) across all q-blocks, ascending
///               [Ha]; length \(\sum_q n_{\mathrm{src}}(q)\cdot n_{\mathrm{ov}}\)
///               (= \(N_k^2 n_{\mathrm{ov}}\) when every ordered pair lands).
/// @throws std::invalid_argument on shape / weight errors.
/// @throws std::runtime_error when (A−B) not PD or any \(\Omega^2\le 0\).
void direct_rpa_rhf_energy_kpoints(
    const UniformGrid& grid, const std::vector<std::vector<Cplx>>& psi_occ_k,
    const std::vector<std::vector<Cplx>>& psi_virt_k,
    const std::vector<std::vector<double>>& eps_occ_k,
    const std::vector<std::vector<double>>& eps_virt_k,
    const std::vector<std::array<double, 3>>& k_frac,
    const std::vector<double>& k_weights, double& e_corr, double& tr_A,
    std::vector<double>& omega);

/// @brief Multi-k **finite-\(q\) inter-\(k\) unrestricted dRPA** (URPA)
///        correlation energy (spin-conserving α→α ⊕ β→β).
///
/// Same q-mesh / source-k construction as
/// `direct_rpa_rhf_energy_kpoints`. Composite OV per source \(k\): α block
/// then β (matches `direct_rpa_uhf` / multi-k UGW dielectric). Coulomb-only
/// Casida with spins explicit (\(A=\Delta\varepsilon+V\), \(B=V\) — no
/// closed-shell factor 2). Furche-averaged
/// \(E_c=\sum_q w_q\,\tfrac12\bigl(\sum_s\Omega_{q,s}-\mathrm{Tr}\,A_q\bigr)\)
/// with \(w_q=n_{\mathrm{src}}(q)/N_k^2\). When \(N_k=1\) and orbitals are
/// real-valued complex with \(C^\alpha=C^\beta\), recovers
/// `direct_rpa_uhf` / closed-shell `direct_rpa_rhf_energy`.
///
/// @param psi_occ_a_k / psi_virt_a_k Per-k α MO grids [nk][n_oσ·npts].
/// @param psi_occ_b_k / psi_virt_b_k Per-k β MO grids (empty channel OK).
/// @param eps_occ_a_k / eps_virt_a_k Per-k α eigenvalues [Ha].
/// @param eps_occ_b_k / eps_virt_b_k Per-k β eigenvalues [Ha].
/// @param k_frac / k_weights Same as RHF multi-k (Σ weights = 1).
/// @param e_corr Out: dRPA \(E_c\) [Hartree].
/// @param tr_A   Out: \(\sum_q w_q\,\mathrm{Tr}\,A_q\) [Hartree].
/// @param omega  Out: positive \(\Omega_{q,s}\) across all q-blocks,
///               ascending [Ha]; length
///               \(\sum_q n_{\mathrm{src}}(q)\cdot(n_{\mathrm{ov}}^\alpha
///               +n_{\mathrm{ov}}^\beta)\).
/// @throws std::invalid_argument on shape / empty both channels.
/// @throws std::runtime_error when (A−B) not PD or any \(\Omega^2\le 0\).
void direct_rpa_uhf_energy_kpoints(
    const UniformGrid& grid, const std::vector<std::vector<Cplx>>& psi_occ_a_k,
    const std::vector<std::vector<Cplx>>& psi_virt_a_k,
    const std::vector<std::vector<Cplx>>& psi_occ_b_k,
    const std::vector<std::vector<Cplx>>& psi_virt_b_k,
    const std::vector<std::vector<double>>& eps_occ_a_k,
    const std::vector<std::vector<double>>& eps_virt_a_k,
    const std::vector<std::vector<double>>& eps_occ_b_k,
    const std::vector<std::vector<double>>& eps_virt_b_k,
    const std::vector<std::array<double, 3>>& k_frac,
    const std::vector<double>& k_weights, double& e_corr, double& tr_A,
    std::vector<double>& omega);

/// @brief Full dRPA result including X/Y amplitudes (G0W0 screening).
///
/// Layout matches `mo_eri_ovov`: composite OV index
/// \(ia = i\cdot n_{\mathrm{virt}}+a\). Matrices `X`/`Y` are row-major
/// `[s, ia]` with shape \(n_{\mathrm{ov}}\times n_{\mathrm{ov}}\)
/// (state s, OV component ia). Same science as molecular
/// `OrbitalHessian::direct_rpa` (including the \(X,Y\) reconstruction).
struct DirectRpaAmplitudes {
    double e_corr = 0.0;       ///< Furche \(E_c\) [Hartree].
    double tr_A = 0.0;         ///< \(\mathrm{Tr}\,A\) [Hartree].
    std::vector<double> omega; ///< \(\Omega_s\) ascending [Hartree].
    std::vector<double> X;     ///< row-major [s, ia].
    std::vector<double> Y;     ///< row-major [s, ia].
};

/// @brief dRPA Hermitian reduction with X/Y amplitudes for spectral G0W0.
/// @see direct_rpa_rhf_energy
void direct_rpa_rhf(const std::vector<double>& v_ovov,
                    const std::vector<double>& eps_occ,
                    const std::vector<double>& eps_virt,
                    DirectRpaAmplitudes& out);

/// @brief Spin-conserving unrestricted dRPA (α→α ⊕ β→β) with X/Y amplitudes
///        for spectral UGW screening.
///
/// Composite OV index: α block first
/// \(ia_\alpha = i\cdot n_{\mathrm{virt}}^\alpha+a\), then β
/// \(ia_\beta = n_{\mathrm{ov}}^\alpha + i\cdot n_{\mathrm{virt}}^\beta+a\).
/// Coulomb-only Casida:
/// \(A=\Delta\varepsilon\,\delta+V\), \(B=V\) with
/// \(V_{ia\sigma,jb\tau}=(ia\sigma|jb\tau)\) — **no** closed-shell factor 2
/// (spins explicit). Same-spin blocks from `mo_eri_ovov` / `mo_eri_ovov_ri`;
/// cross-spin from `mo_eri_ovov_ab` / `mo_eri_ovov_ab_ri`. Empty spin
/// channels (n_occ or n_virt = 0) contribute no OV rows; at least one
/// non-empty channel is required. When \(C^\alpha=C^\beta\) and
/// \(n_\alpha=n_\beta\), Furche \(E_c\) dual-gates `direct_rpa_rhf` on the
/// spatial tensor (charge modes ≡ closed-shell singlet dRPA; spin modes
/// cancel in the plasmon formula).
///
/// @param v_aa Same-spin α (ia|jb) [Ha] (may be empty).
/// @param v_bb Same-spin β (ia|jb) [Ha] (may be empty).
/// @param v_ab Cross-spin (iα aα|jβ bβ) [Ha], layout as `mo_eri_ovov_ab`
///             (may be empty if either channel is empty).
/// @param eps_occ_a / eps_virt_a α orbital energies [Ha].
/// @param eps_occ_b / eps_virt_b β orbital energies [Ha].
/// @param out  Amplitudes; `X`/`Y` row-major `[s, iaσ]` over the composite
///             OV space (α then β).
/// @throws std::invalid_argument on size mismatch or empty both channels.
/// @throws std::runtime_error when (A−B) is not PD or any \(\Omega^2\le 0\).
void direct_rpa_uhf(
    const std::vector<double>& v_aa, const std::vector<double>& v_bb,
    const std::vector<double>& v_ab, const std::vector<double>& eps_occ_a,
    const std::vector<double>& eps_virt_a, const std::vector<double>& eps_occ_b,
    const std::vector<double>& eps_virt_b, DirectRpaAmplitudes& out);

/// @brief Real-space Coulomb potentials \(V_{ia}(r)\) of every OV pair density
///        \(\rho_{ia}=\psi_i\psi_a\) (periodic FFT Poisson, G=0 dropped).
///
/// Shared intermediate for RI-style `(ia|jb)` and spectral G0W0 transition
/// densities. Memory \(\mathcal O(n_{\mathrm{ov}} N_g)\).
/// @return Row-major `(n_occ·n_virt) × npts` potentials [Hartree].
std::vector<double> mo_pair_potentials_ov(const UniformGrid& grid,
                                          const std::vector<double>& psi_occ,
                                          int n_occ,
                                          const std::vector<double>& psi_virt,
                                          int n_virt);

/// @brief Diagonal HF exchange \(\Sigma_{x,p}=-\sum_i^{\mathrm{occ}}(pi|pi)\)
///        [Hartree] via truncated (MT) Coulomb on pair densities
///        \(\rho_{pi}=\psi_p\psi_i\) (G=0-safe Γ foothold).
/// @param psi_mo All MO values, row-major `(n_mo × npts)`.
/// @param n_mo   Total spatial MOs.
/// @param n_occ  Occupied count.
/// @return Length-`n_mo` vector of \(\Sigma_{x,p}\) [Hartree].
std::vector<double> exchange_diag_mt(const UniformGrid& grid,
                                     const std::vector<double>& psi_mo,
                                     int n_mo, int n_occ);

/// @brief Multi-k **finite-\(q\) inter-\(k\)** spectral G0W0 quasiparticle
///        energies (product multi-k path).
///
/// Builds the Coulomb-only Casida spectrum for each transfer
/// \(q=\mathrm{wrap}(k_a-k_i)\) from transitions \(i_k\to a_{k+q}\) (same
/// dielectric as `direct_rpa_rhf_energy_kpoints`), keeps complex \(X+Y\)
/// amplitudes, forms spectral transition potentials \(W_s^q\), and for each
/// target \(k\) contracts intermediates at \(k-q\) into
/// \(\mathrm{Re}\,\Sigma_c\) with weight \(1/N_k\). MT exchange diagonals and
/// local \(v_{\mathrm{xc}}\) diagonals stay per-\(k\). Newton QP
/// orbit-by-orbit. Recovers the real Γ spectral path when \(N_k=1\) and
/// orbitals are real-valued complex.
///
/// @param grid        Uniform density grid (as in the SCF).
/// @param psi_mo_k    Per-k full MO grids [nk][n_mo·npts].
/// @param eps_k       Per-k eigenvalues [nk][n_mo] [Hartree].
/// @param k_frac      Fractional k coordinates (length nk).
/// @param k_weights   Quadrature weights (Σ = 1; shape check).
/// @param n_occ       Occupied spatial MOs.
/// @param n_virt      Virtuals in the dRPA screening space (prefix of the
///                    virtual block; QP covers all `n_mo`).
/// @param vxc_r       Local XC potential on the grid [Hartree].
/// @param qp_energy_k Out: QP energies [nk][n_mo] [Hartree].
/// @param sigma_x_k   Out: MT exchange diagonals [nk][n_mo] [Hartree].
/// @param vxc_diag_k  Out: \(v_{\mathrm{xc}}\) diagonals [nk][n_mo] [Hartree].
/// @param all_converged Out: true when every Newton solve succeeded.
/// @throws std::invalid_argument on shape / weight errors.
/// @throws std::runtime_error when dRPA reduction is ill-conditioned.
void spectral_g0w0_rhf_kpoints(const UniformGrid& grid,
                               const std::vector<std::vector<Cplx>>& psi_mo_k,
                               const std::vector<std::vector<double>>& eps_k,
                               const std::vector<std::array<double, 3>>& k_frac,
                               const std::vector<double>& k_weights, int n_occ,
                               int n_virt, const std::vector<double>& vxc_r,
                               std::vector<std::vector<double>>& qp_energy_k,
                               std::vector<std::vector<double>>& sigma_x_k,
                               std::vector<std::vector<double>>& vxc_diag_k,
                               bool& all_converged);

/// @brief Multi-k **finite-\(q\) inter-\(k\) unrestricted spectral G0W0**
///        (UGW) quasiparticle energies on UKS Bloch orbitals.
///
/// Spin-conserving dRPA over α→α ⊕ β→β transitions \(i_k\to a_{k+q}\)
/// (Coulomb-only Casida; spins explicit — no closed-shell factor 2),
/// spectral \(W_s^q\) without \(\sqrt{2}\), same-spin \(G_\alpha W\) for the
/// **α-channel** correlation self-energy, per-\(k\) MT \(\Sigma_x^\alpha\)
/// and local \(v_{\mathrm{xc}}^\alpha\) diagonals. Newton QP on the α
/// spectrum. Recovers `spectral_g0w0_rhf_kpoints` when \(N_k=1\) and
/// \(C^\alpha=C^\beta\) (unit dual-gate). Requires \(n_\alpha\ge 1\) and
/// \(n_{\mathrm{virt}}^\alpha\ge 1\) (primary report).
///
/// @param psi_mo_a_k / psi_mo_b_k Per-k full MO grids [nk][n_mo_σ·npts].
/// @param eps_a_k / eps_b_k       Per-k eigenvalues [nk][n_mo_σ] [Hartree].
/// @param n_oa / n_va / n_ob / n_vb Occupied / screening-virtual counts.
/// @param vxc_a_r                 Spin-α local XC potential on the grid [Ha].
/// @param qp_energy_k / sigma_x_k / vxc_diag_k Out: α-channel [nk][n_mo_a].
/// @param all_converged Out: true when every α Newton solve succeeded.
void spectral_g0w0_uhf_kpoints(const UniformGrid& grid,
                               const std::vector<std::vector<Cplx>>& psi_mo_a_k,
                               const std::vector<std::vector<Cplx>>& psi_mo_b_k,
                               const std::vector<std::vector<double>>& eps_a_k,
                               const std::vector<std::vector<double>>& eps_b_k,
                               const std::vector<std::array<double, 3>>& k_frac,
                               const std::vector<double>& k_weights, int n_oa,
                               int n_va, int n_ob, int n_vb,
                               const std::vector<double>& vxc_a_r,
                               std::vector<std::vector<double>>& qp_energy_k,
                               std::vector<std::vector<double>>& sigma_x_k,
                               std::vector<std::vector<double>>& vxc_diag_k,
                               bool& all_converged);

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
