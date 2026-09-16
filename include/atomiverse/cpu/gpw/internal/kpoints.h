/// @file cpu/gpw/internal/kpoints.h
/// @brief Monkhorst-Pack k-point sampling machinery for the GPW RKS / UKS SCF:
///        phased (complex) Bloch lattice sums of the analytic one-electron and
///        GTH nonlocal matrices, per-k pre-summed complex basis-on-grid values
///        for density collocation / potential integration, and the per-k
///        generalized eigensolve `diagonalize_k` (built on the shared complex
///        Hermitian eigensolver in `cpu/linalg/eigensolve.h`).
///
/// **Internal GPW helper** — never on the public include path. Owned by the
/// GPW SCF (`GpwDft`) k-point branch and by unit tests (direct include).
///
/// Conventions (all consumers must agree — a mixed convention breaks the
/// Hermiticity of H_k and is caught by the acceptance identities):
///   - Bloch AO:  χ_μ^k(r) = Σ_T e^{i k·T} φ_μ(r − T),  T = (t₁L₁,t₂L₂,t₃L₃),
///     k = 2π diag(1/L) κ with fractional κ ∈ (−1/2, 1/2]³.
///   - Analytic sums: S_k(μν) = Σ_T e^{+2πi κ·t} ⟨φ_μ|φ_ν(·−T)⟩ (image shifts
///     the KET by +T); T_k likewise.
///   - Nonlocal KB:  O_μ^k[A,slot] = ⟨χ_μ^k|p_A⟩ = Σ_T e^{−2πi κ·t}
///     ⟨φ_μ(·−T)|p_A⟩;  V_nl^k(μν) = Σ_A,slot conj(O_μ)·h·O_ν.
///   - Grid values:  χ_μ^k(g) accumulated with e^{−2πi κ·t'} for the unwrapped
///     window cell t' (φ evaluated at r_g + T' is the image T = −T').
///     ρ_k(g) = Σ_μν P^k_μν χ_μ^k(g) conj(χ_ν^k(g));
///     V^k(μν) = Σ_g w V(g) conj(χ_μ^k(g)) χ_ν^k(g).
///
/// Monkhorst-Pack follows CP2K `&KPOINTS SCHEME MONKHORST-PACK`:
/// κ_d = (2r − N_d − 1)/(2N_d), r = 1..N_d (even N ⇒ Γ-avoiding mesh, e.g.
/// N=2 → ±1/4). Optional time-reversal folding keeps one of each ±κ pair with
/// doubled weight (exact for RKS: the density uses only Re P_k).
#pragma once

#include <array>
#include <complex>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/gpw/gpw_solver.h" // GpwAtom, GridBasisFunction
#include "atomiverse/cpu/linalg/eigensolve.h" // eigh_hermitian

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {

/// @brief Complex dense matrix (row-major), the k-space sibling of RealMatrix.
using CplxMatrix = xt::xtensor<std::complex<double>, 2>;

/// @brief One sampled k-point: fractional coordinates and quadrature weight.
struct KPoint {
    double frac[3] = {0, 0, 0}; ///< κ, fractional reciprocal coordinates.
    double weight = 1.0;        ///< Σ w = 1 over the sampled set.
};

/// @brief Monkhorst-Pack mesh (CP2K convention), optionally time-reversal
///        folded. κ canonicalized to (−1/2, 1/2]. Weights sum to 1.
/// @param mesh Per-axis subdivisions N_d ≥ 1.
/// @param time_reversal Fold ±κ pairs (keep one, double its weight).
/// @throws std::invalid_argument when any N_d < 1.
std::vector<KPoint> monkhorst_pack(const int mesh[3], bool time_reversal);

/// @brief Build S_k and T_k for every k in ONE lattice-image sweep.
///
/// Pair loop is screened exactly like the Γ build (Gaussian product magnitude
/// ≥ screen keeps a pair; per-image center-distance ≤ reach keeps an image);
/// each surviving per-image real integral is accumulated into every k with its
/// Bloch phase. Hermiticity is enforced by construction (j ≥ i computed,
/// (i,j) → conj into (j,i)).
///
/// @param basis Global contracted basis (image T shifts the ket center by +T).
/// @param radii Per-function eps-radii [Bohr] (`basis_function_radius`).
/// @param L     Orthorhombic cell edges [Bohr].
/// @param m     Per-axis image half-counts (`pair_image_counts`).
/// @param screen Pair screen threshold on the Gaussian product magnitude.
/// @param kpts  Sampled k-points.
/// @param S_k   Out: per-k overlap, resized to kpts.size() (nb × nb).
/// @param T_k   Out: per-k kinetic energy [Hartree].
void build_overlap_kinetic_k(const std::vector<GridBasisFunction>& basis,
                             const std::vector<double>& radii,
                             const double L[3], const int m[3], double screen,
                             const std::vector<KPoint>& kpts,
                             std::vector<CplxMatrix>& S_k,
                             std::vector<CplxMatrix>& T_k);

/// @brief Phased Kleinman-Bylander GTH nonlocal matrices V_nl^k (adds onto
///        `V_k`, which must be pre-sized nb × nb per k).
///
/// Per (basis function, atom) the projector overlaps are Bloch-accumulated
/// over the μ-image sweep once (phase conj(e^{2πi κ·t})), then the separable
/// assembly conj(O_μ)·h·O_ν fills every pair — the same
/// O(nb·prim·proj) precompute shape as the Γ `add_nonlocal_kb`.
///
/// @param V_k   In/out: per-k matrices to add V_nl^k onto.
/// @param basis Global contracted basis.
/// @param atoms GTH centers (nonlocal channels).
/// @param radii Per-function eps-radii [Bohr].
/// @param L     Cell edges [Bohr]. @param mimg Image half-counts.
/// @param screen Drop threshold for negligible projector overlaps.
/// @param kpts  Sampled k-points.
void add_nonlocal_kb_k(std::vector<CplxMatrix>& V_k,
                       const std::vector<GridBasisFunction>& basis,
                       const std::vector<GpwAtom>& atoms,
                       const std::vector<double>& radii, const double L[3],
                       const int mimg[3], double screen,
                       const std::vector<KPoint>& kpts);

/// @brief Per-k pre-summed complex basis-on-grid values.
///
/// For each k and each active grid point g, stores the dense complex vector
/// χ_μ^k(g) (μ = 0..nb−1) with the image sum and Bloch phases folded at build
/// time. Unlike the Γ `CompactBasisGrid` (which keeps one triple per image and
/// pays the duplication quadratically in the collocation loop), this presums
/// the images — the per-iteration cost is O(n_active · nb²) per k independent
/// of how many lattice images a diffuse function spans (the regime of MOLOPT
/// bases in few-Bohr cells).
struct KpointBasisGrid {
    std::vector<long> gidx; ///< active grid-point linear indices.
    /// Dense per-point values: chi[k][a * nb + u] = χ_u^{k_k}(g_a).
    std::vector<std::vector<std::complex<double>>> chi;
    int nb = 0;          ///< basis size.
    long npts = 0;       ///< total grid points.
    double weight = 0.0; ///< grid integration weight per point.
};

/// @brief Build the per-k pre-summed basis-on-grid values (one window sweep
///        per basis function; Gaussian value reused across k, phase per image).
/// @param grid Uniform density grid. @param basis Global basis.
/// @param kpts Sampled k-points. @param eps Gaussian tail drop threshold.
KpointBasisGrid
build_kpoint_basis_grid(const UniformGrid& grid,
                        const std::vector<GridBasisFunction>& basis,
                        const std::vector<KPoint>& kpts, double eps);

/// @brief Accumulate w_k · ρ_k(r) into `rho` (length npts) from Hermitian P_k.
/// @param kbg Pre-summed basis grid. @param ik k index into kbg.chi.
/// @param P_k Hermitian density matrix at k. @param wk k weight.
void collocate_density_k(const KpointBasisGrid& kbg, int ik,
                         const CplxMatrix& P_k, double wk,
                         std::vector<double>& rho);

/// @brief Integrate V^k(μν) = Σ_g w V(g) conj(χ_μ^k) χ_ν^k for one k.
CplxMatrix integrate_potential_k(const KpointBasisGrid& kbg, int ik,
                                 const std::vector<double>& V_r);

/// @brief Canonical-orthogonalization transform for one k:
///        X_k = U_kept diag(s^{−1/2}) with S_k = U diag(s) U†, dropping
///        overlap eigenvalues s < tau (never amplifying near-null directions).
struct KOrth {
    CplxMatrix X;      ///< nb × m transform, X† S_k X = I_m.
    int n_dropped = 0; ///< dropped directions (s < tau).
    double s_min = 0;  ///< smallest raw overlap eigenvalue.

    /// @param S_k Hermitian overlap at k. @param tau Keep threshold.
    /// @throws std::runtime_error when every direction would drop, or when
    ///         s_min < −1e−8 (a Bloch overlap cannot have a negative
    ///         eigenvalue — the lattice sum would be wrong).
    KOrth(const CplxMatrix& S_k, double tau);
};

/// @brief Diagonalize H_k in the canonical-orthogonal basis:
///        (X† H X) C' = C' ε, C = X C'. Returns ascending ε and C (nb × m).
std::pair<std::vector<double>, CplxMatrix> diagonalize_k(const CplxMatrix& H_k,
                                                         const KOrth& orth);

/// @brief k-space DIIS error vector e = X†(H P S − S P H)X — the complex
///        sibling of `gpw_diis_error`; → 0 at self-consistency.
CplxMatrix k_diis_error(const CplxMatrix& H, const CplxMatrix& P,
                        const CplxMatrix& S, const CplxMatrix& X);

/// @brief k-summed Pulay DIIS on the per-k KS matrices.
///
/// One shared extrapolation: B_ij = Σ_k w_k Re Tr(e_i^k† e_j^k) with the
/// commutator residual e^k = X†(H_k P_k S_k − S_k P_k H_k)X — the standard
/// k-point generalization of the Γ `GpwDiis`.
class KDiis {
  public:
    /// @param max_vectors History depth (≥ 1). @param weights k weights.
    KDiis(int max_vectors, std::vector<double> weights)
        : max_vectors_(max_vectors), weights_(std::move(weights)) {}

    /// @brief Store {H_k}, {e_k} and return the extrapolated {H_k}.
    ///        Latest H during warm-up or on a singular B system.
    std::vector<CplxMatrix> step(const std::vector<CplxMatrix>& H,
                                 const std::vector<CplxMatrix>& e);

  private:
    int max_vectors_;
    std::vector<double> weights_;
    std::vector<std::vector<CplxMatrix>> h_hist_;
    std::vector<std::vector<CplxMatrix>> e_hist_;
};

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
