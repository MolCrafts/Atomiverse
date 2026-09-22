/// @file cpu/scf/internal/integrals.h
/// @brief Host integral **computation** (private). No user pack/bag API.
///
/// Public code: `RHF{basis}.run(frame)` only. Matrices live as **private
/// members of the concrete method** (`MeanFieldMethod` / `cuda::scf::RHF`);
/// this header provides fill helpers and transitional storage used only
/// inside mean-field rebuild. Do **not** invent a parallel public integral
/// bag type.
#pragma once

#include <array>
#include <string>
#include <vector>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/basis_workspace.h"
#include "atomiverse/cpu/scf/internal/ecp_integrals.h"
#include "atomiverse/cpu/scf/internal/integral_view.h"
#include "atomiverse/cpu/scf/internal/point_charges.h"
#include "atomiverse/cpu/scf/native_integrals.h"
#include "atomiverse/cpu/scf/scf_enums.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Transitional storage for mean-field rebuild (not a public bag).
///
/// **Do not construct this from user code.** Prefer free fill helpers or
/// method-owned private matrices. Exists only so MeanFieldMethod /
/// internal engines can rebuild S/Hcore/ERI from Frame without a parallel
/// public pack type.
struct IntegralWorkspace {
    xt::xtensor<double, 2> S; ///< Overlap matrix [n_basis x n_basis].
    xt::xtensor<double, 2> T; ///< Kinetic energy matrix [n_basis x n_basis].
    xt::xtensor<double, 2>
        V; ///< Nuclear attraction matrix [n_basis x n_basis].
    xt::xtensor<double, 2>
        Hcore; ///< Core Hamiltonian T + V [n_basis x n_basis].
    /// @brief Spin-free nuclear pVp matrix W_μν = ⟨∇μ|V|∇ν⟩ [Hartree/Bohr²]
    ///        (PySCF `int1e_pnucp` convention). Empty unless the pack was
    ///        constructed with `compute_pvp=true` (x2c path).
    xt::xtensor<double, 2> W;
    xt::xtensor<double, 4> ERI; ///< Two-electron integrals (ij|kl) [n^4].
    double V_nuc = 0;           ///< Nuclear repulsion energy [Hartree].
    int n_basis = 0;            ///< Number of basis functions (AO dimension
                                ///< of S/T/V/Hcore/ERI: spherical count when
                                ///< the basis convention is Spherical).

    /// @brief Cartesian AO count (== shells.size()). Equals n_basis for
    ///        Cartesian-convention bases; differs (larger) for spherical.
    int n_basis_cart = 0;

    /// @brief Block-diagonal Cartesian→spherical transform (n_basis_cart ×
    ///        n_basis), engine convention (CartToSph::block per shell block).
    ///        Empty ⇒ Cartesian basis, no transform anywhere.
    xt::xtensor<double, 2> c2s;

    /// @brief True when the basis is spherical (c2s is in effect).
    /// @return True iff the pack was sandwiched to spherical AOs.
    [[nodiscard]] bool spherical() const { return c2s.size() > 0; }

    /// @brief AO dimension the Cartesian integral engines work in.
    /// @return n_basis_cart when spherical, else n_basis.
    [[nodiscard]] int n_ao_cart() const {
        return spherical() ? n_basis_cart : n_basis;
    }
    int n_electrons = 0; ///< Total number of electrons (valence only
                         ///< for ECP atoms: core electrons removed).

    /// @brief ECP family name in effect ("" = all-electron pack).
    std::string ecp;

    /// @brief True when at least one atom carries an ECP (U added to Hcore,
    ///        Z reduced to Z_val, n_electrons reduced by n_core). Consumers
    ///        without ECP support (MINAO guess) key off this flag.
    bool has_ecp = false;

    /// @brief ECP-carrying centers (position [Bohr] + borrowed static table
    ///        row), retained so the semi-numerical gradient can rebuild the
    ///        ECP matrix U at displaced geometries for dU/dR. Empty unless
    ///        has_ecp. The table-row pointers reference static storage.
    std::vector<internal::EcpCenter> ecp_centers;

    /// @brief External MM point charges (positions [Å], charge [e]), retained
    ///        by the method when point-charge embedding is active so the
    ///        analytic embedding gradient can assemble QM-center and site
    ///        forces. Empty ⇒ no embedding.
    std::vector<cpu::EmbeddingSite> embed_charges;

    /// @brief When true, include MM–MM self-energy in the energy and its
    ///        site gradient. Default true matches standalone embedding pins;
    ///        additive QM/MM couplings set false so MM–MM is owned once.
    bool embed_self_energy = true;

    /// @brief Uniform external electric field [a.u.], retained by the method
    ///        when a field is active so the semi-numerical gradient can rebuild
    ///        the field 1e matrix + nuclear field energy for dE_field/dR.
    ///        All-zero ⇒ no field.
    std::array<double, 3> external_field{{0.0, 0.0, 0.0}};

    /// @brief Continuum dielectric ε retained when `solvent_eps > 0` so the
    ///        free-energy gradient can rebuild the cavity at displaced
    ///        geometries. Zero ⇒ vacuum (no PCM force term).
    double solvent_eps = 0.0;

    /// @brief Bondi radii scale for the solvent cavity (dimensionless).
    double solvent_radii_scale = 1.2;

    /// @brief Dielectric scaling x in f(ε)=(ε−1)/(ε+x): 0 = C-PCM, 0.5 = COSMO.
    ///        Ignored for IEF-PCM.
    double solvent_x = 0.0;

    /// @brief Continuum model retained for free-energy gradient rebuilds.
    SolventModel solvent_model = SolventModel::CPCM;

    /// @brief Atom descriptors (Z, position in Bohr). Filled by the
    ///        Frame-based constructor.
    std::vector<native::Atom> atoms;

    /// @brief Shell descriptors with owned coefficient storage.
    /// Each shell has n_prim, alpha (pointer into coeff_storage), coeff (pointer into coeff_storage), center.
    std::vector<native::Shell> shells;

    /// @brief Backing storage for shell exponents and normalized coefficients.
    /// One inner vector per shell, length `2 * n_prim`, laid out as
    /// `[alpha_0 ... alpha_{n_prim-1}, coeff_0 ... coeff_{n_prim-1}]`.
    /// `pack.shells[i].alpha` / `.coeff` point into this storage; any code
    /// that copies `shell_data` must also rebind those pointers.
    std::vector<std::vector<double>> shell_data;

    /// @brief Empty (fields filled by the method rebuild / fill helpers).
    IntegralWorkspace() = default;

    /// @brief Fill S, Hcore, optional ERI (+ counts) from Frame for RHF-like paths.
    ///
    /// Out-params only — no pack returned to the caller. Used by CUDA RHF and
    /// any path that must upload dense matrices without owning a bag type.
    /// @param frame Geometry [Å]; Z from type column.
    /// @param basis Basis set.
    /// @param charge Molecular charge [e].
    /// @param compute_eri When true, fill dense ERI [Hartree]; else ERI empty.
    /// @param[out] S Overlap.
    /// @param[out] Hcore Core Hamiltonian [Hartree].
    /// @param[out] ERI Two-electron tensor or empty.
    /// @param[out] V_nuc Nuclear repulsion [Hartree].
    /// @param[out] n_basis AO dimension.
    /// @param[out] n_electrons Electron count (valence if ECP).
    static void fill_rhf_matrices(const Frame& frame, BasisSet basis,
                                  int charge, bool compute_eri,
                                  xt::xtensor<double, 2>& S,
                                  xt::xtensor<double, 2>& Hcore,
                                  xt::xtensor<double, 4>& ERI, double& V_nuc,
                                  int& n_basis, int& n_electrons);

    /// @brief Compute integrals for a molecular geometry via native C++.
    ///
    /// Reads atom positions from the `kAtom_x/y/z` columns (Angstrom) and nuclear
    /// charges from `kAtom_type`. Converts to Bohr internally.
    ///
    /// By default the two-electron ERI tensor is NOT computed (`compute_eri=false`).
    /// The CPU SCF solver uses direct SCF via ShellPairList::g_rhf,
    /// so storing the full O(n^4) ERI tensor is unnecessary for CPU calculations.
    /// Pass `compute_eri=true` when the caller explicitly needs `ERI`:
    ///   - CUDA RHFFock (uploads ERI to device)
    ///   - Tests that verify ERI values directly
    ///
    /// When `ecp` names a supported family ("def2-ECP"), atoms present in
    /// that table switch to the valence-only description: nuclear attraction
    /// and nuclear repulsion use Z_val = Z − n_core, `n_electrons` drops by
    /// n_core per atom, and the semi-local ECP matrix U is added into the
    /// nuclear-attraction block of Hcore (before any spherical sandwich).
    /// Atoms absent from the table stay all-electron (def2 convention:
    /// Z ≤ 36). S/T/ERI are untouched by the ECP.
    ///
    /// @param frame       CPU Frame with atom positions and types (atoms.type = Z).
    /// @param basis       Which basis set to resolve (STO3G, 6-31G, ...).
    /// @param charge      Molecular charge (0 = neutral, +1 = cation, etc.).
    /// @param compute_eri If true, compute and store the full ERI tensor [default false].
    /// @param ecp         ECP family name ("" = all-electron; unknown names throw).
    /// @param ghost_atoms Local atom indices treated as ghosts (basis only;
    ///                    Z → 0 for nuclear attraction / repulsion and electron
    ///                    count). Empty = no ghosts. Out-of-range indices throw.
    /// @param compute_pvp If true, assemble the spin-free nuclear pVp matrix
    ///                    `W` (needed by scalar-relativistic x2c-1e). Default
    ///                    false keeps the non-relativistic path untouched.
    IntegralWorkspace(const Frame& frame, BasisSet basis, int charge = 0,
                      bool compute_eri = false, std::string ecp = {},
                      const std::vector<int>& ghost_atoms = {},
                      bool compute_pvp = false);

    /// @brief Number of doubly-occupied orbitals (RHF closed-shell).
    /// @return `n_electrons / 2`.
    int n_occ() const { return n_electrons / 2; }

    /// @brief Non-owning view of these fields (engines take IntegralView).
    /// @return View bound to this object's members; valid while `*this` lives.
    [[nodiscard]] IntegralView view() const {
        // All fields are const refs into *this — safe while the workspace lives.
        return IntegralView{S,
                            T,
                            V,
                            Hcore,
                            W,
                            ERI,
                            V_nuc,
                            n_basis,
                            n_basis_cart,
                            c2s,
                            n_electrons,
                            ecp,
                            has_ecp,
                            ecp_centers,
                            embed_charges,
                            embed_self_energy,
                            external_field,
                            solvent_eps,
                            solvent_radii_scale,
                            solvent_x,
                            solvent_model,
                            atoms,
                            shells,
                            shell_data};
    }

    /// @brief Implicit conversion for engine APIs that take `IntegralView`.
    ///        Temporary view is valid only for the full-expression that uses it;
    ///        store by value when holding longer than a call.
    operator IntegralView() const { return view(); }

    /// @brief Fused Coulomb + exchange — delegates to `view()`.
    void coulomb_exchange_fused(const RealMatrix& P_total,
                                RealMatrix& G) const {
        view().coulomb_exchange_fused(P_total, G);
    }

    /// @brief Materialize dense ERI — delegates to `view()`.
    [[nodiscard]] xt::xtensor<double, 4> eri_working() const {
        return view().eri_working();
    }

  private:
    friend struct IntegralView; // eri_working uses eri_dense / eri_to_spherical

    /// @brief Overlap matrix S_ij = <φ_i|φ_j>.
    /// @param shells Resolved shell list.
    /// @return (n_basis × n_basis) symmetric matrix [unitless].
    [[nodiscard]] static xt::xtensor<double, 2>
    overlap(const ShellList& shells);

    /// @brief Kinetic-energy matrix T_ij = <φ_i|-½∇²|φ_j>.
    /// @param shells Resolved shell list.
    /// @return (n_basis × n_basis) symmetric matrix [Hartree].
    [[nodiscard]] static xt::xtensor<double, 2>
    kinetic(const ShellList& shells);

    /// @brief Nuclear-attraction matrix V_ij = -Σ_A Z_A <φ_i|1/|r - R_A||φ_j>.
    /// @param shells Resolved shell list.
    /// @param atoms  Atoms supplying nuclear charges and positions (Bohr).
    /// @return (n_basis × n_basis) symmetric matrix [Hartree].
    [[nodiscard]] static xt::xtensor<double, 2>
    nuclear_attraction(const ShellList& shells, const AtomList& atoms);

    /// @brief Spin-free nuclear pVp matrix W_ij = ⟨∇φ_i|V̂|∇φ_j⟩
    ///        = Σ_d ⟨∂φ_i/∂x_d | V̂ | ∂φ_j/∂x_d⟩ with
    ///        V̂ = −Σ_A Z_A / |r−R_A| (PySCF `int1e_pnucp` convention).
    ///
    /// Built from nuclear-attraction OS primitives via the Gaussian derivative
    /// identity ∂_d g(a) = 2α g(a+e_d) − a_d g(a−e_d). Cartesian engine
    /// convention; spherical packs sandwich through `c2s` in the constructor.
    ///
    /// @param shells Resolved shell list (Cartesian AO order).
    /// @param atoms  Atoms supplying nuclear charges and positions [Bohr].
    /// @return (n_basis × n_basis) symmetric matrix [Hartree / Bohr²].
    [[nodiscard]] static xt::xtensor<double, 2> pvp(const ShellList& shells,
                                                    const AtomList& atoms);

    /// @brief Dense two-electron repulsion tensor ERI(i,j,k,l) = (ij|kl)
    ///        [Hartree], chemist notation.
    ///
    /// Assembled by grouping the per-AO shell list into libcint-style l-blocks
    /// (`blocks_from_shells`) and calling the amortized Rys/xsimd shell-block
    /// kernel `native::eri_block` once per canonical block quartet (Bohr-space
    /// Gaussian centers), then scattering each block with full 8-fold
    /// permutation symmetry. The block-pair loop is OpenMP-parallel over the
    /// canonical outer pair; each iteration writes a disjoint set of tensor
    /// cells, so the scatter is a plain race-free assignment (no reduction).
    ///
    /// Dense ERI storage is O(N⁴) in memory and only appropriate for small
    /// systems, validation, or code paths that must hand the full tensor to
    /// another backend (e.g. CUDA RHFFock). The production CPU direct-SCF
    /// path uses `ShellBlockPairList` in internal/shell_pair.h instead.
    ///
    /// @param shells Resolved Cartesian shell list (centers in Bohr).
    /// @return (n × n × n × n) dense tensor (ij|kl) [Hartree].
    [[nodiscard]] static xt::xtensor<double, 4>
    eri_dense(const ShellList& shells);

    /// @brief Four-index Cartesian→spherical sandwich of a dense ERI tensor
    ///        (one index at a time; shared by the constructor and
    ///        `eri_working`).
    /// @param eri_cart Cartesian-dimension tensor (n_cart⁴) [Hartree].
    /// @param c2s      Cartesian→spherical transform (n_cart × n_sph).
    /// @return Spherical-dimension tensor (n_sph⁴) [Hartree].
    [[nodiscard]] static xt::xtensor<double, 4>
    eri_to_spherical(const xt::xtensor<double, 4>& eri_cart,
                     const xt::xtensor<double, 2>& c2s);
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
