/// @file cpu/planewave/internal/pw_setup.h
/// @brief Host-side plane-wave SCF problem assembly and density mixing,
///        shared by the CPU (`cpu::pw::PW`) and CUDA
///        (`cuda::pw::PW`) methods.
#pragma once

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/internal/g_vector.h"
#include "atomiverse/cpu/planewave/internal/kpoint.h"
#include "atomiverse/cpu/planewave/internal/nc_potential.h"
#include "atomiverse/cpu/planewave/internal/pw_workspace.h"
#include <memory>
#include <string>
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// Geometry and radial data admitted for this solve, in frame order.
struct PWIons {
    std::vector<Vec3> positions; ///< Bohr.
    std::vector<std::shared_ptr<const NcPotentialData>> species;
    std::vector<std::size_t> atom_species;
    std::vector<AccReal> ionic_charges;
    AccReal ew_total = 0; ///< Includes the QE G=0 term [Ha].
    AccReal vol_bohr = 0;
    int n_elec = 0;
};

/// One owner for each k-dependent operator table. Keep separate owners for
/// different meshes, such as k and k+q, while their views are in use.
struct PWKMesh {
    KPointGrid kg;
    std::vector<GVectorList> gv_k;
    std::vector<KBProjectors> kb_proj_k;
    std::vector<AtomicOrbitals> atom_wfc_k;

    PWKPointView point(std::size_t ik) const& {
        return {gv_k.at(ik), kb_proj_k.at(ik), atom_wfc_k.at(ik)};
    }
    PWKPointView point(std::size_t) const&& = delete;
};

struct PWStrainReference {
    const PWIons& ions;
    const GVectorList& density_modes;
    const KPointGrid& grid;
    std::span<const GVectorList> wavefunction_modes;
};

/// Static data assembled once for a frame. Consumers borrow it as const;
/// the independent PWWorkspace owns all writable SCF buffers.
struct PWSetup {
    PWIons ions;
    GVectorList gv0; ///< Γ-centred density sphere at four times the PW cutoff.
    std::vector<std::complex<AccReal>> V_loc_G;
    PWKMesh mesh;
    int n_bands = 0;

    PWGridView grid() const& { return {gv0, V_loc_G, ions.vol_bohr}; }
    PWGridView grid() const&& = delete;
    PWStrainReference strain_reference() const& {
        return {ions, gv0, mesh.kg, mesh.gv_k};
    }
    PWStrainReference strain_reference() const&& = delete;
};

/// @brief Refuse an odd valence-electron count on the restricted,
///        fixed-occupation path.
///
/// That path occupies every band with exactly two electrons, so `n_elec / 2`
/// bands hold `n_elec` electrons only when `n_elec` is even. On an odd count
/// the integer division silently drops one electron and the run reports a
/// converged energy for the wrong system — which is why this is a throw, not
/// a clamp, and why `PW::run`, `pw_forces_analytic` and `pw_stress` all call
/// it: an energy the force path refuses to differentiate is not an energy the
/// energy path may quietly report.
///
/// Not called on the spin-polarized path (the α/β split has its own parity
/// rule) nor with metallic smearing (fractional occupations accept an odd
/// count — aluminium's three valence electrons are the standard case).
///
/// @param n_elec Valence-electron count of the cell (dimensionless).
/// @param where  Entry-point name used as the message prefix, e.g. "PW::run".
/// @throws std::invalid_argument when @p n_elec is odd.
void require_closed_shell(int n_elec, const char* where);

/// @brief Radial → G-space form factor v(G) of the NC local potential
///        [Hartree], one value per G of `gv` (QE `vloc_mod.f90` route:
///        erf-screened quadrature + analytic screened-Coulomb transform;
///        the G = 0 entry carries the finite "αZ" part only).
///
/// Per-atom quantity (no structure factor). Shared by problem assembly
/// (`setup_pw_problem`) and the DFPT perturbation builder — dividing the
/// assembled `V_loc_G` by the structure factor is ill-defined on shells
/// where S(G) = 0 (e.g. diamond).
///
/// @param upf Canonical NC radial data in Hartree.
/// @param gv    G-vector list to evaluate on.
/// @param omega Cell volume [Bohr³].
std::vector<AccReal> nc_vloc_to_gspace(const NcPotentialData& upf,
                                       const GVectorList& gv, AccReal omega);

/// Radial local form factors indexed by species and then by G.
std::vector<std::vector<AccReal>> nc_vloc_to_gspace(
    std::span<const std::shared_ptr<const NcPotentialData>> species,
    const GVectorList& gv, AccReal omega);

/// @brief Transform a spherical NC nonlocal potential to diagonal KB channels.
/// @param out Per-atom projectors: angular momentum, strength [Hartree], and
///        radial Fourier integrals and normalized complex β_lm(G) columns.
/// @param gv Wavefunction G-sphere; radial transforms use |G+k| [Bohr^-1].
/// @param upf Radial beta=r*p(r) tables and complete symmetric D_ij [Ha].
/// @param pos_bohr Atomic positions [Bohr]; determines the number of outputs.
/// @param omega Cell volume [Bohr^3], for physical projector normalization.
/// @throws std::invalid_argument for incomplete/nonfinite/asymmetric D_ij or
///         couplings between distinct angular momenta.
void setup_kb_projectors(KBProjectors& out, const GVectorList& gv,
                         const NcPotentialData& upf,
                         const std::vector<Vec3>& pos_bohr, AccReal omega);

/// @brief Assemble the complete SCF problem from a Frame: Ewald energy,
///        G-vectors, k-mesh, FFT grids, V_loc, KB projectors,
///        atomic wavefunctions and the initial density.
///
/// This is the entire pre-loop section of the original monolithic solver; both
/// backends call it and then run their own SCF loop.
///
/// @param frame     Input structure: geometry (positions [Å]), species
///                  (atomic_number = Z) and cell.
/// @param potentials Immutable scalar NC providers, keyed by atomic number.
/// @param ecut_ry   Wavefunction cutoff [Ry].
/// @param k_mesh    Monkhorst-Pack mesh size per dimension.
/// @param out       Operator data to assemble (must outlive the workspace).
/// @param workspace Mutable SCF state seeded on the density sphere.
void setup_pw_problem(const cpu::Frame& frame, NcProviders potentials,
                      double ecut_ry, int k_mesh, PWSetup& out,
                      PWWorkspace& workspace);

/// @brief Rebuild geometry-dependent operators at a fixed reference PW basis.
///
/// The integer density/wavefunction modes, their order, k mesh and FFT shapes
/// are retained. Cartesian G/k vectors, kinetic energies and ionic operators
/// are recomputed for the strained cell. This defines the smooth finite-basis
/// strain derivative without modes entering/leaving the energy cutoff.
/// The immutable admitted species data is shared without repeating admission or
/// copying radial tables. Its form factors are evaluated at the new vectors.
/// @param reference Reference modes and ionic data with independent owners.
void setup_pw_problem_fixed_basis(const cpu::Frame& frame,
                                  PWStrainReference reference, double ecut_ry,
                                  int k_mesh, PWSetup& out,
                                  PWWorkspace& workspace);

/// Assemble a separate wavefunction mesh for frozen-density calculations.
PWKMesh make_pw_k_mesh(const PWIons& ions, const cpu::Box& box, double ecut_ry,
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

/// @brief Pulay/DIIS mixing of a density vector (one or joint spin channels).
/// @param output Output density, replaced with the mixed density.
/// @param input Input density at this iteration.
/// @param residual_history Sliding history of unmixed output-input vectors.
/// @param density_history Corresponding input-density history.
/// @param fraction Linear mixing fraction (dimensionless).
/// @param history_depth Maximum retained entries; zero selects linear mixing.
void pulay_mix(xt::xarray<std::complex<AccReal>>& output,
               const xt::xarray<std::complex<AccReal>>& input,
               std::vector<xt::xarray<std::complex<AccReal>>>& residual_history,
               std::vector<xt::xarray<std::complex<AccReal>>>& density_history,
               AccReal fraction, int history_depth);

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
    AccReal total = 0;    ///< Total energy incl. E_xc and Ewald.
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
/// @param e_xc      XC energy ∫f_xc[ρ] d³r [Hartree] (`ws.E_xc`).
/// @param ew_total  Ewald energy incl. the QE G=0 term [Hartree].
/// @return Filled decomposition [Hartree].
PWEnergyTerms combine_energy_terms(AccReal e_band, AccReal e_h,
                                   AccReal e_xc_int, AccReal e_loc,
                                   AccReal e_xc, AccReal ew_total);

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
