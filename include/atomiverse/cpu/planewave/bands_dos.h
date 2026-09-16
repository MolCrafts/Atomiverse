/// @file cpu/planewave/bands_dos.h
/// @brief Band structure and density-of-states post-processing (CPU PW).
///
/// After a self-consistent density is obtained from `cpu::pw::PW` members,
/// non-self-consistent diagonalization on a frozen \(V_\mathrm{eff}\) yields
/// \(\varepsilon_n(\mathbf{k})\) (bands) and a Gaussian-broadened DOS.
/// Parameters are members of `Bands` / `Dos` — no public `*Config` bag.
///
/// @code
///   cpu::pw::PW pw;
///   pw.upf_path = "Si.upf";
///   pw.ecut_ry = 15.0;
///   pw.k_mesh = 2;
///   pw.xc = cpu::scf::XCType::LSDA;
///
///   cpu::pw::Bands bands;
///   bands.n_bands = 8;
///   bands.set_diamond_path(4);
///   auto br = bands.compute(pw, frame);   // gap_ry, eigenvalues_ry
///
///   // Collinear spin (product): set method.spin_polarized = true
///   // → br.nspin == 2, br.eigenvalues_beta_ry holds spin-down.
///
///   // Metallic occupations (restricted): set method.smearing / degauss /
///   // extra_bands → SCF free-energy density, then NSCF path; br.e_fermi_ry.
///
///   // Collinear spin + metallic smear (product): both flags together →
///   // shared μ, free M, per-spin NSCF path; br.e_fermi_ry from SCF.
///
///   cpu::pw::Dos dos;
///   dos.k_mesh = 4;
///   dos.n_bands = 8;
///   dos.sigma_ry = 0.05;
///   auto dr = dos.compute(pw, frame);     // dos, n_elec_to_vbm
///
///   // Metallic DOS: set method.smearing / degauss / extra_bands;
///   // integral is then to E_F (dr.e_fermi_ry).
///
///   // Collinear spin DOS: method.spin_polarized = true → dr.nspin == 2;
///   // total D = Dα + Dβ (spin-orbital weight 1); integral to joint VBM.
///
///   // Collinear spin + metallic smear: spin_polarized + smearing → shared
///   // μ over α+β; integral to E_F (dr.e_fermi_ry). Bands still rejects
///   // spin+smear (see bands-spin-metallic-si).
/// @endcode
#pragma once

#include <array>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

namespace ATV {
namespace cpu {
namespace pw {

/// @brief Result of a band-structure (NSCF path) calculation.
struct BandsResult {
    /// Fractional k-points used for the path (same order as eigenvalues).
    std::vector<std::array<double, 3>> kfrac;
    /// Eigenvalues \(\varepsilon_n(\mathbf{k})\) [Ry], layout `[ik][ib]`.
    /// Restricted: sole channel. Collinear spin: **spin-up** (\(\alpha\)).
    std::vector<std::vector<double>> eigenvalues_ry;
    /// Spin-down eigenvalues [Ry] when `nspin == 2`; empty when restricted.
    std::vector<std::vector<double>> eigenvalues_beta_ry;
    /// 1 = restricted (closed-shell); 2 = collinear spin-polarized.
    int nspin = 1;
    /// \(N_\uparrow - N_\downarrow\) (fixed occupations; 0 when restricted).
    double magnetization = 0;
    double e_vbm_ry = 0;        ///< Valence-band maximum [Ry].
    double e_cbm_ry = 0;        ///< Conduction-band minimum [Ry].
    double gap_ry = 0;          ///< \(E_\mathrm{cbm}-E_\mathrm{vbm}\) [Ry].
    int vbm_k = 0;              ///< k-index of VBM.
    int cbm_k = 0;              ///< k-index of CBM.
    int vbm_band = 0;           ///< Band index of VBM (0-based).
    int cbm_band = 0;           ///< Band index of CBM (0-based).
    /// Spin channel of VBM/CBM (0=α, 1=β); 0 when restricted.
    int vbm_spin = 0;
    int cbm_spin = 0;
    bool scf_converged = false; ///< Whether the preceding SCF converged.
    double energy_ry = 0;       ///< SCF total / free energy \(F\) [Ry].
    /// @brief Fermi level [Ry] from metallic SCF (0 if fixed occupations).
    ///        Path is not a BZ mesh — \(\mu\) is the SCF value, not re-binned
    ///        from path k-weights.
    double e_fermi_ry = 0;
};

/// @brief Result of a Gaussian-broadened density of states.
struct DosResult {
    std::vector<double> energy_ry; ///< Bin centres [Ry].
    /// DOS \(D(E)\) [states / Ry / cell]. Restricted: each band carries
    /// spin degeneracy 2. Collinear: total \(D = D_\alpha + D_\beta\) with
    /// spin-orbital weight 1.
    std::vector<double> dos;
    /// Spin-up \(D_\alpha(E)\) when `nspin == 2`; empty when restricted.
    std::vector<double> dos_alpha;
    /// Spin-down \(D_\beta(E)\) when `nspin == 2`; empty when restricted.
    std::vector<double> dos_beta;
    /// 1 = restricted; 2 = collinear spin-polarized.
    int nspin = 1;
    /// \(N_\uparrow - N_\downarrow\) (fixed occupations; 0 when restricted).
    double magnetization = 0;
    /// Occupied DOS integral \(\int_{-\infty}^{E_\ast} D(E)\,dE\)
    /// (target \(\approx n_e\)). For fixed occupations \(E_\ast=E_\mathrm{vbm}\);
    /// for metallic smearing \(E_\ast=E_F\) (`e_fermi_ry`).
    double n_elec_to_vbm = 0;
    double e_vbm_ry = 0;  ///< Valence-band maximum [Ry] (fixed-occ / gapped).
    double e_cbm_ry = 0;  ///< Conduction-band minimum [Ry].
    double gap_ry = 0;    ///< \(E_\mathrm{cbm}-E_\mathrm{vbm}\) [Ry] (0 if no empty).
    double e_fermi_ry = 0; ///< Fermi level [Ry] (0 if fixed occupations).
    bool scf_converged = false;
    /// @brief SCF free energy \(F\) (smear) or total energy [Ry].
    double scf_energy_ry = 0;
};

/// @brief High-symmetry-path band structure (NSCF on frozen \(V_\mathrm{eff}\)).
///
/// SCF uses `method` members (`ecut_ry`, `k_mesh`, `xc`, …). Path k-points
/// are `kfrac` (filled by `set_diamond_path` when empty at `compute` time).
///
/// **Collinear spin** (`method.spin_polarized`): fixed-occupation SCF with
/// \(N_\uparrow,N_\downarrow\) from `multiplicity`, then per-spin NSCF on
/// frozen \(V_{\mathrm{eff},\sigma}\).
///
/// **Metallic occupations** (restricted): `method.smearing != none` runs
/// fractional-occupation SCF, freezes \(V_\mathrm{eff}\), NSCF on the path,
/// and reports SCF \(E_F\) on `BandsResult::e_fermi_ry`.
///
/// **Collinear spin + metallic smear** (product): both flags together use
/// a shared chemical potential over spin-orbitals (same as `PW::run`),
/// free \(M\), free energy \(F\), then per-spin NSCF. Path gap reporting
/// uses the integer-fill seed from `multiplicity` when empty states exist.
class Bands {
  public:
    /// @brief Total bands to diagonalize per spin channel (valence + conduction).
    ///        Silicon has 4 occupied valence bands (\(n_e=8\); \(N_\sigma=4\)
    ///        when \(M=0\)). With metallic smear, size Davidson to at least
    ///        \(\lceil n_e/2\rceil + \mathrm{extra\_bands}\) via SCF, and
    ///        keep `n_bands` covering valence + empty states for the gap.
    int n_bands = 8;
    /// @brief Fractional (crystal) k-list. Empty → `set_diamond_path(4)`.
    std::vector<std::array<double, 3>> kfrac;

    /// @brief FCC/diamond high-symmetry path Γ–X–W–L–Γ–K.
    /// @param n_seg Number of samples per segment (including endpoints once).
    void set_diamond_path(int n_seg = 4);

    /// @brief SCF + NSCF bands for the geometry in `frame`.
    /// @param method Plane-wave method (members only; not mutated).
    ///               Metallic path honors `smearing` / `degauss` /
    ///               `extra_bands`; collinear honors `spin_polarized` /
    ///               `multiplicity`; both may be combined.
    /// @param frame  Atom positions [Å], types (Z), simulation cell.
    /// @return Owned `BandsResult` with eigenvalues [Ry], gap, optional \(E_F\).
    BandsResult compute(const PW& method, const cpu::Frame& frame) const;
};

/// @brief Gaussian-broadened DOS from denser Monkhorst–Pack eigenvalues.
///
/// Fixed occupations (`method.smearing == none`): integer-fill SCF;
/// occupied integral is to VBM. Metallic smearing (`fermi_dirac` /
/// `methfessel_paxton` / `gaussian`): fractional occupations; \(\mu\) is
/// recomputed on the DOS mesh and the integral is to \(E_F\).
///
/// **Collinear spin** (`method.spin_polarized`): fixed-occupation SCF with
/// \(N_\uparrow,N_\downarrow\) from `multiplicity`, then per-spin NSCF on
/// frozen \(V_{\mathrm{eff},\sigma}\). Occupied integral is to joint VBM.
///
/// **Collinear spin + metallic smear** (`spin_polarized` and
/// `smearing != none`): fixed \(N_\uparrow,N_\downarrow\) collinear SCF for
/// the density (multiplicity seed; same GS policy as DFPT spin+smear), freeze
/// \(V_{\mathrm{eff},\sigma}\), denser-mesh NSCF, recompute shared \(\mu\) on
/// the DOS mesh over \(\alpha+\beta\) (`spin_degen=1`), and integrate total
/// \(D\) to \(E_F\). Self-consistent Mermin spin-density for open metals is
/// `PW::run` (spin+smear); this post-processor enables the product DOS path.
class Dos {
  public:
    /// @brief DOS k-mesh per dimension (MP \(k_\mathrm{mesh}^3\)).
    int k_mesh = 4;
    /// @brief Bands per k-point (must cover valence + some conduction).
    int n_bands = 8;
    /// @brief Gaussian broadening width [Ry].
    double sigma_ry = 0.05;
    /// @brief Number of energy bins for \(D(E)\).
    int n_bins = 200;
    /// @brief Energy window [Ry]; if `e_max_ry <= e_min_ry`, auto from evals.
    double e_min_ry = 0;
    double e_max_ry = 0;

    /// @brief SCF (method.k_mesh) + denser-mesh NSCF + Gaussian DOS.
    /// @param method Plane-wave method (members; SCF uses method.k_mesh;
    ///               metallic path honors `smearing` / `degauss` / `extra_bands`;
    ///               collinear spin honors `spin_polarized` / `multiplicity`;
    ///               spin + metallic smear is supported on this path).
    /// @param frame  Geometry [Å].
    /// @return Owned `DosResult`.
    DosResult compute(const PW& method, const cpu::Frame& frame) const;
};

/// @brief Locate VBM/CBM from a spectrum with `n_occ` occupied bands.
///
/// Pure helper for unit tests and both post-processors. Eigenvalues must be
/// sorted ascending per k-point. Units are those of the input (Ry or Ha).
///
/// @param evals  `[ik][ib]` eigenvalues (sorted ascending in `ib`).
/// @param n_occ  Number of occupied bands (\(n_e/2\) for closed shell).
/// @param e_vbm  Output valence-band maximum.
/// @param e_cbm  Output conduction-band minimum.
/// @param vbm_k  Output k-index of VBM.
/// @param cbm_k  Output k-index of CBM.
/// @param vbm_b  Output band index of VBM.
/// @param cbm_b  Output band index of CBM.
/// @return gap = e_cbm − e_vbm.
double gap_from_spectrum(const std::vector<std::vector<double>>& evals,
                         int n_occ, double& e_vbm, double& e_cbm, int& vbm_k,
                         int& cbm_k, int& vbm_b, int& cbm_b);

/// @brief Gaussian DOS on a uniform energy grid (unit-testable free math).
///
/// \(D(E_i)=\sum_{n k} w_{nk}\, (g/\sqrt{2\pi}\sigma)\,
/// \exp(-(E_i-\varepsilon)^2/(2\sigma^2))\). Default \(g=2\) is closed-shell
/// spin degeneracy; collinear spin-orbitals use \(g=1\).
///
/// @param evals            Flat list of eigenvalues [energy units].
/// @param weights          Per-eigenvalue k-weight (same length as evals).
/// @param e_min            Grid start.
/// @param e_max            Grid end.
/// @param n_bins           Number of bins.
/// @param sigma            Gaussian width (same energy units).
/// @param energy           Output bin centres.
/// @param dos              Output \(D(E)\).
/// @param spin_degeneracy  Band spin factor \(g\) (default 2; spin channels use 1).
void gaussian_dos(const std::vector<double>& evals,
                  const std::vector<double>& weights, double e_min,
                  double e_max, int n_bins, double sigma,
                  std::vector<double>& energy, std::vector<double>& dos,
                  double spin_degeneracy = 2.0);

/// @brief Trapezoid \(\int_{-\infty}^{E_\mathrm{cut}} D(E)\,dE\) on a uniform grid.
///
/// Unit-testable free math shared by fixed-occupation (cut = VBM) and
/// metallic (cut = \(E_F\)) DOS post-processing.
///
/// @param energy  Bin centres (ascending, uniform spacing).
/// @param dos     \(D(E)\) at each bin.
/// @param e_cut   Upper integration limit (same units as `energy`).
/// @return Integrated electron count (states / cell).
double integrate_dos_to(const std::vector<double>& energy,
                        const std::vector<double>& dos, double e_cut);

} // namespace pw
} // namespace cpu
} // namespace ATV
