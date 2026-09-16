/// @file cpu/scf/scf_enums.h
/// @brief The light SCF selector enums — zero-dependency public header so
///        internal headers can take the enums without dragging in result
///        types (rehomed from the since-deleted scf_result.h, 02-rehome).
#pragma once

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Mean-field flavor: Hartree-Fock or Kohn-Sham (dimensionless tag).
enum class MeanFieldType { HF, KS };

/// @brief Spin treatment: restricted (shared spatial orbitals),
///        unrestricted (independent alpha/beta channels),
///        restricted-open (single MO set with split alpha/beta occupation —
///        ROHF; Roothaan effective-Fock coupling), or generalized (GHF:
///        two-component real spinor orbitals φ_i = (φ_i^α; φ_i^β), one
///        2n×2n density with a spin-off-diagonal P^αβ block — the
///        noncollinear-spin mean field; Fukutome, Int. J. Quantum Chem. 20,
///        955 (1981), DOI 10.1002/qua.560200502).
///
/// Paths that dispatch on this enum and do not implement the
/// restricted-open or generalized case (analytic gradients, MP2/FCI packs,
/// SOSCF, level shift) throw explicitly for `RestrictedOpen` /
/// `Generalized` — no silent Restricted/Unrestricted branch is ever taken.
enum class SpinType { Restricted, Unrestricted, RestrictedOpen, Generalized };

/// @brief SCF initial-guess flavor.
///
/// - `Hcore`:   diagonalize the bare core Hamiltonian (worst quality).
/// - `MINAO`:   project MINAO atomic fragment densities (PySCF default).
/// - `Hueckel`: diagonalize the extended-Hückel Fock built from the static
///              VSIP table (`internal::HueckelGuess`); Lehtola,
///              JCTC 15, 1593 (2019), DOI 10.1021/acs.jctc.8b01089.
/// - `SAP`:     diagonalize F = T + V_SAP, the superposition of tabulated
///              atomic potentials on the DFT grid (`internal::SapGuess`);
///              same reference.
enum class InitGuess { Hcore, MINAO, Hueckel, SAP };

/// @brief Density-fitting (resolution-of-the-identity) mode for the
///        two-electron Fock build.
///
/// - `None`: exact 4-index ERIs (the shipping path, bit-identical).
/// - `J`:    fit only the Coulomb matrix (RI-J, γ_P route); exchange
///           stays exact. Vahtras-Almlöf-Feyereisen robust Coulomb fit,
///           CPL 213, 514 (1993), DOI 10.1016/0009-2614(93)89151-7.
/// - `JK`:   fit Coulomb AND exchange (RI-JK, B half-transform route);
///           Weigend, PCCP 4, 4285 (2002), DOI 10.1039/B204199P.
///
/// The auxiliary basis is selected by `AuxBasis` (def2 universal JKFIT —
/// one table serves both modes; see basis_set.h for the no-JFIT
/// decision). Fit errors are second order in the fitting residual under
/// the Coulomb metric: tens of μHa/atom for matched orbital/auxiliary
/// pairs (Weigend, PCCP 8, 1057 (2006), DOI 10.1039/b515623h).
enum class RI { None, J, JK };

/// @brief Coulomb-matrix (J) algorithm for the two-electron Fock build.
///
/// - `Direct`: Schwarz-screened analytic ERIs (shipping default;
///             bit-identical when `ri == RI::None`).
/// - `RI`:     density-fitted J (same engine as `ri = RI::J` / `RI::JK`);
///             selecting this with `ri == RI::None` is treated as RI-J.
/// - `CFMM`:   continuous fast multipole method for J (White–Johnson–Gill–
///             Head-Gordon extent classification + multipole far field;
///             Chem. Phys. Lett. 230, 8 (1994),
///             DOI 10.1016/0009-2614(94)01128-1). Mutually exclusive with
///             `ri = RI::J/JK` — both set throws. Exchange (K) never uses CFMM.
enum class Coulomb { Direct, RI, CFMM };

/// @brief SCF Fock-mixer family for the mix stage.
///
/// - `CDIIS`:       Pulay commutator-DIIS only (the shipping default).
/// - `EDIIS_CDIIS`: energy-DIIS far from convergence, permanently switching
///                  to CDIIS once max|DIIS error| < `mixer_switch_tol`
///                  (Kudin-Scuseria-Cancès, DOI 10.1063/1.1470195).
/// - `ADIIS_CDIIS`: augmented-Roothaan-Hall DIIS far from convergence with
///                  the same permanent CDIIS switch (Hu-Yang,
///                  DOI 10.1063/1.3304922).
///
/// EDIIS/ADIIS are far-field mixers only — run to the end they stall well
/// above tight convergence, hence the escalation-to-CDIIS design.
enum class MixerKind { CDIIS, EDIIS_CDIIS, ADIIS_CDIIS };

/// @brief Implicit-solvent continuum model attached to a mean-field method.
///
/// Active when `solvent_eps > 0`. Conductor-like models share the S-matrix
/// equation \(S q = -f(\varepsilon) V\) with \(f=(\varepsilon-1)/(\varepsilon+x)\):
/// - `CPCM`: \(x=0\) (Barone–Cossi; use with `solvent_x = 0`).
/// - `COSMO`: \(x=\tfrac12\) (Klamt; use with `solvent_x = 0.5`).
/// - `IEFPCM`: integral-equation formalism (Cancès–Mennucci–Tomasi) with
///   \(f=(\varepsilon-1)/(\varepsilon+1)\) and the \(K q = R V\) system
///   involving the double-layer operator \(D\) (PySCF `method='IEF-PCM'`).
///
/// Non-electrostatic terms (cavitation, dispersion, repulsion) are never
/// included. Default `CPCM` keeps the historical vacuum/C-PCM path.
enum class SolventModel { CPCM, COSMO, IEFPCM };

/// @brief Maximum-overlap occupation selection for ΔSCF excited states.
///
/// - `Off`:  energy-ordered Aufbau (shipping default; bit-identical).
/// - `MOM`:  Gilbert–Besley–Gill maximum-overlap method — each SCF
///           iteration occupies the n_occ new MOs with largest projection
///           onto the **previous** iteration's occupied subspace
///           (J. Phys. Chem. A **112**, 13164 (2008),
///           DOI 10.1021/jp801738f).
/// - `IMOM`: initial maximum-overlap method — same projection scores, but
///           the reference occupied subspace is **frozen** at the seed
///           (Barca, Gilbert & Gill, JCTC **14**, 1501 (2018),
///           DOI 10.1021/acs.jctc.7b00994). Matches PySCF `scf.addons.mom_occ_`.
enum class MomStyle { Off, MOM, IMOM };

} // namespace scf
} // namespace cpu
} // namespace ATV
