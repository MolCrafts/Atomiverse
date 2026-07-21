/// @file cpu/scf/internal/hueckel_guess.h
/// @brief internal::HueckelGuess — extended-Hückel initial-guess Fock matrix
///        for the far-from-convergence SCF start.
///
/// **Internal SCF helper.** Does NOT inherit `cpu::Component` and is not
/// registered into any Plan. Owned transiently by the concrete method's
/// initial-guess dispatch (`MeanFieldMethod::drive_scf_core_`) and by tests.
///
/// Extended Hückel theory as an SCF starting point (assessed in S. Lehtola,
/// J. Chem. Theory Comput. 15, 1593 (2019), DOI 10.1021/acs.jctc.8b01089):
///
///   F_μμ = H_μμ (the VSIP of the AO's element + shell type)     [Hartree]
///   F_μν = ½ K (H_μμ + H_νν) S_μν,  μ ≠ ν,  K = 1.75  (Wolfsberg-Helmholz)
///
/// One diagonalization of `F` (done by the caller, reusing the SCF eigensolver
/// + DensityBuilder) yields the initial density.
///
/// **Spin-averaging limitation:** the Hückel Fock is spin-symmetric, so
/// open-shell runs derive both spin channels from the same orbitals (the
/// existing split + HOMO-LUMO-mixing rule at the dispatch site); the guess
/// cannot break spin symmetry by itself.
///
/// The VSIP (valence-state ionization potential) table is static and inline
/// below (Hartree); unsupported (Z, l) throws — no fallback values.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

/// @brief Wolfsberg-Helmholz constant K in F_μν = ½K(H_μμ+H_νν)S_μν
///        (dimensionless).
inline constexpr double kWolfsbergHelmholz = 1.75;

/// @brief Highest atomic number with a tabulated VSIP row.
inline constexpr int kVsipMaxZ = 18;

/// @brief Static VSIP table [Hartree], rows Z = 1..18, columns l = s, p, d.
///
/// Valence s/p entries are the standard extended-Hückel parameters
/// (R. Hoffmann, J. Chem. Phys. 39, 1397 (1963), as collected in the
/// Alvarez/YAeHMOP compilation), converted from eV at 1 Ha = 27.211386245988
/// eV; noble-gas rows use the atomic valence ionization energies. Shell types
/// above an element's tabulated valence set (polarization functions: p/d on
/// H-He, d on main-group rows) REUSE the element's highest tabulated
/// valence-shell VSIP — that convention is written out per entry below (it
/// only sets the energy scale of polarization diagonals in a guess; it is a
/// table entry, not a runtime fallback).
inline constexpr double kVsipHartree[kVsipMaxZ][3] = {
    // s, p, d — p/d repeats are the recorded polarization convention.
    {-0.4997915, -0.4997915, -0.4997915}, // H : 1s -13.60 eV
    {-0.9040340, -0.9040340, -0.9040340}, // He: 1s -24.60 eV
    {-0.1984466, -0.1286228, -0.1286228}, // Li: 2s -5.40, 2p -3.50 eV
    {-0.3674932, -0.2204959, -0.2204959}, // Be: 2s -10.00, 2p -6.00 eV
    {-0.5585897, -0.3123692, -0.3123692}, // B : 2s -15.20, 2p -8.50 eV
    {-0.7864355, -0.4189423, -0.4189423}, // C : 2s -21.40, 2p -11.40 eV
    {-0.9554824, -0.4924409, -0.4924409}, // N : 2s -26.00, 2p -13.40 eV
    {-1.1870032, -0.5438900, -0.5438900}, // O : 2s -32.30, 2p -14.80 eV
    {-1.4699729, -0.6651627, -0.6651627}, // F : 2s -40.00, 2p -18.10 eV
    {-1.7823421, -0.7937854, -0.7937854}, // Ne: 2s -48.50, 2p -21.60 eV
    {-0.1874216, -0.1102480, -0.1102480}, // Na: 3s -5.10, 3p -3.00 eV
    {-0.3307439, -0.1653719, -0.1653719}, // Mg: 3s -9.00, 3p -4.50 eV
    {-0.4520167, -0.2388705, -0.2388705}, // Al: 3s -12.30, 3p -6.50 eV
    {-0.6357633, -0.3380938, -0.3380938}, // Si: 3s -17.30, 3p -9.20 eV
    {-0.6835374, -0.5145240, -0.5145240}, // P : 3s -18.60, 3p -14.00 eV
    {-0.7349865, -0.4887660, -0.4887660}, // S : 3s -20.00, 3p -13.30 eV
    {-0.9665072, -0.5218404, -0.5218404}, // Cl: 3s -26.30, 3p -14.20 eV
    {-1.0730803, -0.5806393, -0.5806393}, // Ar: 3s -29.20, 3p -15.80 eV
};

/// @brief Extended-Hückel guess Fock matrix in the working AO basis.
///
/// Constructor-only lookup class (no factory free function). Groups the
/// Cartesian shell list into angular-momentum shell blocks, maps each working
/// AO (Cartesian or spherical, decided by the dimension of `S`) to its
/// (element, l) pair, and assembles `F` from the static VSIP table and `S`.
class HueckelGuess {
  public:
    /// @brief Assemble the extended-Hückel Fock matrix.
    /// @param shells Cartesian AO shell descriptors (one entry per Cartesian
    ///               AO; centers in Bohr) — `IntegralPack::shells`.
    /// @param S      Overlap matrix (dimensionless) in the WORKING AO basis
    ///               (spherical when the basis convention is spherical); its
    ///               dimension selects the Cartesian/spherical AO mapping.
    /// @param atoms  Atom descriptors (Z, position in Bohr) supplying the
    ///               element of each shell center — `IntegralPack::atoms`.
    /// @throws std::invalid_argument on an unsupported element/shell type
    ///         (Z > 18 or l > 2), on an S dimension matching neither AO
    ///         convention, or on a shell centered on no atom. No fallback.
    HueckelGuess(const std::vector<native::Shell>& shells, const RealMatrix& S,
                 const std::vector<native::Atom>& atoms);

    /// @brief VSIP diagonal element for one element + shell type.
    /// @param Z Atomic number (1..18 supported).
    /// @param l Angular momentum of the shell (0=s, 1=p, 2=d).
    /// @return H_μμ = VSIP [Hartree] (negative).
    /// @throws std::invalid_argument outside the table — no fallback.
    static double vsip(int Z, int l) {
        if (Z < 1 || Z > kVsipMaxZ)
            throw std::invalid_argument(
                "HueckelGuess: no VSIP tabulated for element Z="
                + std::to_string(Z) + " (supported: 1.."
                + std::to_string(kVsipMaxZ) + ")");
        if (l < 0 || l > 2)
            throw std::invalid_argument(
                "HueckelGuess: no VSIP tabulated for shell type l="
                + std::to_string(l) + " (supported: s, p, d)");
        return kVsipHartree[Z - 1][l];
    }

    /// @brief The assembled extended-Hückel Fock matrix [Hartree], in the
    ///        working AO dimension of `S`.
    RealMatrix F;
};

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
