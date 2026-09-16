/// @file cpu/potentials/pair_born.h
/// @brief Born–Mayer–Huggins pair: U = A exp((σ−r)/ρ) − C/r⁶ + D/r⁸ (CPU).
///
/// Matches LAMMPS `pair_style born` with energy shift at cutoff
/// (`pair_modify shift yes`): \(U(r_c)=0\). Units: energy [eV], length [Å].
/// Multi-type tables enable Fumi–Tosi molten-salt short-range (compose with
/// `PairCoulCut` / `coul/long` for the Coulomb term).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/cpu/potentials/pair.h"

namespace ATV {
namespace cpu {

/// @brief One Born–Mayer–Huggins type-pair coefficient entry.
///
/// Frame `kAtom_type` values are used as-is and are never renumbered. In this
/// tree that column carries the ATOMIC NUMBER — `cpu::from_molrs` fills it from
/// the molrs `atomic_number` column — so a Na–Cl row is `{11, 17}` and an argon
/// dimer is `{18, 18}`, not a 0-based dense index. Symmetric table: (i,j) and
/// (j,i) share the same parameters.
struct BornTypeCoeff {
    int type_i;   ///< first atom type (frame `kAtom_type`)
    int type_j;   ///< second atom type
    double A;     ///< Born–Mayer prefactor [eV]
    double rho;   ///< Softness length [Å] (> 0)
    double sigma; ///< Ionic-pair radius parameter [Å]
    double C;     ///< Dispersion coefficient [eV·Å⁶]
    double D;     ///< \(r^{-8}\) coefficient [eV·Å⁸]
};

/// @brief Born–Mayer–Huggins pair potential (shifted at cutoff).
///
/// \[
/// U(r)=A\,e^{(\sigma-r)/\rho}-\frac{C}{r^{6}}+\frac{D}{r^{8}}-U_{\mathrm{shift}},
/// \quad r < r_c
/// \]
/// with
/// \(U_{\mathrm{shift}}=A e^{(\sigma-r_c)/\rho}-C/r_c^{6}+D/r_c^{8}\).
///
/// Force: \(F=-\nabla U\) with
/// \(\mathrm{d}U/\mathrm{d}r=-(A/\rho)\,e^{(\sigma-r)/\rho}+6C/r^{7}-8D/r^{9}\)
/// (unshifted; shift is constant).
///
/// Two walks, and which one runs is decided by the CONSTRUCTOR, not only by
/// what the run publishes: the SINGLE-TYPE arm uses `ctx.md.pair_list` when a
/// `NeighborList` published one (the production bulk path) and falls back to an
/// O(N²) all-pairs walk otherwise, while the MULTI-TYPE table arm is always
/// O(N²) — it has no per-type list to walk. So a one-row table is not a
/// performance substitute for the single-type constructor, only an energetic
/// one. Either way `as_neighbor_consumer()` returns null, so no run is ever
/// REQUIRED to install a `NeighborList` for this style.
///
/// Coulomb is a separate component (`PairCoulCut` / `CoulLong` + a reciprocal
/// partner).
///
/// Writes energy to `kEnergy` + `kEnergyPair` and the pair virial
/// \(\sum \Delta r \otimes f\) to `kStress` (Voigt xx yy zz xy xz yz [eV],
/// LAMMPS sign). The virial is what makes this style usable under a barostat:
/// in a molten/crystalline salt the Born–Mayer wall is the entire repulsive
/// half of the pressure, and a constant-pressure run without it collapses the
/// cell onto the Madelung attraction.
class PairBorn : public Pair {
  public:
    /// @param A      Born–Mayer prefactor [eV] (must be ≥ 0).
    /// @param rho    Softness length [Å] (must be > 0).
    /// @param sigma  Ionic-pair radius parameter [Å].
    /// @param C      Dispersion coefficient [eV·Å⁶] (must be ≥ 0).
    /// @param D      \(r^{-8}\) coefficient [eV·Å⁸].
    /// @param cutoff Pair cutoff [Å] (must be > 0).
    PairBorn(double A, double rho, double sigma, double C, double D,
             double cutoff);

    /// @brief Multi-type Fumi–Tosi table constructor.
    ///
    /// Every interacting type pair must be listed. Unlisted pairs do not
    /// interact (no silent mixing — Fumi–Tosi parameters are not Lorentz–
    /// Berthelot).
    ///
    /// @param coeffs Type-pair coefficients; must not be empty.
    /// @param cutoff Interaction cutoff [Å] (> 0).
    /// @throws std::invalid_argument if `coeffs` is empty or a parameter is invalid.
    PairBorn(std::vector<BornTypeCoeff> coeffs, double cutoff);

    std::string name() const override { return "born"; }

    /// @brief True when a per-type coefficient table is active.
    bool is_multi_type() const { return multi_; }

    /// @brief O(N²) path: withdraw neighbor-list gate.
    const NeighborConsumer* as_neighbor_consumer() const final {
        return nullptr;
    }

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    struct Entry {
        double A = 0;
        double inv_rho = 0;
        double sigma = 0;
        double C = 0;
        double D = 0;
        double v_shift = 0;
        bool set = false;
    };

    const Entry& entry_for_(int ti, int tj) const;

    // Single-type path
    double A_ = 0;
    double rho_ = 0;
    double sigma_ = 0;
    double C_ = 0;
    double D_ = 0;
    double inv_rho_ = 0;
    double v_shift_ = 0;

    // Multi-type path
    bool multi_ = false;
    int n_types_ = 0;
    std::vector<Entry> table_; ///< dense n_types_ × n_types_

    double cutoff_sq_ = 0;
};

} // namespace cpu
} // namespace ATV
