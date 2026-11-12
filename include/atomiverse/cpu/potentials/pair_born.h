/// @file cpu/potentials/pair_born.h
/// @brief Born–Mayer–Huggins pair: U = A exp((σ−r)/ρ) − C/r⁶ + D/r⁸ (CPU).
///
/// Matches LAMMPS `pair_style born` with energy shift at cutoff
/// (`pair_modify shift yes`, the default here; `set_energy_shift(false)` is
/// `shift no`) and, optionally, the analytic tail correction (`pair_modify
/// tail yes`, `set_tail_correction`). Units: energy [eV], length [Å].
/// Multi-type tables enable Fumi–Tosi molten-salt short-range (compose with
/// `PairCoulCut` / `coul/long` for the Coulomb term).
#pragma once

#include <array>
#include <span>
#include <string>
#include <vector>

#include "atomiverse/cpu/component/curvature_protocol.h"
#include "atomiverse/cpu/potentials/pair.h"
#include "atomiverse/cpu/potentials/pair_loop.h"

namespace ATV {
namespace cpu {

/// @brief One Born–Mayer–Huggins type-pair coefficient entry.
///
/// Frame `kAtom_type` values are used as-is and are never renumbered. That
/// column is the FORCE-FIELD TYPE index, not the species: `cpu::from_molpy`
/// (and so the Python binding) fills it from the molpy `atoms.type_id` column,
/// while the atomic number lives in `kAtom_atomic_number`. A Na–Cl row keys
/// whatever type ids the frame gives Na and Cl, e.g. `{0, 1}`. Symmetric
/// table: (i,j) and (j,i) share the same parameters.
struct BornTypeCoeff {
    int type_i;   ///< first atom type (frame `kAtom_type`)
    int type_j;   ///< second atom type
    double A;     ///< Born–Mayer prefactor [eV]
    double rho;   ///< Softness length [Å] (> 0)
    double sigma; ///< Ionic-pair radius parameter [Å]
    double C;     ///< Dispersion coefficient [eV·Å⁶]
    double D;     ///< \(r^{-8}\) coefficient [eV·Å⁸]
};

/// @brief Born–Mayer–Huggins pair potential (shifted at cutoff by default).
///
/// \[
/// U(r)=A\,e^{(\sigma-r)/\rho}-\frac{C}{r^{6}}+\frac{D}{r^{8}}-U_{\mathrm{shift}},
/// \quad r < r_c
/// \]
/// with
/// \(U_{\mathrm{shift}}=A e^{(\sigma-r_c)/\rho}-C/r_c^{6}+D/r_c^{8}\), or
/// \(U_{\mathrm{shift}}=0\) after `set_energy_shift(false)`.
///
/// Force: \(F=-\nabla U\) with
/// \(\mathrm{d}U/\mathrm{d}r=-(A/\rho)\,e^{(\sigma-r)/\rho}+6C/r^{7}-8D/r^{9}\)
/// (unshifted; shift is constant).
///
/// **Tail correction** (`set_tail_correction(true)`, LAMMPS `pair_modify tail
/// yes`): with a homogeneous fluid beyond the cutoff (g(r) = 1) every ordered
/// type pair (i, j) with global counts \(N_i, N_j\) adds
/// \(e_{ij}/V\) to the energy and \(p_{ij}/V\) to each diagonal virial
/// component, where
/// \(e_{ij}=2\pi N_iN_j\int_{r_c}^{\infty}r^2U(r)\,\mathrm{d}r\) and
/// \(p_{ij}=-\tfrac{2\pi}{3}N_iN_j\int_{r_c}^{\infty}r^3U'(r)\,\mathrm{d}r\)
/// (the `PairLJCut` convention), in closed form
/// \(e_{ij}=2\pi N_iN_j[A\rho e^{(\sigma-r_c)/\rho}(r_c^2+2\rho r_c+2\rho^2)
///   -C/(3r_c^3)+D/(5r_c^5)]\) and
/// \(p_{ij}=-\tfrac{2\pi}{3}N_iN_j[-Ae^{(\sigma-r_c)/\rho}(r_c^3+3\rho r_c^2
///   +6\rho^2r_c+6\rho^3)+2C/r_c^3-8D/(5r_c^5)]\). The single-type form
/// counts every atom as its one type. Independent of the energy shift.
///
/// Two walks: both arms take the published half list `ctx.md.pair_list` when
/// it covers the cutoff (`PairList::cutoff`, set by `NeighborList`; the
/// production bulk path, the partner's type read from the ghost rows), and
/// fall back to an O(N²) all-pairs walk at minimum image otherwise. A list
/// that does not cover the cutoff is never walked: `as_neighbor_consumer()`
/// returns null, so no run is ever REQUIRED to install a `NeighborList` for
/// this style and no build-time check compares the two radii.
///
/// Coulomb is a separate component (`PairCoulCut` / `CoulLong` + a reciprocal
/// partner).
///
/// **Curvature** (`CurvatureSource`, configurational temperature): over the
/// pairs the force walk visits,
/// \(\varphi' = -\tfrac{A}{\rho}E + 6C/r^7 - 8D/r^9\),
/// \(\varphi'' = \tfrac{A}{\rho^2}E - 42C/r^8 + 72D/r^{10}\),
/// \(\varphi''' = -\tfrac{A}{\rho^3}E + 336C/r^9 - 720D/r^{11}\)
/// (\(E = e^{(\sigma-r)/\rho}\)) give the Hessian, the Laplacian
/// \(L = \varphi'' + 2\varphi'/r\) and \(L' = \varphi''' + 2\varphi''/r
/// - 2\varphi'/r^2\) (`pair_curvature.h`). The tail correction depends on
/// the volume only and adds nothing.
///
/// Writes energy to `kEnergy` + `kEnergyPair` and the pair virial
/// \(\sum \Delta r \otimes f\) to `kStress` (Voigt xx yy zz xy xz yz [eV],
/// LAMMPS sign). The virial is what makes this style usable under a barostat:
/// in a molten/crystalline salt the Born–Mayer wall is the entire repulsive
/// half of the pressure, and a constant-pressure run without it collapses the
/// cell onto the Madelung attraction.
class PairBorn : public Pair, public CurvatureSource {
  public:
    bool requires_full_frame() const override { return true; }
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

    /// @brief LAMMPS `pair_modify shift`: subtract \(U(r_c)\) from every
    ///        counted pair so the energy is continuous at the cutoff.
    ///        Forces are unaffected. Default on.
    /// @param on Whether the shift is subtracted.
    /// @return `*this` (lvalue) or the moved evaluator (rvalue).
    PairBorn& set_energy_shift(bool on) &;
    PairBorn set_energy_shift(bool on) &&;

    /// @brief LAMMPS `pair_modify tail yes` — analytic homogeneous-fluid tail
    ///        energy and pressure beyond the cutoff (see the class comment).
    ///        Type counts are taken on the first evaluation after `build()`;
    ///        every evaluation adds them at the current box volume. Default
    ///        off.
    /// @param on Whether the tail is added.
    /// @return `*this` (lvalue) or the moved evaluator (rvalue).
    PairBorn& set_tail_correction(bool on) &;
    PairBorn set_tail_correction(bool on) &&;

    CurvatureSource* as_curvature_source() override { return this; }
    void add_curvature(const LaunchContext& ctx,
                       const CurvatureRows& rows) override;

    /// @brief True when a per-type coefficient table is active.
    bool is_multi_type() const { return multi_; }

    /// @brief The per-type table reads the neighbour's force-field type.
    static constexpr std::array kTypedGhosts = {
        GhostField::read_stable(kAtom_type, "PairBorn"),
    };
    std::span<const GhostField> declare_ghosts() const override {
        if (multi_)
            return kTypedGhosts;
        return {};
    }

    /// @brief O(N²) path: withdraw neighbor-list gate.
    const NeighborConsumer* as_neighbor_consumer() const final {
        return nullptr;
    }

    /// @brief Forget the walk's cached row partition and the tail's type
    ///        counts (they belong to the frame of this build).
    void build(const BuildContext&) override {
        loop_.reset();
        tail_ready_ = false;
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
    /// @brief Gather the type counts and set `etail_pref_` / `ptail_pref_`.
    void prepare_tail_(const Frame& frame);
    /// The published half list if it lists every pair inside the cutoff,
    /// else null (the all-pairs walk runs).
    const md::PairList* covering_list_(const LaunchContext& ctx) const;

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

    bool energy_shift_ = true; ///< `pair_modify shift` (set_energy_shift).
    bool tail_ = false;        ///< `pair_modify tail` (set_tail_correction).
    bool tail_ready_ = false;
    /// E_tail = etail_pref_/V; each diagonal virial += ptail_pref_/V.
    double etail_pref_ = 0.0, ptail_pref_ = 0.0;

    md::PairLoop loop_; ///< shared CPU pair walk (cached row partition)
};

} // namespace cpu
} // namespace ATV
