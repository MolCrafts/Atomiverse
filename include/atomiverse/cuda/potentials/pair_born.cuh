/// @file cuda/potentials/pair_born.cuh
/// @brief Born–Mayer–Huggins pair potential (LAMMPS `pair_style born`) — CUDA.
///
/// The CUDA twin of `cpu::PairBorn` (per-type table form):
/// \f[
///   U(r) = A\,e^{(\sigma - r)/\rho} - \frac{C}{r^6} + \frac{D}{r^8}
///          - U(r_c)\ \text{(with the energy shift)}
/// \f]
/// on the cluster-tile engine shared with `PairLJCut` and `CoulLong`. A type
/// pair without a row does not interact. With Tosi–Fumi rows and the Ewald
/// pair (`CoulLong` + `KSpaceEwald`) this is the molten-salt model of
/// Petravic & Delhommelle, J. Chem. Phys. 118, 7477 (2003).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "atomiverse/cuda/component/curvature_protocol.cuh"
#include "atomiverse/cuda/potentials/pair.cuh"

namespace ATV {
namespace cuda {

/// @brief Born–Mayer–Huggins coefficients of one (type_i, type_j) pair;
///        the fields of `cpu::BornTypeCoeff`.
struct BornTypeCoeff {
    int type_i;   ///< first atom type (frame `type` column)
    int type_j;   ///< second atom type
    double A;     ///< Born–Mayer prefactor [eV]
    double rho;   ///< Softness length [Å] (> 0)
    double sigma; ///< Ionic-pair radius parameter [Å]
    double C;     ///< Dispersion coefficient [eV·Å⁶]
    double D;     ///< \(r^{-8}\) coefficient [eV·Å⁸]
};

/// @brief Per-type Born–Mayer–Huggins pair force, energy, virial and
///        curvature.
class PairBorn : public Pair, public CurvatureSource {
  public:
    /// @param coeffs One row per interacting type pair; the table is
    ///        symmetric in (type_i, type_j).
    /// @param cutoff Interaction cutoff [Å] (> 0), shared by every row.
    /// @throws std::invalid_argument on an empty table, a negative type,
    ///         rho <= 0, A < 0, C < 0 or cutoff <= 0.
    PairBorn(std::vector<BornTypeCoeff> coeffs, double cutoff);
    ~PairBorn() override;
    PairBorn(PairBorn&&) noexcept;
    PairBorn& operator=(PairBorn&&) noexcept;

    std::string name() const override { return "born"; }
    bool graph_capturable() const override { return true; }

    /// @brief LAMMPS `pair_modify shift`: subtract U(r_c) per pair
    ///        (default on).
    PairBorn& set_energy_shift(bool on) &;
    PairBorn set_energy_shift(bool on) &&;

    /// @brief LAMMPS `pair_modify tail yes`: the homogeneous energy and
    ///        pressure tail beyond the cutoff from the build-time type
    ///        counts and the live cell volume (default off). Single GPU.
    PairBorn& set_tail_correction(bool on) &;
    PairBorn set_tail_correction(bool on) &&;

    CurvatureSource* as_curvature_source() override { return this; }
    void add_curvature(const LaunchContext& ctx,
                       const CurvatureRows& rows) override;

    void build(const BuildContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

  protected:
    void compute(const LaunchContext& ctx) override;

  private:
    std::vector<BornTypeCoeff> coeffs_;
    bool energy_shift_ = true;
    bool tail_ = false;
    double etail_pref_ = 0.0; ///< Σ 2π N_i N_j ∫ r² U [eV·Å³].
    double ptail_pref_ = 0.0; ///< Σ −(2π/3) N_i N_j ∫ r³ U′ [eV·Å³].

    /// Cluster-tile engine (packed tiles, slot types, partials) and the
    /// device coefficient table, sized in build(); compute() only launches.
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace cuda
} // namespace ATV
