/// @file cpu/dynamics/qeq.h
/// @brief QEq / fluctuating-charge equilibration (LAMMPS `fix qeq/point` twin).
///
/// Electronegativity-equalization charges for electrolyte / molecular systems
/// (Rappé–Goddard 1991 QEq, MetalWalls capability map §2 "fluctuating-charge /
/// QEq"). Deliberately **not** the constant-potential electrode solvers
/// (`ConstPotBO` / `ConstPotCG` / `ConstPotMaZe` solve fixed-Ψ Gaussian
/// electrodes); QEq equilibrates every parameterized atom against its own
/// electronegativity. Component on `cpu::md::MD` at `pre_force` — never a
/// Driver category, never inside `KSpace*`. Writes `kAtom_charge`.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Charge equilibration minimizing
/// \(E(q)=\sum_i(\chi_i q_i + \tfrac12\eta_i q_i^2)
///   + \sum_{i<j,\,r\le r_c} q_i q_j / r_{ij}\)
/// subject to \(\sum_i q_i = q_{\mathrm{tot}}\).
///
/// The coupling kernel is exactly LAMMPS `fix qeq/point`: bare \(1/r\) inside
/// a hard cutoff with **no** Coulomb prefactor on the off-diagonal (the
/// literal LAMMPS convention — `compute_H` fills `0.5/r` on a full list),
/// diagonal \(\eta_i\), RHS \(-\chi_i\), and the two-vector constraint
/// \(q = s - ((\Sigma s - q_{\mathrm{tot}})/\Sigma t)\,t\) where
/// \(H s = -\chi\) and \(H t = -\mathbf 1\). Distances use the frame box
/// minimum image. Dense Cholesky solve, rebuilt every launch (charges follow
/// the moving geometry).
///
/// Parameters are constructor arguments and public members — no Config bag:
/// @code
/// cpu::md::QEq qeq(/*cutoff [Å]=*/10.0);
/// qeq.set_type(/*type=*/1, /*chi [eV]=*/5.343, /*eta [eV/e²]=*/10.126);
/// qeq.set_type(2, 8.741, 13.364);
/// sim.on(md::Phase::pre_force).add(std::move(qeq));
/// @endcode
class QEq : public cpu::Component {
  public:
    /// @param cutoff Coulomb coupling cutoff \(r_c\) [Å] (> 0).
    explicit QEq(double cutoff);

    /// @brief Register electronegativity / hardness for one atom type.
    ///        Every type present in the frame must be registered (no
    ///        fallback — `build()` throws on an unparameterized type).
    /// @param type Atom type id (matches `kAtom_type`).
    /// @param chi Electronegativity \(\chi\) [eV] (LAMMPS qeq param file col 2).
    /// @param eta Hardness \(\eta\) [eV/e²] (> 0; param file col 3).
    QEq& set_type(int type, double chi, double eta) &;
    QEq set_type(int type, double chi, double eta) &&;

    std::string name() const override { return "qeq"; }

    /// @brief QEq rewrites owned `kAtom_charge` at `pre_force`, then posts
    ///        `ChargePacker` through `ctx.md.comm->forward_comm`.
    bool updates_owned_charges() const override { return true; }

    std::vector<DefaultBinding> default_bindings() const override {
        // Before pair/kspace force components so they see equilibrated
        // charges; same slot as the const-potential electrode solvers.
        return {ATV::make_binding(ATV::md::Phase::pre_force, -50)};
    }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    /// @brief Equilibrated charges [e] in frame-row order (last launch).
    const std::vector<double>& charges() const { return q_; }

    /// @brief QEq objective \(\sum\chi q + \tfrac12 q^{\mathsf T}Hq\) at the
    ///        last solve [eV] (bare-\(1/r\) kernel convention, as LAMMPS).
    double qeq_energy() const { return energy_; }

    /// @brief \(\sum_i q_i\) after the last solve [e] (equals `qtot`).
    double net_charge() const;

    /// @brief Number of atoms in the last solve.
    int n_solved() const { return static_cast<int>(q_.size()); }

    // ── Public parameters (mutable after construction) ───────────────
    double cutoff = 10.0; ///< Coupling cutoff [Å].
    /// Constrained total charge \(q_{\mathrm{tot}}\) [e]. LAMMPS `fix qeq`
    /// always drives \(\sum q\to 0\); nonzero models a net-charged system.
    double qtot = 0.0;

  private:
    struct TypeParam {
        int type;
        double chi; ///< [eV]
        double eta; ///< [eV/e²]
    };
    std::vector<TypeParam> params_;

    std::vector<double> chi_i_; ///< Per-atom χ [eV].
    std::vector<double> eta_i_; ///< Per-atom η [eV/e²].
    std::vector<double> H_;     ///< Row-major (η δ + 1/r) [eV/e²].
    std::vector<double> q_;     ///< Solved charges [e].
    double energy_ = 0.0;

    void collect_params_(const cpu::Frame& frame);
    void build_matrix_(const cpu::Frame& frame);
    void solve_and_write_(cpu::Frame& frame);
};

} // namespace md
} // namespace cpu
} // namespace ATV
