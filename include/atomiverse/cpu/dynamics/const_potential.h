/// @file cpu/dynamics/const_potential.h
/// @brief Constant-potential electrode charge solve (BO dense inverse).
///
/// E1 Siepmann–Sprik / Reed / Scalfi fluctuating-charge electrodes held at
/// fixed potentials. Component on `cpu::md::MD` — **not** a Driver category
/// and **not** welded into `KSpace*`. Writes electrode `kAtom_charge` each
/// `pre_force` step; private \(A\), \(Q\), Cholesky workspace.
///
/// Model (MetalWalls / Scalfi form):
/// \[
///   U = \tfrac12 Q^{\mathsf T} A Q - Q^{\mathsf T}(\Psi - \Phi_{\mathrm{elec}})
/// \]
/// with Gaussian electrode width \(w\) [Å]:
/// \[
///   A_{ii} = k_e\sqrt{2/\pi}/w,\quad
///   A_{ij} = k_e\,\mathrm{erf}(r_{ij}/(w\sqrt2))/r_{ij}\ (i\neq j),
/// \]
/// \(\Phi_i\) from non-electrode (point) charges via min-image Coulomb, and
/// electrode electroneutrality \(\sum Q_i = -\sum q_{\mathrm{elec}}\) (global
/// neutrality when electrolyte charge is free; for neutral electrolyte
/// \(\sum Q = 0\)).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {

/// @brief Constant-potential BO electrode charge solver (dense, frozen walls).
///
/// Construct with parallel-plate or multi-electrode type/potential lists.
/// Parameters are ctor args — no public `*Config` bag.
///
/// Default binding: `md::Phase::pre_force @ -500` (after Zero\*, before force).
class ConstPotentialBO : public cpu::Component {
  public:
    /// @param electrode_types Atom type id per electrode (order = electrode index).
    /// @param potentials_V    Applied potential \(\Psi\) [V = eV/e] per electrode.
    /// @param gaussian_width_A Gaussian charge width \(w\) [Å] (MetalWalls-class).
    ConstPotentialBO(std::vector<int> electrode_types,
                     std::vector<double> potentials_V,
                     double gaussian_width_A = 0.5);

    std::string name() const override { return "const_potential_bo"; }

    /// @brief Declares the mid-step charge rewrite: under MPI DD the MD
    /// solver posts `ChargePacker` through `ctx.md.comm->forward_comm`.
    bool updates_owned_charges() const override { return true; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::pre_force, -500)};
    }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    /// @brief Total charge on electrode `e` after the last solve [e].
    double total_charge(int electrode) const;

    /// @brief Number of electrodes.
    int n_electrodes() const {
        return static_cast<int>(electrode_types_.size());
    }

    /// @brief Number of electrode atoms in the last build.
    int n_electrode_atoms() const { return n_elec_; }

    /// @brief Residual \(|\sum Q_i + Q_{\mathrm{electrolyte}}|\) after last solve [e].
    double neutrality_residual() const { return neutrality_residual_; }

  private:
    void rebuild_A_(const Frame& frame);
    void solve_(Frame& frame);

    std::vector<int> electrode_types_;
    std::vector<double> potentials_; ///< Ψ [V]
    double w_;                       ///< Gaussian width [Å]

    /// Local owned electrode atom indices and electrode id per site.
    std::vector<int> elec_idx_;
    std::vector<int> elec_id_; ///< electrode index for each entry of elec_idx_
    int n_elec_ = 0;

    /// Dense A (n×n, row-major) and Cholesky factor L (lower).
    std::vector<double> A_;
    std::vector<double> L_;
    std::vector<double> b_;
    std::vector<double> q_;
    /// Augmented system size n+1 for neutrality Lagrange multiplier.
    std::vector<double> Aaug_;
    std::vector<double> Laug_;
    std::vector<double> baug_;
    std::vector<double> xaug_;

    double neutrality_residual_ = 0.0;
    std::vector<double> Q_per_electrode_;
    bool a_built_ = false;
};

} // namespace cpu
} // namespace ATV
