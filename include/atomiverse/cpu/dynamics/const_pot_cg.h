/// @file cpu/dynamics/const_pot_cg.h
/// @brief Constant-potential electrode charge solve via conjugate gradient (E2).
///
/// Same Gaussian elastance \(A\) and electrolyte RHS \(b\) as `ConstPotBO`, but
/// solves \(A Q = \Psi-b\) (and the neutrality projector) with Jacobi-PCG
/// instead of dense inverse. Component on `cpu::md::MD` at `pre_force`; writes
/// electrode charges to `kAtom_charge`. Not a Driver category and not part of
/// `KSpace*`.
#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h" // ElectrodeGroup

namespace ATV {
namespace cpu {
namespace md {

/// @brief Constant-potential CG electrode charge solver (large \(N_{\mathrm{elec}}\)).
///
/// Parameters are constructor arguments and public members — no Config bag.
///
/// @code
/// cpu::md::ConstPotCG conp(/*eta [1/Å]=*/2.0);
/// conp.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
/// conp.add_electrode(/*type=*/2, /*Psi [V]=*/1.0);
/// conp.electroneutrality = true;
/// conp.cg_tol = 1e-12;
/// sim.on(md::Phase::pre_force).add(std::move(conp));
/// @endcode
class ConstPotCG : public cpu::Component {
  public:
    /// @param eta Gaussian charge width \(\eta\) [1/Å] (shared by all
    ///            electrode sites in v1).
    explicit ConstPotCG(double eta);

    /// @brief Register one electrode group (type + applied potential).
    /// @param type Atom type of electrode sites.
    /// @param psi Applied potential [V].
    ConstPotCG& add_electrode(int type, double psi) &;
    ConstPotCG add_electrode(int type, double psi) &&;

    std::string name() const override { return "const_pot_cg"; }

    /// @brief Declares the mid-step charge rewrite: under MPI DD the MD
    /// solver posts `ChargePacker` through `ctx.md.comm->forward_comm`.
    bool updates_owned_charges() const override { return true; }

    std::vector<DefaultBinding> default_bindings() const override {
        // Same slot as ConstPotBO: before pair/kspace; after Zero*.
        return {ATV::make_binding(ATV::md::Phase::pre_force, -50)};
    }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    /// @brief Total charge on electrode group `ie` [e] after last launch.
    /// @param ie Electrode index in registration order.
    double total_charge(std::size_t ie) const;

    /// @brief \(\Psi_{\mathrm{last}}-\Psi_{\mathrm{first}}\) [V].
    double delta_psi() const;

    /// @brief \(\lvert\sum_i Q_i\rvert\) [e] over all electrode sites.
    double neutrality_residual() const;

    /// @brief \(\lVert AQ-(\Psi-b)\rVert_\infty\) after last solve.
    double solve_residual() const { return residual_; }

    /// @brief Number of electrode atoms in the last solve.
    int n_electrode() const { return static_cast<int>(elec_atoms_.size()); }

    /// @brief CG iterations used by the primary \(A v=\Psi-b\) solve (last launch).
    int cg_iterations() const { return cg_iters_primary_; }

    // ── Public parameters (mutable after construction) ───────────────
    double eta = 1.979;            ///< Gaussian width [1/Å].
    bool electroneutrality = true; ///< Project \(\sum Q_i = 0\).
    /// Half-range of 2D \(xy\) lattice images (0 = min-image / primary).
    int xy_image_range = 0;
    /// Rebuild \(A\) every launch (movable electrodes).
    bool rebuild_matrix_every_step = false;
    /// Relative CG residual tolerance [dimensionless].
    double cg_tol = 1e-12;
    /// Max CG iterations per linear solve [dimensionless].
    int max_cg_iter = 1000;

  private:
    std::vector<ElectrodeGroup> groups_;

    std::vector<int> elec_atoms_;
    std::vector<int> elec_group_;
    std::vector<double> psi_vec_;
    std::vector<double> Q_;
    std::vector<double> b_;
    std::vector<double> A_; ///< Row-major elastance [V/e].
    std::vector<double> total_Q_;
    bool matrix_ready_ = false;
    double residual_ = 0.0;
    int cg_iters_primary_ = 0;

    void collect_electrodes_(const cpu::Frame& frame);
    void build_matrix_(const cpu::Frame& frame);
    void build_rhs_(const cpu::Frame& frame);
    void solve_();
    void write_charges_(cpu::Frame& frame) const;
    double pair_A_(double r, double eta_i, double eta_j) const;
    double pair_b_(double r, double q, double eta_i) const;

    /// Jacobi-PCG for dense SPD \(A x = rhs\). Returns iteration count.
    int pcg_solve_(const std::vector<double>& rhs,
                   std::vector<double>& x) const;
};

} // namespace md
} // namespace cpu
} // namespace ATV
