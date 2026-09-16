/// @file cpu/dynamics/const_pot_maze.h
/// @brief Mass-zero constrained (MaZe) electrode charge dynamics.
///
/// True symplectic mass-zero constrained charge dynamics (Coretti, Scalfi,
/// Bacon, Rotenberg, Vuilleumier, Salanne, Ciccotti, Bonella, J. Chem.
/// Phys. 152, 194701 (2020)): electrode charges are auxiliary degrees of
/// freedom evolved by Verlet extrapolation \(\tilde Q = 2Q_n - Q_{n-1}\)
/// with SHAKE-like corrections **along the constraint gradients**
/// (\(Q_{n+1} = \tilde Q + A\gamma\), \(A^2\gamma = -\sigma(\tilde Q)\))
/// until the constant-potential constraint
/// \(\sigma_i = (AQ - (\Psi - b))_i - \chi = 0\) holds to `maze_tol`
/// (\(\chi\) = uniform potential shift enforcing electroneutrality).
/// Solved tightly the ionic dynamics is time-reversible with no secular
/// energy drift — unlike dissipative predictor / loose-SCF schemes.
///
/// `symplectic = false` keeps the wave-4 adiabatic foothold twin
/// (CG + ASPC on `ConstPotBO` each `pre_force`). Component on
/// `cpu::md::MD` at `pre_force`; never a Driver category, never in KSpace*.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Mass-zero constrained (SHAKE-like) constant-potential charges.
///
/// Parameters are public members — no Config bag. Composes as a thin owner
/// of `ConstPotBO` (elastance build, RHS, write-back are reused, never
/// duplicated).
class ConstPotMaZe : public cpu::Component {
  public:
    /// @param eta Gaussian charge width \(\eta\) [1/Å] for electrode sites.
    explicit ConstPotMaZe(double eta);

    /// @brief Register one electrode group (type + applied potential).
    /// @param type Atom type of electrode sites.
    /// @param psi Applied potential [V].
    ConstPotMaZe& add_electrode(int type, double psi) &;
    ConstPotMaZe add_electrode(int type, double psi) &&;

    std::string name() const override { return "const_pot_maze"; }

    /// @brief Declares the mid-step charge rewrite (owned ConstPotBO writes).
    bool updates_owned_charges() const override { return true; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::pre_force, -50)};
    }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    /// @brief Underlying BO solver (electrode registration, eta, images…).
    ConstPotBO& solver() { return bo_; }
    const ConstPotBO& solver() const { return bo_; }

    /// @brief Max \(|Q_i(t)-Q_i(t-1)|\) after last launch [e].
    double max_dQ() const { return max_dQ_; }

    /// @brief \(\lVert\sigma\rVert_\infty\) of the constant-potential
    ///        constraint after the last symplectic launch [V].
    double constraint_residual() const { return constraint_residual_; }

    /// @brief Constraint-CG iterations on the last symplectic launch.
    /// @return 0 when the Verlet guess already satisfied the constraint.
    int last_maze_iters() const { return last_maze_iters_; }

    /// @brief Drop the charge history — the next launch bootstraps cold.
    void clear_history();

    // ── Public parameters (members, no Config bag) ───────────────────
    /// True mass-zero constrained SHAKE path (Coretti et al. 2020).
    /// `false` = wave-4 CG+ASPC adiabatic foothold twin.
    bool symplectic = true;
    /// Constraint residual tolerance \(\lVert\sigma\rVert_\infty\) [V].
    double maze_tol = 1e-10;
    /// Hard cap on constraint-CG iterations; throws if exceeded.
    int max_maze_iters = 2000;

  private:
    ConstPotBO bo_;
    /// Verlet history of the **unconstrained** solve \(A Q = \Psi - b\)
    /// [e] — the χ neutrality shift is re-applied on commit (extrapolating
    /// committed charges would leave a constant \(|\chi|\) violation).
    std::vector<double> Q_prev_; ///< \(Q_{n-1}\) [e].
    std::vector<double> Q_curr_; ///< \(Q_n\) [e].
    std::vector<double> u_;      ///< Cached \(A^{-1}\mathbf{1}\) [e/V].
    bool have_u_ = false;
    int hist_count_ = 0; ///< 0 = none, 1 = `Q_curr_` only, 2 = both.
    double max_dQ_ = 0.0;
    double constraint_residual_ = 0.0;
    int last_maze_iters_ = 0;

    void launch_symplectic_(const LaunchContext& ctx);
    /// @brief SHAKE constraint solve: CG on \(A^2\gamma = \mathrm{rhs} -
    ///        A\tilde Q\); updates `Q` to \(\tilde Q + A\gamma\) in place.
    /// @return Iteration count.
    int shake_cg_(std::vector<double>& Q, const std::vector<double>& rhs);
    void ensure_u_();
};

} // namespace md
} // namespace cpu
} // namespace ATV
