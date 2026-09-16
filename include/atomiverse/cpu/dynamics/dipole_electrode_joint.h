/// @file cpu/dynamics/dipole_electrode_joint.h
/// @brief Coupled PIM induced dipoles ⊕ constant-potential electrode \(Q\) (E5).
///
/// MetalWalls main hardness (JCP 2022 / map E5): alternating BO electrode
/// solve and Thole-damped induced-dipole SCC until joint residual converges.
/// Component on `cpu::md::MD` at `pre_force` — **not** a Driver category and
/// **not** inside `KSpace*`. Writes electrode charges to `kAtom_charge`.
/// Parameters are ctor args / public members (no Config bag).
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"

namespace ATV {
namespace cpu {
class Box; // frame/box.h — Spans holds a non-owning pointer
namespace md {

/// @brief Joint electrode \(Q\) + electrolyte induced-dipole Component.
///
/// Parallel-plate (E5 foothold):
/// @code
/// cpu::md::DipoleElectrodeJoint joint(/*eta=*/2.0, /*alpha=*/{0,0,0,1.0,1.5});
/// joint.add_electrode(1, 0.0);
/// joint.add_electrode(2, 1.0);
/// joint.thole_a = 0.39;
/// joint.electroneutrality = true;
/// sim.on(md::Phase::pre_force).add(std::move(joint));
/// @endcode
class DipoleElectrodeJoint : public cpu::Component {
  public:
    /// @param eta Gaussian electrode width \(\eta\) [1/Å].
    /// @param alpha_by_type Isotropic polarizability [Å³] by `kAtom_type`
    ///        index. Electrode types should have \(\alpha=0\). Types outside
    ///        the vector → \(\alpha=0\).
    DipoleElectrodeJoint(double eta, std::vector<double> alpha_by_type);

    /// @brief Register one electrode group (type + applied potential).
    /// @param type Atom type of electrode sites.
    /// @param psi Applied potential [V].
    DipoleElectrodeJoint& add_electrode(int type, double psi) &;
    DipoleElectrodeJoint add_electrode(int type, double psi) &&;

    std::string name() const override { return "dipole_electrode_joint"; }

    /// @brief Declares the mid-step charge rewrite: under MPI DD the MD
    /// solver posts `ChargePacker` through `ctx.md.comm->forward_comm`.
    bool updates_owned_charges() const override { return true; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::pre_force, -50)};
    }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;
    void destroy(const DestroyContext& ctx) override;

    /// @brief Total charge on electrode group `ie` [e] after last launch.
    double total_charge(std::size_t ie) const;

    /// @brief \(\lvert\sum_i Q_i\rvert\) [e] over electrode sites.
    double neutrality_residual() const;

    /// @brief \(\lVert A Q - (\Psi - b)\rVert_\infty\) after last outer iter.
    double charge_residual() const { return charge_residual_; }

    /// @brief Max \(|\Delta\mu|\) [e·Å] of the last outer iteration.
    double dipole_residual() const { return dipole_residual_; }

    /// @brief Outer joint iterations used on the last launch.
    int last_joint_iters() const { return last_joint_iters_; }

    /// @brief Whether the joint loop met `joint_tol` on the last launch.
    bool last_joint_converged() const { return last_joint_converged_; }

    /// @brief Induced dipoles of the last launch (packed xyz, e·Å; full nlocal).
    const std::vector<double>& last_mu() const { return last_mu_; }

    /// @brief Number of electrode atoms in the last solve.
    int n_electrode() const { return static_cast<int>(elec_atoms_.size()); }

    // ── Public parameters ────────────────────────────────────────────
    double eta = 1.979;            ///< Gaussian width [1/Å].
    bool electroneutrality = true; ///< Project \(\sum Q_i = 0\).
    int xy_image_range = 0; ///< 2D \(xy\) lattice half-range (0 = min-image).
    bool rebuild_matrix_every_step = false;
    double thole_a = 0.39;  ///< Thole \(a\) (0 = undamped multipoles).
    double pol_tol = 1e-10; ///< Inner PIM SCC tol [e·Å].
    int pol_max_iter = 100; ///< Inner PIM SCC budget.
    double joint_tol =
        1e-10; ///< Outer max |\(\Delta Q\)| [e] and |\(\Delta\mu\)| [e·Å].
    int joint_max_iter = 50;           ///< Outer joint iteration budget.
    std::vector<double> alpha_by_type; ///< \(\alpha\) [Å³] by type index.

  private:
    std::vector<ElectrodeGroup> groups_;

    std::vector<int> elec_atoms_;
    std::vector<int> elec_group_;
    std::vector<double> psi_vec_;
    std::vector<double> Q_;
    std::vector<double> b_;
    std::vector<double> A_;
    std::vector<double> A_work_;
    std::vector<double> total_Q_;
    std::vector<char> is_elec_;
    std::vector<double> last_mu_;
    bool matrix_ready_ = false;

    double charge_residual_ = 0.0;
    double dipole_residual_ = 0.0;
    int last_joint_iters_ = 0;
    bool last_joint_converged_ = false;

    /// Cached Frame column pointers for one launch (budget-safe: resolve once).
    struct Spans {
        const StorageReal* x = nullptr;
        const StorageReal* y = nullptr;
        const StorageReal* z = nullptr;
        StorageReal* charge = nullptr;
        const int* type = nullptr;
        int n = 0;
        const cpu::Box* box = nullptr;
    };

    void collect_electrodes_(const cpu::Frame& frame);
    void build_matrix_(const cpu::Frame& frame);
    void build_rhs_(const Spans& s, const std::vector<double>& mu);
    void solve_Q_();
    void write_charges_(const Spans& s) const;
    /// @brief Inner PIM SCC; updates `last_mu_`. Returns max |\(\Delta\mu\)|.
    double scc_mu_(const Spans& s);
    double alpha_of_(int type) const;
    double pair_A_(double r, double eta_i, double eta_j) const;
    double pair_b_(double r, double q, double eta_i) const;
};

} // namespace md
} // namespace cpu
} // namespace ATV
