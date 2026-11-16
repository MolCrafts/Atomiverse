/// @file cpu/dynamics/configurational_temperature.h
/// @brief First-order configurational temperature \(T_{C1}\) and its
///        gradient (CPU): the assembly `CurvatureAssembly` and the observer
///        `ConfigurationalTemperature`.
///
/// J. Petravic and J. Delhommelle, J. Chem. Phys. 118, 7477 (2003), eq. (7):
/// \[
///   k_B T_{C1} = \frac{N}{D},\qquad N = \sum_i |F_i|^2,\quad
///   D = \sum_i \nabla_i^2 U = -\sum_i \nabla_i\cdot F_i,
/// \]
/// \(F_i\) the force of the interaction potential \(U\) alone: every
/// `ExternalDrive` (e.g. `FixEfield`) is subtracted from the frame forces
/// first. Its gradient, which the configurational thermostat feeds back into
/// the positions (eq. 12a), is
/// \[
///   \frac{\partial T_{C1}}{\partial r_i}
///   = -\frac{2 (H F)_i\, D + N\, \nabla_i D}{k_B D^2},
/// \]
/// \(H = \nabla\nabla U\) (\(\partial N/\partial r_i = -2(HF)_i\)).
///
/// **Electrostatics.** The paper drops the Coulomb terms because
/// \(\nabla^2 (1/r) = 0\). Under Ewald summation that is not true term by
/// term: the real-space \(\operatorname{erfc}(\alpha r)/r\) pair has
/// \(\nabla^2 = (4\alpha^3/\sqrt\pi)\,e^{-\alpha^2 r^2}\) and the reciprocal
/// sum has its own Laplacian; only their total is simple. For the periodic
/// (tinfoil) Coulomb energy of a neutral cell it is the constant
/// \(\sum_i \nabla_i^2 U_\mathrm{C} = -4\pi k_e \sum_i q_i^2 / V\) (the
/// neutralising background of the periodic Green's function,
/// \(\nabla^2\psi = -4\pi\delta + 4\pi/V\)), so its gradient vanishes but
/// the constant stays in \(D\) (about 0.1 % of \(D\) for molten NaCl). The
/// Hessian of the Coulomb energy enters \((HF)_i\) in full. Every term is
/// taken from its own `CurvatureSource`, so \(T_{C1}\) is exact for the
/// Hamiltonian the forces integrate.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/component/curvature_protocol.h"
#include "atomiverse/cpu/component/observer.h"

namespace ATV {
namespace cpu {

/// @brief One evaluation of \(T_{C1}\).
struct ConfigurationalSample {
    double force_sq = 0.0;    ///< \(N = \sum_i |F_i|^2\) [force²]
    double laplacian = 0.0;   ///< \(D = \sum_i \nabla_i^2 U\) [energy/length²]
    double temperature = 0.0; ///< \(T_{C1} = N/(k_B D)\) [K]
};

/// @brief Gathers the interaction forces and every source's curvature for
///        the owned atoms (not a component: owned by its consumers).
///
/// Ghost rows receive the interaction force by forward communication before
/// the sources run, and the sources' ghost-row accumulations return to their
/// owners by reverse communication, as forces do. Single process (serial or
/// full replica) only.
class CurvatureAssembly : public ForwardPacker, public ReversePacker {
  public:
    /// @brief The run's force terms (from `CurvatureConsumer`).
    void set_terms(std::vector<CurvatureSource*> sources,
                   std::vector<const ExternalDrive*> drives);

    /// @brief Evaluate at the frame's positions and forces.
    /// @return \(N\), \(D\) and \(T_{C1}\) (meaningful only for
    ///         \(D > 0\); a lone Coulomb system has \(D < 0\)).
    /// @throws std::runtime_error under spatial decomposition or without
    ///         sources.
    ConfigurationalSample evaluate(const LaunchContext& ctx);

    /// @brief \(\partial T_{C1}/\partial r_i\) of the last `evaluate` for
    ///        owned row i and axis d [K/length], layout `d·n + i`.
    void temperature_gradient(double kB, std::vector<double>& out) const;

    /// @brief Owned-row results of the last `evaluate` (layout `d·n + i`
    ///        for the vectors): the interaction force, \(HF\),
    ///        \(\nabla_i^2 U\) and \(\nabla_i D\).
    int n_owned() const { return n_owned_; }
    const std::vector<double>& force() const { return u_; }
    const std::vector<double>& hessian_force() const { return h_; }
    const std::vector<double>& laplacian() const { return lap_; }
    const std::vector<double>& laplacian_gradient() const { return g_; }

    int comm_forward() const override { return 3; }
    int pack_forward_comm(int n, const int* list, double* buf, int pbc_flag,
                          const int* pbc) override;
    void unpack_forward_comm(int n, int first, const double* buf) override;
    int comm_reverse() const override { return 7; }
    int pack_reverse_comm(int n, int first, double* buf) override;
    void unpack_reverse_comm(int n, const int* list,
                             const double* buf) override;

  private:
    std::vector<CurvatureSource*> sources_;
    std::vector<const ExternalDrive*> drives_;
    int n_owned_ = 0;
    int n_total_ = 0;
    /// Work rows [0, n_total): wu_/wh_/wg_ by axis (d·n_total + i), wlap_.
    std::vector<double> wu_, wh_, wlap_, wg_;
    /// Owned results (d·n_owned + i for the vectors).
    std::vector<double> u_, h_, lap_, g_;
    ConfigurationalSample last_{};
};

/// @brief Observer: \(T_{C1}\), \(N\) and \(D\) of the current
///        configuration every `every` steps, and their run averages.
///
/// Bound at `post_force` after the force reduction and the drives (order
/// +200), where positions and forces belong to the same configuration under
/// every integrator (with `GearPC`, the predicted one). The averages of `N`
/// and `D` over the evaluations since the current `run()` began (a
/// continued run starts a new average, as on CUDA) give the paper's
/// \(T_\mathrm{CONF}\) (eq. 5) as \(\langle N\rangle / (k_B\langle
/// D\rangle)\) (`mean()`), which is not the mean of \(T_{C1}\).
class ConfigurationalTemperature : public Observer, public CurvatureConsumer {
  public:
    /// @param every Evaluate every `every` steps (> 0).
    explicit ConfigurationalTemperature(int every = 1);

    std::string name() const override { return "configurational_temperature"; }
    int frequency() const override { return every_; }
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::post_force, 200)};
    }
    CurvatureConsumer* as_curvature_consumer() override { return this; }
    void set_curvature_terms(std::vector<CurvatureSource*> sources,
                             std::vector<const ExternalDrive*> drives) override;
    /// @brief Start of every `run()`: clear the averages, a new average.
    void setup_output(const LaunchContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    /// @brief The last evaluation.
    const ConfigurationalSample& last() const { return last_; }
    /// @brief Run averages \(\langle N\rangle\), \(\langle D\rangle\) and
    ///        \(T_\mathrm{CONF} = \langle N\rangle / (k_B\langle
    ///        D\rangle)\) over the `samples()` evaluations since the
    ///        current `run()` began (all zero before the first).
    const ConfigurationalSample& mean() const { return mean_; }
    /// @brief Number of evaluations in `mean()`.
    long long samples() const { return samples_; }
    /// @brief Per-atom data of the last evaluation.
    const CurvatureAssembly& assembly() const { return assembly_; }

  private:
    int every_;
    CurvatureAssembly assembly_;
    ConfigurationalSample last_{};
    ConfigurationalSample mean_{};
    double sum_force_sq_ = 0.0, sum_laplacian_ = 0.0;
    long long samples_ = 0;
};

} // namespace cpu
} // namespace ATV
