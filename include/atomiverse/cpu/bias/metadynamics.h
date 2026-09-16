/// @file cpu/bias/metadynamics.h
/// @brief Well-tempered multi-CV metadynamics Bias (CPU).
///
/// History-dependent bias on \(\mathbf{s}\in\mathbb{R}^{n}\):
/// \[
/// V(\mathbf{s})=\sum_k h_k\exp\Big(-\sum_\alpha\frac{(s_\alpha-s_{k\alpha})^2}
/// {2\sigma_\alpha^2}\Big),\quad
/// h_k=h_0\exp\!\Big(-\frac{V(\mathbf{s}_k)}{(γ-1)k_BT}\Big).
/// \]
/// Forces: \(F_i=-\sum_\alpha(\partial V/\partial s_\alpha)(\partial s_\alpha/\partial x_i)\).
///
/// Laio & Parrinello, *PNAS* **99**, 12562 (2002); Barducci, Bussi, Parrinello,
/// *Phys. Rev. Lett.* **100**, 020603 (2008). Units: eV, CV-native, K.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "atomiverse/cpu/bias/collective_variable.h"
#include "atomiverse/cpu/component/bias.h"

namespace ATV {
namespace cpu {
namespace bias {

/// @brief N-dimensional well-tempered metadynamics Bias.
class Metadynamics : public cpu::Bias {
  public:
    /// @param cvs Owned collective variables (size n ≥ 1).
    /// @param height Initial hill height h0 [eV].
    /// @param sigma Per-CV hill widths [CV unit], size n.
    /// @param temperature T [K] for the well-tempered factor.
    /// @param bias_factor γ ≥ 1 (γ=1 → constant height).
    /// @param pace Deposit one hill every this many launches (≥ 1).
    Metadynamics(std::vector<std::unique_ptr<CollectiveVariable>> cvs,
                 double height, std::vector<double> sigma, double temperature,
                 double bias_factor = 10.0, int pace = 10);

    std::string name() const override { return "metadynamics"; }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    int ndim() const { return static_cast<int>(cvs_.size()); }
    /// @brief Number of collective variables (alias of `ndim`).
    int n_cv() const { return ndim(); }
    int n_hills() const { return static_cast<int>(heights_.size()); }
    /// @brief Last CV vector (size ndim).
    const std::vector<double>& last_cv() const { return last_s_; }
    double last_bias_energy() const { return last_u_; }

  private:
    double bias_at_(const std::vector<double>& s) const;
    void dV_ds_(const std::vector<double>& s, std::vector<double>& out) const;

    std::vector<std::unique_ptr<CollectiveVariable>> cvs_;
    double h0_ = 0;
    std::vector<double> sigma_;
    double temperature_ = 300;
    double gamma_ = 10;
    int pace_ = 10;
    int step_ = 0;
    /// Flat hill centers: hill k at centers_[k*ndim + α].
    std::vector<double> centers_;
    std::vector<double> heights_;
    /// Per-CV gradient buffers: grad_[α] holds 3*nlocal ∂s_α/∂x.
    std::vector<std::vector<double>> grads_;
    std::vector<double> last_s_;
    double last_u_ = 0;
};

} // namespace bias
} // namespace cpu
} // namespace ATV
