/// @file electronic/xc_functional.h
/// @brief Backend-independent, batched semi-local XC extension contract.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace ATV::electronic {

/// @brief Physical ingredients required by an XC functional.
enum class XcIngredients : std::uint8_t {
    Rho = 1,
    Sigma = 2,
    Tau = 4,
    Laplacian = 8
};
/// @brief Union of physical XC ingredient requirements.
/// @param a First set. @param b Second set. @return Their union.
constexpr XcIngredients operator|(XcIngredients a, XcIngredients b) noexcept {
    return static_cast<XcIngredients>(static_cast<unsigned>(a)
                                      | static_cast<unsigned>(b));
}
/// @brief Spin convention of a batch (both density spans are always present).
enum class XcSpin { Restricted, Collinear };
/// @brief Requested derivative order; second derivatives are not yet provided.
enum class XcOrder { Energy, First, Second };

/// @brief Call-scoped physical samples in atomic units, never a solver workspace.
struct XcBatch {
    XcSpin spin = XcSpin::Collinear; ///< Restricted means rho_alpha = rho_beta.
    std::span<const double> rho_alpha, rho_beta; ///< Density [Bohr^-3].
    std::span<const double> sigma_aa, sigma_ab,
        sigma_bb; ///< Gradient products [Bohr^-8]; ab has no factor two.
    std::span<const double> tau_alpha,
        tau_beta; ///< 1/2 sum f |grad psi|^2 [Bohr^-5].
    std::span<const double> laplacian_alpha,
        laplacian_beta; ///< Laplacian rho [Bohr^-5].
};

/// @brief Caller-owned output spans; energy is per volume, never per electron.
struct XcOutput {
    std::span<double> energy_density; ///< f, E_xc = integral f dr [Ha/Bohr^3].
    std::span<double> vrho_alpha, vrho_beta; ///< df/drho [Ha].
    std::span<double> vsigma_aa, vsigma_ab,
        vsigma_bb;                           ///< df/dsigma [Ha Bohr^5].
    std::span<double> vtau_alpha, vtau_beta; ///< df/dtau [Ha Bohr^2].
    std::span<double> vlaplacian_alpha,
        vlaplacian_beta; ///< df/d(laplacian rho) [Ha Bohr^2].
};

/// @brief Immutable semi-local XC mathematics; no Driver, geometry or history.
class XcFunctional {
  public:
    virtual ~XcFunctional() = default;
    /// @brief Required physical input channels. @return Ingredient set.
    virtual XcIngredients ingredients() const noexcept = 0;
    /// @brief Whether a spin/derivative request is implemented.
    /// @param spin Spin convention. @param order Derivative order.
    /// @return True only for implemented requests.
    virtual bool supports(XcSpin spin, XcOrder order) const noexcept = 0;

    /// @brief Validate, evaluate one batch and reject incomplete/nonfinite outputs.
    /// @param in Physical input views; valid for this call only.
    /// @param order Requested energy/derivative order.
    /// @param out Preallocated physical output views; no allocation by the provider.
    /// @throws std::invalid_argument for unsupported requests or invalid shapes.
    /// @throws std::runtime_error for unwritten or nonfinite requested outputs.
    void evaluate(const XcBatch& in, XcOrder order, XcOutput& out) const {
        if ((in.spin != XcSpin::Restricted && in.spin != XcSpin::Collinear)
            || (order != XcOrder::Energy && order != XcOrder::First)
            || !supports(in.spin, order))
            throw std::invalid_argument(
                "XC: unsupported spin or derivative order");
        const auto mask = static_cast<unsigned>(ingredients());
        if ((mask & 1u) == 0 || (mask & ~15u) != 0)
            throw std::invalid_argument("XC: invalid ingredient declaration");
        const auto n = in.rho_alpha.size();
        auto input = [n](std::span<const double> values) {
            if (values.size() != n)
                throw std::invalid_argument("XC: input batch size mismatch");
            for (double value : values)
                if (!std::isfinite(value))
                    throw std::invalid_argument("XC: nonfinite input sample");
        };
        input(in.rho_alpha);
        input(in.rho_beta);
        if (mask & 2u) {
            input(in.sigma_aa);
            input(in.sigma_ab);
            input(in.sigma_bb);
        }
        if (mask & 4u) {
            input(in.tau_alpha);
            input(in.tau_beta);
        }
        if (mask & 8u) {
            input(in.laplacian_alpha);
            input(in.laplacian_beta);
        }
        auto prepare = [](std::span<double> values) {
            std::fill(values.begin(), values.end(),
                      std::numeric_limits<double>::quiet_NaN());
        };
        // Validate every shape before touching any output.
        auto shape = [n](std::span<double> values) {
            if (values.size() != n)
                throw std::invalid_argument("XC: output batch size mismatch");
        };
        auto outputs = [&](auto&& visit) {
            visit(out.energy_density);
            if (order == XcOrder::First) {
                visit(out.vrho_alpha);
                visit(out.vrho_beta);
                if (mask & 2u) {
                    visit(out.vsigma_aa);
                    visit(out.vsigma_ab);
                    visit(out.vsigma_bb);
                }
                if (mask & 4u) {
                    visit(out.vtau_alpha);
                    visit(out.vtau_beta);
                }
                if (mask & 8u) {
                    visit(out.vlaplacian_alpha);
                    visit(out.vlaplacian_beta);
                }
            }
        };
        outputs(shape);
        outputs(prepare);
        evaluate_batch(in, order, out);
        outputs([](std::span<double> values) {
            for (double value : values)
                if (!std::isfinite(value))
                    throw std::runtime_error(
                        "XC: incomplete or nonfinite output");
        });
    }

  protected:
    /// @brief Implement the requested batch; all input/output shapes are checked.
    /// @param in Physical input samples. @param order Validated derivative order.
    /// @param out Write every requested output; retain no spans after returning.
    virtual void evaluate_batch(const XcBatch& in, XcOrder order,
                                XcOutput& out) const = 0;
};

} // namespace ATV::electronic
