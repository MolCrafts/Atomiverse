/// @file cpu/electronic/aimd_method.h
#include <cmath>
/// @brief CarParrinelloMethod / XlBomdMethod — AIMD wrappers over mean-field methods.
#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ATV {
namespace cpu {
namespace electronic {

/// @brief Optional electronic thermostat for AO-CPMD (same numbers as GPW).
struct ScfElectronicThermostat {
    double ekin_target_ha = 1e-5;
    double tau_e_fs = 50.0;
};

/// @brief Car–Parrinello over a concrete mean-field method (`RHF`/`RKS`/…).
///
/// Publicly inherits `Method` so nested-driver binding and HF kind stay
/// intact. AIMD flags are set only in this constructor (base flags are
/// protected on `MeanFieldMethod`).
template <typename Method>
class CarParrinelloMethod : public Method {
  public:
    /// @brief Move a configured method and enable CP with mass \(\mu\).
    CarParrinelloMethod(Method method, double mu_au)
        : Method(std::move(method)) {
        if (!(mu_au > 0.0) || !std::isfinite(mu_au))
            throw std::invalid_argument(
                "CarParrinelloMethod: mu_au must be finite and > 0");
        this->cpmd = true;
        this->cpmd_mu = mu_au;
        this->cpmd_thermostat = false;
    }

    CarParrinelloMethod(Method method, double mu_au, ScfElectronicThermostat th)
        : CarParrinelloMethod(std::move(method), mu_au) {
        if (!(th.ekin_target_ha > 0.0) || !std::isfinite(th.ekin_target_ha))
            throw std::invalid_argument(
                "CarParrinelloMethod: ekin_target_ha must be finite and > 0");
        if (!(th.tau_e_fs > 0.0) || !std::isfinite(th.tau_e_fs))
            throw std::invalid_argument(
                "CarParrinelloMethod: tau_e_fs must be finite and > 0");
        this->cpmd_thermostat = true;
        this->cpmd_ekin_target_ha = th.ekin_target_ha;
        this->cpmd_tau_e = th.tau_e_fs;
    }
};

/// @brief XL-BOMD over a concrete mean-field method.
template <typename Method>
class XlBomdMethod : public Method {
  public:
    explicit XlBomdMethod(Method method, int k = 5)
        : Method(std::move(method)) {
        this->xl_bomd = true;
        this->xl_bomd_k = k;
    }
};

} // namespace electronic
} // namespace cpu
} // namespace ATV
