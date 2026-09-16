/// @file cpu/observe/elastance_observer.h
/// @brief Empty-capacitance / elastance \(\mathrm{cond}(A)\) debug observer.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/observer.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"

namespace ATV {
namespace cpu {

/// @brief Samples \(\kappa(A)\) and empty-plate \(C=|Q|/|\Delta\Psi|\) from ConstPotBO.
class ElastanceObserver : public cpu::Observer {
  public:
    explicit ElastanceObserver(md::ConstPotBO* conp) : conp_(conp) {
        if (!conp_)
            throw std::invalid_argument("ElastanceObserver: null ConstPotBO");
    }

    std::string name() const override { return "elastance_observer"; }

    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +25)};
    }

    void launch(const LaunchContext& ctx) override;

    double last_condition() const { return last_cond_; }
    double last_capacitance() const { return last_C_; }

  private:
    md::ConstPotBO* conp_ = nullptr;
    double last_cond_ = 0.0;
    double last_C_ = 0.0;
};

} // namespace cpu
} // namespace ATV
