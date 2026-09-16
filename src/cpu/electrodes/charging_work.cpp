/// @file src/cpu/electrodes/charging_work.cpp
/// @brief ChargingWork implementation. @see charging_work.h

#include "atomiverse/cpu/electrodes/charging_work.h"

#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/owned_range.h"

namespace ATV {
namespace cpu {
namespace electrodes {

ChargingWork::ChargingWork(std::vector<int> electrode_types,
                           std::vector<double> potential_V)
    : types_(std::move(electrode_types)), psi_V_(std::move(potential_V)) {
    if (types_.size() != psi_V_.size()) {
        throw std::invalid_argument(
            "ChargingWork: electrode_types and potential_V must have equal "
            "length");
    }
    for (std::size_t k = 0; k < types_.size(); ++k) {
        const int t = types_[k];
        if (psi_of_type_.count(t) != 0) {
            throw std::invalid_argument(
                "ChargingWork: duplicate electrode type " + std::to_string(t));
        }
        psi_of_type_[t] = psi_V_[k];
    }
}

void ChargingWork::build(const BuildContext& ctx) {
    (void)ctx;
    // Type→Ψ map is fixed at construction; rebuild is a no-op for v1.
    psi_of_type_.clear();
    for (std::size_t k = 0; k < types_.size(); ++k)
        psi_of_type_[types_[k]] = psi_V_[k];
}

void ChargingWork::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    const auto type = f.span(kAtom_type);
    const auto charge = f.span(kAtom_charge);

    double work = 0.0;
    for (int i = 0; i < n; ++i) {
        const auto it = psi_of_type_.find(static_cast<int>(type[i]));
        if (it == psi_of_type_.end())
            continue;
        // E = -Σ Ψ_i Q_i  with Ψ [V], Q [e] → eV.
        work -= it->second * static_cast<double>(charge[i]);
    }

    last_work_ = work;
    f.meta.get(kEnergy) += work;
    f.meta.get(kEnergyChargingWork) += work;
}

} // namespace electrodes
} // namespace cpu
} // namespace ATV
