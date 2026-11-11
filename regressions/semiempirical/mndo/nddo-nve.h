#pragma once
#include "../nddo_mopac_fixtures.h"
#include "../nddo_nve_cpu.h"
#include "atomiverse/cpu/semiempirical/mndo_solver.h"
namespace atv_reg_semiempirical_mndo_nve {
inline const auto& water_pin() { return atv_reg_nddo_mopac::kMndoPins[2]; }
inline int run_cpu() {
    return atv_reg_nddo_nve::run_cpu_case<ATV::cpu::semiempirical::MNDO>(
        water_pin());
}
} // namespace atv_reg_semiempirical_mndo_nve
