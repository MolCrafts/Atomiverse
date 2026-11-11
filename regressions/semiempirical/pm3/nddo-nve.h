#pragma once
#include "../nddo_mopac_fixtures.h"
#include "../nddo_nve_cpu.h"
#include "atomiverse/cpu/semiempirical/pm3_solver.h"
namespace atv_reg_semiempirical_pm3_nve {
inline const auto& water_pin() { return atv_reg_nddo_mopac::kPm3Pins[2]; }
inline int run_cpu() {
    return atv_reg_nddo_nve::run_cpu_case<ATV::cpu::semiempirical::PM3>(
        water_pin());
}
} // namespace atv_reg_semiempirical_pm3_nve
