#pragma once
#include "../am1/am1-water-mopac-energy.h"
#include "../nddo_nve_cpu.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"
namespace atv_reg_semiempirical_rm1_nve {
inline const auto& water_pin() {
    return atv_reg_semiempirical_am1_water_mopac_energy::kModernPins[5];
}
inline int run_cpu() {
    return atv_reg_nddo_nve::run_cpu_case<ATV::cpu::semiempirical::RM1>(
        water_pin());
}
} // namespace atv_reg_semiempirical_rm1_nve
