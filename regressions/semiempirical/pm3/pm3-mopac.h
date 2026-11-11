#pragma once
#include "../nddo_mopac_cpu.h"
#include "atomiverse/cpu/semiempirical/pm3_solver.h"
namespace atv_reg_semiempirical_pm3_mopac {
inline int run_cpu() {
    return atv_reg_nddo_mopac::run_cpu_case<ATV::cpu::semiempirical::PM3>(
        atv_reg_nddo_mopac::kPm3Pins);
}
} // namespace atv_reg_semiempirical_pm3_mopac
