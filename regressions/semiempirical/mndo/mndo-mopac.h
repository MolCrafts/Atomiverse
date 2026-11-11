#pragma once
#include "../nddo_mopac_cpu.h"
#include "atomiverse/cpu/semiempirical/mndo_solver.h"
namespace atv_reg_semiempirical_mndo_mopac {
inline int run_cpu() {
    return atv_reg_nddo_mopac::run_cpu_case<ATV::cpu::semiempirical::MNDO>(
        atv_reg_nddo_mopac::kMndoPins);
}
} // namespace atv_reg_semiempirical_mndo_mopac
