/// @file src/cpu/dynamics/cpmd_conserved.cpp
/// @brief Implementation of cpu::md::CpmdConserved (see header).

#include "atomiverse/cpu/dynamics/cpmd_conserved.h"

#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"

namespace ATV {
namespace cpu {
namespace md {

void CpmdConserved::launch(const cpu::LaunchContext& ctx) {
    cpu::Frame& f = ctx.frame;

    // Both electronic terms must be on the frame BEFORE anything is published:
    // a refused launch leaves kConserved untouched (no half-assembled value).
    for (const auto& key :
         {kEnergyElectronKinetic.key, kEnergyElectronThermostat.key}) {
        if (!f.meta.contains(key))
            throw std::runtime_error(
                "CpmdConserved: frame.meta has no '" + std::string(key)
                + "' — no CPMD-enabled evaluator ran this step (a GPW "
                  "evaluator with cpmd=true publishes both electronic energy "
                  "keys on every launch). Refusing to substitute 0: a CPMD "
                  "conserved energy missing a fictitious-electron term looks "
                  "conserved only because the drifting term was dropped.");
    }

    // E_cons = E_KS + T_nuclear + T_e + E_therm^e  [eV]. Replaced, not
    // accumulated — kConserved has one owner per run (see header).
    f.meta.set(kConserved, f.meta.get(kEnergy) + f.meta.get(kKinetic)
                               + f.meta.get(kEnergyElectronKinetic)
                               + f.meta.get(kEnergyElectronThermostat));
}

} // namespace md
} // namespace cpu
} // namespace ATV
