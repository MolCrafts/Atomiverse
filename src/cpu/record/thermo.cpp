/// @file cpu/record/thermo.cpp
/// @brief CPU thermodynamic output component.

#include "atomiverse/cpu/record/thermo.h"
#include "atomiverse/core/log.h"

#include <utility>

namespace ATV {
namespace cpu {

Thermo::Thermo(int freq, std::vector<std::string> keywords)
    : freq_(freq),
      style_(keywords.empty() ? ThermoStyle{} : ThermoStyle{keywords}) {}

void Thermo::launch(const cpu::LaunchContext& ctx) {
    if (ctx.runtime.mpi_rank != 0)
        return;
    const auto& f = ctx.frame;
    ThermoSample sample;
    sample.step = ctx.step;
    sample.pe = f.meta.get(kEnergy);
    sample.ke = f.meta.get(kKinetic);
    // The integrator's conserved quantity (extended Hamiltonian for
    // Nosé-Hoover NVT/NPT); 0 when no integrator populates it.
    sample.conserved = f.meta.get(kConserved);
    sample.volume = f.box.volume();
    const auto virial = f.meta.get(kStress);
    const auto kinetic = f.meta.get(kStressKinetic);
    for (int a = 0; a < 6; ++a) {
        sample.virial[a] = virial[a];
        sample.kinetic[a] = kinetic[a];
    }
    // Ndof has ONE owner (Driver::publish_dof_): 3N - 3 - n_constraints.
    sample.ndof = ctx.runtime.md.n_dof_global;
    if (!header_printed_) {
        ATV_INFO("{}", style_.header());
        header_printed_ = true;
    }
    ATV_INFO("{}", style_.line(sample));
}

} // namespace cpu
} // namespace ATV
