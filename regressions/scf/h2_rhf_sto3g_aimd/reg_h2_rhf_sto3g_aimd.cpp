/// @file regressions/scf/h2_rhf_sto3g_aimd/reg_h2_rhf_sto3g_aimd.cpp
/// @brief Short H2 / STO-3G RHF BOMD — **CPU device** (CUDA-off builds).
///
/// With `ATV_BUILD_CUDA=ON`, CMake builds `reg_h2_rhf_sto3g_aimd.cu` instead
/// (`--device=cpu|cuda|both`). Public MD + RHF composition only; no AimdDriver,
/// IntegralWorkspace, SCFSolver, or user Context. See REFERENCE.md.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

int main() {
    using namespace ATV;

    cpu::Frame frame(2);
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = 0.74;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.0;
    frame.span(kAtom_mass)[0] = 1.008;
    frame.span(kAtom_mass)[1] = 1.008;
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_vx)[0] = 0.001;
    frame.span(kAtom_vx)[1] = -0.001;
    frame.box = cpu::Box::cubic(100.0);

    // Composition: MD force slot + concrete RHF (no AimdDriver).
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.5});
    sim.on(md::Phase::force)
        .add(cpu::scf::RHF{cpu::scf::BasisSet::STO3G}.with_gradient(true));
    sim.build();
    sim.run(frame, /*n_steps=*/5);

    const double pe_ev = static_cast<double>(frame.meta.get(kEnergy));
    const double pe_ha = pe_ev / units::kHartreeInEv;
    if (!std::isfinite(pe_ev) || pe_ev >= 0.0) {
        std::fprintf(stderr,
                     "FAIL [cpu]: H2 BOMD PE not finite/bound: %.6f eV\n",
                     pe_ev);
        return 1;
    }
    constexpr double kPeMinEv = -35.0;
    constexpr double kPeMaxEv = -25.0;
    if (pe_ev < kPeMinEv || pe_ev > kPeMaxEv) {
        std::fprintf(stderr,
                     "FAIL [cpu]: PE outside literature envelope: %.6f eV\n",
                     pe_ev);
        return 1;
    }
    std::printf("PASS [cpu]: H2 RHF/STO-3G short BOMD PE=%.6f eV (%.6f Ha)\n",
                pe_ev, pe_ha);
    return 0;
}
