/// @file regressions/scf/h2co_rks_tda/reg_h2co_rks_tda.cpp
/// H2 RKS-LSDA/STO-3G CIS@RKS first positive singlet — framework gate.
///
/// Skeleton: HF-like TDA on KS orbitals (ALDA deferred). Formaldehyde can
/// show negative roots under this kernel; H2 is the clean textbook pin.
#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

int main() {
    cpu::Frame frame(2);
    frame.span(kAtom_x)[0] = 0;
    frame.span(kAtom_y)[0] = 0;
    frame.span(kAtom_z)[0] = 0;
    frame.span(kAtom_x)[1] = 0.74;
    frame.span(kAtom_y)[1] = 0;
    frame.span(kAtom_z)[1] = 0;
    frame.span(kAtom_type)[0] = frame.span(kAtom_type)[1] = 1;
    frame.box = cpu::Box::cubic(20.0);

    cpu::scf::RKS rks(cpu::scf::XCType::LSDA);
    rks.basis = cpu::scf::BasisSet::STO3G;
    rks.tol = 1e-8;
    rks.max_iter = 80;
    rks.run(frame);
    if (!rks.converged()) {
        std::printf("FAIL: RKS did not converge\n");
        return 1;
    }
    auto es = rks.tda(1, cpu::scf::ExcitationChannel::singlet);
    if (es.energy_ev.empty() || !std::isfinite(es.energy_ev[0])) {
        std::printf("FAIL: empty/non-finite excitation\n");
        return 1;
    }
    const double eV = es.energy_ev[0];
    std::printf("H2 RKS-LSDA/STO-3G CIS@RKS S1 = %.4f eV\n", eV);
    if (!(eV > 1.0 && eV < 30.0)) {
        std::printf("FAIL: S1 out of [1,30] eV band\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
