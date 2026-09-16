/// @file regressions/scf/h2_full_grid_alda/reg_h2_full_grid_alda.cpp
/// @brief H2/STO-3G LSDA TDA with full MO-product ALDA — hard-coded S1 gate.
///
/// Oracle: Atomiverse full-grid ALDA path (Dirac-exchange f^xc on SG0 grid),
/// 2026-07-23. Window is relative to the diagonal-only skeleton path and the
/// CIS@RKS (PBE) continuum — absolute S1 pinned from this engine.
#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

int main() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.box = cpu::Box::cubic(20);

    cpu::scf::RKS rks(cpu::scf::XCType::LSDA);
    rks.basis = cpu::scf::BasisSet::STO3G;
    rks.tol = 1e-9;
    rks.run(f);
    if (!rks.converged()) {
        std::printf("FAIL scf\n");
        return 1;
    }
    auto es = rks.tda(1);
    std::printf("full-grid ALDA S1=%.6f eV alda=%d\n", es.energy_ev[0],
                (int)es.alda);
    if (!es.alda) {
        std::printf("FAIL alda flag\n");
        return 1;
    }
    // Hard-coded S1 [eV]: Atomiverse full-grid ALDA (H2 r=0.74 Å, STO-3G,
    // LSDA, SG0 50×110), 2026-07-23. Tolerance 0.05 eV covers grid noise.
    constexpr double kS1Ref = 7.653049; // eV
    constexpr double kTol = 0.05;       // eV
    if (!(std::fabs(es.energy_ev[0] - kS1Ref) < kTol)) {
        std::printf("FAIL S1 out of window\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
