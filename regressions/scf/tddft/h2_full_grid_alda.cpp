/// @file regressions/scf/h2_full_grid_alda/reg_h2_full_grid_alda.cpp
/// @brief H2/STO-3G LSDA TDA with full MO-product ALDA — hard-coded S1 gate.
///
/// Oracle: PySCF 2.14.0 `dft.RKS(xc="lda,vwn")` + `tdscf.TDA` (external).
/// The former self-pin (7.653049 eV) came from a kernel that kept full HF
/// exchange and only the Dirac part of f^xc; see REFERENCE.md.
#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "harness/registry.h"

namespace atv_reg_scf_h2_full_grid_alda {

using namespace ATV;

int run() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_atomic_number)[0] = f.span(kAtom_atomic_number)[1] = 1;
    f.box = cpu::Box::cubic(20);

    cpu::scf::RKS rks(cpu::scf::BasisSet::STO3G, cpu::xc::XCType::LSDA);
    rks.tol = 1e-9;
    rks.run(f);
    if (!rks.converged()) {
        std::printf("FAIL scf\n");
        return 1;
    }
    auto es = cpu::scf::TDA<cpu::scf::RKS>{rks}.run(1);
    std::printf("full-grid ALDA S1=%.6f eV alda=%d\n", es.energy_ev[0],
                (int)es.alda);
    if (!es.alda) {
        std::printf("FAIL alda flag\n");
        return 1;
    }
    // S1 [eV]: PySCF 2.14.0 LDA,VWN TDA (H2 r=0.74 Å, STO-3G). Tolerance
    // 0.05 eV covers the SG0 vs PySCF default grid difference.
    constexpr double kS1Ref = 26.443003; // eV
    constexpr double kTol = 0.05;        // eV
    if (!(std::fabs(es.energy_ev[0] - kS1Ref) < kTol)) {
        std::printf("FAIL S1 out of window\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

} // namespace atv_reg_scf_h2_full_grid_alda

ATV_REGRESSION("scf/tddft", "h2_full_grid_alda",
               atv_reg_scf_h2_full_grid_alda::run);
