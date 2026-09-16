#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include <cmath>
#include <cstdio>
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
    // CIS@RHF reference
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-8;
    rhf.run(f);
    const double e_cis = rhf.tda(1).energy_ev[0];
    cpu::scf::RKS rks(cpu::scf::XCType::LSDA);
    rks.basis = cpu::scf::BasisSet::STO3G;
    rks.tol = 1e-8;
    rks.run(f);
    auto es = rks.tda(1);
    std::printf("CIS-RHF=%.4f ALDA-RKS=%.4f alda=%d\n", e_cis, es.energy_ev[0],
                (int)es.alda);
    if (!es.alda || !std::isfinite(es.energy_ev[0])) {
        std::printf("FAIL\n");
        return 1;
    }
    if (std::fabs(es.energy_ev[0] - e_cis) < 1e-6) {
        std::printf("FAIL: ALDA identical to RHF CIS\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
