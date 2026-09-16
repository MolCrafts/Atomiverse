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
    cpu::scf::RKS rks(cpu::scf::XCType::LSDA);
    rks.basis = cpu::scf::BasisSet::STO3G;
    rks.tol = 1e-8;
    rks.run(f);
    if (!rks.converged()) {
        std::printf("FAIL scf\n");
        return 1;
    }
    auto es = rks.tda(1);
    std::printf("ALDA-path S1=%.4f eV alda=%d\n", es.energy_ev[0],
                (int)es.alda);
    if (!es.alda || !(es.energy_ev[0] > 1.0 && es.energy_ev[0] < 30.0)) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
