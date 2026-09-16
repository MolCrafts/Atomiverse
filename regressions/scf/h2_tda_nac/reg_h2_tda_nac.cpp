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
    f.span(kAtom_x)[1] = 0.9;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.box = cpu::Box::cubic(15);
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::Pople631G;
    rhf.tol = 1e-7;
    rhf.run(f);
    if (!rhf.converged()) {
        std::printf("FAIL scf\n");
        return 1;
    }
    auto es = rhf.tda(2);
    if (es.energy_ev.size() < 2) {
        std::printf("FAIL nstates\n");
        return 1;
    }
    const double s = rhf.tda_nac_strength(f, 0, 1, 5e-3);
    std::printf("NAC strength=%.6e 1/A\n", s);
    if (!std::isfinite(s) || !(s > 0.0)) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
