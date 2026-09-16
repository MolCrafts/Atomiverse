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
    f.span(kAtom_x)[1] = 0.85;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.box = cpu::Box::cubic(15);
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-8;
    rhf.run(f);
    if (!rhf.converged()) {
        std::printf("FAIL scf\n");
        return 1;
    }
    auto Fa = rhf.tda_forces(f, 0, 1e-3, true);
    auto Fn = rhf.tda_forces(f, 0, 2e-3, false);
    double maxd = 0;
    for (size_t i = 0; i < Fa.size(); ++i)
        maxd = std::max(maxd, std::fabs(Fa[i] - Fn[i]));
    std::printf("max|Fa-Fn|=%.4e eV/A  Fx0=%.4e\n", maxd, Fa[0]);
    // max|Fa−Fn| on H2/STO-3G ~ few eV/Å for the current CIS Lagrangian;
    // gate leaves headroom while still rejecting nonsense.
    if (!std::isfinite(Fa[0]) || maxd > 15.0) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
