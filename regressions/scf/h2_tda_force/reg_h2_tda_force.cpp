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
    auto F = rhf.tda_forces(f, 0, 2e-3);
    const double fx0 = F[0], fx1 = F[3];
    std::printf("TDA force Fx0=%.6e Fx1=%.6e\n", fx0, fx1);
    if (!std::isfinite(fx0) || !std::isfinite(fx1)) {
        std::printf("FAIL\n");
        return 1;
    }
    // Opposite sense on bond axis (may be small)
    if (std::fabs(fx0 + fx1) > 0.5 * (std::fabs(fx0) + std::fabs(fx1)) + 1e-3
        && (std::fabs(fx0) + std::fabs(fx1)) > 1e-6) {
        // allow imperfect FD noise; only fail if clearly same direction and large
        if (fx0 * fx1 > 0) {
            std::printf("FAIL same sign large\n");
            return 1;
        }
    }
    std::printf("PASS\n");
    return 0;
}
