/// @file regressions/pw/h_stress_cubic/reg_h_stress_cubic.cpp
/// @brief END-TO-END: cubic monoatomic PW stress off-diagonals vanish.
#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/pw/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

int main() {
    cpu::Frame frame(1);
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_mass)[0] = 1.008;
    frame.span(kAtom_x)[0] = frame.span(kAtom_y)[0] = frame.span(kAtom_z)[0] =
        0;
    frame.simbox = cpu::Box::cubic(6.0);

    PWConfig cfg;
    cfg.ecut_ry = 5.0;
    cfg.k_mesh = 1;
    cfg.tol = 1e-4;
    cfg.max_iter = 8;
    cfg.mix_beta = 0.5;

    const auto s = pw_stress(frame, cfg);
    const double tol = 0.5;
    for (int k = 3; k < 6; ++k) {
        if (!(std::abs(s[k]) < tol)) {
            std::printf("FAIL: off-diag stress[%d]=%.6g\n", k, s[k]);
            return 1;
        }
    }
    if (!(std::abs(s[0] - s[1]) < tol && std::abs(s[1] - s[2]) < tol)) {
        std::printf("FAIL: diagonal not equal xx=%.4g yy=%.4g zz=%.4g\n", s[0],
                    s[1], s[2]);
        return 1;
    }
    std::printf("ok: stress Voigt = %.4g %.4g %.4g | %.4g %.4g %.4g eV/A3\n",
                s[0], s[1], s[2], s[3], s[4], s[5]);
    std::printf("PASS: pw cubic H stress symmetry\n");
    return 0;
}
