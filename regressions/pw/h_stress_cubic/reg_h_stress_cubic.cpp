/// @file regressions/pw/h_stress_cubic/reg_h_stress_cubic.cpp
/// @brief END-TO-END: cubic monoatomic PW stress off-diagonals vanish.
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

int main() {
    cpu::Frame frame(1);
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_mass)[0] = 1.008;
    frame.span(kAtom_x)[0] = frame.span(kAtom_y)[0] = frame.span(kAtom_z)[0] =
        0;
    frame.box = cpu::Box::cubic(6.0);

    PW cfg;
    cfg.gth_name = "GTH-PBE-q1";
    cfg.ecut_ry = 5.0;
    cfg.k_mesh = 1;
    cfg.tol = 1e-4;
    cfg.max_iter = 8;
    cfg.mix_beta = 0.5;
    // One valence electron is open-shell: the restricted stress path rejects an
    // odd electron count (src/cpu/planewave/internal/pw_stress.cpp:625). A lone
    // H is a doublet (S = 1/2), so N↑−N↓ = 1 ⇒ multiplicity 2; the default 1
    // would fail the parity check in the same guard.
    cfg.spin_polarized = true;
    cfg.multiplicity = 2;

    const auto s = pw_stress(frame, cfg);
    // eV/Å³. Cubic cell + one atom at the origin: the off-diagonal stress
    // vanishes and the three diagonals are equal, both by symmetry rather than
    // by convergence. The two are NOT the same kind of bound, so they do not
    // share a tolerance:
    //
    //   * tol_offdiag — the off-diagonals are structurally zero, so this is a
    //     machine-precision identity. Measured 2026-08-15: |s| <= 1.0e-14.
    //   * tol_diag    — `pw_stress` takes central finite differences of the
    //     SCF energy under strain, and each diagonal component comes from an
    //     independent FD evaluation. The floor is therefore set by the FD step
    //     and the 1e-4 / 8-iteration SCF, NOT by machine epsilon. Measured
    //     2026-08-15 (spin-polarized doublet) spread: 1.0e-9 against a signal
    //     of 0.005085 (2.0e-7 relative), and identical to the printed 10
    //     significant digits at 1, 4 and 8 OpenMP threads.
    //
    // The pre-GTH value of 0.5 was sized for the deleted toy potential and was
    // ~7 orders too loose to catch an asymmetric stress term.
    const double tol_offdiag = 1e-9;
    const double tol_diag = 1e-6;
    for (int k = 3; k < 6; ++k) {
        if (!(std::abs(s[k]) < tol_offdiag)) {
            std::printf("FAIL: off-diag stress[%d]=%.6g tol=%g\n", k, s[k],
                        tol_offdiag);
            return 1;
        }
    }
    const double spread =
        std::max(std::abs(s[0] - s[1]),
                 std::max(std::abs(s[1] - s[2]), std::abs(s[0] - s[2])));
    if (!(spread < tol_diag)) {
        std::printf("FAIL: diagonal not equal xx=%.17g yy=%.17g zz=%.17g "
                    "spread=%.6g tol=%g\n",
                    s[0], s[1], s[2], spread, tol_diag);
        return 1;
    }
    std::printf("ok: stress Voigt = %.10g %.10g %.10g | %.4g %.4g %.4g eV/A3\n",
                s[0], s[1], s[2], s[3], s[4], s[5]);
    std::printf(
        "ok: diag spread=%.4g (tol %g)  max|off-diag|=%.4g (tol %g)\n", spread,
        tol_diag,
        std::max(std::abs(s[3]), std::max(std::abs(s[4]), std::abs(s[5]))),
        tol_offdiag);
    std::printf("PASS: pw cubic H stress symmetry\n");
    return 0;
}
