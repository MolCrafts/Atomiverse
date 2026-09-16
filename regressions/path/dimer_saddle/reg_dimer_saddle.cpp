/// @file regressions/path/dimer_saddle/reg_dimer_saddle.cpp
/// @brief Dimer method finds inverted-harmonic saddle at origin.
/// Public API only. See REFERENCE.md.
///
/// V = ½ x² − ½ y²  (saddle at 0). Start near (0.3, 0.4); dimer converges to
/// origin with |F_mod| small.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/path/dimer.h"

using namespace ATV;

int main() {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 0.35;
    f.span(kAtom_y)[0] = 0.25;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_mass)[0] = 1.0;
    f.span(kAtom_type)[0] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(20.0);

    cpu::path::Dimer dimer(0.01, 0.03, 12, 1e-3, 0.02, 42);
    dimer.use_inverted_harmonic(1.0, 1.0);
    const int steps = dimer.run(f, 300);

    const double r =
        std::hypot(f.span(kAtom_x)[0], f.span(kAtom_y)[0], f.span(kAtom_z)[0]);
    std::printf("Dimer: steps=%d converged=%d r=%.6f f_mod=%.4e\n", steps,
                (int)dimer.converged(), r, dimer.max_force());

    bool ok = true;
    if (!dimer.converged()) {
        std::printf("FAIL: not converged\n");
        ok = false;
    }
    if (!(r < 0.1)) {
        std::printf("FAIL: not near saddle origin\n");
        ok = false;
    }
    if (!ok)
        return 1;
    std::printf("PASS\n");
    return 0;
}
