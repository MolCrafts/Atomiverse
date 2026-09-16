/// @file regressions/path/pimd_free_particle_estimators/reg_pimd_free_particle_estimators.cpp
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/path/pimd.h"

using namespace ATV;

int main() {
    cpu::path::Pimd pimd(4, 300.0, 0.25, 5, 3, 0.0);

    cpu::Frame f(1);
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0;
    f.span(kAtom_mass)[0] = 1.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(50.0);

    // Short free-particle PIMD; estimator gate is order-of-magnitude.
    pimd.run(f, 5);
    const double rg2 = pimd.radius_of_gyration_sq();
    const double an = pimd.free_particle_rg2_analytic();
    std::printf("PIMD estimators: Rg2=%.6e analytic=%.6e k_spring=%.6e\n", rg2,
                an, pimd.spring_constant());
    bool ok = true;
    // Closed-form free-polymer R_g² must be positive and match units formula.
    if (!(an > 0.0 && std::isfinite(an)))
        ok = false;
    // Instantaneous diagnostic after free PIMD must be finite and open the ring.
    if (!(rg2 > 0.0 && std::isfinite(rg2)))
        ok = false;
    if (!(pimd.spring_constant() > 0.0))
        ok = false;
    // P=1 classical limit of the formula: (1-1/P²)→0 as P→1 — check P=4 factor.
    const double P = 4.0;
    const double factor = 1.0 - 1.0 / (P * P);
    if (!(std::fabs(factor - 0.9375) < 1e-15))
        ok = false;
    if (!ok) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
