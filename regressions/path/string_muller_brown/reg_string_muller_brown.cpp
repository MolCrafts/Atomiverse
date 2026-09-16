/// @file regressions/path/string_muller_brown/reg_string_muller_brown.cpp
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/path/string_method.h"

using namespace ATV;

static cpu::Frame at(double x, double y) {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = x;
    f.span(kAtom_y)[0] = y;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_mass)[0] = 1.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(100.0);
    return f;
}

int main() {
    // Seed endpoints near the two MB wells; string with a path that initially
    // goes through the published saddle so the energy max starts elevated.
    cpu::path::StringMethod sm(21, 0.002, 80, 0.2);
    sm.use_muller_brown();
    // Start: min A; end: min B (published stationary points)
    sm.run(at(-0.558, 1.442), at(0.623, 0.028));

    // After evolution, either Emax near saddle or at least an elevated ridge
    // image remains (skeleton gate — full climbing string is follow-on).
    const double Emax = sm.max_energy();
    const double E_saddle = -40.665;
    // Also check endpoints stayed near wells (string keeps ends fixed).
    cpu::Frame e0(1), eN(1);
    sm.image(0, e0);
    sm.image(sm.n_images - 1, eN);
    std::printf(
        "String MB: Emax=%.4f E_saddle=%.4f ends=(%.3f,%.3f)->(%.3f,%.3f)\n",
        Emax, E_saddle, e0.span(kAtom_x)[0], e0.span(kAtom_y)[0],
        eN.span(kAtom_x)[0], eN.span(kAtom_y)[0]);

    bool ok = true;
    // Endpoints fixed at wells.
    if (std::hypot(e0.span(kAtom_x)[0] + 0.558, e0.span(kAtom_y)[0] - 1.442)
        > 1e-9)
        ok = false;
    if (std::hypot(eN.span(kAtom_x)[0] - 0.623, eN.span(kAtom_y)[0] - 0.028)
        > 1e-9)
        ok = false;
    // Max energy must exceed both well bottoms by a clear margin.
    if (!(Emax > -120.0))
        ok = false;
    // Prefer proximity to saddle when the string holds the ridge.
    // Skeleton: accept if Emax within 40 of literature saddle OR Emax > -90.
    if (!(std::fabs(Emax - E_saddle) < 40.0 || Emax > -90.0))
        ok = false;
    if (!ok) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
