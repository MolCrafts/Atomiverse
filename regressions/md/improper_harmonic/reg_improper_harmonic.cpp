/// @file regressions/md/improper_harmonic/reg_improper_harmonic.cpp
/// @brief END-TO-END: harmonic improper energy at χ = π/2.
///
/// See REFERENCE.md. Public API only, one flat main().

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/improper_harmonic.h"

using namespace ATV;

int main() {
    const double K = 0.4; // [eV/rad²]
    const double tol = 1e-10;

    struct Case {
        double chi0;
        double want;
        const char* label;
    };
    const Case cases[] = {
        {M_PI / 2.0, 0.0, "at equilibrium"},
        {0.0, K * (M_PI / 2.0) * (M_PI / 2.0), "chi0=0"},
    };

    for (const auto& c : cases) {
        cpu::Frame f(4);
        f.span(ATV::kAtom_x)[0] = 0.0;
        f.span(ATV::kAtom_y)[0] = 0.0;
        f.span(ATV::kAtom_z)[0] = 0.0;
        f.span(ATV::kAtom_x)[1] = 1.0;
        f.span(ATV::kAtom_y)[1] = 0.0;
        f.span(ATV::kAtom_z)[1] = 0.0;
        f.span(ATV::kAtom_x)[2] = 1.0;
        f.span(ATV::kAtom_y)[2] = 1.0;
        f.span(ATV::kAtom_z)[2] = 0.0;
        f.span(ATV::kAtom_x)[3] = 1.0;
        f.span(ATV::kAtom_y)[3] = 1.0;
        f.span(ATV::kAtom_z)[3] = 1.0;
        for (int i = 0; i < 4; ++i) {
            f.span(ATV::kAtom_mass)[i] = 12.0;
            f.span(ATV::kAtom_type)[i] = 0;
        }
        f.simbox = cpu::Box::cubic(40.0);
        f.clear_tag_index();
        f.rebuild_tag_index(0, f.nlocal());
        f.add_improper(0, 1, 2, 3, 0);

        cpu::Evaluate eval;
        eval.on(evaluate::Phase::evaluate)
            .add(cpu::ImproperHarmonic{{K}, {c.chi0}});
        eval.enable_timing(false);
        eval.build();
        eval.run(f);

        const double got = f.meta.get(ATV::kEnergy);
        const double imp = f.meta.get(ATV::kEnergyImproper);
        if (std::abs(got - c.want) > tol || std::abs(imp - c.want) > tol) {
            std::printf("FAIL [%s]: energy got=%.16g want=%.16g tol=%g "
                        "(energy_improper=%.16g)\n",
                        c.label, got, c.want, tol, imp);
            return 1;
        }
        std::printf("ok [%s]: E=%.12g eV\n", c.label, got);
    }

    std::printf("PASS: improper_harmonic analytic energies at chi=pi/2\n");
    return 0;
}
