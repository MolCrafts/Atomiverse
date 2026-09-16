/// @file regressions/md/dihedral_charmm/reg_dihedral_charmm.cpp
/// @brief END-TO-END: CHARMM periodic dihedral energy at φ = π/2.
///
/// See REFERENCE.md. Public API only, one flat main().

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/dihedral_charmm.h"

using namespace ATV;

int main() {
    const double K = 0.5; // [eV]
    const double tol = 1e-10;

    struct Case {
        int n;
        double delta;
        double want;
        const char* label;
    };
    const Case cases[] = {
        {1, 0.0, K, "n=1 delta=0"},
        {2, 0.0, 0.0, "n=2 delta=0"},
        {1, M_PI, K, "n=1 delta=pi"},
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
        f.box = cpu::Box::cubic(40.0);
        f.clear_tag_index();
        f.rebuild_tag_index(0, f.nlocal());
        f.add_dihedral(0, 1, 2, 3, 0);

        cpu::Evaluate eval;
        eval.on(evaluate::Phase::evaluate)
            .add(cpu::DihedralCharmm{{K}, {c.n}, {c.delta}});
        eval.enable_timing(false);
        eval.build();
        eval.run(f);

        const double got = f.meta.get(ATV::kEnergy);
        const double dih = f.meta.get(ATV::kEnergyDihedral);
        if (std::abs(got - c.want) > tol || std::abs(dih - c.want) > tol) {
            std::printf("FAIL [%s]: energy got=%.16g want=%.16g tol=%g "
                        "(energy_dihedral=%.16g)\n",
                        c.label, got, c.want, tol, dih);
            return 1;
        }
        std::printf("ok [%s]: E=%.12g eV\n", c.label, got);
    }

    std::printf("PASS: dihedral_charmm analytic energies at phi=pi/2\n");
    return 0;
}
