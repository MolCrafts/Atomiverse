/// @file regressions/scf/h2o_rhf_dipole/reg_h2o_rhf_dipole.cpp
/// @brief H2O / 6-31G* RHF dipole moment vs the externally pinned PySCF
///        `mf.dip_moment()` — the multipole-observer chain link's e2e.
///
/// Out-of-library consumer: public headers + link atomiverse only. Runs the
/// concrete RHF, wires a MultipoleObserver to its result_handle(), launches
/// it once (the same call a host driver's end_of_step would make), and
/// checks each dipole component against PySCF 2.14.0 to 1e-6 a.u.
/// (REFERENCE.md). HF overestimates the H2O dipole (~2.23 D vs 1.85 D
/// experiment) — the comparison is engine-vs-engine at identical theory.
///
/// Exit non-zero on mismatch. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observer/multipole.h"
#include "atomiverse/cpu/scf/methods.h"

int main() {
    using namespace ATV;

    cpu::Frame frame(3);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_x)[2] = 0.0;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.simbox = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::Pople631Gs;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.run(frame);

    cpu::MultipoleObserver obs{rhf.result_handle()};
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, frame, 0, 0};
    obs.launch(lc);

    // PySCF 2.14.0, RHF/6-31G* (cart), conv_tol=1e-12 — REFERENCE.md.
    const double want[3] = {0.0, 0.0, -0.8757858707840785}; // a.u.
    const double tol = 1e-6;
    const auto got = obs.dipole_au();
    bool ok = true;
    for (int a = 0; a < 3; ++a) {
        const double diff = std::fabs(got[a] - want[a]);
        std::printf("mu[%d] got=%.10f want=%.10f |diff|=%.3e\n", a, got[a],
                    want[a], diff);
        ok = ok && diff < tol;
    }
    std::printf("mu_z = %.6f D (HF overestimates vs 1.85 D exp)\n",
                obs.dipole_debye()[2]);
    if (!ok) {
        std::printf("FAIL: dipole off PySCF oracle (tol %.0e a.u.)\n", tol);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
