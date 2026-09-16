/// @file regressions/scf/h2o_rhf_def2svp_sph/reg_h2o_rhf_def2svp_sph.cpp
/// @brief H2O / def2-SVP (spherical 5d) RHF total energy vs the externally
///        pinned PySCF `cart=False` value — the spherical-AO chain link's e2e.
///
/// Out-of-library consumer: public headers + link atomiverse only. def2-SVP
/// is DEFINED with real spherical harmonics; this run proves the whole
/// spherical pipeline (c2s sandwich + Fock-build bridge) end to end:
/// E_RHF = −75.96098398708 Ha, PySCF 2.14.0, conv_tol=1e-12 (REFERENCE.md).
/// Tolerance 1e-8 Ha per the acceptance contract.
///
/// Exit non-zero on mismatch, printing got / want / tol. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

int main() {
    using namespace ATV;

    // Pinned geometry (identical to the PySCF run in REFERENCE.md):
    // O(0,0,0.1173), H(0,±0.7572,−0.4692) Å.
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
    frame.box = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::Def2SVP;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.run(frame);

    const double e_ha = frame.meta.get(kEnergy) / units::kHartreeInEv;
    const double want = -75.96098398708; // PySCF 2.14.0, cart=False
    const double tol = 1e-8;
    const double diff = std::fabs(e_ha - want);
    std::printf("E_RHF(def2-SVP, spherical) got=%.11f want=%.11f "
                "|diff|=%.3e tol=%.0e\n",
                e_ha, want, diff, tol);
    if (!(diff < tol)) {
        std::printf("FAIL: spherical def2-SVP RHF energy off oracle\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
