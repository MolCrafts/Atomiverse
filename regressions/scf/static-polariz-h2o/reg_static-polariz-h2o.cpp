/// @file regressions/scf/static-polariz-h2o/reg_static-polariz-h2o.cpp
/// @brief H₂O RHF/STO-3G static polarizability via CPHF electric-field
///        response (`RHF::polarizability`) vs the externally pinned PySCF
///        CPHF/finite-field oracle (REFERENCE.md).
///
/// Out-of-library consumer: public headers + link atomiverse only. Gates
/// the three diagonal α_ii components to 1e-5 a.u. Exit non-zero on
/// mismatch. ≤ 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
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
    frame.box = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.run(frame);

    const auto alpha = rhf.polarizability(); // 3×3 [a.u.]

    // PySCF 2.14.0 analytic CPHF (and FD dμ/dF) — REFERENCE.md.
    const double want[3] = {0.04190863, 5.20973255, 2.12715820};
    const double tol = 1e-5;
    bool ok = true;
    for (int a = 0; a < 3; ++a) {
        const double got = alpha(a, a);
        const double diff = std::fabs(got - want[a]);
        std::printf("alpha[%d%d] got=%.10f want=%.10f |diff|=%.3e\n", a, a, got,
                    want[a], diff);
        ok = ok && diff < tol;
    }
    // Off-diagonal should be ~0 for C2v water in this frame.
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < a; ++b) {
            const double off = std::fabs(alpha(a, b));
            std::printf("alpha[%d%d] |got|=%.3e (expect ~0)\n", a, b, off);
            ok = ok && off < 1e-4;
        }
    }
    const double iso = (alpha(0, 0) + alpha(1, 1) + alpha(2, 2)) / 3.0;
    std::printf("alpha_iso = %.8f a.u. (STO-3G, unphysical vs exp ~9–10)\n",
                iso);
    if (!ok) {
        std::printf("FAIL: polarizability off PySCF oracle (tol %.0e a.u.)\n",
                    tol);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
