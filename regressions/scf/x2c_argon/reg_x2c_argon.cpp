/// @file regressions/scf/x2c_argon/reg_x2c_argon.cpp
/// @brief Ar / def2-TZVP RHF x2c-1e total energy vs externally pinned PySCF.
///
/// Out-of-library consumer: public headers + link atomiverse only.
///
/// Oracle (PySCF 2.14.0, xuncontract=False — **required**; the default
/// xuncontract=True path is out of scope for this link):
/// @code
///   from pyscf import gto, scf
///   mol = gto.M(atom='Ar 0 0 0', basis='def2-TZVP', verbose=0)
///   e_nr = scf.RHF(mol).run(conv_tol=1e-12).e_tot
///   mf = scf.RHF(mol).x2c1e()
///   mf.with_x2c.xuncontract = False
///   e_x2c = mf.run(conv_tol=1e-12).e_tot
///   # e_nr  = -526.8027603325265
///   # e_x2c = -528.6442231685685
///   # delta = -1.8414628360419556
/// @endcode
///
/// Acceptance: |E_x2c − pin| ≤ 1e-5 Ha; E_x2c − E_nonrel < 0 and |Δ|
/// within 5% of the pinned PySCF difference. ≤ 60 s.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

int main() {
    using namespace ATV;

    // Pinned PySCF 2.14.0 values (xuncontract=False, see file header).
    constexpr double kE_nr_pyscf = -526.8027603325265;
    constexpr double kE_x2c_pyscf = -528.6442231685685;
    constexpr double kDelta_pyscf = kE_x2c_pyscf - kE_nr_pyscf; // ~−1.841 Ha
    constexpr double kTolAbs = 1e-5; // Ha

    cpu::Frame frame(1);
    frame.span(kAtom_type)[0] = 18; // Ar
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.simbox = cpu::Box::cubic(30.0);

    // Non-relativistic reference.
    cpu::scf::RHF nr;
    nr.basis = cpu::scf::BasisSet::Def2TZVP;
    nr.tol = 1e-12;
    nr.max_iter = 200;
    nr.x2c = false;
    nr.run(frame);
    if (!nr.converged()) {
        std::printf("FAIL: non-relativistic RHF did not converge\n");
        return 1;
    }
    const double e_nr = frame.meta.get(kEnergy) / units::kHartreeInEv;

    // x2c-1e.
    cpu::scf::RHF x2c;
    x2c.basis = cpu::scf::BasisSet::Def2TZVP;
    x2c.tol = 1e-12;
    x2c.max_iter = 200;
    x2c.x2c = true;
    x2c.run(frame);
    if (!x2c.converged()) {
        std::printf("FAIL: x2c RHF did not converge\n");
        return 1;
    }
    const double e_x2c = frame.meta.get(kEnergy) / units::kHartreeInEv;
    const double delta = e_x2c - e_nr;

    const double diff_x2c = std::fabs(e_x2c - kE_x2c_pyscf);
    const double diff_nr = std::fabs(e_nr - kE_nr_pyscf);
    const double rel_delta =
        std::fabs(delta - kDelta_pyscf) / std::fabs(kDelta_pyscf);

    std::printf("Ar/def2-TZVP RHF\n");
    std::printf("  E_nr   got=%.12f  pin=%.12f  |diff|=%.3e\n", e_nr,
                kE_nr_pyscf, diff_nr);
    std::printf("  E_x2c  got=%.12f  pin=%.12f  |diff|=%.3e  tol=%.0e\n",
                e_x2c, kE_x2c_pyscf, diff_x2c, kTolAbs);
    std::printf("  delta  got=%.12f  pin=%.12f  rel_err=%.3e\n", delta,
                kDelta_pyscf, rel_delta);

    int rc = 0;
    if (!(diff_x2c <= kTolAbs)) {
        std::printf("FAIL: |E_x2c − pin| = %.3e > %.0e Ha\n", diff_x2c,
                    kTolAbs);
        rc = 1;
    }
    if (!(delta < 0.0)) {
        std::printf("FAIL: E_x2c − E_nonrel = %.6f is not negative\n", delta);
        rc = 1;
    }
    if (!(rel_delta <= 0.05)) {
        std::printf("FAIL: |delta − pin_delta| / |pin_delta| = %.3e > 5%%\n",
                    rel_delta);
        rc = 1;
    }
    if (rc == 0)
        std::printf("PASS\n");
    return rc;
}
