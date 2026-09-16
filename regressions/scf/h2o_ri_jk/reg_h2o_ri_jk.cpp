/// @file regressions/scf/h2o_ri_jk/reg_h2o_ri_jk.cpp
/// @brief H2O / def2-TZVP RI-JK density-fitted RHF — the link-12 e2e
///        double gate:
///          1. |E(RI-JK) − E(exact-ERI HF)| ≤ 5e-5 Ha (50 μHa): the
///             Coulomb-metric robust fit leaves an error second order in
///             the fitting residual (Vahtras, Almlöf, Feyereisen, CPL
///             213, 514 (1993), DOI 10.1016/0009-2614(93)89151-7); with
///             the matched def2 universal JKFIT auxiliary basis
///             (Weigend, PCCP 4, 4285 (2002), DOI 10.1039/B204199P;
///             Weigend, PCCP 8, 1057 (2006), DOI 10.1039/b515623h) the
///             residual energy shift is tens of μHa for a molecule of
///             this size.
///          2. |E(RI-JK) − E_PySCF_DF| ≤ 1e-6 Ha against the externally
///             pinned PySCF `scf.RHF(mol).density_fit()` value
///             (REFERENCE.md) — the fit must live in the SAME spherical
///             auxiliary span PySCF uses, not merely "a" fit.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Exit non-zero on mismatch, printing got / want / tol. <= 60 s.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

int main() {
    using namespace ATV;

    // Pinned geometry (identical to the PySCF runs in REFERENCE.md):
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

    // Reference: own exact-ERI RHF (def2-TZVP is spherical).
    cpu::scf::RHF exact;
    exact.basis = cpu::scf::BasisSet::Def2TZVP;
    exact.tol = 1e-12;
    exact.max_iter = 200;
    exact.run(frame);
    const double e_exact = exact.energy();

    // Under test: the same molecule with ri = RI::JK (member-driven).
    cpu::scf::RHF ri;
    ri.basis = cpu::scf::BasisSet::Def2TZVP;
    ri.tol = 1e-12;
    ri.max_iter = 200;
    ri.ri = cpu::scf::RI::JK;                     // fitted J and K
    ri.aux_basis = cpu::scf::AuxBasis::Def2JKFIT; // the default, explicit
    ri.run(frame);
    const double e_ri = ri.energy();

    int rc = 0;

    // Gate 1: second-order fit error vs own exact-ERI HF.
    const double d_fit = std::fabs(e_ri - e_exact);
    std::printf("E_RHF(exact)  = %.12f Ha\n", e_exact);
    std::printf("E_RHF(RI-JK)  = %.12f Ha\n", e_ri);
    std::printf("gate 1: |E_RI - E_exact| = %.3e Ha (tol 5e-5)\n", d_fit);
    if (!(d_fit <= 5e-5)) {
        std::printf("FAIL: RI-JK fit error beyond 50 uHa\n");
        rc = 1;
    }

    // Gate 2: external oracle — PySCF 2.14.0 density_fit() (REFERENCE.md).
    const double want_pyscf = -76.058999347289; // Ha
    const double d_oracle = std::fabs(e_ri - want_pyscf);
    std::printf("gate 2: |E_RI - E_PySCF_DF| = %.3e Ha (tol 1e-6), "
                "want %.12f\n",
                d_oracle, want_pyscf);
    if (!(d_oracle <= 1e-6)) {
        std::printf("FAIL: RI-JK energy off the PySCF density_fit oracle\n");
        rc = 1;
    }

    std::printf(rc == 0 ? "PASS\n" : "FAIL\n");
    return rc;
}
