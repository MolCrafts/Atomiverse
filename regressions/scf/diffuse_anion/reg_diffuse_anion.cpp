/// @file regressions/scf/diffuse_anion/reg_diffuse_anion.cpp
/// @brief Diffuse-function physics for anions (Kendall-Dunning-Harrison) —
///        OH⁻ ε_HOMO binding vs basis, plus Cl⁻/6-31+G* and H2O/aug-cc-pVDZ
///        RHF total energies vs externally pinned PySCF 2.14.0 values.
///
/// Out-of-library consumer: public headers + link atomiverse only. Anions
/// hold their extra electron loosely; compact bases lack the slow-decay
/// functions needed to describe it, and the HOMO comes out unbound or
/// barely bound (Kendall, Dunning, Harrison, J. Chem. Phys. 96, 6796
/// (1992), DOI 10.1063/1.462569). Assertions (all PySCF pins in
/// REFERENCE.md; conv_tol=1e-12, Pople cart=True, aug-cc sph):
///   - OH⁻/STO-3G: ε_HOMO = +0.251911715713 Ha > 0 — the unbound artifact.
///   - OH⁻/6-31G*: ε_HOMO = −0.037774390226 Ha (barely bound; pinned).
///   - OH⁻/6-31+G*: ε_HOMO = −0.103579773338 Ha — strictly lower than the
///     6-31G* value and same sign as the PySCF pin (diffuse sp restores
///     binding); each ε matches its PySCF pin to 1e-6 Ha.
///   - Cl⁻/6-31+G*: E = −459.539661044387 Ha to 1e-7 Ha.
///   - H2O/aug-cc-pVDZ (spherical): E = −76.041393519987 Ha to 1e-7 Ha.
///
/// Exit non-zero on mismatch, printing got / want / tol. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace {

/// OH⁻: O(0,0,0), H(0,0,0.9640) Å — identical to the PySCF pin runs.
ATV::cpu::Frame oh_frame() {
    ATV::cpu::Frame frame(2);
    frame.span(ATV::kAtom_type)[0] = 8;
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_type)[1] = 1;
    frame.span(ATV::kAtom_x)[1] = 0.0;
    frame.span(ATV::kAtom_y)[1] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 0.9640;
    frame.simbox = ATV::cpu::Box::cubic(20.0);
    return frame;
}

/// One RHF run; returns {E_total [Ha], ε_HOMO [Ha]}.
struct RhfOut {
    double e;
    double homo;
};

RhfOut run_rhf(ATV::cpu::Frame frame, ATV::cpu::scf::BasisSet basis,
               int charge) {
    ATV::cpu::scf::RHF rhf;
    rhf.basis = basis;
    rhf.charge = charge;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.run(frame);
    const double homo = rhf.determinant().homo(
        ATV::cpu::scf::SlaterDeterminant::Spin::alpha);
    return {rhf.energy(), homo};
}

bool check(const char* label, double got, double want, double tol) {
    const double diff = std::fabs(got - want);
    std::printf("%-34s got=%.12f want=%.12f |diff|=%.3e tol=%.0e\n", label,
                got, want, diff, tol);
    return diff < tol;
}

} // namespace

int main() {
    using ATV::cpu::scf::BasisSet;

    bool ok = true;

    // ── OH⁻ ε_HOMO across three bases (binding restored by diffuse) ────
    const auto sto = run_rhf(oh_frame(), BasisSet::STO3G, -1);
    const auto gs = run_rhf(oh_frame(), BasisSet::Pople631Gs, -1);
    const auto pgs = run_rhf(oh_frame(), BasisSet::Pople631pGs, -1);

    ok &= check("eps_HOMO(OH-, STO-3G)", sto.homo, 0.251911715713, 1e-6);
    ok &= check("eps_HOMO(OH-, 6-31G*)", gs.homo, -0.037774390226, 1e-6);
    ok &= check("eps_HOMO(OH-, 6-31+G*)", pgs.homo, -0.103579773338, 1e-6);
    ok &= check("E(OH-, 6-31+G*)", pgs.e, -75.376293363750, 1e-7);

    // The compact-basis artifact: STO-3G leaves the extra electron unbound.
    if (!(sto.homo > 0.0)) {
        std::printf("FAIL: expected the STO-3G HOMO artifact (eps > 0)\n");
        ok = false;
    }
    // Diffuse sp strictly deepens the HOMO, same sign as the PySCF pin.
    if (!(pgs.homo < gs.homo && pgs.homo < 0.0)) {
        std::printf("FAIL: diffuse functions must lower eps_HOMO below the "
                    "6-31G* value and bind the electron\n");
        ok = false;
    }

    // ── Cl⁻ / 6-31+G* total energy (third-period diffuse row) ──────────
    ATV::cpu::Frame cl(1);
    cl.span(ATV::kAtom_type)[0] = 17;
    cl.span(ATV::kAtom_x)[0] = 0.0;
    cl.span(ATV::kAtom_y)[0] = 0.0;
    cl.span(ATV::kAtom_z)[0] = 0.0;
    cl.simbox = ATV::cpu::Box::cubic(20.0);
    const auto clm = run_rhf(cl, BasisSet::Pople631pGs, -1);
    ok &= check("E(Cl-, 6-31+G*)", clm.e, -459.539661044387, 1e-7);

    // ── H2O / aug-cc-pVDZ (spherical) total energy ──────────────────────
    ATV::cpu::Frame h2o(3);
    h2o.span(ATV::kAtom_type)[0] = 8;
    h2o.span(ATV::kAtom_x)[0] = 0.0;
    h2o.span(ATV::kAtom_y)[0] = 0.0;
    h2o.span(ATV::kAtom_z)[0] = 0.1173;
    h2o.span(ATV::kAtom_type)[1] = 1;
    h2o.span(ATV::kAtom_x)[1] = 0.0;
    h2o.span(ATV::kAtom_y)[1] = 0.7572;
    h2o.span(ATV::kAtom_z)[1] = -0.4692;
    h2o.span(ATV::kAtom_type)[2] = 1;
    h2o.span(ATV::kAtom_x)[2] = 0.0;
    h2o.span(ATV::kAtom_y)[2] = -0.7572;
    h2o.span(ATV::kAtom_z)[2] = -0.4692;
    h2o.simbox = ATV::cpu::Box::cubic(20.0);
    const auto wat = run_rhf(h2o, BasisSet::AugCcPVDZ, 0);
    ok &= check("E(H2O, aug-cc-pVDZ)", wat.e, -76.041393519987, 1e-7);

    if (!ok) {
        std::printf("FAIL: diffuse-anion regression off the PySCF oracle\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
