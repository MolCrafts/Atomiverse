/// @file regressions/scf/b3lyp-fused-jk-h2o/reg_b3lyp_fused_jk_h2o.cpp
/// @brief Restricted B3LYP / def2-SVP H2O single point through the restricted-
///        hybrid Fock build (the path the fused one-sweep jk_rhf touches).
///
/// Out-of-library consumer: public headers + link atomiverse only. Concrete
/// method `cpu::scf::RKS{XCType::B3LYP}`, Frame in / Frame out — no `*Config`
/// bag, no factory, no live PySCF at ctest time.
///
/// The B3LYP restricted-hybrid Fock is `F = Hcore + J - 0.5*alpha_hf*K + Vxc`
/// with alpha_hf = 0.20. This is precisely the branch that spec
/// scf-review-06-scf-perf fuses into a single ShellBlockPairList sweep
/// (jk_rhf) — so this regression runs TODAY on the two-pass g_rhf + k path and
/// must keep reproducing the pinned energy after the fused path lands
/// (algebraically exact, ≤1e-12 Ha per-element Fock invariant).
///
/// def2-SVP puts a spherical d-shell on oxygen, so the run exercises d-block
/// exchange quartets — the exact multiplicity-unpack the ≤1e-12 fused-JK gate
/// guards. Golden is a hard-coded PySCF 2.14.0 total energy (REFERENCE.md:
/// tool, version, command, date). <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

int main() {
    using namespace ATV;

    // Pinned geometry (identical to the PySCF `water` preset in REFERENCE.md):
    // O(0,0,0.1173), H(0,±0.7572,−0.4692) Å.
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.1173;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.7572;
    f.span(kAtom_z)[1] = -0.4692;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_x)[2] = 0.0;
    f.span(kAtom_y)[2] = -0.7572;
    f.span(kAtom_z)[2] = -0.4692;
    f.box = cpu::Box::cubic(20.0);

    // PySCF 2.14.0 pin (REFERENCE.md; def2-SVP spherical, conv_tol=1e-12,
    // default DFT grid; ATV default RKS grid is MatchPySCF).
    constexpr double kEB3lypHa = -76.35814183099154; // Ha
    constexpr double kTolHa = 1e-6;                  // Ha

    cpu::scf::RKS rks(cpu::scf::XCType::B3LYP);
    rks.basis = cpu::scf::BasisSet::Def2SVP;
    rks.tol = 1e-10;
    rks.max_iter = 200;
    rks.run(f);

    if (!rks.converged()) {
        std::fprintf(stderr, "FAIL: RKS/B3LYP def2-SVP did not converge\n");
        return 1;
    }

    const double e_ha = rks.energy();
    const double diff = std::fabs(e_ha - kEB3lypHa);
    std::printf("E_RKS(B3LYP/def2-SVP H2O) got=%.11f want=%.11f "
                "|diff|=%.3e tol=%.0e\n",
                e_ha, kEB3lypHa, diff, kTolHa);
    if (!std::isfinite(e_ha) || diff > kTolHa) {
        std::fprintf(stderr, "FAIL: B3LYP/def2-SVP H2O energy off PySCF pin\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
