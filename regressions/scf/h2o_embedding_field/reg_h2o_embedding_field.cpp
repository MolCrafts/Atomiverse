/// @file regressions/scf/h2o_embedding_field/reg_h2o_embedding_field.cpp
/// @brief H₂O / STO-3G: finite-field dipole identity + point-charge
///        embedding vs pinned PySCF `qmmm.mm_charge` (see REFERENCE.md).
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Exit non-zero on mismatch. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observer/multipole.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/core/units.h"

namespace {

ATV::cpu::Frame make_h2o() {
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
    return frame;
}

} // namespace

int main() {
    using namespace ATV;

    auto frame = make_h2o();

    // ── Analytic dipole (link 03) ──────────────────────────────────────
    cpu::scf::RHF rhf0;
    rhf0.basis = cpu::scf::BasisSet::STO3G;
    rhf0.tol = 1e-12;
    rhf0.max_iter = 200;
    rhf0.run(frame);
    if (!rhf0.converged()) {
        std::printf("FAIL: vacuum SCF did not converge\n");
        return 1;
    }

    cpu::MultipoleObserver obs{rhf0.result_handle()};
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, frame, 0, 0};
    obs.launch(lc);
    const auto mu = obs.dipole_au();
    std::printf("mu_z analytic = %.12f a.u.\n", mu[2]);

    // ── Finite-field central difference ────────────────────────────────
    constexpr double F = 1e-4;
    cpu::scf::RHF rhf_p;
    rhf_p.basis = cpu::scf::BasisSet::STO3G;
    rhf_p.tol = 1e-12;
    rhf_p.max_iter = 200;
    rhf_p.external_field = {0.0, 0.0, F};
    rhf_p.run(frame);

    cpu::scf::RHF rhf_m;
    rhf_m.basis = cpu::scf::BasisSet::STO3G;
    rhf_m.tol = 1e-12;
    rhf_m.max_iter = 200;
    rhf_m.external_field = {0.0, 0.0, -F};
    rhf_m.run(frame);

    if (!rhf_p.converged() || !rhf_m.converged()) {
        std::printf("FAIL: finite-field SCF did not converge\n");
        return 1;
    }

    const double dEdF = (rhf_p.energy() - rhf_m.energy()) / (2.0 * F);
    // E(F) = E0 − μ·F  ⇒  dE/dF = −μ
    const double mu_ff = -dEdF;
    std::printf("mu_z finite-field = %.12f a.u.  (dE/dF = %.12f)\n", mu_ff,
                dEdF);
    const double dmu = std::fabs(mu_ff - mu[2]);
    std::printf("|mu_ff - mu_analytic|_z = %.3e  (tol 1e-5)\n", dmu);
    if (dmu > 1e-5) {
        std::printf("FAIL: finite-field dipole off analytic\n");
        return 1;
    }

    // ── Point-charge embedding vs pinned PySCF ─────────────────────────
    // PySCF qmmm.mm_charge excludes MM–MM; Atomiverse includes it in v_nuc.
    // Pin (PySCF 2.14.0, REFERENCE.md):
    constexpr double kEPyscfMm = -74.98418931920534; // no MM–MM
    constexpr double kMmMm =
        (0.5 * -0.5)
        / (6.0 * units::kAngstromInBohr); // classical MM–MM [Ha]
    constexpr double kEWant = kEPyscfMm + kMmMm;

    cpu::scf::RHF rhf_mm;
    rhf_mm.basis = cpu::scf::BasisSet::STO3G;
    rhf_mm.tol = 1e-12;
    rhf_mm.max_iter = 200;
    rhf_mm.point_charges = {{0.0, 0.0, 3.0, 0.5}, {0.0, 0.0, -3.0, -0.5}};
    rhf_mm.run(frame);
    if (!rhf_mm.converged()) {
        std::printf("FAIL: embedded SCF did not converge\n");
        return 1;
    }

    const double e_mm = rhf_mm.energy();
    std::printf("E_embed = %.12f Ha  want = %.12f Ha  (PySCF+MM-MM)\n", e_mm,
                kEWant);
    std::printf("  MM-MM classical = %.12f Ha  PySCF mm = %.12f Ha\n", kMmMm,
                kEPyscfMm);
    const double de = std::fabs(e_mm - kEWant);
    std::printf("|E - want| = %.3e  (tol 1e-8)\n", de);
    if (de > 1e-8) {
        std::printf("FAIL: embedded energy off PySCF pin (+ MM-MM)\n");
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
