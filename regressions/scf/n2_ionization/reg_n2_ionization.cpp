/// @file regressions/scf/n2_ionization/reg_n2_ionization.cpp
/// @brief N2 / 6-31G* ionization potential two ways — Koopmans −ε_HOMO and
///        ΔSCF — vs externally pinned PySCF 2.14.0 values.
///
/// Out-of-library consumer: public headers + link atomiverse only.
///   - Koopmans: RHF on N2 (R = 1.0977 Å), IP = −ε_HOMO read through
///     `OrbitalInfoObserver::ip_koopmans_eV()`, vs the PySCF pin to
///     1e-3 eV (16.6467 eV at this basis; the HF-limit literature value
///     is ≈17.3 eV and experiment 15.58 eV — Szabo-Ostlund §3.3).
///   - ΔSCF: E(N2⁺, UHF) − E(N2, RHF), the cation configured ONLY through
///     the existing `charge` / `n_alpha` / `n_beta` members (no new Config
///     type, no factory), vs the PySCF ΔSCF pin to 1e-3 eV.
///
/// Physics check carried by the pins themselves: ΔSCF (15.69 eV) < Koopmans
/// (16.65 eV) because the cation relaxes its orbitals — and lands closer to
/// the 15.58 eV experiment. Exit non-zero on mismatch. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/scf/orbital_info_observer.h"

namespace {

ATV::cpu::Frame n2_frame() {
    ATV::cpu::Frame frame(2);
    frame.span(ATV::kAtom_type)[0] = 7;
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_type)[1] = 7;
    frame.span(ATV::kAtom_x)[1] = 0.0;
    frame.span(ATV::kAtom_y)[1] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 1.0977;
    frame.simbox = ATV::cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;

    // ── Neutral N2: RHF/6-31G* ─────────────────────────────────────────
    auto frame = n2_frame();
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::Pople631Gs;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.run(frame);
    const double e_n2 = rhf.energy();
    std::printf("E(N2, RHF) = %.10f Ha (PySCF -108.942662347983)\n", e_n2);

    // Koopmans IP through the observer extension (ip_koopmans_eV = −HOMO).
    cpu::scf::OrbitalInfoObserver orb{rhf.result_handle()};
    cpu::Context ctx;
    cpu::LaunchContext lc{ctx, frame, 0, 0};
    orb.launch(lc);
    const double ip_koopmans = orb.ip_koopmans_eV();

    // PySCF 2.14.0 pin: −ε_HOMO = 0.611756029712 Ha = 16.64672961 eV.
    const double want_koopmans = 16.64672961;
    std::printf("IP(Koopmans) = %.6f eV (want %.6f, exp 15.58, HF-limit "
                "literature ~17.3)\n",
                ip_koopmans, want_koopmans);
    bool ok = std::fabs(ip_koopmans - want_koopmans) < 1e-3;

    // Sanity: inside the HF band around experiment (Szabo-Ostlund §3.3).
    ok = ok && ip_koopmans > 15.5 && ip_koopmans < 17.5;

    // ── Cation N2⁺: UHF, members only (charge / n_alpha / n_beta) ──────
    auto frame_cat = n2_frame();
    cpu::scf::UHF uhf;
    uhf.basis = cpu::scf::BasisSet::Pople631Gs;
    uhf.charge = 1;
    uhf.n_alpha = 7;
    uhf.n_beta = 6;
    uhf.tol = 1e-12;
    uhf.max_iter = 400;
    uhf.run(frame_cat);
    const double e_n2p = uhf.energy();
    std::printf("E(N2+, UHF) = %.10f Ha (PySCF -108.365971102534)\n", e_n2p);

    const double ip_delta = (e_n2p - e_n2) * units::kHartreeInEv;
    const double want_delta = 15.69256822; // PySCF 2.14.0 ΔSCF pin
    std::printf("IP(dSCF) = %.6f eV (want %.6f)\n", ip_delta, want_delta);
    ok = ok && std::fabs(ip_delta - want_delta) < 1e-3;

    // Relaxation lowers the cation: ΔSCF must sit below Koopmans.
    ok = ok && ip_delta < ip_koopmans;

    if (!ok) {
        std::printf("FAIL: ionization potentials off the PySCF oracle\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
