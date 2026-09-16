/// @file regressions/scf/uks-full-lr-o2-lsda/reg_uks-full-lr-o2-lsda.cpp
/// @brief O₂ UKS-LSDA/STO-3G full LR-TDDFT roots vs PySCF pin.
///
/// Public-API consumer only. Geometry, basis, and oracle numbers are pinned
/// in REFERENCE.md (PySCF 2.14.0 offline; no live import in ctest).
///
/// Open-shell Casida A/B on a UKS/LSDA reference with spin-resolved ALDA
/// \(f^{\mathrm{xc}}\) — full RPA, not TDA (`UKS::tddft`, not `UKS::tda`).
#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

ATV::cpu::Frame o2_frame() {
    ATV::cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_type)[i] = 8;
        frame.span(ATV::kAtom_mass)[i] = 15.999;
        frame.span(ATV::kAtom_x)[i] = 0.0;
        frame.span(ATV::kAtom_y)[i] = 0.0;
    }
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 1.2075;
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

bool check(const char* what, double got, double want, double tol) {
    if (!std::isfinite(got) || std::abs(got - want) > tol) {
        std::fprintf(stderr, "FAIL: %s got=%.12f want=%.12f tol=%.1e\n", what,
                     got, want, tol);
        return false;
    }
    std::printf("  %-28s %14.8f (want %14.8f +/- %.1e)\n", what, got, want,
                tol);
    return true;
}

} // namespace

int main() {
    using namespace ATV;

    auto frame = o2_frame();

    // ROKS warm-start → UKS (same multi-minima caution as uks-tda-o2-lsda).
    cpu::scf::ROKS roks(9, 7, cpu::scf::XCType::LSDA);
    roks.basis = cpu::scf::BasisSet::STO3G;
    roks.tol = 1e-12;
    roks.max_iter = 200;
    roks.grid_level = cpu::scf::GridLevel::MatchPySCF;
    roks.run(frame);
    if (!roks.converged()) {
        std::fprintf(stderr, "FAIL: ROKS warm-start did not converge\n");
        return 1;
    }

    using Spin = cpu::scf::SlaterDeterminant::Spin;
    cpu::scf::SCFInitialGuess guess;
    guess.P_alpha = roks.determinant().density(Spin::alpha);
    guess.P_beta = roks.determinant().density(Spin::beta);
    guess.has_alpha = true;
    guess.has_beta = true;

    cpu::scf::UKS uks(9, 7, cpu::scf::XCType::LSDA);
    uks.basis = cpu::scf::BasisSet::STO3G;
    uks.tol = 1e-12;
    uks.max_iter = 200;
    uks.grid_level = cpu::scf::GridLevel::MatchPySCF;
    uks.run(frame, &guess);
    if (!uks.converged()) {
        std::fprintf(stderr, "FAIL: UKS/LSDA did not converge\n");
        return 1;
    }

    // PySCF 2.14.0 pins (REFERENCE.md; cart=True, conv_tol=1e-12, LDA,VWN TDDFT).
    constexpr double kEUksHa = -147.193805922283;
    constexpr double kRootsEv[5] = {
        7.620917323839, 7.642419562717, 7.642420073101,
        8.386321248270, 8.386321248273,
    };
    constexpr double kTolEv = 1e-4;  // eV — job gate
    constexpr double kTolEHa = 1e-4; // Ha — ground-state sanity
    // TDA root-3 pin (must NOT match full RPA if B is dropped).
    constexpr double kTdaRoot3Ev = 8.403284923812;

    std::printf("O2 UKS-LSDA/STO-3G full LR-TDDFT (Casida A,B):\n");
    if (!check("E_UKS [Ha]", uks.energy(), kEUksHa, kTolEHa))
        return 1;

    const cpu::scf::ExcitedStatesResult es = uks.tddft(5);
    if (es.energy_ev.size() < 5) {
        std::fprintf(stderr, "FAIL: expected 5 roots, got %zu\n",
                     es.energy_ev.size());
        return 1;
    }
    if (es.tda) {
        std::fprintf(stderr, "FAIL: tda flag true on full RPA TDDFT path\n");
        return 1;
    }
    if (!es.alda) {
        std::fprintf(stderr, "FAIL: alda flag false on LSDA UKS TDDFT\n");
        return 1;
    }

    bool ok = true;
    for (int i = 0; i < 5; ++i) {
        char label[32];
        std::snprintf(label, sizeof(label), "omega[%d] [eV]", i);
        ok = check(label, es.energy_ev[static_cast<std::size_t>(i)],
                   kRootsEv[i], kTolEv)
             && ok;
        std::printf("    root %d = %.8f eV  (%.10f Ha)\n", i,
                    es.energy_ev[static_cast<std::size_t>(i)],
                    es.energy_ev[static_cast<std::size_t>(i)]
                        / units::kHartreeInEv);
    }

    // Guard: root 3 must not accidentally match pure TDA on this pin.
    const double r3 = es.energy_ev[3];
    if (std::abs(r3 - kTdaRoot3Ev) < 0.005
        && std::abs(r3 - kRootsEv[3]) > kTolEv) {
        std::fprintf(stderr,
                     "FAIL: root 3 matches UKS-TDA pin (%.6f eV) — used tda?\n",
                     kTdaRoot3Ev);
        return 1;
    }

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
