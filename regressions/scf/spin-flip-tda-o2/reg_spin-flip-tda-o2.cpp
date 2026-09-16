/// @file regressions/scf/spin-flip-tda-o2/reg_spin-flip-tda-o2.cpp
/// @brief O₂ UHF/STO-3G collinear spin-flip TDA (SF-CIS) vs offline pin.
///
/// Public-API consumer only. Geometry, basis, and oracle numbers are pinned
/// in REFERENCE.md (PySCF 2.14.0 UHF MOs + ao2mo SF-CIS matrix; no live
/// import in ctest).
///
/// High-spin collinear SF-TDA: Ms-decreasing α→β singles on UHF (Shao,
/// Head-Gordon & Krylov 2003 exchange-only A matrix).
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

    // ROHF warm-start → UHF (same multi-minima path as ump2-o2-sto3g).
    cpu::scf::ROHF rohf(9, 7);
    rohf.basis = cpu::scf::BasisSet::STO3G;
    rohf.tol = 1e-12;
    rohf.max_iter = 200;
    rohf.run(frame);
    if (!rohf.converged()) {
        std::fprintf(stderr, "FAIL: ROHF warm-start did not converge\n");
        return 1;
    }

    using Spin = cpu::scf::SlaterDeterminant::Spin;
    cpu::scf::SCFInitialGuess guess;
    guess.P_alpha = rohf.determinant().density(Spin::alpha);
    guess.P_beta = rohf.determinant().density(Spin::beta);
    guess.has_alpha = true;
    guess.has_beta = true;

    cpu::scf::UHF uhf(9, 7);
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.tol = 1e-12;
    uhf.max_iter = 200;
    uhf.run(frame, &guess);
    if (!uhf.converged()) {
        std::fprintf(stderr, "FAIL: UHF did not converge\n");
        return 1;
    }

    // PySCF 2.14.0 UHF + collinear SF-CIS pin (REFERENCE.md).
    constexpr double kEUhfHa = -147.633946785450;
    constexpr double kRootsEv[5] = {
        0.045146996469, 1.567707515056, 1.567707515056,
        3.038448959259, 7.848628442263,
    };
    constexpr double kTolEv = 1e-4;  // eV — job gate
    constexpr double kTolEHa = 1e-4; // Ha — ground-state sanity

    std::printf("O2 UHF/STO-3G collinear SF-TDA (α→β):\n");
    if (!check("E_UHF [Ha]", uhf.energy(), kEUhfHa, kTolEHa))
        return 1;

    const cpu::scf::ExcitedStatesResult es = uhf.sf_tda(5);
    if (es.energy_ev.size() < 5) {
        std::fprintf(stderr, "FAIL: expected 5 roots, got %zu\n",
                     es.energy_ev.size());
        return 1;
    }
    if (!es.tda) {
        std::fprintf(stderr, "FAIL: tda flag false\n");
        return 1;
    }
    if (es.alda) {
        std::fprintf(stderr, "FAIL: alda flag true on HF SF-TDA\n");
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

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
