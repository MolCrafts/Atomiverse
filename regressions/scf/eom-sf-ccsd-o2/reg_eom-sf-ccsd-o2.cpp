/// @file regressions/scf/eom-sf-ccsd-o2/reg_eom-sf-ccsd-o2.cpp
/// @brief O₂ EOM-SF-CCSD/STO-3G spin-flip excitation energies vs offline
///        PySCF pin (REFERENCE.md).
///
/// Out-of-library consumer: public headers only
/// (`cpu/scf/methods.h`, `cpu/frame/frame.h`). Product gate: ω to 1e-4 eV.

#include <cmath>
#include <cstdio>

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
        std::fprintf(stderr, "FAIL: %s got=%.15f want=%.15f tol=%.1e\n", what,
                     got, want, tol);
        return false;
    }
    std::printf("  %-28s %18.12f (want %18.12f +/- %.1e)\n", what, got, want,
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

    // Pins — REFERENCE.md (PySCF 2.14.0 UCCSD.eomsf_ccsd nroots=15, lowest 5).
    constexpr double kECcsd = -147.741936323434; // Ha
    constexpr double kOmegaEv[5] = {
        0.004154770937, 1.024028019354, 1.024028354230,
        1.586396740693, 6.184403293079,
    };
    constexpr double kTolEv = 1e-4; // eV
    constexpr double kTolHa = 1e-6; // Ha

    cpu::scf::UHF uhf(9, 7);
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.tol = 1e-12;
    uhf.max_iter = 200;

    const cpu::scf::EOMSFCCSDResult eom =
        uhf.eom_sf_ccsd(frame, /*n_states=*/5, &guess);
    if (!uhf.converged()) {
        std::fprintf(stderr, "FAIL: UHF/STO-3G (O2) did not converge\n");
        return 1;
    }
    if (!eom.converged) {
        std::fprintf(stderr, "FAIL: EOM-SF-CCSD did not converge\n");
        return 1;
    }
    if (static_cast<int>(eom.energy_ev.size()) < 5) {
        std::fprintf(stderr, "FAIL: expected 5 SF roots, got %zu\n",
                     eom.energy_ev.size());
        return 1;
    }

    std::printf("O2 EOM-SF-CCSD/STO-3G (triplet UHF → SF):\n");
    if (!check("E_UCCSD [Ha]", eom.e_ccsd_ha, kECcsd, kTolHa))
        return 1;
    for (int k = 0; k < 5; ++k) {
        char label[32];
        std::snprintf(label, sizeof(label), "omega_%d [eV]", k);
        if (!check(label, eom.energy_ev[static_cast<std::size_t>(k)],
                   kOmegaEv[k], kTolEv))
            return 1;
    }

    std::printf("  ccsd_iterations            %d\n", eom.ccsd_iterations);
    std::printf("\nPASS: eom-sf-ccsd-o2\n");
    return 0;
}
