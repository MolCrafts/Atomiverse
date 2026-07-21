/// @file regressions/scf/h2o_scf_convergence/reg_h2o_scf_convergence.cpp
/// @brief SCF convergence-layer e2e (hf-capability-gaps-07-guess):
///        (1) initial-guess quality — H2O/def2-SVP RHF iteration counts must
///        order Hueckel < Hcore and SAP < Hcore (Lehtola,
///        DOI 10.1021/acs.jctc.8b01089);
///        (2) far-field mixer contrast — on 2.6x-stretched H2O/def2-SVP UHF
///        plain CDIIS stalls within the 60-iteration budget on the unstable
///        spin-symmetric solution, while ADIIS->CDIIS (Hu-Yang,
///        DOI 10.1063/1.3304922; EDIIS: Kudin-Scuseria-Cancès,
///        DOI 10.1063/1.1470195) converges to the PySCF-pinned
///        broken-symmetry UHF minimum within 1e-6 Ha.
///
/// Out-of-library consumer: public headers + link atomiverse only. Oracle
/// values pinned externally with PySCF 2.14.0, conv_tol=1e-12 (REFERENCE.md).
/// Exit non-zero on failure, printing got / want. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;
using cpu::scf::BasisSet;
using cpu::scf::InitGuess;
using cpu::scf::MixerKind;

namespace {

/// @brief H2O frame; `stretch` scales both O-H bonds (1 = equilibrium).
cpu::Frame make_h2o(double stretch) {
    cpu::Frame frame(3);
    const double hx = 0.7572 * stretch;
    const double oz = 0.1173;
    const double hz = oz + (-0.4692 - oz) * stretch;
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = oz;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = hx;
    frame.span(kAtom_z)[1] = hz;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_x)[2] = 0.0;
    frame.span(kAtom_y)[2] = -hx;
    frame.span(kAtom_z)[2] = hz;
    frame.simbox = cpu::Box::cubic(20.0);
    return frame;
}

struct RunResult {
    int iterations;
    bool converged;
    double energy_ha;
};

RunResult run_rhf(InitGuess guess, MixerKind mixer) {
    cpu::Frame frame = make_h2o(1.0);
    cpu::scf::RHF rhf;
    rhf.basis = BasisSet::Def2SVP;
    rhf.tol = 1e-9;
    rhf.max_iter = 100;
    rhf.init_guess = guess;
    rhf.mixer = mixer;
    rhf.run(frame);
    return {rhf.iterations(), rhf.converged(), rhf.energy()};
}

RunResult run_uhf_stretched(MixerKind mixer) {
    cpu::Frame frame = make_h2o(2.6);
    cpu::scf::UHF uhf(/*n_alpha=*/5, /*n_beta=*/5);
    uhf.basis = BasisSet::Def2SVP;
    uhf.tol = 1e-9;
    uhf.max_iter = 60; // the budget CDIIS-alone must exceed
    uhf.init_guess = InitGuess::Hcore;
    uhf.damping_factor = 0.0;
    uhf.diis_start_iter = 1;
    uhf.mixer = mixer;
    uhf.run(frame);
    return {uhf.iterations(), uhf.converged(), uhf.energy()};
}

const char* gname(InitGuess g) {
    switch (g) {
    case InitGuess::Hcore:
        return "hcore";
    case InitGuess::MINAO:
        return "minao";
    case InitGuess::Hueckel:
        return "hueckel";
    default:
        return "sap";
    }
}

} // namespace

int main() {
    int failures = 0;

    // ── Part 1: guess quality — H2O/def2-SVP RHF iteration counts ────────
    // PySCF 2.14.0 cycles (REFERENCE.md): hcore 10, minao 8, huckel 8,
    // sap 8 — same ordering this part asserts on the engine's own counts.
    const double e_eq = -75.96098398708; // PySCF 2.14.0, conv_tol=1e-12
    std::printf("H2O/def2-SVP RHF (tol=1e-9), iterations per guess:\n");
    RunResult r[4];
    const InitGuess order[4] = {InitGuess::Hcore, InitGuess::MINAO,
                                InitGuess::Hueckel, InitGuess::SAP};
    for (int i = 0; i < 4; ++i) {
        r[i] = run_rhf(order[i], MixerKind::CDIIS);
        std::printf("  %-8s iters=%3d converged=%d E=%.11f Ha\n",
                    gname(order[i]), r[i].iterations, r[i].converged,
                    r[i].energy_ha);
        if (!r[i].converged || std::fabs(r[i].energy_ha - e_eq) > 1e-7) {
            std::printf("FAIL: %s guess did not converge to the pinned "
                        "energy %.11f\n",
                        gname(order[i]), e_eq);
            ++failures;
        }
    }
    if (!(r[2].iterations < r[0].iterations)) {
        std::printf("FAIL: iters(Hueckel)=%d !< iters(Hcore)=%d\n",
                    r[2].iterations, r[0].iterations);
        ++failures;
    }
    if (!(r[3].iterations < r[0].iterations)) {
        std::printf("FAIL: iters(SAP)=%d !< iters(Hcore)=%d\n",
                    r[3].iterations, r[0].iterations);
        ++failures;
    }

    // ── Part 2: far-field mixer contrast on 2.6x-stretched H2O UHF ───────
    // PySCF 2.14.0 (REFERENCE.md): the stable broken-symmetry UHF minimum is
    // E = -75.71896632261 Ha (<S^2> ~ 1.97); PySCF's own default-guess UHF
    // first lands on the UNSTABLE spin-symmetric solution -75.35728186304 —
    // the state plain CDIIS oscillates around here without converging.
    const double e_bs = -75.71896632261;
    std::printf("\n2.6x-stretched H2O/def2-SVP UHF (tol=1e-9, budget 60):\n");
    RunResult cd = run_uhf_stretched(MixerKind::CDIIS);
    RunResult ad = run_uhf_stretched(MixerKind::ADIIS_CDIIS);
    std::printf("  CDIIS alone : iters=%3d converged=%d E=%.11f Ha\n",
                cd.iterations, cd.converged, cd.energy_ha);
    std::printf("  ADIIS->CDIIS: iters=%3d converged=%d E=%.11f Ha "
                "(want %.11f, tol 1e-6)\n",
                ad.iterations, ad.converged, ad.energy_ha, e_bs);
    if (cd.converged) {
        std::printf("FAIL: plain CDIIS unexpectedly converged in the budget "
                    "(the stall contrast is gone — re-tune the case)\n");
        ++failures;
    }
    if (!ad.converged) {
        std::printf("FAIL: ADIIS->CDIIS did not converge in the budget\n");
        ++failures;
    }
    if (std::fabs(ad.energy_ha - e_bs) > 1e-6) {
        std::printf("FAIL: ADIIS->CDIIS energy off the PySCF pin: |%.3e| > "
                    "1e-6 Ha\n",
                    ad.energy_ha - e_bs);
        ++failures;
    }

    if (failures) {
        std::printf("FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
