/// @file regressions/scf/soscf_hard_case/reg_soscf_hard_case.cpp
/// @brief SOSCF hard-case e2e (hf-capability-gaps-09-soscf): second-order
///        CIAH orbital rotations (Sun, arXiv:1610.08423; Helgaker,
///        Jørgensen, Olsen, *Molecular Electronic-Structure Theory*, ch. 10)
///        as the SCF finishing step, on two cases pinned against external
///        PySCF 2.14.0 `mf.newton()` runs (REFERENCE.md, commands verbatim):
///
///        Case A (RHF pin) — N2 at R = 2.5 Å / 6-31G (Cartesian), Hcore
///        guess. PySCF newton: 4 macro cycles, E = -108.21122307939 Ha,
///        HOMO = -0.304093695 Ha. The engine's `soscf = true` run must land
///        on the SAME solution (energy ≤ 1e-8 Ha, HOMO ≤ 1e-6 Ha — stretched
///        N2 RHF has multiple SCF solutions, so the HOMO check pins the
///        basin), with Newton-phase macro count ≤ 2x the PySCF pin. CDIIS
///        also converges this case (engine 13 iterations) — the second-order
///        payoff here is the iteration count, not rescue.
///
///        Case B (UHF stall contrast) — 2.6x-stretched H2O / def2-SVP UHF
///        from the Hcore guess (the 07-link hard case, per the spec's
///        fallback allowance): plain CDIIS oscillates around the unstable
///        spin-symmetric solution for the whole 60-iteration budget, while
///        the DIIS-early/SOSCF-finish escalation converges onto the
///        broken-symmetry minimum. PySCF pins (REFERENCE.md): the BS
///        minimum E = -75.71896632262 Ha, <S^2> = 1.9710; the
///        escalation-equivalent newton macro count is 8 (6 DIIS warmup
///        cycles from the engine-like mixed-Hcore guess, then mf.newton()
///        — the ONLY warmup depth from which PySCF newton reaches the BS
///        state at all; colder or shallower seeds land on spin-symmetric
///        stationary points, see REFERENCE.md). The BS state is a
///        mirror-degenerate pair (alpha/beta swap), so solution identity
///        is pinned by energy + <S^2>, not per-channel HOMOs.
///
/// Out-of-library consumer: public headers + link atomiverse only. Exit
/// non-zero on failure, printing got / want. <= 60 s, single process.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;
using cpu::scf::BasisSet;
using cpu::scf::InitGuess;
using Spin = cpu::scf::SlaterDeterminant::Spin;

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
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

/// @brief N2 at bond length R [Angstrom] along z.
cpu::Frame make_n2(double R) {
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 7;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_type)[1] = 7;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 2.5;
    (void)R;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

int main() {
    int failures = 0;

    // ── Case A: stretched N2 RHF — energy/HOMO/iteration pin ─────────────
    // PySCF 2.14.0 (REFERENCE.md): RHF(N2, R=2.5, 6-31G, cart=True).newton()
    // from the hcore guess, conv_tol=1e-12: 4 macro cycles,
    // E = -108.21122307939 Ha, HOMO = -0.304093695 Ha.
    const double eA_pin = -108.21122307939;
    const double homoA_pin = -0.304093695;
    const int macroA_pin = 4;

    std::printf("Case A: N2 R=2.5 A, 6-31G RHF, hcore guess (tol 1e-9)\n");
    cpu::Frame fA1 = make_n2(2.5);
    cpu::scf::RHF cdiisA;
    cdiisA.basis = BasisSet::Pople631G;
    cdiisA.tol = 1e-9;
    cdiisA.max_iter = 60;
    cdiisA.init_guess = InitGuess::Hcore;
    cdiisA.run(fA1);

    cpu::Frame fA2 = make_n2(2.5);
    cpu::scf::RHF soscfA;
    soscfA.basis = BasisSet::Pople631G;
    soscfA.tol = 1e-9;
    soscfA.max_iter = 60;
    soscfA.init_guess = InitGuess::Hcore;
    soscfA.soscf = true;
    soscfA.run(fA2);
    const double homoA = soscfA.determinant().homo(Spin::alpha);

    std::printf("  CDIIS : iters=%3d conv=%d E=%.11f Ha\n", cdiisA.iterations(),
                cdiisA.converged(), cdiisA.energy());
    std::printf("  SOSCF : iters=%3d (newton %d) conv=%d E=%.11f Ha "
                "HOMO=%.8f (pins: E=%.11f HOMO=%.8f macro=%d)\n",
                soscfA.iterations(), soscfA.soscf_iterations(),
                soscfA.converged(), soscfA.energy(), homoA, eA_pin, homoA_pin,
                macroA_pin);

    if (!soscfA.converged()) {
        std::printf("FAIL[A]: soscf=true did not converge\n");
        ++failures;
    }
    if (std::fabs(soscfA.energy() - eA_pin) > 1e-8) {
        std::printf("FAIL[A]: |E - pin| = %.3e > 1e-8 Ha\n",
                    soscfA.energy() - eA_pin);
        ++failures;
    }
    if (std::fabs(homoA - homoA_pin) > 1e-6) {
        std::printf("FAIL[A]: HOMO off the pinned solution: %.8f vs %.8f "
                    "(different SCF basin?)\n",
                    homoA, homoA_pin);
        ++failures;
    }
    if (soscfA.soscf_iterations() > 2 * macroA_pin) {
        std::printf("FAIL[A]: newton-phase iters %d > 2 x pinned macro %d\n",
                    soscfA.soscf_iterations(), macroA_pin);
        ++failures;
    }
    if (!(soscfA.iterations() < cdiisA.iterations())) {
        std::printf("FAIL[A]: SOSCF total %d !< CDIIS total %d — the "
                    "second-order payoff is gone\n",
                    soscfA.iterations(), cdiisA.iterations());
        ++failures;
    }

    // ── Case B: 2.6x-stretched H2O UHF — CDIIS stalls, SOSCF converges ───
    // PySCF 2.14.0 (REFERENCE.md): the broken-symmetry UHF minimum
    // E = -75.71896632262 Ha, <S^2> = 1.9710 (mf.newton() from the
    // stability-rotated density: 4 macros). Escalation-equivalent macro
    // pin: 6 DIIS warmup cycles from the mixed-Hcore guess + mf.newton()
    // = 8 macro cycles to the same state.
    const double eB_pin = -75.71896632262;
    const double s2B_pin = 1.9710;
    const int macroB_pin = 8;
    const int budget = 60;

    std::printf("Case B: H2O 2.6x, def2-SVP UHF, hcore guess (budget %d)\n",
                budget);
    auto run_uhf = [&](bool soscf) {
        cpu::Frame frame = make_h2o(2.6);
        cpu::scf::UHF uhf(/*n_alpha=*/5, /*n_beta=*/5);
        uhf.basis = BasisSet::Def2SVP;
        uhf.tol = 1e-9;
        uhf.max_iter = budget;
        uhf.init_guess = InitGuess::Hcore;
        uhf.damping_factor = 0.0;
        uhf.diis_start_iter = 1;
        uhf.soscf = soscf;
        uhf.run(frame);
        return uhf;
    };
    cpu::scf::UHF cdiisB = run_uhf(false);
    cpu::scf::UHF soscfB = run_uhf(true);
    const double s2B = soscfB.determinant().spin_contamination();

    std::printf("  CDIIS : iters=%3d conv=%d E=%.11f Ha\n", cdiisB.iterations(),
                cdiisB.converged(), cdiisB.energy());
    std::printf("  SOSCF : iters=%3d (newton %d) conv=%d E=%.11f Ha "
                "<S2>=%.4f (pins: E=%.11f <S2>=%.4f macro=%d)\n",
                soscfB.iterations(), soscfB.soscf_iterations(),
                soscfB.converged(), soscfB.energy(), s2B, eB_pin, s2B_pin,
                macroB_pin);

    if (cdiisB.converged()) {
        std::printf("FAIL[B]: plain CDIIS unexpectedly converged in the "
                    "budget (the stall contrast is gone — re-tune)\n");
        ++failures;
    }
    if (!soscfB.converged()) {
        std::printf("FAIL[B]: soscf=true did not converge in the same "
                    "budget\n");
        ++failures;
    }
    if (std::fabs(soscfB.energy() - eB_pin) > 1e-8) {
        std::printf("FAIL[B]: |E - pin| = %.3e > 1e-8 Ha\n",
                    soscfB.energy() - eB_pin);
        ++failures;
    }
    if (std::fabs(s2B - s2B_pin) > 1e-3) {
        std::printf("FAIL[B]: <S^2> = %.4f off the pinned broken-symmetry "
                    "state %.4f (different solution?)\n",
                    s2B, s2B_pin);
        ++failures;
    }
    if (soscfB.soscf_iterations() > 2 * macroB_pin) {
        std::printf("FAIL[B]: newton-phase iters %d > 2 x pinned macro %d\n",
                    soscfB.soscf_iterations(), macroB_pin);
        ++failures;
    }

    if (failures) {
        std::printf("FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
