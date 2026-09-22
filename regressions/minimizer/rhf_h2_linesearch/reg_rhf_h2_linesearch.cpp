/// @file regressions/minimizer/rhf_h2_linesearch/reg_rhf_h2_linesearch.cpp
/// @brief END-TO-END: L-BFGS + StrongWolfe line search over a NESTED RHF
///        force source relaxes H2 to the RHF/STO-3G equilibrium bond length.
///
/// The geometry pin is literature (NIST CCCBDB, RHF/STO-3G H2 r_e = 0.712 A);
/// the iteration pin is the witness that every line-search trial carried a
/// real SCF energy and slope. Public headers only. See REFERENCE.md.

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

int main() {
    // ── Reference ────────────────────────────────────────────────────────
    // NIST CCCBDB (https://cccbdb.nist.gov), calculated bond length,
    // HF/STO-3G, H2 (1-Sigma-g+): r_e = 0.712 A. The four-digit value
    // 0.7122 A is the same table at full precision; the tolerance below is
    // an order of magnitude coarser than the table's own resolution.
    const double r_ref = 0.7122; // [A]
    const double r_tol = 5.0e-3; // [A]
    // Frozen iteration bound — see REFERENCE.md "Iteration bound". Measured
    // 5 iterations with a working StrongWolfe search over the nested RHF;
    // frozen at 5 + 2 headroom. The same case with a step-scaled search
    // (LineSearchMode::Fixed, same alpha0) needs > 200 and does not converge;
    // with the pre-fix replay (zeroed trials) the search accepted alpha = 0
    // and the geometry never moved at all.
    const int max_iterations = 7;
    const int budget = 30; // generous; hitting it is a failure

    // ── H2, stretched 0.19 A past equilibrium ────────────────────────────
    const double r0 = 0.90; // [A]
    cpu::Frame frame(2);
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = r0;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.0;
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1; // Z = 1 (hydrogen)
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_tag)[i] = i + 1;
    }
    frame.box = cpu::Box::cubic(20.0);

    // ── RHF/STO-3G as a NESTED force source at the minimizer force phase ──
    // tol is tight so the trial energies/slopes the line search reads back
    // are converged well below the Wolfe tests' own resolution.
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-10; // [Hartree]
    rhf.max_iter = 100;

    // Borrowed so the accepted-step trace survives the run (diagnostics).
    // Declared before the driver: the driver must die first.
    cpu::minimizer::LBFGS lbfgs{/*memory=*/5, /*alpha0=*/0.01, /*max_ls=*/20};
    // No line_search() call: StrongWolfe is the DEFAULT mode, and it is the
    // mode under test — each trial geometry re-enters the nested RHF.

    cpu::minimizer::Minimize opt;
    opt.set_optimizer_ref(lbfgs);
    opt.on(ATV::minimizer::Phase::force).add(std::move(rhf));
    opt.add_criterion(cpu::ForceConverged{/*ftol=*/0.01, /*rms_ftol=*/0.005,
                                          /*max_dtol=*/5e-3, /*rms_dtol=*/2e-3,
                                          /*etol=*/1e-5});
    opt.build();

    const int iterations = opt.run(frame, budget);

    const double dx = frame.span(kAtom_x)[1] - frame.span(kAtom_x)[0];
    const double dy = frame.span(kAtom_y)[1] - frame.span(kAtom_y)[0];
    const double dz = frame.span(kAtom_z)[1] - frame.span(kAtom_z)[0];
    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double energy = frame.meta.get(kEnergy); // [eV]
    const double fmax =
        std::max(std::hypot(frame.span(kAtom_fx)[0], frame.span(kAtom_fy)[0],
                            frame.span(kAtom_fz)[0]),
                 std::hypot(frame.span(kAtom_fx)[1], frame.span(kAtom_fy)[1],
                            frame.span(kAtom_fz)[1]));
    const std::string reason = opt.stop_reason();

    std::printf("H2 RHF/STO-3G + LBFGS/StrongWolfe over a nested SCF\n");
    std::printf("  r_start   = %.6f A\n", r0);
    std::printf("  r_final   = %.6f A   (CCCBDB HF/STO-3G r_e = %.4f A, "
                "tol %.1e A)\n",
                r, r_ref, r_tol);
    std::printf("  E_final   = %.8f eV\n", energy);
    std::printf("  fmax      = %.3e eV/A\n", fmax);
    std::printf("  iterations= %d           (frozen bound %d, budget %d)\n",
                iterations, max_iterations, budget);
    std::printf("  converged = %d   reason=%s\n", (int)opt.converged(),
                reason.c_str());
    // Accepted step per iteration: a degenerate search (the pre-fix replay,
    // which never re-entered the nested SCF) either froze at alpha = 0 or
    // took the untested initial trial step every time.
    for (const auto& t : lbfgs.trace())
        std::printf("    step %2d  E=%.8f eV  fmax=%.4e eV/A  "
                    "max_disp=%.6e A\n",
                    t.step, t.energy, t.max_force, t.max_displacement);

    bool ok = true;
    if (!(std::fabs(r - r_ref) < r_tol)) {
        std::printf("FAIL: bond length off the CCCBDB RHF/STO-3G equilibrium: "
                    "got %.6f A, want %.4f A, tol %.1e A\n",
                    r, r_ref, r_tol);
        ok = false;
    }
    if (!(iterations <= max_iterations)) {
        std::printf("FAIL: too many iterations: got %d, want <= %d — the line "
                    "search is not exploiting real trial energies/slopes\n",
                    iterations, max_iterations);
        ok = false;
    }
    if (!opt.converged()
        || reason.find("force_converged") == std::string::npos) {
        std::printf("FAIL: expected a force_converged stop, got converged=%d "
                    "reason=%s\n",
                    (int)opt.converged(), reason.c_str());
        ok = false;
    }
    if (!ok)
        return 1;
    std::printf("PASS: H2 relaxed to the RHF/STO-3G equilibrium bond length "
                "in %d line-searched iterations\n",
                iterations);
    return 0;
}
