/// @file regressions/forcefield/lj_soft_dimer/reg_lj_soft_dimer.cpp
/// @brief Soft-core LJ two-atom single-point vs analytic Beutler one-leg.
/// Public headers only. See REFERENCE.md.
///
/// Goldens computed offline 2026-07-23 from Domain basis:
///   r_λ = (α σ⁶ (1−λ)^p + r⁶)^{1/6}
///   V = λ [4ε((σ/r_λ)¹² − (σ/r_λ)⁶) − V_shift]
///   F_r = λ F_LJ(r_λ) (r/r_λ)^5
/// with V_shift = hard LJ at raw rc (PairLJCut convention).

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_soft.h"

using namespace ATV;

int main() {
    const double eps = 0.5;   // eV
    const double sigma = 3.0; // Å
    const double rc = 8.0;    // Å
    const double r = 3.0;     // Å
    const double lam = 0.5;
    const double alpha = 0.5;
    const double p = 1.0;

    // Hard-coded analytic goldens (python3, 2026-07-23).
    // r_λ = 3.11367244666864 Å
    // V = −0.15722681917773984 eV
    // F_r = 0.7680000000000029 eV/Å  (repulsive; atom0 force −F_r x-hat)
    const double E_ref = -0.15722681917773984;
    const double Fr_ref = 0.7680000000000029;

    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(50.0);

    cpu::PairLJSoft soft(eps, sigma, rc, alpha, p);
    soft.lambda = lam;
    cpu::Evaluate eval;
    eval.set_evaluator(std::move(soft));
    eval.set_neighbor(cpu::md::NeighborList<>{rc, 0.3});
    eval.build();
    eval.run(f);

    const double E = f.meta.get(kEnergy);
    const double fx0 = f.span(kAtom_fx)[0];
    const double fx1 = f.span(kAtom_fx)[1];

    std::printf("PairLJSoft dimer: E=%.16e (ref=%.16e)  "
                "fx0=%.16e fx1=%.16e  Fr_ref=%.16e\n",
                E, E_ref, fx0, fx1, Fr_ref);

    bool ok = true;
    if (!(std::fabs(E - E_ref) <= 1e-10)) {
        std::printf("FAIL: energy\n");
        ok = false;
    }
    // Atom 0 force along −x is −Fr (dx = x0−x1 = −r, fpair>0 → f0 = fpair*dx < 0)
    // |F| = Fr.
    if (!(std::fabs(std::fabs(fx0) - Fr_ref) <= 1e-10)) {
        std::printf("FAIL: |fx0| vs Fr\n");
        ok = false;
    }
    if (!(std::fabs(fx0 + fx1) <= 1e-12)) {
        std::printf("FAIL: Newton III\n");
        ok = false;
    }
    if (!ok)
        return 1;
    std::printf("PASS\n");
    return 0;
}
