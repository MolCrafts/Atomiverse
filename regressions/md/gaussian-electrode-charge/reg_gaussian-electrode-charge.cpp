/// @file regressions/md/gaussian-electrode-charge/reg_gaussian-electrode-charge.cpp
/// Gaussian electrode charge kernels vs published Reed / MetalWalls form.
///
/// Offline pin: analytic evaluation of the literature kernel (no live MetalWalls
/// or LAMMPS ELECTRODE). See REFERENCE.md.
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/gaussian_charge.h"
#include "atomiverse/cpu/potentials/pair_gaussian_coul.h"

using namespace ATV;

namespace {

// ── Fixed-Q 5-atom parallel-plate toy (REFERENCE.md) ───────────────────
// Electrode η = 1.0 Å⁻¹ (type 1); ion is point (type 0).
// q_L = +0.25 on L0,L1; q_R = −0.25 on R0,R1; q_ion = +1.
// Literature kernel energy pin [eV]:
constexpr double kEFixedPin = 1.938689655050537;

// ── Toy 2-site BO: A Q = Ψ, neutrality, antisymmetric Ψ ───────────────
// L = 10 Å, η = 1.0 Å⁻¹, Ψ = (+0.5, −0.5) eV/e
// Q_L = (Ψ_L − Ψ_R) / (2 (A_ii − A_ij)) with A = coulomb * G
constexpr double kQGaussPin = 0.049754759157599825;

bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

/// Solve 2-site neutrality-constrained A Q = Ψ with equal diagonal.
/// A11 Q1 + A12 Q2 + λ = Ψ1; A12 Q1 + A11 Q2 + λ = Ψ2; Q1+Q2=0.
double solve_two_site_q1(double A11, double A12, double Psi1, double Psi2) {
    // Antisymmetric Ψ ⇒ λ = 0; Q1 = (Ψ1 − Ψ2) / (2 (A11 − A12))
    return (Psi1 - Psi2) / (2.0 * (A11 - A12));
}

} // namespace

int main() {
    const double eta = 1.0;
    const double K = ATV::units::coulomb;
    int nfail = 0;

    // ── 1. Fixed-Q energy via public Evaluate + PairGaussianCoul ────────
    cpu::Frame f(5);
    // L0, L1, R0, R1, ion
    const double xyz[5][3] = {
        {0.0, 0.0, 0.0},  {2.0, 0.0, 0.0}, {0.0, 0.0, 10.0},
        {2.0, 0.0, 10.0}, {1.0, 0.0, 5.0},
    };
    const double qq[5] = {0.25, 0.25, -0.25, -0.25, 1.0};
    const int tt[5] = {1, 1, 1, 1, 0};
    for (int i = 0; i < 5; ++i) {
        f.span(kAtom_x)[i] = xyz[i][0];
        f.span(kAtom_y)[i] = xyz[i][1];
        f.span(kAtom_z)[i] = xyz[i][2];
        f.span(kAtom_charge)[i] = qq[i];
        f.span(kAtom_type)[i] = tt[i];
        f.span(kAtom_mass)[i] = 12.0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(100.0);

    cpu::Evaluate eval;
    // type 0 = point (η≤0), type 1 = Gaussian η=1
    eval.set_evaluator(cpu::PairGaussianCoul{40.0, {0.0, eta}, true});
    // PairGaussianCoul is a NeighborConsumer (it has a neighbour-listed bulk
    // path), so the driver requires a list rather than letting it fall back to
    // an O(N²) sweep unannounced. At L = 100 Å and r_c = 40 Å every pair of
    // this 5-atom cell is inside the cutoff, so the list changes which loop
    // runs, never which pairs are summed — the pin below is the same number.
    eval.set_neighbor(cpu::md::NeighborList<>{40.0, /*skin=*/1.0});
    eval.build();
    eval.run(f);

    const double E = f.meta.get(kEnergy);
    std::printf("fixed-Q energy: E=%.12e pin=%.12e |dE|=%.3e\n", E, kEFixedPin,
                std::fabs(E - kEFixedPin));
    if (!near(E, kEFixedPin, 1e-9)) {
        std::printf("FAIL energy pin\n");
        ++nfail;
    }

    // ── 2. Toy Q: Gaussian A vs point-charge A ─────────────────────────
    const double L = 10.0;
    const double Psi1 = 0.5, Psi2 = -0.5;
    const double Gii = cpu::GaussianChargeKernel::self_greens(eta);
    const double Gij = cpu::GaussianChargeKernel::greens(L, eta, eta);
    const double A11g = K * Gii;
    const double A12g = K * Gij;
    const double Qg = solve_two_site_q1(A11g, A12g, Psi1, Psi2);

    // Point-charge A: G=1/r off-diagonal, no self (A11=0) — documented approx.
    const double A11p = 0.0;
    const double A12p = K / L;
    const double Qp = solve_two_site_q1(A11p, A12p, Psi1, Psi2);

    std::printf("toy Q gaussian: Q1=%.12e pin=%.12e\n", Qg, kQGaussPin);
    std::printf("toy Q point-A:  Q1=%.12e  |Qg-Qp|=%.6e\n", Qp,
                std::fabs(Qg - Qp));

    if (!near(Qg, kQGaussPin, 1e-9)) {
        std::printf("FAIL Q gaussian pin\n");
        ++nfail;
    }
    if (!(std::fabs(Qg - Qp) > 1e-6)) {
        std::printf("FAIL expected Gaussian Q ≠ point-charge Q\n");
        ++nfail;
    }

    // Sanity: kernel self vs pair limit consistency
    const double e_eff = cpu::GaussianChargeKernel::eta_eff(eta, eta);
    const double g_near = cpu::GaussianChargeKernel::greens(1e-8, eta, eta);
    const double g_self = cpu::GaussianChargeKernel::self_greens(eta);
    // lim r→0 erf(e r)/r = 2 e / √π = G_self
    (void)e_eff;
    if (!near(g_near, g_self, 1e-6)) {
        std::printf("FAIL self limit: near=%.12e self=%.12e\n", g_near, g_self);
        ++nfail;
    }

    if (nfail == 0) {
        std::printf("PASS\n");
        return 0;
    }
    std::printf("FAIL (%d checks)\n", nfail);
    return 1;
}
