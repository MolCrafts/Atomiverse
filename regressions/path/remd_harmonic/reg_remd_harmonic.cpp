/// @file regressions/path/remd_harmonic/reg_remd_harmonic.cpp
/// @brief END-TO-END REGRESSION: replica-exchange MD on a harmonic dimer —
///        the measured exchange acceptance against the EXACT canonical value.
///
/// The Sugita–Okamoto exchange test
///
///     P_acc = min(1, exp[(β_i − β_j)(U_i − U_j)])
///
/// is the Metropolis rule for swapping configurations between two systems
/// that each sample the **canonical** ensemble at their own β. So its mean
/// acceptance is not a free parameter: with U known analytically the average
/// is a two-dimensional integral over the two canonical distributions, and
/// this case computes it by quadrature and compares.
///
/// See REFERENCE.md. Out-of-library consumer, one flat main().

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/bias/distance_cv.h"
#include "atomiverse/cpu/bias/harmonic_bias.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/path/remd.h"

using namespace ATV;

namespace {

// A single harmonic bond between the two atoms: U(r) = ½ k (r − r₀)².
constexpr double kSpring = 5.0; // [eV/Å²]
constexpr double kR0 = 4.0;     // [Å]

void build_spring(cpu::md::MD& md, double /*temperature*/, void*) {
    md.set_evaluator(cpu::bias::HarmonicBias{
        std::make_unique<cpu::bias::DistanceCV>(1, 2), kSpring, kR0});
}

/// Canonical distribution of the bond length at temperature T, up to
/// normalisation: P(r) ∝ r² exp(−βU(r)) — the r² is the spherical Jacobian of
/// the relative coordinate, which is why ⟨U⟩ is not exactly ½k_BT.
double weight(double r, double beta) {
    const double u = 0.5 * kSpring * (r - kR0) * (r - kR0);
    return r * r * std::exp(-beta * u);
}

/// Exact mean acceptance of one exchange attempt between slots at Ti and Tj:
///     ⟨min(1, exp[(β_i−β_j)(U_i−U_j)])⟩ over P_i(r) P_j(r').
double accept_exact(double t_i, double t_j) {
    const double kb = units::kB_ev_per_k;
    const double bi = 1.0 / (kb * t_i), bj = 1.0 / (kb * t_j);
    const double s_max = std::sqrt(kb * std::max(t_i, t_j) / kSpring);
    const double lo = std::max(0.0, kR0 - 9.0 * s_max);
    const double hi = kR0 + 9.0 * s_max;
    const int n = 4000;
    const double h = (hi - lo) / n;
    std::vector<double> u(n + 1), wi(n + 1), wj(n + 1);
    double zi = 0.0, zj = 0.0;
    for (int a = 0; a <= n; ++a) {
        const double r = lo + a * h;
        u[a] = 0.5 * kSpring * (r - kR0) * (r - kR0);
        wi[a] = weight(r, bi);
        wj[a] = weight(r, bj);
        zi += wi[a];
        zj += wj[a];
    }
    double acc = 0.0;
    for (int a = 0; a <= n; ++a) {
        if (wi[a] / zi < 1e-14)
            continue;
        double inner = 0.0;
        for (int b = 0; b <= n; ++b) {
            const double arg = (bi - bj) * (u[a] - u[b]);
            inner += wj[b] * (arg >= 0.0 ? 1.0 : std::exp(arg));
        }
        acc += wi[a] * inner / zj;
    }
    return acc / zi;
}

cpu::Frame dimer() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = kR0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 39.948;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(30.0);
    f.rebuild_tag_index(0, 2);
    return f;
}

} // namespace

int main() {
    // Geometric ladder, ratio 3. Wide enough that the exact acceptance sits
    // near ⅔ rather than against the ceiling: a criterion that accepts
    // everything (what an UNTHERMOSTATTED ladder degenerates to) is then 21
    // standard errors away instead of a couple.
    const std::vector<double> T = {200.0, 600.0, 1800.0, 5400.0};
    constexpr int n_ex = 600;
    constexpr int steps_per_exchange = 800; // ≫ the 128 fs spring period

    // Even/odd schedule over 4 replicas: even rounds attempt (0,1) and (2,3),
    // odd rounds attempt (1,2). Over `n_ex` rounds each of the three edges is
    // attempted n_ex/2 times, so the expected overall ratio is their mean.
    const double a01 = accept_exact(T[0], T[1]);
    const double a12 = accept_exact(T[1], T[2]);
    const double a23 = accept_exact(T[2], T[3]);
    const double a_pred = (a01 + a12 + a23) / 3.0;

    cpu::Frame f = dimer();
    cpu::path::Remd remd(T, /*dt=*/1.0, steps_per_exchange, /*seed=*/7);
    remd.set_program_builder(&build_spring, nullptr);
    remd.run(f, n_ex);

    const int attempts = remd.exchange_attempts();
    const int accepts = remd.exchange_accepts();
    const double ratio = attempts ? double(accepts) / attempts : 0.0;
    // Exchange attempts are Bernoulli and — at 800 fs between rounds, six
    // spring periods — effectively independent, so the binomial standard
    // error is the right scale.
    const double se = std::sqrt(a_pred * (1.0 - a_pred) / double(attempts));
    const double tol = 4.0 * se;

    std::printf("REMD harmonic dimer: k = %.1f eV/A^2, r0 = %.1f A, ladder "
                "%.0f/%.0f/%.0f/%.0f K\n",
                kSpring, kR0, T[0], T[1], T[2], T[3]);
    std::printf("  exact <P_acc>: edge(0,1) %.4f  edge(1,2) %.4f  edge(2,3) "
                "%.4f  -> %.4f\n",
                a01, a12, a23, a_pred);
    std::printf("  measured: attempts=%d accepts=%d ratio=%.4f  (diff %+.4f = "
                "%.2f SE, tol %.4f)\n",
                attempts, accepts, ratio, ratio - a_pred,
                (ratio - a_pred) / se, tol);

    // Structural: even/odd alternation must exercise every edge.
    const int expect_attempts = (n_ex / 2) * 2 + (n_ex / 2) * 1;
    if (attempts != expect_attempts) {
        std::printf("FAIL: expected %d exchange attempts for a 4-replica "
                    "even/odd schedule over %d rounds, got %d\n",
                    expect_attempts, n_ex, attempts);
        return 1;
    }
    // The physics gate. An unthermostatted ladder — replicas integrated at
    // constant energy, β_i a label rather than a distribution — degenerates
    // to accepting every swap; that is ratio = 1.000, and it is 21 SE outside
    // this window.
    if (!(std::fabs(ratio - a_pred) <= tol)) {
        std::printf("FAIL: exchange acceptance %.4f is %.2f SE off the exact "
                    "canonical value %.4f — the replicas are not sampling the "
                    "canonical ensemble at their own beta, or the Metropolis "
                    "criterion is wrong\n",
                    ratio, (ratio - a_pred) / se, a_pred);
        return 1;
    }
    if (f.nlocal() != 2) {
        std::printf("FAIL: collapse nlocal=%d\n", f.nlocal());
        return 1;
    }
    // Collapse writes the lowest-T slot; positions must remain finite, and the
    // bond must still be a bond (the spring cannot break, so anything outside
    // r0 +- 9 sigma at T0 means the trajectory left the model).
    const double sigma0 = std::sqrt(units::kB_ev_per_k * T[0] / kSpring);
    double dx = f.span(kAtom_x)[0] - f.span(kAtom_x)[1];
    double dy = f.span(kAtom_y)[0] - f.span(kAtom_y)[1];
    double dz = f.span(kAtom_z)[0] - f.span(kAtom_z)[1];
    f.box.min_image(dx, dy, dz);
    const double r_final = std::sqrt(dx * dx + dy * dy + dz * dz);
    std::printf("  collapsed slot 0: r = %.4f A  (r0 %.2f, sigma(T0) %.4f)\n",
                r_final, kR0, sigma0);
    if (!std::isfinite(r_final)
        || std::fabs(r_final - kR0) > 9.0 * sigma0) {
        std::printf("FAIL: collapsed configuration is not a thermal sample of "
                    "the lowest-temperature slot\n");
        return 1;
    }

    // Determinism: same seed → same accept count.
    cpu::Frame f2 = dimer();
    cpu::path::Remd remd2(T, /*dt=*/1.0, steps_per_exchange, /*seed=*/7);
    remd2.set_program_builder(&build_spring, nullptr);
    remd2.run(f2, n_ex);
    if (remd2.exchange_accepts() != accepts
        || remd2.exchange_attempts() != attempts) {
        std::printf("FAIL: seed-7 not deterministic (%d/%d vs %d/%d)\n",
                    remd2.exchange_accepts(), remd2.exchange_attempts(),
                    accepts, attempts);
        return 1;
    }

    std::printf("PASS: REMD acceptance matches the exact canonical value\n");
    return 0;
}
