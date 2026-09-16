/// @file regressions/path/hremd_soft_lambda/reg_hremd_soft_lambda.cpp
/// @brief END-TO-END REGRESSION: Hamiltonian replica exchange on a λ ladder of
///        harmonic dimers — the measured acceptance against the EXACT
///        canonical value.
///
/// H-REMD swaps configurations between replicas that share one temperature and
/// differ in their Hamiltonian, with
///
///     P_acc = min(1, exp[−β( (U_i(x_j) + U_j(x_i)) − (U_i(x_i) + U_j(x_j)) )])
///
/// (Fukunishi, Watanabe & Takada, *J. Chem. Phys.* **116**, 9058 (2002);
/// Sugita & Okamoto's temperature ladder is the sibling case). That is the
/// Metropolis rule for swapping between two systems each sampling the
/// **canonical** ensemble at the same β, so its mean acceptance is not a free
/// parameter: with `U_λ` analytic the average is a two-dimensional integral
/// over the two canonical distributions, which this case evaluates by
/// quadrature and compares.
///
/// See REFERENCE.md. Out-of-library consumer, one flat main().

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/bias/distance_cv.h"
#include "atomiverse/cpu/bias/harmonic_bias.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/path/hremd.h"

using namespace ATV;

namespace {

// The λ ladder IS the Hamiltonian: one harmonic bond whose stiffness is λ.
//     U_λ(r) = ½ λ (r − r₀)²
// so the exchange argument collapses to (λ_i − λ_j)·(u(r') − u(r)) with
// u(r) = ½(r − r₀)² — an exactly integrable two-dimensional average.
constexpr double kR0 = 4.0;  // [Å]
constexpr double kT = 300.0; // [K]
const std::vector<double> kLambda = {1.0, 4.0, 16.0}; // [eV/Å²]

void build_spring(cpu::md::MD& md, double lambda, void*) {
    md.set_evaluator(cpu::bias::HarmonicBias{
        std::make_unique<cpu::bias::DistanceCV>(1, 2), lambda, kR0});
}

double energy_at_lambda(const cpu::Frame& frame, double lambda, void*) {
    cpu::Frame copy = frame;
    cpu::Evaluate eval;
    eval.set_evaluator(cpu::bias::HarmonicBias{
        std::make_unique<cpu::bias::DistanceCV>(1, 2), lambda, kR0});
    eval.build();
    eval.run(copy);
    return copy.meta.get(kEnergy);
}

/// Canonical distribution of the bond length at stiffness λ, up to
/// normalisation: P(r) ∝ r² exp(−βU_λ(r)) — the r² is the spherical Jacobian
/// of the relative coordinate.
double weight(double r, double lambda, double beta) {
    const double u = 0.5 * lambda * (r - kR0) * (r - kR0);
    return r * r * std::exp(-beta * u);
}

/// Exact mean acceptance of one exchange attempt between slots λ_i and λ_j.
///
///   ⟨min(1, exp[−β(λ_i−λ_j)(u(r')−u(r))])⟩  over  P_i(r)·P_j(r')
double accept_exact(double li, double lj) {
    const double beta = 1.0 / (units::kB_ev_per_k * kT);
    const double s_max = std::sqrt(units::kB_ev_per_k * kT / std::min(li, lj));
    const double lo = std::max(0.0, kR0 - 9.0 * s_max);
    const double hi = kR0 + 9.0 * s_max;
    constexpr int n = 4000;
    const double h = (hi - lo) / n;
    std::vector<double> u(n + 1), wi(n + 1), wj(n + 1);
    double zi = 0.0, zj = 0.0;
    for (int a = 0; a <= n; ++a) {
        const double r = lo + a * h;
        u[a] = 0.5 * (r - kR0) * (r - kR0);
        wi[a] = weight(r, li, beta);
        wj[a] = weight(r, lj, beta);
        zi += wi[a];
        zj += wj[a];
    }
    double acc = 0.0;
    for (int a = 0; a <= n; ++a) {
        if (wi[a] / zi < 1e-14)
            continue;
        double inner = 0.0;
        for (int b = 0; b <= n; ++b) {
            // dH = (λ_i − λ_j)(u(r_b) − u(r_a)); accept with exp(−β dH).
            const double arg = -beta * (li - lj) * (u[b] - u[a]);
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
    // Three λ so the even/odd schedule exercises an edge on EVERY round:
    // even rounds attempt (0,1), odd rounds attempt (1,2).
    constexpr int n_ex = 1000;
    constexpr int steps_per_exchange = 800; // ≫ the 285 fs softest-spring period

    const double a01 = accept_exact(kLambda[0], kLambda[1]);
    const double a12 = accept_exact(kLambda[1], kLambda[2]);
    const double a_pred = 0.5 * (a01 + a12);

    cpu::Frame f = dimer();
    cpu::path::HRemd h(kLambda, kT, /*dt=*/1.0, steps_per_exchange,
                       /*seed=*/11);
    h.thermostat_tau = 100.0; // [fs]
    h.set_program_builder(&build_spring, nullptr);
    h.set_energy_fn(&energy_at_lambda, nullptr);
    h.run(f, n_ex);

    const int attempts = h.exchange_attempts();
    const int accepts = h.exchange_accepts();
    const double ratio = attempts ? double(accepts) / attempts : 0.0;
    // Attempts are Bernoulli and — at 800 fs between rounds, nearly three
    // periods of the softest spring with a 100 fs CSVR coupling — effectively
    // independent, so the binomial standard error is the right scale. The
    // degenerate answer an UNTHERMOSTATTED ladder collapses to (accept
    // everything, ratio 1.000) is 26 SE outside this window.
    const double se = std::sqrt(a_pred * (1.0 - a_pred) / double(attempts));
    const double tol = 4.0 * se;

    std::printf("H-REMD harmonic dimer: r0 = %.1f A, T = %.0f K, lambda "
                "ladder %.1f/%.1f/%.1f eV/A^2\n",
                kR0, kT, kLambda[0], kLambda[1], kLambda[2]);
    std::printf("  exact <P_acc>: edge(0,1) %.4f  edge(1,2) %.4f  -> %.4f\n",
                a01, a12, a_pred);
    std::printf("  measured: attempts=%d accepts=%d ratio=%.4f  (diff %+.4f = "
                "%.2f SE, tol %.4f)\n",
                attempts, accepts, ratio, ratio - a_pred,
                (ratio - a_pred) / se, tol);

    // Structural: with three replicas, every round attempts exactly one edge.
    if (attempts != n_ex) {
        std::printf("FAIL: expected %d exchange attempts for a 3-replica "
                    "even/odd schedule over %d rounds, got %d\n",
                    n_ex, n_ex, attempts);
        return 1;
    }
    // The physics gate. Replicas integrated at constant energy do not sample
    // the canonical ensemble the exchange rule is written for, and the ratio
    // stops matching this integral.
    if (!(std::fabs(ratio - a_pred) <= tol)) {
        std::printf("FAIL: exchange acceptance %.4f is %.2f SE off the exact "
                    "canonical value %.4f — the replicas are not sampling the "
                    "canonical ensemble at beta, or the Metropolis criterion "
                    "is wrong\n",
                    ratio, (ratio - a_pred) / se, a_pred);
        return 1;
    }
    if (f.nlocal() != 2) {
        std::printf("FAIL: collapse nlocal=%d\n", f.nlocal());
        return 1;
    }
    // Collapse writes slot 0; the bond must still be a thermal sample of the
    // softest slot (the spring cannot break).
    const double sigma0 = std::sqrt(units::kB_ev_per_k * kT / kLambda[0]);
    double dx = f.span(kAtom_x)[0] - f.span(kAtom_x)[1];
    double dy = f.span(kAtom_y)[0] - f.span(kAtom_y)[1];
    double dz = f.span(kAtom_z)[0] - f.span(kAtom_z)[1];
    f.box.min_image(dx, dy, dz);
    const double r_final = std::sqrt(dx * dx + dy * dy + dz * dz);
    std::printf("  collapsed slot 0: r = %.4f A  (r0 %.2f, sigma %.4f)\n",
                r_final, kR0, sigma0);
    if (!std::isfinite(r_final) || std::fabs(r_final - kR0) > 9.0 * sigma0) {
        std::printf("FAIL: collapsed configuration is not a thermal sample of "
                    "the softest slot\n");
        return 1;
    }

    // Determinism: same seed → same accept count.
    cpu::Frame f2 = dimer();
    cpu::path::HRemd h2(kLambda, kT, /*dt=*/1.0, steps_per_exchange,
                        /*seed=*/11);
    h2.thermostat_tau = 100.0;
    h2.set_program_builder(&build_spring, nullptr);
    h2.set_energy_fn(&energy_at_lambda, nullptr);
    h2.run(f2, n_ex);
    if (h2.exchange_accepts() != accepts
        || h2.exchange_attempts() != attempts) {
        std::printf("FAIL: seed-11 not deterministic (%d/%d vs %d/%d)\n",
                    h2.exchange_accepts(), h2.exchange_attempts(), accepts,
                    attempts);
        return 1;
    }

    std::printf("PASS: H-REMD acceptance matches the exact canonical value\n");
    return 0;
}
