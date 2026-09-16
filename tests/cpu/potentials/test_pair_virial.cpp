/// @file tests/cpu/potentials/test_pair_virial.cpp
/// @brief The pair virial each style writes to `kStress` is
///        \(\sum_{\text{pairs}} \Delta r \otimes f\), and its trace is
///        \(-3V\,\mathrm{d}E/\mathrm{d}V\).
///
/// Two independent statements per style, because they fail independently:
///
/// 1. **Against a direct pair sum.** The six Voigt components must equal
///    \(\sum f_{\text{pair}}\,\Delta_a \Delta_b\) recomputed here over the same
///    pair list. This is what catches a kernel that gets energy and forces
///    right and the virial wrong — the shape of the defect that shipped: the
///    `PairLJCut` xsimd ket loop guarded its six batch reduction chains with a
///    loop-invariant `if (do_virial)`, and GCC 14.3 at the project's own
///    Release flags (`-O3 -march=native`, AVX-512) dropped most of the
///    accumulation. Energy and forces were exact to 1e-17; the virial trace was
///    60 % short.
///
/// 2. **Against \(-3V\,\mathrm{d}E/\mathrm{d}V\).** Scale every coordinate and
///    the cell by the same \(\lambda\) and the virial trace is fixed by the
///    energy alone. This is what catches a style that writes no virial at all —
///    the shape `PairBorn`, `PairBuck`, `PairMorse`, `PairWCA`, `PairYukawa`,
///    `PairGaussianCoul`, `PairDpd` and `PairLJCutTip4pCut` all shipped with.
///    Both statements are exact algebra: no literature value, no golden, and
///    no dependence on optimisation flags.
///
/// The cluster is small and sits well inside a large cell with a cutoff larger
/// than its diameter, so no pair crosses the cutoff under the finite difference
/// and the identity is exact, not asymptotic. `PairWCA` and `PairDpd` carry
/// their own cutoff, but both are \(C^1\) there (energy *and* force vanish at
/// \(r_c\)), so a pair crossing under the finite difference is harmless.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/gaussian_charge.h"
#include "atomiverse/cpu/potentials/pair_born.h"
#include "atomiverse/cpu/potentials/pair_buck.h"
#include "atomiverse/cpu/potentials/pair_dpd.h"
#include "atomiverse/cpu/potentials/pair_gaussian_coul.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_tip4p_cut.h"
#include "atomiverse/cpu/potentials/pair_morse.h"
#include "atomiverse/cpu/potentials/pair_wca.h"
#include "atomiverse/cpu/potentials/pair_yukawa.h"

using namespace ATV;

namespace {

constexpr int kN = 24;      ///< ≥ 8 neighbours for the low rows ⇒ the xsimd
                            ///< ket loop runs (W = 8 on AVX-512) plus a tail.
constexpr double kBox = 40.0;  ///< [Å] — cutoff stays < L/2.
constexpr double kCutoff = 14.0; ///< [Å] — larger than the cluster diameter.

/// @brief Deterministic pseudo-random cluster, all pairs inside the cutoff.
///
/// Spread over ~7 Å so the LJ pair force spans repulsive and attractive
/// branches; nothing sits near the cutoff, so an affine scale never changes
/// the pair set.
cpu::Frame cluster(double lambda) {
    cpu::Frame f(kN);
    for (int i = 0; i < kN; ++i) {
        const double a = 1.7 * i + 0.3;
        f.span(kAtom_x)[i] =
            lambda * (3.5 * std::sin(a) + 1.1 * std::cos(2.9 * a));
        f.span(kAtom_y)[i] =
            lambda * (3.5 * std::cos(1.3 * a) + 0.9 * std::sin(3.7 * a));
        f.span(kAtom_z)[i] =
            lambda * (3.5 * std::sin(2.1 * a + 0.5) + 1.3 * std::cos(a));
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(kBox * lambda);
    return f;
}

/// @brief The same cluster, alternating ±1 e and two types.
///
/// Type 0 is a Gaussian electrode site, type 1 a point charge, so one run
/// exercises all three `GaussianChargeKernel` branches (Gauss–Gauss,
/// Gauss–point, point–point). Net charge is zero.
cpu::Frame charged_cluster(double lambda) {
    cpu::Frame f = cluster(lambda);
    for (int i = 0; i < kN; ++i) {
        f.span(kAtom_type)[i] = i % 2;
        f.span(kAtom_charge)[i] = (i % 2 == 0) ? 1.0 : -1.0;
    }
    return f;
}

/// @brief Dense half list: every ordered pair i < j, exactly once.
cpu::md::PairList all_pairs_half(int n) {
    cpu::md::PairList p;
    p.inum = n;
    p.n_rows = n;
    p.ilist.resize(n);
    p.numneigh.resize(n);
    p.firstneigh.resize(n);
    int off = 0;
    for (int i = 0; i < n; ++i) {
        p.ilist[static_cast<std::size_t>(i)] = i;
        p.firstneigh[static_cast<std::size_t>(i)] = off;
        p.numneigh[static_cast<std::size_t>(i)] = n - 1 - i;
        for (int j = i + 1; j < n; ++j)
            p.neighbors.push_back(j);
        off += n - 1 - i;
    }
    p.total_pairs = off;
    p.has_shifts = false;
    p.version = 1;
    return p;
}

struct Voigt {
    double v[6] = {0, 0, 0, 0, 0, 0};
    double trace() const { return v[0] + v[1] + v[2]; }
};

/// @brief `sum f_pair * d_a * d_b` over every pair inside @p cutoff.
///
/// @param fp Coefficient of \(\Delta_a\Delta_b\) for pair (i, j) at r², i.e.
///           \(-U'(r)/r\) — whatever sign convention the kernel uses for its
///           own separation vector, the virial is quadratic in it.
template <class FPair>
Voigt reference_virial(const cpu::Frame& f, double cutoff, FPair fp) {
    Voigt w;
    const int n = static_cast<int>(f.nlocal());
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double dx = f.span(kAtom_x)[i] - f.span(kAtom_x)[j];
            double dy = f.span(kAtom_y)[i] - f.span(kAtom_y)[j];
            double dz = f.span(kAtom_z)[i] - f.span(kAtom_z)[j];
            f.box.min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= cutoff * cutoff)
                continue;
            const double c = fp(i, j, r2);
            w.v[0] += c * dx * dx;
            w.v[1] += c * dy * dy;
            w.v[2] += c * dz * dz;
            w.v[3] += c * dx * dy;
            w.v[4] += c * dx * dz;
            w.v[5] += c * dy * dz;
        }
    return w;
}

/// @brief `sum_k x_k ⊗ f_k` from the per-atom forces the style just wrote.
///
/// Independent of any pair bookkeeping: it only knows where each force ended
/// up. Equal to the pair virial whenever the total force vanishes and no
/// separation was min-imaged, which is why the cluster sits far inside the
/// cell. This is the gate for a style that redistributes forces onto sites
/// other than the ones it measured the separation between (TIP4P's M site).
Voigt atomic_virial(const cpu::Frame& f) {
    Voigt w;
    const int n = static_cast<int>(f.nlocal());
    for (int i = 0; i < n; ++i) {
        const double x = f.span(kAtom_x)[i];
        const double y = f.span(kAtom_y)[i];
        const double z = f.span(kAtom_z)[i];
        const double fx = f.span(kAtom_fx)[i];
        const double fy = f.span(kAtom_fy)[i];
        const double fz = f.span(kAtom_fz)[i];
        w.v[0] += x * fx;
        w.v[1] += y * fy;
        w.v[2] += z * fz;
        w.v[3] += x * fy;
        w.v[4] += x * fz;
        w.v[5] += y * fz;
    }
    return w;
}

/// @brief Run one pair style over the dense half list; return energy + stress.
template <class Style>
void run_style(Style& style, cpu::Frame& f, double& energy, Voigt& stress) {
    cpu::md::PairList pairs = all_pairs_half(static_cast<int>(f.nlocal()));
    cpu::Context ctx;
    ctx.md.pair_list = &pairs;
    style.build({ctx, f});
    style.launch({ctx, f, 0, 0});
    energy = f.meta.get(kEnergy);
    for (int k = 0; k < 6; ++k)
        stress.v[k] = f.meta.get(kStress)[k];
}

/// @brief tr(σ) at λ=1, and −3V dE/dV by central difference in λ.
///
/// @param make       Builds a fresh style (components carry state; one per run).
/// @param make_frame Builds the configuration scaled by λ, cell included.
template <class Make, class MakeFrame>
void fd_identity(Make make, MakeFrame make_frame, double& trace,
                 double& minus_3v_dedv) {
    constexpr double h = 1e-6;
    double e_p = 0, e_m = 0, e_0 = 0;
    Voigt s_p, s_m, s_0;
    cpu::Frame f_p = make_frame(1.0 + h);
    cpu::Frame f_m = make_frame(1.0 - h);
    cpu::Frame f_0 = make_frame(1.0);
    auto a = make();
    auto b = make();
    auto c = make();
    run_style(a, f_p, e_p, s_p);
    run_style(b, f_m, e_m, s_m);
    run_style(c, f_0, e_0, s_0);
    trace = s_0.trace();
    minus_3v_dedv = -3.0 * f_0.box.volume() * (e_p - e_m)
                    / (f_p.box.volume() - f_m.box.volume());
}

constexpr double kEps = 0.0104; // [eV]
constexpr double kSig = 3.4;    // [Å]

// Born–Mayer–Huggins, Tosi–Fumi Na+/Cl- unlike-pair row (see pair_born.h).
constexpr double kA = 0.21096;   // [eV]
constexpr double kRho = 0.317;   // [Å]
constexpr double kSigma = 2.755; // [Å]
constexpr double kC = 4.3691;    // [eV Å⁶]
constexpr double kD = 5.4176;    // [eV Å⁸]

} // namespace

// ── PairLJCut ───────────────────────────────────────────────────────────

TEST(PairLJCut, SimdKetLoopVirialMatchesPairSum) {
    cpu::Frame f = cluster(1.0);
    cpu::PairLJCut lj{kEps, kSig, kCutoff};
    double e = 0;
    Voigt got;
    run_style(lj, f, e, got);

    const double sig6 = std::pow(kSig, 6.0);
    const Voigt want = reference_virial(f, kCutoff, [&](int, int, double r2) {
        const double r6inv = sig6 / (r2 * r2 * r2);
        return 24.0 * kEps / r2 * (2.0 * r6inv * r6inv - r6inv);
    });

    // The pair sum is exact arithmetic over the identical pair set; only
    // summation order differs, so the tolerance is rounding, not physics.
    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(got.v[k], want.v[k], 1e-11 * std::max(1.0, std::fabs(want.v[k])))
            << "Voigt component " << k;
    // Guard the guard: a virial of zero must not pass by accident.
    EXPECT_GT(std::fabs(want.trace()), 1.0);
}

TEST(PairLJCut, VirialTraceIsMinusThreeVdEdV) {
    double trace = 0, dedv = 0;
    fd_identity([] { return cpu::PairLJCut{kEps, kSig, kCutoff}; },
                [](double l) { return cluster(l); }, trace, dedv);
    EXPECT_NEAR(trace, dedv, 1e-6 * std::fabs(dedv));
    EXPECT_GT(std::fabs(dedv), 1.0);
}

// ── PairBorn ────────────────────────────────────────────────────────────

TEST(PairBorn, VirialMatchesPairSum) {
    cpu::Frame f = cluster(1.0);
    cpu::PairBorn born{kA, kRho, kSigma, kC, kD, kCutoff};
    double e = 0;
    Voigt got;
    run_style(born, f, e, got);

    const Voigt want = reference_virial(f, kCutoff, [&](int, int, double r2) {
        const double r = std::sqrt(r2);
        const double r6 = r2 * r2 * r2;
        const double r8 = r6 * r2;
        const double dUdr = -kA / kRho * std::exp((kSigma - r) / kRho)
                            + 6.0 * kC / (r6 * r) - 8.0 * kD / (r8 * r);
        return -dUdr / r;
    });

    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(got.v[k], want.v[k],
                    1e-11 * std::max(1.0, std::fabs(want.v[k])))
            << "Voigt component " << k;
    EXPECT_GT(std::fabs(want.trace()), 1e-3);
}

TEST(PairBorn, VirialTraceIsMinusThreeVdEdV) {
    double trace = 0, dedv = 0;
    fd_identity([] { return cpu::PairBorn{kA, kRho, kSigma, kC, kD, kCutoff}; },
                [](double l) { return cluster(l); }, trace, dedv);
    EXPECT_NEAR(trace, dedv, 1e-6 * std::fabs(dedv));
    EXPECT_GT(std::fabs(dedv), 1e-3);
}

// ── Multi-type PairBorn takes the same route ───────────────────────────

TEST(PairBorn, MultiTypeTableWritesTheSameVirial) {
    cpu::Frame f_single = cluster(1.0);
    cpu::Frame f_multi = cluster(1.0);
    cpu::PairBorn single{kA, kRho, kSigma, kC, kD, kCutoff};
    cpu::PairBorn multi{
        std::vector<cpu::BornTypeCoeff>{{0, 0, kA, kRho, kSigma, kC, kD}},
        kCutoff};
    double e_s = 0, e_m = 0;
    Voigt s_s, s_m;
    run_style(single, f_single, e_s, s_s);
    run_style(multi, f_multi, e_m, s_m);
    EXPECT_NEAR(e_s, e_m, 1e-12 * std::max(1.0, std::fabs(e_s)));
    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(s_s.v[k], s_m.v[k],
                    1e-11 * std::max(1.0, std::fabs(s_s.v[k])));
}

// ── PairBuck ────────────────────────────────────────────────────────────

namespace {
constexpr double kBuckA = 100.0;  // [eV]
constexpr double kBuckRho = 0.3;  // [Å]
constexpr double kBuckC = 5.0;    // [eV Å⁶]
} // namespace

TEST(PairBuck, VirialMatchesPairSum) {
    cpu::Frame f = cluster(1.0);
    cpu::PairBuck buck{kBuckA, kBuckRho, kBuckC, kCutoff};
    double e = 0;
    Voigt got;
    run_style(buck, f, e, got);

    const Voigt want = reference_virial(f, kCutoff, [&](int, int, double r2) {
        const double r = std::sqrt(r2);
        const double r6 = r2 * r2 * r2;
        // dU/dr = −(A/ρ) e^{−r/ρ} + 6C/r⁷
        const double dUdr = -kBuckA / kBuckRho * std::exp(-r / kBuckRho)
                            + 6.0 * kBuckC / (r6 * r);
        return -dUdr / r;
    });

    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(got.v[k], want.v[k],
                    1e-11 * std::max(1.0, std::fabs(want.v[k])))
            << "Voigt component " << k;
    EXPECT_GT(std::fabs(want.trace()), 1.0);
}

TEST(PairBuck, VirialTraceIsMinusThreeVdEdV) {
    double trace = 0, dedv = 0;
    fd_identity([] { return cpu::PairBuck{kBuckA, kBuckRho, kBuckC, kCutoff}; },
                [](double l) { return cluster(l); }, trace, dedv);
    EXPECT_NEAR(trace, dedv, 1e-6 * std::fabs(dedv));
    EXPECT_GT(std::fabs(dedv), 1.0);
}

// ── PairMorse ───────────────────────────────────────────────────────────

namespace {
constexpr double kMorseD = 0.15;    // [eV]
constexpr double kMorseAlpha = 1.5; // [Å⁻¹]
constexpr double kMorseR0 = 3.0;    // [Å]
} // namespace

TEST(PairMorse, VirialMatchesPairSum) {
    cpu::Frame f = cluster(1.0);
    cpu::PairMorse morse{kMorseD, kMorseAlpha, kMorseR0, kCutoff};
    double e = 0;
    Voigt got;
    run_style(morse, f, e, got);

    const Voigt want = reference_virial(f, kCutoff, [&](int, int, double r2) {
        const double r = std::sqrt(r2);
        const double x = std::exp(-kMorseAlpha * (r - kMorseR0));
        // dV/dr = 2 D α (1 − x) x
        const double dVdr = 2.0 * kMorseD * kMorseAlpha * (1.0 - x) * x;
        return -dVdr / r;
    });

    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(got.v[k], want.v[k],
                    1e-11 * std::max(1.0, std::fabs(want.v[k])))
            << "Voigt component " << k;
    EXPECT_GT(std::fabs(want.trace()), 1.0);
}

TEST(PairMorse, VirialTraceIsMinusThreeVdEdV) {
    double trace = 0, dedv = 0;
    fd_identity(
        [] { return cpu::PairMorse{kMorseD, kMorseAlpha, kMorseR0, kCutoff}; },
        [](double l) { return cluster(l); }, trace, dedv);
    EXPECT_NEAR(trace, dedv, 1e-6 * std::fabs(dedv));
    EXPECT_GT(std::fabs(dedv), 1.0);
}

// ── PairWCA ─────────────────────────────────────────────────────────────
//
// The cutoff is not a free parameter: r_c = 2^{1/6} σ, where both the energy
// and the force vanish. The reference sum uses the same r_c.

namespace {
constexpr double kWcaEps = 0.05;  // [eV]
constexpr double kWcaSigma = 3.4; // [Å]
double wca_cutoff() { return std::pow(2.0, 1.0 / 6.0) * kWcaSigma; }
} // namespace

TEST(PairWCA, VirialMatchesPairSum) {
    cpu::Frame f = cluster(1.0);
    cpu::PairWCA wca{kWcaEps, kWcaSigma};
    double e = 0;
    Voigt got;
    run_style(wca, f, e, got);

    const double sig2 = kWcaSigma * kWcaSigma;
    const Voigt want =
        reference_virial(f, wca_cutoff(), [&](int, int, double r2) {
            const double s2 = sig2 / r2;
            const double s6 = s2 * s2 * s2;
            return 24.0 * kWcaEps / r2 * (2.0 * s6 * s6 - s6);
        });

    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(got.v[k], want.v[k],
                    1e-11 * std::max(1.0, std::fabs(want.v[k])))
            << "Voigt component " << k;
    EXPECT_GT(std::fabs(want.trace()), 1.0);
}

TEST(PairWCA, VirialTraceIsMinusThreeVdEdV) {
    double trace = 0, dedv = 0;
    fd_identity([] { return cpu::PairWCA{kWcaEps, kWcaSigma}; },
                [](double l) { return cluster(l); }, trace, dedv);
    EXPECT_NEAR(trace, dedv, 1e-6 * std::fabs(dedv));
    EXPECT_GT(std::fabs(dedv), 1.0);
}

// ── PairYukawa ──────────────────────────────────────────────────────────

namespace {
constexpr double kYukA = 10.0;    // [eV Å]
constexpr double kYukKappa = 0.5; // [Å⁻¹]
} // namespace

TEST(PairYukawa, VirialMatchesPairSum) {
    cpu::Frame f = cluster(1.0);
    cpu::PairYukawa yuk{kYukA, kYukKappa, kCutoff};
    double e = 0;
    Voigt got;
    run_style(yuk, f, e, got);

    const Voigt want = reference_virial(f, kCutoff, [&](int, int, double r2) {
        const double r = std::sqrt(r2);
        const double ex = std::exp(-kYukKappa * r);
        // U = A e^{−κr}/r → dU/dr = −A e^{−κr} (κ/r + 1/r²)
        const double dUdr = -kYukA * ex * (kYukKappa / r + 1.0 / r2);
        return -dUdr / r;
    });

    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(got.v[k], want.v[k],
                    1e-11 * std::max(1.0, std::fabs(want.v[k])))
            << "Voigt component " << k;
    EXPECT_GT(std::fabs(want.trace()), 1.0);
}

TEST(PairYukawa, VirialTraceIsMinusThreeVdEdV) {
    double trace = 0, dedv = 0;
    fd_identity([] { return cpu::PairYukawa{kYukA, kYukKappa, kCutoff}; },
                [](double l) { return cluster(l); }, trace, dedv);
    EXPECT_NEAR(trace, dedv, 1e-6 * std::fabs(dedv));
    EXPECT_GT(std::fabs(dedv), 1.0);
}

// ── PairGaussianCoul ────────────────────────────────────────────────────
//
// The style is a self-contained real-space Coulomb with a hard cutoff, not a
// `LongRangeParticipant`: no reciprocal partner carries any part of this
// energy, so the whole pair sum is its own virial. Its Gaussian self-energy
// depends only on η and q — it is independent of the cell, contributes nothing
// to dE/dV, and must therefore contribute nothing to the virial. It is left
// enabled here so a leak would show up as a broken identity.

namespace {
constexpr double kEtaGauss = 0.9; // [Å⁻¹] type 0 — Gaussian electrode site
constexpr double kEtaPoint = 0.0; // type 1 — point charge
std::vector<double> gauss_eta() { return {kEtaGauss, kEtaPoint}; }
} // namespace

TEST(PairGaussianCoul, VirialMatchesPairSum) {
    cpu::Frame f = charged_cluster(1.0);
    cpu::PairGaussianCoul gc{kCutoff, gauss_eta()};
    double e = 0;
    Voigt got;
    run_style(gc, f, e, got);

    const double K = units::coulomb;
    const Voigt want = reference_virial(f, kCutoff, [&](int i, int j,
                                                        double r2) {
        const double r = std::sqrt(r2);
        const double qi = f.span(kAtom_charge)[i];
        const double qj = f.span(kAtom_charge)[j];
        const double ei = gauss_eta()[static_cast<std::size_t>(i % 2)];
        const double ej = gauss_eta()[static_cast<std::size_t>(j % 2)];
        const double dG = cpu::GaussianChargeKernel::dgreens_dr(r, ei, ej);
        // U = K q_i q_j G(r) → −U'/r = −K q_i q_j G'(r)/r
        return -K * qi * qj * dG / r;
    });

    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(got.v[k], want.v[k],
                    1e-11 * std::max(1.0, std::fabs(want.v[k])))
            << "Voigt component " << k;
    EXPECT_GT(std::fabs(want.trace()), 1.0);
}

TEST(PairGaussianCoul, VirialTraceIsMinusThreeVdEdV) {
    double trace = 0, dedv = 0;
    fd_identity([] { return cpu::PairGaussianCoul{kCutoff, gauss_eta()}; },
                [](double l) { return charged_cluster(l); }, trace, dedv);
    EXPECT_NEAR(trace, dedv, 1e-6 * std::fabs(dedv));
    EXPECT_GT(std::fabs(dedv), 1.0);
}

// ── PairDpd ─────────────────────────────────────────────────────────────
//
// Only the conservative force has a virial. `F^D` and `F^R` are thermostat
// forces: both average to zero at equilibrium (`⟨r̂·v_ij⟩ = 0`, `⟨θ⟩ = 0`), the
// invariant measure of the DPD thermostat is the Gibbs measure of the
// conservative potential alone, and `F^R ∝ 1/√dt` diverges as the timestep
// shrinks. Tallying them would put unbounded noise into a barostat and break
// the only exact statement available here, `tr(σ) = −3V dE/dV` — the style
// reports the conservative energy and nothing else. Pinned by
// `VirialExcludesDissipativeAndRandomForces` below.

namespace {
constexpr double kDpdA = 25.0;  // [eV/Å]
constexpr double kDpdRc = 6.0;  // [Å]
constexpr double kDpdT = 300.0; // [K]
constexpr double kDpdDt = 1.0;  // [fs]

/// @brief The cluster with a deterministic non-zero velocity field.
cpu::Frame moving_cluster(double lambda) {
    cpu::Frame f = cluster(lambda);
    for (int i = 0; i < kN; ++i) {
        const double a = 0.9 * i + 0.15;
        f.span(kAtom_vx)[i] = 0.02 * std::sin(a);
        f.span(kAtom_vy)[i] = 0.02 * std::cos(1.7 * a);
        f.span(kAtom_vz)[i] = 0.02 * std::sin(2.3 * a);
    }
    return f;
}
} // namespace

TEST(PairDpd, ConservativeVirialMatchesPairSum) {
    cpu::Frame f = cluster(1.0);
    cpu::PairDpd dpd{kDpdA, /*gamma=*/0.0, kDpdT, kDpdRc, kDpdDt, 1};
    double e = 0;
    Voigt got;
    run_style(dpd, f, e, got);

    const Voigt want = reference_virial(f, kDpdRc, [&](int, int, double r2) {
        const double r = std::sqrt(r2);
        // F^C = a w r̂, w = 1 − r/r_c ⇒ −U'/r = a w / r
        return kDpdA * (1.0 - r / kDpdRc) / r;
    });

    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(got.v[k], want.v[k],
                    1e-11 * std::max(1.0, std::fabs(want.v[k])))
            << "Voigt component " << k;
    EXPECT_GT(std::fabs(want.trace()), 1.0);
}

TEST(PairDpd, VirialTraceIsMinusThreeVdEdV) {
    double trace = 0, dedv = 0;
    fd_identity(
        [] {
            return cpu::PairDpd{kDpdA, /*gamma=*/0.0, kDpdT, kDpdRc, kDpdDt, 1};
        },
        [](double l) { return cluster(l); }, trace, dedv);
    EXPECT_NEAR(trace, dedv, 1e-6 * std::fabs(dedv));
    EXPECT_GT(std::fabs(dedv), 1.0);
}

TEST(PairDpd, VirialExcludesDissipativeAndRandomForces) {
    cpu::Frame f_c = moving_cluster(1.0);
    cpu::Frame f_dr = moving_cluster(1.0);
    cpu::PairDpd conservative{kDpdA, /*gamma=*/0.0, kDpdT, kDpdRc, kDpdDt, 1};
    cpu::PairDpd thermostatted{kDpdA, /*gamma=*/2.0, kDpdT, kDpdRc, kDpdDt, 1};
    double e_c = 0, e_dr = 0;
    Voigt s_c, s_dr;
    run_style(conservative, f_c, e_c, s_c);
    run_style(thermostatted, f_dr, e_dr, s_dr);

    // The thermostat must actually be doing something, or this proves nothing.
    double max_df = 0.0;
    for (int i = 0; i < kN; ++i)
        max_df = std::max(max_df, std::fabs(f_dr.span(kAtom_fx)[i]
                                            - f_c.span(kAtom_fx)[i]));
    EXPECT_GT(max_df, 1e-6);
    EXPECT_GT(std::fabs(s_c.trace()), 1.0);

    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(s_dr.v[k], s_c.v[k],
                    1e-12 * std::max(1.0, std::fabs(s_c.v[k])))
            << "Voigt component " << k;
    EXPECT_NEAR(e_dr, e_c, 1e-12 * std::max(1.0, std::fabs(e_c)));
}

// ── PairLJCutTip4pCut ───────────────────────────────────────────────────
//
// The Coulomb force is measured between M sites and then redistributed onto
// O, H1, H2. Because the redistribution weights are exactly the barycentric
// weights that place M ( (1−α)x_O + (α/2)(x_H1 + x_H2) = x_M ), the virial of
// the three redistributed forces equals x_M ⊗ f — so the pair virial is the
// M-site separation times the M-site force, and using the O positions instead
// would be wrong. `atomic_virial` checks that from the forces alone, without
// knowing any of this.

namespace {
constexpr double kROH = 0.9572;            // [Å]
constexpr double kThetaHOH = 1.82421813;   // [rad] = 104.52°
constexpr double kQDist = 0.15;            // [Å]
constexpr double kTipEps = 0.006721;       // [eV] ≈ 0.155 kcal/mol
constexpr double kTipSigma = 3.15365;      // [Å]
constexpr double kTipCut = 25.0;           // [Å] — beyond the cluster diameter
constexpr double kTipBox = 100.0;          // [Å] — cutoff stays < L/2
constexpr double kQO = -1.04, kQH = 0.52;  // [e]
constexpr int kNWater = 4;

/// @brief Four rigid TIP4P waters, deterministic centres and orientations.
///
/// Tags are consecutive per molecule (O = t, H = t+1, t+2) as the style
/// requires. Everything — centres, intramolecular geometry and the cell —
/// scales with λ, so the M site scales with it too (it is affine in the atom
/// positions) and the whole configuration is an affine image of λ = 1.
cpu::Frame water_cluster(double lambda) {
    const double half = 0.5 * kThetaHOH;
    const double sh = std::sin(half), ch = std::cos(half);
    cpu::Frame f(3 * kNWater);
    for (int m = 0; m < kNWater; ++m) {
        const double a = 1.9 * m + 0.4;
        const double ox = 0.45 * (4.6 * std::sin(a) + 1.2 * std::cos(2.3 * a));
        const double oy = 0.45 * (4.6 * std::cos(1.7 * a) + 0.8 * std::sin(3.1 * a));
        const double oz = 0.45 * (4.6 * std::sin(2.3 * a + 0.7) + 1.1 * std::cos(a));
        const double phi = 0.83 * m + 0.21, psi = 0.57 * m + 1.13;
        auto rot = [&](double vx, double vy, double vz, double& rx, double& ry,
                       double& rz) {
            const double x1 = std::cos(phi) * vx - std::sin(phi) * vy;
            const double y1 = std::sin(phi) * vx + std::cos(phi) * vy;
            rx = x1;
            ry = std::cos(psi) * y1 - std::sin(psi) * vz;
            rz = std::sin(psi) * y1 + std::cos(psi) * vz;
        };
        double h1x, h1y, h1z, h2x, h2y, h2z;
        rot(kROH * sh, 0.0, kROH * ch, h1x, h1y, h1z);
        rot(-kROH * sh, 0.0, kROH * ch, h2x, h2y, h2z);

        const int iO = 3 * m;
        const double px[3] = {ox, ox + h1x, ox + h2x};
        const double py[3] = {oy, oy + h1y, oy + h2y};
        const double pz[3] = {oz, oz + h1z, oz + h2z};
        const int ty[3] = {0, 1, 1};
        const double mass[3] = {15.9994, 1.008, 1.008};
        const double q[3] = {kQO, kQH, kQH};
        for (int k = 0; k < 3; ++k) {
            f.span(kAtom_x)[iO + k] = lambda * px[k];
            f.span(kAtom_y)[iO + k] = lambda * py[k];
            f.span(kAtom_z)[iO + k] = lambda * pz[k];
            f.span(kAtom_type)[iO + k] = ty[k];
            f.span(kAtom_mass)[iO + k] = mass[k];
            f.span(kAtom_charge)[iO + k] = q[k];
            f.span(kAtom_tag)[iO + k] = iO + k + 1;
        }
    }
    f.box = cpu::Box::cubic(kTipBox * lambda);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    for (int m = 0; m < kNWater; ++m) {
        const TagInt t = 3 * m + 1;
        f.add_bond(t, t + 1, 0);
        f.add_bond(t, t + 2, 0);
        f.add_angle(t + 1, t, t + 2, 0);
    }
    f.build_special_list();
    return f;
}

cpu::PairLJCutTip4pCut make_tip4p() {
    return cpu::PairLJCutTip4pCut(/*type_O=*/0, /*type_H=*/1, kQDist, kROH,
                                  kThetaHOH, kTipEps, kTipSigma, kTipCut,
                                  kTipCut)
        .set_special_bonds(0.0, 0.0, 0.0);
}
} // namespace

TEST(PairLJCutTip4pCut, VirialMatchesRedistributedAtomicSum) {
    cpu::Frame f = water_cluster(1.0);
    auto tip4p = make_tip4p();
    double e = 0;
    Voigt got;
    run_style(tip4p, f, e, got);

    // Σ x ⊗ f is origin-independent only when the total force vanishes.
    double sfx = 0, sfy = 0, sfz = 0;
    for (int i = 0; i < 3 * kNWater; ++i) {
        sfx += f.span(kAtom_fx)[i];
        sfy += f.span(kAtom_fy)[i];
        sfz += f.span(kAtom_fz)[i];
    }
    EXPECT_NEAR(sfx, 0.0, 1e-12);
    EXPECT_NEAR(sfy, 0.0, 1e-12);
    EXPECT_NEAR(sfz, 0.0, 1e-12);

    const Voigt want = atomic_virial(f);
    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(got.v[k], want.v[k],
                    1e-9 * std::max(1.0, std::fabs(want.v[k])))
            << "Voigt component " << k;
    EXPECT_GT(std::fabs(want.trace()), 0.1);
}

TEST(PairLJCutTip4pCut, VirialTraceIsMinusThreeVdEdV) {
    double trace = 0, dedv = 0;
    fd_identity([] { return make_tip4p(); },
                [](double l) { return water_cluster(l); }, trace, dedv);
    EXPECT_NEAR(trace, dedv, 1e-6 * std::fabs(dedv));
    EXPECT_GT(std::fabs(dedv), 0.1);
}
