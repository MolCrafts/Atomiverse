/// @file tests/cpu/dynamics/test_configurational_temperature.cpp
/// @brief Second derivatives of the interaction energy (`CurvatureSource`)
///        against central differences of the forces, the Ewald Laplacian sum
///        rule, and ∂T_C1/∂r against differences of T_C1 (Petravic &
///        Delhommelle, JCP 118, 7477 (2003), eq. 7).
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/unit_system.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/dynamics/configurational_temperature.h"
#include "atomiverse/cpu/dynamics/fix_efield.h"
#include "atomiverse/cpu/dynamics/gear_pc.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/ewald.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/pair_born.h"

#include "nacl_fixture.h"

using namespace ATV;
using namespace nacl_fixture;

namespace {

/// Which force terms an evaluation mounts.
enum class Terms {
    BornTable,      ///< multi-type PairBorn (all pairs, minimum image)
    BornListed,     ///< single-type PairBorn on the neighbor list (ghost rows)
    Coulomb,        ///< CoulLong + KSpaceEwald
    CoulombSpecial, ///< Coulomb with 1-2/1-3/1-4 scales on a bonded chain
    Melt,           ///< BornTable + Coulomb + FixEfield
};

constexpr double kTightAlpha = 4.0 / kCut; // erfc(4) ~ 2e-8
constexpr double kField = 0.3;             // [V/Å] along x

struct Result {
    int n = 0;
    std::vector<double> force;  ///< total frame force, d·n + i
    std::vector<double> charge; ///< per row [e]
    std::vector<double> u, h, lap, g, grad_t;
    cpu::ConfigurationalSample sample;
};

Result evaluate_terms(const std::vector<double>& pos, Terms terms,
                      int side = kSide) {
    cpu::Frame f = nacl(1500.0, 3, 0.25, side);
    const int n = f.nlocal();
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] = pos[i];
        f.span(kAtom_y)[i] = pos[n + i];
        f.span(kAtom_z)[i] = pos[2 * n + i];
    }
    const bool coulomb = terms == Terms::Coulomb
                         || terms == Terms::CoulombSpecial
                         || terms == Terms::Melt;
    if (terms == Terms::CoulombSpecial) {
        // A chain 1-2-3-4 and a pair 10-11 (tags) of neighbouring ions.
        f.add_bond(1, 2, 0);
        f.add_bond(2, 3, 0);
        f.add_bond(3, 4, 0);
        f.add_bond(10, 11, 0);
        f.build_special_list();
    }
    cpu::Evaluate eval;
    eval.set_units("metal");
    if (terms != Terms::BornTable)
        eval.set_neighbor(cpu::md::NeighborList<>{kCut, 0.5}.set_atom_sort(
            1000, 0.0, /*enabled=*/false));
    if (terms == Terms::BornTable || terms == Terms::Melt)
        eval.set_evaluator(cpu::PairBorn{tosi_fumi(), kCut});
    if (terms == Terms::BornListed) {
        const cpu::BornTypeCoeff nacl_row = tosi_fumi()[1];
        eval.set_evaluator(cpu::PairBorn{nacl_row.A, nacl_row.rho,
                                         nacl_row.sigma, nacl_row.C, nacl_row.D,
                                         kCut});
    }
    if (coulomb) {
        const double alpha = terms == Terms::Melt ? kAlpha : kTightAlpha;
        cpu::CoulLong real{alpha, kCut};
        if (terms == Terms::CoulombSpecial)
            real.set_special_bonds(0.0, 0.5, 0.8);
        eval.set_evaluator(std::move(real));
        eval.set_evaluator(cpu::KSpaceEwald{alpha, 1e-8});
    }
    if (terms == Terms::Melt)
        eval.on(ATV::evaluate::Phase::evaluate)
            .add(cpu::md::FixEfield{kField, 0.0, 0.0}, 10);
    cpu::ConfigurationalTemperature observer;
    eval.on(ATV::evaluate::Phase::finalize).add_ref(observer);
    eval.build();
    eval.run(f);

    Result r;
    r.n = n;
    r.force.resize(3 * n);
    r.charge.resize(n);
    for (int i = 0; i < n; ++i) {
        r.charge[i] = f.span(kAtom_charge)[i];
        r.force[i] = f.span(kAtom_fx)[i];
        r.force[n + i] = f.span(kAtom_fy)[i];
        r.force[2 * n + i] = f.span(kAtom_fz)[i];
    }
    const cpu::CurvatureAssembly& a = observer.assembly();
    r.u = a.force();
    r.h = a.hessian_force();
    r.lap = a.laplacian();
    r.g = a.laplacian_gradient();
    a.temperature_gradient(units::from_style(units::Style::metal).boltz,
                           r.grad_t);
    r.sample = observer.last();
    return r;
}

std::vector<double> positions(int side = kSide) {
    const cpu::Frame f = nacl(1500.0, 3, 0.25, side);
    const int n = f.nlocal();
    std::vector<double> pos(3 * n);
    for (int i = 0; i < n; ++i) {
        pos[i] = f.span(kAtom_x)[i];
        pos[n + i] = f.span(kAtom_y)[i];
        pos[2 * n + i] = f.span(kAtom_z)[i];
    }
    return pos;
}

double max_abs(const std::vector<double>& v) {
    double m = 0;
    for (double x : v)
        m = std::max(m, std::fabs(x));
    return m;
}

constexpr double kFdStep = 1e-4;           // [Å]
constexpr int kProbe[] = {0, 1, 2, 3, 37}; // rows checked by differences

/// Every per-atom curvature quantity of `terms` against central
/// differences: H·F of the forces along F, ∇ᵢ²U of the force divergence,
/// ∇ᵢD of the source's own total Laplacian.
void check_against_differences(Terms terms) {
    const std::vector<double> pos = positions();
    const Result base = evaluate_terms(pos, terms);
    const int n = base.n;

    // (H F) = −dF(r + εF)/dε.
    const double eps = kFdStep / max_abs(base.u);
    std::vector<double> plus = pos, minus = pos;
    for (int k = 0; k < 3 * n; ++k) {
        plus[k] += eps * base.u[k];
        minus[k] -= eps * base.u[k];
    }
    const Result fp = evaluate_terms(plus, terms),
                 fm = evaluate_terms(minus, terms);
    std::vector<double> hf(3 * n);
    for (int k = 0; k < 3 * n; ++k)
        hf[k] = -(fp.force[k] - fm.force[k]) / (2 * eps);
    const double h_scale = max_abs(hf);
    for (int k = 0; k < 3 * n; ++k)
        EXPECT_NEAR(base.h[k], hf[k], 1e-6 * h_scale) << "HF row " << k;

    const double lap_scale = max_abs(base.lap);
    const double g_scale = std::max(max_abs(base.g), 1e-3 * lap_scale);
    for (const int i : kProbe) {
        double divergence = 0;
        for (int d = 0; d < 3; ++d) {
            std::vector<double> p = pos, m = pos;
            p[d * n + i] += kFdStep;
            m[d * n + i] -= kFdStep;
            const Result rp = evaluate_terms(p, terms),
                         rm = evaluate_terms(m, terms);
            divergence +=
                (rp.force[d * n + i] - rm.force[d * n + i]) / (2 * kFdStep);
            const double dD =
                (rp.sample.laplacian - rm.sample.laplacian) / (2 * kFdStep);
            EXPECT_NEAR(base.g[d * n + i], dD, 1e-5 * g_scale)
                << "grad D, atom " << i << " axis " << d;
        }
        EXPECT_NEAR(base.lap[i], -divergence, 1e-6 * lap_scale)
            << "laplacian, atom " << i;
    }
}

} // namespace

TEST(Curvature, PairBornTableMatchesDifferencesOfItsForces) {
    check_against_differences(Terms::BornTable);
}

TEST(Curvature, PairBornOnTheNeighborListMatchesDifferencesOfItsForces) {
    check_against_differences(Terms::BornListed);
}

TEST(Curvature, EwaldCoulombMatchesDifferencesOfItsForces) {
    check_against_differences(Terms::Coulomb);
}

TEST(Curvature, SpecialBondCoulombRemaindersMatchDifferencesOfTheForces) {
    // Scaled and excluded pairs carry −(1−s)/r: Hessian, no Laplacian.
    check_against_differences(Terms::CoulombSpecial);
}

#ifdef _OPENMP
// The pair walks open the whole team once their pairs repay it
// (`md::CurvatureLoop`, `PairRowPartition::team_for`: 6·10³ entries for 1000
// rows). 1000 ions give the list ≈ 1.7·10⁴ entries and the all-pairs walk
// (BornTable) 5·10⁵, so both run threaded here, each thread into its own
// accumulator. Every thread count must reproduce the one-thread sums to
// rounding.
TEST(Curvature, EveryThreadCountGivesTheOneThreadSums) {
    constexpr int side = 10;
    const std::vector<double> pos = positions(side);
    const int saved = omp_get_max_threads();
    for (const Terms terms :
         {Terms::BornTable, Terms::BornListed, Terms::Melt}) {
        omp_set_num_threads(1);
        const Result one = evaluate_terms(pos, terms, side);
        const double h_tol = 1e-11 * max_abs(one.h);
        const double lap_tol = 1e-11 * max_abs(one.lap);
        const double g_tol = 1e-11 * max_abs(one.g);
        const double t_tol = 1e-11 * max_abs(one.grad_t);
        for (const int threads : {2, 4, 7}) {
            omp_set_num_threads(threads);
            const Result many = evaluate_terms(pos, terms, side);
            ASSERT_EQ(many.n, one.n);
            for (int k = 0; k < 3 * one.n; ++k) {
                ASSERT_NEAR(many.h[k], one.h[k], h_tol)
                    << "HF row " << k << ", " << threads << " threads";
                ASSERT_NEAR(many.g[k], one.g[k], g_tol)
                    << "grad D row " << k << ", " << threads << " threads";
                ASSERT_NEAR(many.grad_t[k], one.grad_t[k], t_tol)
                    << "grad T row " << k << ", " << threads << " threads";
            }
            for (int i = 0; i < one.n; ++i)
                ASSERT_NEAR(many.lap[i], one.lap[i], lap_tol)
                    << "laplacian, atom " << i << ", " << threads << " threads";
            EXPECT_NEAR(many.sample.laplacian, one.sample.laplacian,
                        1e-11 * std::fabs(one.sample.laplacian));
            EXPECT_NEAR(many.sample.force_sq, one.sample.force_sq,
                        1e-11 * one.sample.force_sq);
        }
    }
    omp_set_num_threads(saved);
}
#endif

TEST(Curvature, EwaldCoulombLaplacianIsTheUniformBackgroundConstant) {
    // Periodic (tinfoil) Coulomb: ∇²ψ = −4πδ + 4π/V, so for a neutral cell
    // Σᵢ∇ᵢ²U = −4π k_e Σq²/V (−8.07 eV/Å² here). The real-space erfc and
    // the reciprocal Laplacians are each ~4 k_e α³ Σq²/√π ≈ 800 eV/Å² and
    // cancel to that constant within the Ewald accuracy (1.5e-5 of it).
    const Result r = evaluate_terms(positions(), Terms::Coulomb);
    const units::UnitSystem metal = units::from_style(units::Style::metal);
    const double volume = std::pow(kSide * kSpacing, 3);
    const double want = -4.0 * units::pi * metal.qqrd2e() * kN / volume;
    EXPECT_NEAR(r.sample.laplacian, want, 1e-4 * std::fabs(want));
}

TEST(Curvature, TemperatureGradientMatchesDifferencesOfTheTemperature) {
    const std::vector<double> pos = positions();
    const Result base = evaluate_terms(pos, Terms::Melt);
    const int n = base.n;
    // T_C1 sees the interaction force only: the field force is removed.
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(base.u[i], base.force[i] - kField * base.charge[i], 1e-12);
        EXPECT_EQ(base.u[n + i], base.force[n + i]);
    }
    const double scale = max_abs(base.grad_t);
    for (const int i : kProbe)
        for (int d = 0; d < 3; ++d) {
            std::vector<double> p = pos, m = pos;
            p[d * n + i] += kFdStep;
            m[d * n + i] -= kFdStep;
            const double dT =
                (evaluate_terms(p, Terms::Melt).sample.temperature
                 - evaluate_terms(m, Terms::Melt).sample.temperature)
                / (2 * kFdStep);
            EXPECT_NEAR(base.grad_t[d * n + i], dT, 1e-5 * scale)
                << "atom " << i << " axis " << d;
        }
}

namespace {

/// A force term with no curvature: a uniform push that is not declared an
/// external drive.
class UndeclaredPush : public cpu::Potential {
  public:
    std::string name() const override { return "undeclared_push"; }
    void launch(const cpu::LaunchContext& ctx) override {
        for (int i = 0; i < ctx.frame.nlocal(); ++i)
            ctx.frame.span(kAtom_fx)[i] += 0.01;
    }
};

struct MeltRun {
    double mean_t_c1 = 0; ///< over the second half [K]
};

/// Tosi–Fumi NaCl integrated by `gear` for `steps` steps; T_C1 from an
/// observer on the same (predicted) configuration as the forces.
MeltRun run_melt(cpu::md::GearPC gear, int steps) {
    cpu::Frame frame = nacl(1500.0, 7);
    cpu::md::MD sim;
    sim.set_units("metal");
    sim.set_neighbor(cpu::md::NeighborList<>{kCut, 0.5});
    sim.set_evaluator(cpu::PairBorn{tosi_fumi(), kCut});
    sim.set_evaluator(cpu::CoulLong{kAlpha, kCut});
    sim.set_evaluator(cpu::KSpaceEwald{kAlpha, 1e-4});
    cpu::ConfigurationalTemperature observer;
    sim.on(ATV::md::Phase::post_force).add_ref(observer, 200);
    sim.set_integrator_ref(gear);
    sim.build();
    MeltRun out;
    int samples = 0;
    for (int step = 0; step < steps; ++step) {
        sim.run(frame, 1);
        if (step < steps / 2)
            continue;
        out.mean_t_c1 += observer.last().temperature;
        ++samples;
    }
    out.mean_t_c1 /= samples;
    return out;
}

} // namespace

TEST(Curvature, AForceTermWithoutCurvatureIsRefused) {
    cpu::Evaluate eval;
    eval.set_units("metal");
    eval.set_evaluator(cpu::PairBorn{tosi_fumi(), kCut});
    eval.on(ATV::evaluate::Phase::evaluate).add(UndeclaredPush{}, 5);
    cpu::ConfigurationalTemperature observer;
    eval.on(ATV::evaluate::Phase::finalize).add_ref(observer);
    EXPECT_THROW(eval.build(), std::runtime_error);
}

/// Sums the observer's per-step evaluations from the same phase, after it.
class SampleSums final : public cpu::Component {
  public:
    explicit SampleSums(const cpu::ConfigurationalTemperature& observer)
        : observer_(observer) {}
    std::string name() const override { return "sample_sums"; }
    std::vector<ATV::DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::post_force, 300)};
    }
    void launch(const cpu::LaunchContext& /*ctx*/) override {
        n += observer_.last().force_sq;
        d += observer_.last().laplacian;
        t += observer_.last().temperature;
        ++count;
    }
    void reset() { n = d = t = 0.0, count = 0; }
    double n = 0.0, d = 0.0, t = 0.0;
    int count = 0;

  private:
    const cpu::ConfigurationalTemperature& observer_;
};

TEST(ConfigurationalTemperature, RunAverageIsTheRatioOfTheMeans) {
    // T_CONF of eq. (5) is <N>/(k_B <D>) over the evaluations of the run,
    // not the mean of T_C1; every run() starts a new average, a continued
    // one included (the CUDA observer's rule).
    cpu::Frame frame = nacl(1500.0, 11);
    cpu::md::MD sim;
    sim.set_units("metal");
    sim.set_neighbor(cpu::md::NeighborList<>{kCut, 0.5});
    sim.set_evaluator(cpu::PairBorn{tosi_fumi(), kCut});
    sim.set_evaluator(cpu::CoulLong{kAlpha, kCut});
    sim.set_evaluator(cpu::KSpaceEwald{kAlpha, 1e-4});
    cpu::ConfigurationalTemperature observer;
    sim.on(ATV::md::Phase::post_force).add_ref(observer, 200);
    SampleSums sums{observer};
    sim.on(ATV::md::Phase::post_force).add_ref(sums, 300);
    cpu::md::GearPC gear{0.001};
    sim.set_integrator_ref(gear);
    sim.build();
    const double kB = units::from_style(units::Style::metal).boltz;
    for (const int steps : {30, 10}) {
        SCOPED_TRACE("run of " + std::to_string(steps) + " steps");
        sums.reset();
        sim.run(frame, steps);
        ASSERT_EQ(sums.count, steps);
        ASSERT_EQ(observer.samples(), steps);
        EXPECT_NEAR(observer.mean().force_sq, sums.n / steps,
                    1e-13 * sums.n / steps);
        EXPECT_NEAR(observer.mean().laplacian, sums.d / steps,
                    1e-13 * sums.d / steps);
        const double t_conf = sums.n / (kB * sums.d);
        EXPECT_NEAR(observer.mean().temperature, t_conf, 1e-12 * t_conf);
        // Over the first run the melt's T_C1 moves enough that the two
        // averages differ, so the check above tells them apart; the second
        // run continues it and averages only its own steps.
        if (steps == 30)
            EXPECT_GT(std::fabs(sums.t / steps - t_conf), 1e-3 * t_conf);
    }
}

TEST(ConfigurationalThermostat,
     DrivesTheConfigurationalTemperatureToTheTarget) {
    // Unthermostatted, this start relaxes to T_C1 ≈ 1080 K. Asked for a
    // target on either side, the feedback of eqs. (12a–c) takes the
    // second-half mean of T_C1 there (within the Nosé–Hoover oscillation:
    // 643 K and 1540 K measured), heating or cooling as needed.
    constexpr double kDt = 0.001; // [ps]
    constexpr int kSteps = 500;
    constexpr double kQ = 20.0; // [Å²/ps²]
    for (const double t0 : {700.0, 1500.0}) {
        const MeltRun held = run_melt(
            cpu::md::GearPC{kDt, cpu::md::ConfigurationalThermostat{t0, kQ}},
            kSteps);
        EXPECT_NEAR(held.mean_t_c1, t0, 0.1 * t0) << "target " << t0;
    }
}
