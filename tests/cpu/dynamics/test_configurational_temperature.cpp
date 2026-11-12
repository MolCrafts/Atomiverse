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

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/unit_system.h"
#include "atomiverse/core/units.h"
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

Result evaluate_terms(const std::vector<double>& pos, Terms terms) {
    cpu::Frame f = nacl(1500.0, 3);
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

std::vector<double> positions() {
    const cpu::Frame f = nacl(1500.0, 3);
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
