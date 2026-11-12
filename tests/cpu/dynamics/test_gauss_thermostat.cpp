/// @file tests/cpu/dynamics/test_gauss_thermostat.cpp
/// @brief Gaussian kinetic thermostats of `cpu::md::GearPC` (Petravic &
///        Delhommelle, JCP 118, 7477 (2003), eqs. 9–11): the multiplier law,
///        and the constrained temperature held exactly while a strong field
///        drives a Tosi–Fumi NaCl configuration.
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/unit_system.h"
#include "atomiverse/cpu/dynamics/fix_efield.h"
#include "atomiverse/cpu/dynamics/gear_pc.h"
#include "atomiverse/cpu/dynamics/internal/gauss_constraint.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/ewald.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/pair_born.h"

#include "nacl_fixture.h"

using namespace ATV;
using cpu::md::GaussKind;
using cpu::md::internal::GaussConstraint;

namespace {

/// Random two-species data with zero total momentum, as the paper assumes.
struct Ions {
    int n = 0;
    std::vector<double> mass, q, v[3], force[3];
    std::vector<int> type;
};

Ions random_ions(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    Ions s;
    s.n = n;
    s.mass.resize(n);
    s.q.resize(n);
    s.type.resize(n);
    for (int d = 0; d < 3; ++d) {
        s.v[d].resize(n);
        s.force[d].resize(n);
    }
    double p[3] = {0, 0, 0}, m_sum = 0;
    for (int i = 0; i < n; ++i) {
        const bool na = i % 2 == 0;
        s.type[i] = na ? 0 : 1;
        s.mass[i] = na ? 22.99 : 35.45;
        s.q[i] = na ? 1.0 : -1.0;
        m_sum += s.mass[i];
        for (int d = 0; d < 3; ++d) {
            // A species drift along x, as a field builds one.
            s.v[d][i] = g(rng) + (d == 0 ? (na ? 0.7 : -0.4) : 0.0);
            s.force[d][i] = 3.0 * g(rng);
            p[d] += s.mass[i] * s.v[d][i];
        }
    }
    for (int d = 0; d < 3; ++d)
        for (int i = 0; i < n; ++i)
            s.v[d][i] -= p[d] / m_sum;
    return s;
}

/// α the way the paper writes it: momenta p = m v, interaction force F,
/// field E along x acting on charge q.
double paper_alpha(const Ions& s, GaussKind kind, double E) {
    double num = 0, den = 0;
    if (kind == GaussKind::Kinetic) { // eq. (9)
        for (int i = 0; i < s.n; ++i) {
            const double m = s.mass[i];
            for (int d = 0; d < 3; ++d) {
                const double p = m * s.v[d][i];
                num += p * s.force[d][i] / m;
                den += p * p / m;
            }
            num += E * s.q[i] * m * s.v[0][i] / m;
        }
    } else if (kind == GaussKind::Flow) { // eqs. (3), (10)
        double mom[2] = {0, 0}, count[2] = {0, 0}, fx[2] = {0, 0};
        for (int i = 0; i < s.n; ++i) {
            mom[s.type[i]] += s.mass[i] * s.v[0][i];
            fx[s.type[i]] += s.force[0][i];
            count[s.type[i]] += 1;
        }
        for (int i = 0; i < s.n; ++i) {
            const int nu = s.type[i];
            const double m = s.mass[i];
            const double u = mom[nu] / (count[nu] * m); // eq. (3)
            const double pi_x = m * s.v[0][i] - m * u;
            const double phi_x = s.force[0][i] - fx[nu] / count[nu];
            const double py = m * s.v[1][i], pz = m * s.v[2][i];
            num += (pi_x * phi_x + py * s.force[1][i] + pz * s.force[2][i]) / m;
            den += (pi_x * pi_x + py * py + pz * pz) / m;
        }
    } else { // eq. (11)
        for (int i = 0; i < s.n; ++i) {
            const double m = s.mass[i];
            const double py = m * s.v[1][i], pz = m * s.v[2][i];
            num += (py * s.force[1][i] + pz * s.force[2][i]) / m;
            den += (py * py + pz * pz) / m;
        }
    }
    return num / den;
}

/// Mass-weighted mean of `col` over the rows of `type` (−1: all rows).
double group_mean(const std::vector<double>& mass, const double* col,
                  const std::vector<int>& types, int type) {
    double m = 0, p = 0;
    for (std::size_t i = 0; i < mass.size(); ++i)
        if (type < 0 || types[i] == type) {
            m += mass[i];
            p += mass[i] * col[i];
        }
    return p / m;
}

} // namespace

TEST(GaussConstraint, MultiplierIsPetravicDelhommelleEquationsNineToEleven) {
    constexpr double kField = 0.8; // along x [force/charge]
    const Ions s = random_ions(40, 11);
    for (const GaussKind kind :
         {GaussKind::Kinetic, GaussKind::Flow, GaussKind::Transverse}) {
        GaussConstraint c(kind, /*axis=*/0);
        c.bind(s.n, s.type.data());
        std::vector<double> w[3], a[3], vdot[3];
        for (int d = 0; d < 3; ++d) {
            w[d].resize(s.n);
            a[d].resize(s.n);
            vdot[d].resize(s.n);
            for (int i = 0; i < s.n; ++i)
                a[d][i] = (s.force[d][i] + (d == 0 ? kField * s.q[i] : 0.0))
                          / s.mass[i];
        }
        c.peculiar(s.n, s.mass.data(), s.type.data(),
                   {s.v[0].data(), s.v[1].data(), s.v[2].data()},
                   {w[0].data(), w[1].data(), w[2].data()});
        const double alpha = GaussConstraint::multiplier(
            s.n, s.mass.data(), {w[0].data(), w[1].data(), w[2].data()},
            {a[0].data(), a[1].data(), a[2].data()});
        const double want = paper_alpha(s, kind, kField);
        EXPECT_NEAR(alpha, want, 1e-12 * std::fabs(want))
            << "kind " << static_cast<int>(kind);

        // v̇ = a − α w leaves the unconstrained motion to the forces alone.
        for (int d = 0; d < 3; ++d)
            for (int i = 0; i < s.n; ++i)
                vdot[d][i] = a[d][i] - alpha * w[d][i];
        for (int d = 0; d < 3; ++d) {
            const double com_a = group_mean(s.mass, a[d].data(), s.type, -1);
            EXPECT_NEAR(group_mean(s.mass, vdot[d].data(), s.type, -1), com_a,
                        1e-12)
                << "centre of mass, axis " << d;
        }
        if (kind == GaussKind::Flow)
            for (int nu = 0; nu < 2; ++nu)
                EXPECT_NEAR(group_mean(s.mass, vdot[0].data(), s.type, nu),
                            group_mean(s.mass, a[0].data(), s.type, nu), 1e-12)
                    << "species flow " << nu;
        if (kind == GaussKind::Transverse)
            for (int i = 0; i < s.n; ++i)
                EXPECT_EQ(vdot[0][i], a[0][i]) << "field axis, atom " << i;
    }
}

namespace {

/// Kinetic temperatures computed from the frame alone [K], each with its
/// own degrees of freedom: barycentric (3N−3), flow along x (3N−4 for two
/// species) and yz (2N−2).
struct Temperatures {
    double kinetic, flow, transverse;
    double flow_split; ///< u_Na,x − u_Cl,x [Å/ps]
};

Temperatures measure(const cpu::Frame& f) {
    const units::UnitSystem metal = units::from_style(units::Style::metal);
    const int n = f.nlocal();
    std::vector<double> mass(n), v[3];
    std::vector<int> type(n);
    for (int d = 0; d < 3; ++d)
        v[d].resize(n);
    for (int i = 0; i < n; ++i) {
        mass[i] = f.span(kAtom_mass)[i];
        type[i] = f.span(kAtom_type)[i];
        v[0][i] = f.span(kAtom_vx)[i];
        v[1][i] = f.span(kAtom_vy)[i];
        v[2][i] = f.span(kAtom_vz)[i];
    }
    double bary[3], species[2];
    for (int d = 0; d < 3; ++d)
        bary[d] = group_mean(mass, v[d].data(), type, -1);
    for (int nu = 0; nu < 2; ++nu)
        species[nu] = group_mean(mass, v[0].data(), type, nu);
    double sum_bary[3] = {0, 0, 0}, sum_flow_x = 0;
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d)
            sum_bary[d] += mass[i] * (v[d][i] - bary[d]) * (v[d][i] - bary[d]);
        const double wx = v[0][i] - species[type[i]];
        sum_flow_x += mass[i] * wx * wx;
    }
    const double to_t = metal.mvv2e / metal.boltz;
    Temperatures t;
    t.kinetic = (sum_bary[0] + sum_bary[1] + sum_bary[2]) * to_t / (3 * n - 3);
    t.flow = (sum_flow_x + sum_bary[1] + sum_bary[2]) * to_t / (3 * n - 4);
    t.transverse = (sum_bary[1] + sum_bary[2]) * to_t / (2 * n - 2);
    t.flow_split = species[nacl_fixture::kNa] - species[nacl_fixture::kCl];
    return t;
}

double constrained(const Temperatures& t, GaussKind kind) {
    if (kind == GaussKind::Kinetic)
        return t.kinetic;
    if (kind == GaussKind::Flow)
        return t.flow;
    return t.transverse;
}

class GearPCGauss : public ::testing::TestWithParam<GaussKind> {};

} // namespace

TEST_P(GearPCGauss, HoldsTheConstrainedTemperatureUnderAStrongField) {
    using namespace nacl_fixture;
    constexpr double kT0 = 1500.0; // [K]
    constexpr double kField = 0.5; // [V/Å] along x
    constexpr double kDt = 0.001;  // [ps]
    constexpr int kSteps = 200;
    // The Gear start (zero higher derivatives) needs ~40 steps to settle;
    // its corrections reach 3e-4 before that.
    constexpr int kStartup = 50;
    const GaussKind kind = GetParam();
    cpu::Frame frame = nacl(1800.0, 5);

    cpu::md::GearPC gear{kDt, cpu::md::GaussThermostat{kind, kT0, 0}};
    cpu::md::MD sim;
    sim.set_units("metal");
    // Rebuild and spatially re-sort every step: the rows are permuted
    // between the predictor and the corrector, and the history must follow.
    sim.set_neighbor(
        cpu::md::NeighborList<>{kCut, 0.5, 1, 0, false}.set_atom_sort(1, 0.4));
    sim.set_evaluator(cpu::PairBorn{tosi_fumi(), kCut});
    sim.set_evaluator(cpu::CoulLong{kAlpha, kCut});
    sim.set_evaluator(cpu::KSpaceEwald{kAlpha, 1e-6});
    sim.on(ATV::md::Phase::post_force)
        .add(cpu::md::FixEfield{kField, 0, 0}, 100);
    sim.set_integrator_ref(gear);
    sim.build();

    double worst_t = 0, worst_drift = 0;
    int permutations = 0;
    std::vector<TagInt> previous(frame.span(kAtom_tag).data(),
                                 frame.span(kAtom_tag).data() + kN);
    for (int step = 0; step < kSteps; ++step) {
        sim.run(frame, 1);
        const Temperatures t = measure(frame);
        worst_t = std::max(worst_t, std::fabs(constrained(t, kind) / kT0 - 1));
        if (step >= kStartup)
            worst_drift = std::max(worst_drift, gear.constraint_drift());

        const auto tags = frame.span(kAtom_tag);
        bool moved = false;
        for (int i = 0; i < kN; ++i) {
            moved = moved || tags[i] != previous[i];
            previous[i] = tags[i];
        }
        permutations += moved ? 1 : 0;
    }
    const Temperatures end = measure(frame);
    // The constrained temperature, measured independently from the frame
    // with its own degrees of freedom, is exact at every step.
    EXPECT_LT(worst_t, 1e-12);
    // What the projection removes is the integrator's truncation error
    // (≤ 2e-7 per step here), not the field's work: an isokinetic α without
    // the field term leaves 1e-2. Flow and transverse α carry no field term
    // (the field does no work on the species-peculiar or transverse motion).
    EXPECT_LT(worst_drift, 1e-6);
    // The rows really were permuted under the integrator.
    EXPECT_GT(permutations, 10);
    // The field separates the species flows (Na⁺ along +x).
    EXPECT_GT(end.flow_split, 0.0);
}

INSTANTIATE_TEST_SUITE_P(Kinds, GearPCGauss,
                         ::testing::Values(GaussKind::Kinetic, GaussKind::Flow,
                                           GaussKind::Transverse),
                         [](const auto& info) {
                             switch (info.param) {
                             case GaussKind::Kinetic:
                                 return std::string("Kinetic");
                             case GaussKind::Flow:
                                 return std::string("Flow");
                             default:
                                 return std::string("Transverse");
                             }
                         });
