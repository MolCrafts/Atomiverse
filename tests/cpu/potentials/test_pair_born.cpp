/// @file tests/cpu/potentials/test_pair_born.cpp
/// @brief Unit checks for cpu::PairBorn energy/force vs analytic shifted form.
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/pair_born.h"

using namespace ATV;

namespace {

cpu::Frame dimer(double r, int t0 = 0, int t1 = 0) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.0;
    f.span(kAtom_type)[0] = t0;
    f.span(kAtom_type)[1] = t1;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(50.0);
    return f;
}

double U_un(double A, double rho, double sigma, double C, double D, double r) {
    return A * std::exp((sigma - r) / rho) - C / std::pow(r, 6)
           + D / std::pow(r, 8);
}

double dUdr(double A, double rho, double sigma, double C, double D, double r) {
    return -A / rho * std::exp((sigma - r) / rho) + 6.0 * C / std::pow(r, 7)
           - 8.0 * D / std::pow(r, 9);
}

} // namespace

TEST(PairBorn, SpatialHalfRowsMayPointToALowerLocalIndex) {
    auto frame = dimer(2.5);
    cpu::md::PairList pairs;
    pairs.inum = 1;
    pairs.ilist = {1};
    pairs.numneigh = {0, 1};
    pairs.firstneigh = {0, 0};
    pairs.neighbors = {0};
    pairs.total_pairs = 1;
    cpu::Context context;
    context.md.pair_list = &pairs;
    cpu::PairBorn born(1000.0, 3.0, 1.0, 50.0, 20.0, 8.0);
    born.build({context, frame});
    born.launch({context, frame, 0, 0});
    EXPECT_NEAR(frame.meta.get(kEnergy),
                U_un(1000.0, 3.0, 1.0, 50.0, 20.0, 2.5)
                    - U_un(1000.0, 3.0, 1.0, 50.0, 20.0, 8.0),
                1e-12);
    EXPECT_NEAR(frame.span(kAtom_fx)[0],
                dUdr(1000.0, 3.0, 1.0, 50.0, 20.0, 2.5), 1e-12);
    born.destroy({context, frame});
}

// Offline LAMMPS born dimer foothold coeffs (born-repulsion-foothold).
TEST(PairBorn, AnalyticEnergyAndForceLammpsDimer) {
    const double A = 1000.0, rho = 3.0, sigma = 1.0, C = 50.0, D = 20.0;
    const double rc = 8.0, r = 2.5;
    const double U =
        U_un(A, rho, sigma, C, D, r) - U_un(A, rho, sigma, C, D, rc);
    const double du = dUdr(A, rho, sigma, C, D, r);

    auto f = dimer(r);
    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;

    cpu::Context ctx;
    cpu::PairBorn born(A, rho, sigma, C, D, rc);
    born.build({ctx, f});
    born.launch({ctx, f, 0, 0});

    EXPECT_NEAR(f.meta.get(kEnergy), U, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[0], du, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[1], -du, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fy)[0], 0.0, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fz)[0], 0.0, 1e-15);
    born.destroy({ctx, f});
}

TEST(PairBorn, AnalyticEnergyAndForce) {
    const double A = 20.3548, rho = 0.317, sigma = 2.755;
    const double C = 48.1771, D = 48.1771, rc = 10.0, r = 2.8;
    const double U =
        U_un(A, rho, sigma, C, D, r) - U_un(A, rho, sigma, C, D, rc);
    const double du = dUdr(A, rho, sigma, C, D, r);

    auto f = dimer(r);
    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;

    cpu::Context ctx;
    cpu::PairBorn born(A, rho, sigma, C, D, rc);
    born.build({ctx, f});
    born.launch({ctx, f, 0, 0});

    EXPECT_NEAR(f.meta.get(kEnergy), U, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[0], du, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[1], -du, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fy)[0], 0.0, 1e-15);
    EXPECT_NEAR(f.span(kAtom_fz)[0], 0.0, 1e-15);
    born.destroy({ctx, f});
}

TEST(PairBorn, ZeroAtCutoff) {
    const double A = 1000.0, rho = 3.0, sigma = 1.0, C = 50.0, D = 20.0;
    const double rc = 8.0;
    const double r = rc - 1e-6;
    const double U =
        U_un(A, rho, sigma, C, D, r) - U_un(A, rho, sigma, C, D, rc);

    auto f = dimer(r);
    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;

    cpu::Context ctx;
    cpu::PairBorn born(A, rho, sigma, C, D, rc);
    born.build({ctx, f});
    born.launch({ctx, f, 0, 0});
    EXPECT_NEAR(f.meta.get(kEnergy), U, 1e-12);
    EXPECT_LT(std::fabs(f.meta.get(kEnergy)), 1e-3);
    born.destroy({ctx, f});
}

TEST(PairBorn, MultiTypeSelectsPairCoeffs) {
    // Na-Cl coeffs only — Na-Na must not interact when types differ.
    const double A = 20.3548, rho = 0.317, sigma = 2.755;
    const double C = 48.1771, D = 48.1771, rc = 10.0, r = 2.8;
    const double U =
        U_un(A, rho, sigma, C, D, r) - U_un(A, rho, sigma, C, D, rc);
    const double du = dUdr(A, rho, sigma, C, D, r);

    cpu::PairBorn born(
        {{0, 1, A, rho, sigma, C, D},
         {0, 0, 999.0, 0.3, 1.0, 0.0, 0.0}, // unused for unlike pair
         {1, 1, 999.0, 0.3, 1.0, 0.0, 0.0}},
        rc);
    EXPECT_TRUE(born.is_multi_type());

    auto f = dimer(r, /*t0=*/0, /*t1=*/1);
    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;

    cpu::Context ctx;
    born.build({ctx, f});
    born.launch({ctx, f, 0, 0});

    EXPECT_NEAR(f.meta.get(kEnergy), U, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[0], du, 1e-12);
    EXPECT_NEAR(f.span(kAtom_fx)[1], -du, 1e-12);
    born.destroy({ctx, f});
}

TEST(PairBorn, RejectsInvalidParams) {
    EXPECT_THROW(cpu::PairBorn(1.0, 0.0, 1.0, 1.0, 0.0, 5.0),
                 std::invalid_argument);
    EXPECT_THROW(cpu::PairBorn(1.0, 0.3, 1.0, 1.0, 0.0, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(cpu::PairBorn(-1.0, 0.3, 1.0, 1.0, 0.0, 5.0),
                 std::invalid_argument);
    EXPECT_THROW(cpu::PairBorn(1.0, 0.3, 1.0, -1.0, 0.0, 5.0),
                 std::invalid_argument);
    EXPECT_THROW(cpu::PairBorn(std::vector<cpu::BornTypeCoeff>{}, 5.0),
                 std::invalid_argument);
}

TEST(PairBorn, NameIsBorn) {
    cpu::PairBorn born(1.0, 0.3, 1.0, 1.0, 0.0, 5.0);
    EXPECT_EQ(born.name(), "born");
    EXPECT_EQ(born.as_neighbor_consumer(), nullptr);
}

namespace {

/// One Born evaluation of `f` (forces, energy and virial reset first).
void evaluate_born(cpu::PairBorn& born, cpu::Frame& f) {
    f.meta.get(kEnergy) = 0;
    for (int d = 0; d < 6; ++d)
        f.meta.get(kStress)[d] = 0;
    for (int i = 0; i < f.nlocal(); ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    cpu::Context ctx;
    born.build({ctx, f});
    born.launch({ctx, f, 0, 0});
    born.destroy({ctx, f});
}

/// ∫_rc^∞ g(r) dr by composite Simpson in u = rc / r ∈ (0, 1]; the
/// integrands here decay like r⁻⁴, so g(rc/u)·rc/u² → 0 as u → 0.
template <class G>
double integrate_to_infinity(double rc, G g) {
    const int n = 20000; // even
    const double h = 1.0 / n;
    auto w = [&](double u) {
        if (u <= 0.0)
            return 0.0;
        const double r = rc / u;
        return g(r) * rc / (u * u);
    };
    double sum = w(0.0) + w(1.0);
    for (int k = 1; k < n; ++k)
        sum += (k % 2 ? 4.0 : 2.0) * w(k * h);
    return sum * h / 3.0;
}

/// Tosi–Fumi NaCl (`regressions/md/npt/npt_pme_nacl.cpp`): D enters
/// negated because PairBorn's dispersion term is +D/r⁸.
std::vector<cpu::BornTypeCoeff> tosi_fumi() {
    constexpr double rho = 0.317;
    return {
        {0, 0, 0.2637038, rho, 2.340, 0.6553585, -0.3114513},
        {0, 1, 0.2109630, rho, 2.755, 4.3690564, -5.4176299},
        {1, 1, 0.1582223, rho, 3.170, 45.2509408, -90.8139570},
    };
}

/// Ions in a cubic cell; the tail depends on the type counts and the volume
/// only, not on where they sit.
cpu::Frame ions(int n_na, int n_cl, double box) {
    cpu::Frame f(n_na + n_cl);
    for (int i = 0; i < n_na + n_cl; ++i) {
        f.span(kAtom_x)[i] = 0.5 + 2.9 * i;
        f.span(kAtom_y)[i] = 0.3 * i;
        f.span(kAtom_z)[i] = 0.1;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = i < n_na ? 0 : 1;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(box);
    return f;
}

} // namespace

TEST(PairBorn, WithoutEnergyShiftTheEnergyIsTheBarePotential) {
    const double A = 1000.0, rho = 3.0, sigma = 1.0, C = 50.0, D = 20.0;
    const double rc = 8.0, r = 2.5;
    auto shifted = dimer(r);
    auto bare = dimer(r);
    cpu::PairBorn on(A, rho, sigma, C, D, rc);
    cpu::PairBorn off =
        cpu::PairBorn(A, rho, sigma, C, D, rc).set_energy_shift(false);
    evaluate_born(on, shifted);
    evaluate_born(off, bare);
    EXPECT_NEAR(bare.meta.get(kEnergy), U_un(A, rho, sigma, C, D, r), 1e-12);
    EXPECT_NEAR(shifted.meta.get(kEnergy) - bare.meta.get(kEnergy),
                -U_un(A, rho, sigma, C, D, rc), 1e-12);
    // The shift is a constant: identical forces.
    EXPECT_EQ(bare.span(kAtom_fx)[0], shifted.span(kAtom_fx)[0]);

    // Same switch on the per-type table.
    const auto table = tosi_fumi();
    auto f_on = dimer(2.9, 0, 1);
    auto f_off = dimer(2.9, 0, 1);
    cpu::PairBorn t_on(table, 9.0);
    cpu::PairBorn t_off = cpu::PairBorn(table, 9.0).set_energy_shift(false);
    evaluate_born(t_on, f_on);
    evaluate_born(t_off, f_off);
    const auto& e = table[1];
    EXPECT_NEAR(f_off.meta.get(kEnergy),
                U_un(e.A, e.rho, e.sigma, e.C, e.D, 2.9), 1e-13);
    EXPECT_NEAR(f_on.meta.get(kEnergy) - f_off.meta.get(kEnergy),
                -U_un(e.A, e.rho, e.sigma, e.C, e.D, 9.0), 1e-13);
}

// Oracle: the homogeneous-fluid tail as a numerical quadrature of the
// potential itself, E = Σ_ij 2π N_i N_j/V ∫ r²U and W_αα = Σ_ij −(2π/3)
// N_i N_j/V ∫ r³U' over [r_c, ∞), ordered type pairs.
TEST(PairBorn, TailCorrectionEqualsTheIntegralOfThePotential) {
    const auto table = tosi_fumi();
    const double rc = 5.0, box = 31.0;
    const int n_na = 4, n_cl = 6;
    const double counts[2] = {n_na, n_cl};

    double e_want = 0.0, w_want = 0.0;
    for (int ti = 0; ti < 2; ++ti)
        for (int tj = 0; tj < 2; ++tj) {
            const auto& c = table[ti + tj]; // rows 00, 01, 11
            const auto u = [&](double r) {
                return U_un(c.A, c.rho, c.sigma, c.C, c.D, r);
            };
            const auto du = [&](double r) {
                return dUdr(c.A, c.rho, c.sigma, c.C, c.D, r);
            };
            const double pairs = counts[ti] * counts[tj];
            e_want +=
                2.0 * M_PI * pairs * integrate_to_infinity(rc, [&](double r) {
                    return r * r * u(r);
                });
            w_want += -(2.0 / 3.0) * M_PI * pairs
                      * integrate_to_infinity(
                          rc, [&](double r) { return r * r * r * du(r); });
        }
    const double vol = box * box * box;

    for (const bool shift : {true, false}) {
        auto f_on = ions(n_na, n_cl, box);
        auto f_off = ions(n_na, n_cl, box);
        cpu::PairBorn with = cpu::PairBorn(table, rc)
                                 .set_energy_shift(shift)
                                 .set_tail_correction(true);
        cpu::PairBorn without =
            cpu::PairBorn(table, rc).set_energy_shift(shift);
        evaluate_born(with, f_on);
        evaluate_born(without, f_off);
        EXPECT_NEAR(f_on.meta.get(kEnergy) - f_off.meta.get(kEnergy),
                    e_want / vol, 1e-10 * std::abs(e_want / vol));
        for (int d = 0; d < 3; ++d)
            EXPECT_NEAR(f_on.meta.get(kStress)[d] - f_off.meta.get(kStress)[d],
                        w_want / vol, 1e-10 * std::abs(w_want / vol));
        for (int d = 3; d < 6; ++d)
            EXPECT_EQ(f_on.meta.get(kStress)[d], f_off.meta.get(kStress)[d]);
    }

    // Single-type form: every atom is the one Born type.
    const auto& c = table[2];
    const double n = n_na + n_cl;
    const double e1 =
        2.0 * M_PI * n * n * integrate_to_infinity(rc, [&](double r) {
            return r * r * U_un(c.A, c.rho, c.sigma, c.C, c.D, r);
        });
    auto f_on = ions(n_na, n_cl, box);
    auto f_off = ions(n_na, n_cl, box);
    cpu::PairBorn with = cpu::PairBorn(c.A, c.rho, c.sigma, c.C, c.D, rc)
                             .set_tail_correction(true);
    cpu::PairBorn without(c.A, c.rho, c.sigma, c.C, c.D, rc);
    evaluate_born(with, f_on);
    evaluate_born(without, f_off);
    EXPECT_NEAR(f_on.meta.get(kEnergy) - f_off.meta.get(kEnergy), e1 / vol,
                1e-10 * std::abs(e1 / vol));
}
