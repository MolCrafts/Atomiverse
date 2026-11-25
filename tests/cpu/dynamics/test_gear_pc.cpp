/// @file tests/cpu/dynamics/test_gear_pc.cpp
/// @brief Fifth-order Gear predictor–corrector: the core law converges at
///        sixth order from an exact start, and `cpu::md::GearPC` integrates
///        ṙ = v, v̇ = F/m with exactly that law (units, masses, start-up).
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/gear_pc.h"
#include "atomiverse/core/unit_system.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/dynamics/gear_pc.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

/// x'' = −ω² x as the first-order system (x, v), integrated with the core
/// law from the exact Nordsieck vectors of x(t) = cos t (ω = 1). Returns the
/// position error at t_end.
double oscillator_error(double h, double t_end) {
    const int steps = static_cast<int>(std::lround(t_end / h));
    // d^k cos t / dt^k at 0: 1, 0, −1, 0, 1, 0, −1.
    const double deriv[7] = {1, 0, -1, 0, 1, 0, -1};
    double zx[6], zv[6];
    double scale = 1.0;
    for (int k = 0; k < 6; ++k) {
        zx[k] = scale * deriv[k];
        zv[k] = scale * deriv[k + 1];
        scale *= h / (k + 1);
    }
    for (int s = 0; s < steps; ++s) {
        md::gear_predict(zx);
        md::gear_predict(zv);
        const double rate_x = zv[0];
        const double rate_v = -zx[0];
        md::gear_correct(zx, h * rate_x);
        md::gear_correct(zv, h * rate_v);
    }
    return std::fabs(zx[0] - std::cos(steps * h));
}

constexpr double kSpring = 1.3;  // [eV/Å²]
constexpr double kCentre = 50.0; // tether point, mid-box [Å]

/// Harmonic tether F = −k (r − c) on every owned atom: a force term whose
/// trajectory the test can integrate independently.
class Tether : public cpu::Potential {
  public:
    std::string name() const override { return "tether"; }
    void launch(const cpu::LaunchContext& ctx) override {
        auto& f = ctx.frame;
        for (int i = 0; i < f.nlocal(); ++i) {
            f.span(kAtom_fx)[i] -= kSpring * (f.span(kAtom_x)[i] - kCentre);
            f.span(kAtom_fy)[i] -= kSpring * (f.span(kAtom_y)[i] - kCentre);
            f.span(kAtom_fz)[i] -= kSpring * (f.span(kAtom_z)[i] - kCentre);
        }
    }
};

cpu::Frame tethered_atoms() {
    cpu::Frame f(3);
    const double mass[3] = {22.9898, 35.453, 6.94};
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_x)[i] = kCentre + 0.1 * (i + 1);
        f.span(kAtom_y)[i] = kCentre - 0.05 * (i + 2);
        f.span(kAtom_z)[i] = kCentre + 0.07 * (2 - i);
        f.span(kAtom_vx)[i] = 1.5 - i;
        f.span(kAtom_vy)[i] = 0.4 * (i + 1);
        f.span(kAtom_vz)[i] = -0.8 + 0.3 * i;
        f.span(kAtom_mass)[i] = mass[i];
        f.span(kAtom_type)[i] = i;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::cubic(100.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    return f;
}

/// The documented start and step of `GearPC`, written against the tether
/// force law directly: one displacement (x, v) from the tether point with
/// v̇ = −ω² x.
std::array<double, 2> reference(double x0, double v0, double w2, double h,
                                int steps) {
    double zx[6] = {x0, h * v0, 0, 0, 0, 0};
    double zv[6] = {v0, -h * w2 * x0, 0, 0, 0, 0};
    zx[2] = 0.5 * h * zv[1];
    for (int s = 0; s < steps; ++s) {
        md::gear_predict(zx);
        md::gear_predict(zv);
        const double rate_x = zv[0];
        const double rate_v = -w2 * zx[0];
        md::gear_correct(zx, h * rate_x);
        md::gear_correct(zv, h * rate_v);
    }
    return {zx[0], zv[0]};
}

} // namespace

TEST(GearCoreLaw, GlobalErrorFallsAsTheSixthPowerOfTheStep) {
    const double coarse = oscillator_error(0.05, 10.0);
    const double fine = oscillator_error(0.025, 10.0);
    // Sixth order: halving h divides the error by 2⁶ = 64 (56 measured at
    // these steps, error 3.8e-10). A 2 % error in c0 or c2 lowers the
    // observed order and more than doubles the error.
    EXPECT_GT(std::log2(coarse / fine), 5.5)
        << "coarse " << coarse << " fine " << fine;
    EXPECT_LT(fine, 5e-10);
}

TEST(GearCoreLaw, CorrectorIsTheNordsieckFormOfFifthOrderAdamsMoulton) {
    // Gear's first-order set is Adams–Moulton 5 in Nordsieck form, whose
    // step matrix (I − c e₁ᵀ)P has eigenvalues 1, 0, 0, 0, 0, 0 with the
    // zeros nilpotent: for ẏ = 0 any start-up history is gone after five
    // steps, exactly. Any wrong c1..c5 leaves a non-zero eigenvalue.
    double z[6] = {1.0, 0.3, -0.2, 0.5, 0.1, -0.4};
    for (int step = 0; step < 5; ++step) {
        md::gear_predict(z);
        md::gear_correct(z, 0.0);
    }
    for (int k = 1; k < 6; ++k)
        EXPECT_NEAR(z[k], 0.0, 1e-14) << "z" << k;
}

TEST(GearPC, IntegratesTheEquationsOfMotionWithTheCoreLaw) {
    const units::UnitSystem metal = units::from_style(units::Style::metal);
    constexpr double kDt = 0.002; // [ps]
    constexpr int kSteps = 150;
    cpu::Frame frame = tethered_atoms();
    const cpu::Frame start = tethered_atoms();

    cpu::md::MD sim;
    sim.set_units("metal");
    sim.set_evaluator(Tether{});
    sim.set_integrator(cpu::md::GearPC{kDt});
    sim.build();
    sim.run(frame, kSteps);

    const StorageReal* cols0[6] = {
        start.span(kAtom_x).data(),  start.span(kAtom_y).data(),
        start.span(kAtom_z).data(),  start.span(kAtom_vx).data(),
        start.span(kAtom_vy).data(), start.span(kAtom_vz).data()};
    const StorageReal* cols[6] = {
        frame.span(kAtom_x).data(),  frame.span(kAtom_y).data(),
        frame.span(kAtom_z).data(),  frame.span(kAtom_vx).data(),
        frame.span(kAtom_vy).data(), frame.span(kAtom_vz).data()};
    for (int i = 0; i < 3; ++i) {
        const double w2 = kSpring * metal.velocity_impulse.scale()
                          / static_cast<double>(start.span(kAtom_mass)[i]);
        const double w = std::sqrt(w2);
        const double t = kSteps * kDt;
        for (int d = 0; d < 3; ++d) {
            const double x0 = cols0[d][i] - kCentre, v0 = cols0[3 + d][i];
            const auto want = reference(x0, v0, w2, kDt, kSteps);
            const double x = cols[d][i] - kCentre;
            EXPECT_NEAR(x, want[0], 1e-11) << "atom " << i << " axis " << d;
            EXPECT_NEAR(cols[3 + d][i], want[1], 1e-10)
                << "atom " << i << " axis " << d;
            // And the law tracks the oscillation itself (the O(h²) start-up
            // offset of the documented start is ~1e-5 Å here).
            const double exact =
                x0 * std::cos(w * t) + v0 / w * std::sin(w * t);
            EXPECT_NEAR(x, exact, 1e-4) << "atom " << i << " axis " << d;
        }
    }
}
