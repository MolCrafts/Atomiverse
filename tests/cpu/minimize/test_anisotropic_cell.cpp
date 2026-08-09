/// @file tests/cpu/minimize/test_anisotropic_cell.cpp
/// @brief Unit gates for AnisotropicCellRelax (mask, stress convention) and
///        the StressConverged criterion. GPW-driven e2e lives in
///        regressions/gpw/gpw-variable-cell-ice/.
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/anisotropic_cell_relax.h"
#include "atomiverse/cpu/minimize/minimize.h"

using namespace ATV;

namespace {

/// Energy source writing a constant INTENSIVE Voigt stress [eV/Å³] — the
/// PW/GPW `kStress` convention — so cell-relax steps are exactly predictable.
class ConstStress : public cpu::Potential {
  public:
    double sxx = 0.0, syy = 0.0, szz = 0.0;

    ConstStress(double sxx, double syy, double szz)
        : sxx(sxx), syy(syy), szz(szz) {}

    std::string name() const override { return "const_stress"; }

    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::minimizer::Phase::force))
            return;
        ctx.frame.meta.get(kEnergy) = 0.0;
        // (sxx, syy, szz) is the constant *intensive* stress this stand-in
        // represents [eV/Å³]. kStress carries the extensive Voigt virial [eV]
        // (core/frame_schema.h), so it is published as σ·V against the frame's
        // current volume — which keeps the represented stress constant while
        // the relaxer rescales the cell.
        const double vol = ctx.frame.box.volume();
        auto& st = ctx.frame.meta.get(kStress);
        st[0] = sxx * vol;
        st[1] = syy * vol;
        st[2] = szz * vol;
        st[3] = st[4] = st[5] = 0.0;
    }
};

cpu::Frame one_atom_box(double L) {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0.25 * L;
    f.span(kAtom_mass)[0] = 1.0;
    f.span(kAtom_type)[0] = 0;
    f.span(kAtom_tag)[0] = 1;
    f.box = cpu::Box::cubic(L);
    return f;
}

} // namespace

TEST(AnisotropicCellRelax, Name) {
    cpu::minimizer::AnisotropicCellRelax a(0.0, 0.05);
    EXPECT_EQ(a.name(), "anisotropic_cell_relax");
}

TEST(AnisotropicCellRelax, IntensiveStressStepAndMask) {
    // relax_x only, intensive convention: after the seed iteration the next
    // scale step is exactly 1 + alpha0*sxx/bulk_mod; frozen y/z stay
    // bit-identical even though syy/szz are nonzero.
    const double L0 = 10.0;
    auto frame = one_atom_box(L0);

    cpu::minimizer::Minimize opt;
    cpu::minimizer::AnisotropicCellRelax cell(0.0, /*alpha*/ 0.5);
    cell.bulk_mod = 1.0;
    cell.max_scale = 0.05;
    cell.relax_y = false;
    cell.relax_z = false;
    opt.set_optimizer_ref(cell);
    opt.on(minimizer::Phase::force).add(ConstStress{0.02, 0.04, -0.04});
    opt.enable_timing(false);
    opt.build();
    opt.run(frame, 2); // iter 0 seeds the stress; iter 1 applies one step

    const double expected = L0 * (1.0 + 0.5 * 0.02 / 1.0); // 10.1, unclamped
    EXPECT_NEAR(frame.box.length(0), expected, 1e-12 * L0);
    EXPECT_EQ(frame.box.length(1), L0); // masked: bit-identical
    EXPECT_EQ(frame.box.length(2), L0);
    // Residual reads the relaxed axis only (0.02), not the masked 0.04.
    EXPECT_NEAR(cell.pressure_residual(), 0.02, 1e-15);
}

TEST(AnisotropicCellRelax, MaxScaleClampsIntensiveStep) {
    const double L0 = 10.0;
    auto frame = one_atom_box(L0);

    cpu::minimizer::Minimize opt;
    cpu::minimizer::AnisotropicCellRelax cell(0.0, /*alpha*/ 1.0);
    cell.bulk_mod = 1.0;
    cell.max_scale = 0.03;
    cell.relax_y = false;
    cell.relax_z = false;
    opt.set_optimizer_ref(cell);
    opt.on(minimizer::Phase::force).add(ConstStress{-5.0, 0.0, 0.0});
    opt.enable_timing(false);
    opt.build();
    opt.run(frame, 2);

    EXPECT_NEAR(frame.box.length(0), L0 * (1.0 - 0.03), 1e-12 * L0);
}

TEST(AnisotropicCellRelax, AllAxesMaskedThrowsAtComponentBuild) {
    // Component build() runs at the driver's setup pass (frame binds at
    // run()), so the all-masked configuration error surfaces there.
    auto frame = one_atom_box(10.0);
    cpu::minimizer::Minimize opt;
    cpu::minimizer::AnisotropicCellRelax cell(0.0, 0.05);
    cell.relax_x = cell.relax_y = cell.relax_z = false;
    opt.set_optimizer_ref(cell);
    opt.on(minimizer::Phase::force).add(ConstStress{0.0, 0.0, 0.0});
    opt.enable_timing(false);
    opt.build();
    EXPECT_THROW(opt.run(frame, 1), std::invalid_argument);
}

TEST(StressConverged, Name) {
    cpu::StressConverged sc(0.0, 1e-4);
    EXPECT_EQ(sc.name(), "stress_converged");
}

TEST(StressConverged, VerdictMaskedPerAxis) {
    cpu::Frame frame(1);
    frame.box = cpu::Box::cubic(10.0);
    const double V = frame.box.volume(); // 1000 Å³
    // kStress is the extensive Voigt virial [eV]; the criterion judges the
    // intensive W_dd/V [eV/Å³] against its tolerance, so the stresses under
    // test are published as σ·V.
    auto& st = frame.meta.get(kStress);
    st[0] = 5e-5 * V; // σ_xx under tol
    st[1] = 0.5 * V;  // way out — but masked below
    st[2] = -0.5 * V; // way out — masked
    cpu::Context rt;
    ConvergenceMetrics metrics;
    cpu::LoopState s{/*iteration=*/0, frame, rt, metrics};

    cpu::StressConverged x_only(0.0, 1e-4, true, false, false);
    EXPECT_EQ(x_only.verdict(s), Verdict::Converged);

    cpu::StressConverged all(0.0, 1e-4);
    EXPECT_EQ(all.verdict(s), Verdict::Continue);

    st[0] = 2e-4 * V; // σ_xx now out of tol
    EXPECT_EQ(x_only.verdict(s), Verdict::Continue);
}

TEST(StressConverged, InvalidParametersThrow) {
    EXPECT_THROW(cpu::StressConverged(0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(cpu::StressConverged(0.0, 1e-4, false, false, false),
                 std::invalid_argument);
}
