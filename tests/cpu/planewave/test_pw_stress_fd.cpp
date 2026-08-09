/// @file tests/cpu/planewave/test_pw_stress_fd.cpp
/// @brief Analytic-vs-strain-FD cell stress gate for plane-wave DFT.
///
/// Split out of `test_pw_stress.cpp` for the same reason as
/// `test_pw_forces_fd.cpp`: a strain-FD oracle costs 12 SCF solves (plus the
/// analytic path's own internal NSCF strain sweep), which does not fit
/// alongside the cheap gates inside `ATV_TEST_TIMEOUT_SECONDS`. Still
/// heavy-gated — the target name matches `pw_stress` in
/// `ATV_TEST_HEAVY_REGEX`.

#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

namespace {

cpu::Frame make_h_cubic(double L) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_mass)[0] = 1.008;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.box = cpu::Box::cubic(L);
    return f;
}

PW cheap_cfg() {
    PW cfg;
    cfg.gth_name = "GTH-PBE-q1";
    cfg.ecut_ry = 5.0;
    cfg.k_mesh = 1;
    cfg.tol = 1e-4;
    cfg.max_iter = 8;
    cfg.mix_beta = 0.5;
    cfg.xc = cpu::scf::XCType::LSDA; // LDA XC stress term is complete
    return cfg;
}

} // namespace

TEST(PWStressFd, AnalyticMatchesFdCheapH) {
    // Analytic (Nielsen–Martin) vs strain-FD of the SCF energy: a
    // potential-independent self-consistency gate, the cell-derivative twin
    // of the analytic-vs-FD force test. Measured max|Δ| ≈ 4e-5 eV/Å³ against
    // a stress of ~1e-2 eV/Å³; the old 0.15 eV/Å³ bound was 15× the stress
    // being measured and asserted essentially nothing.
    //
    // Unlike the force gate this one is strongly cutoff-sensitive: at fixed
    // ecut the plane-wave count changes with cell volume, so the strain
    // derivative carries a Pulay (basis-incompleteness) term the analytic
    // expression does not model. It falls off with cutoff — dropping to
    // 4 Ry raises the residual to ~3e-2 — so this bound belongs to *these*
    // settings and is not a statement about the pseudopotential.
    auto f = make_h_cubic(6.0);
    auto cfg = cheap_cfg();
    const auto san = pw_stress(f, cfg);
    const auto sfd = pw_stress_fd(f, cfg, 2e-3);
    double max_abs = 0.0;
    for (int v = 0; v < 6; ++v) {
        const double d = std::abs(san[static_cast<std::size_t>(v)]
                                  - sfd[static_cast<std::size_t>(v)]);
        max_abs = std::max(max_abs, d);
        EXPECT_TRUE(std::isfinite(san[static_cast<std::size_t>(v)]));
        EXPECT_TRUE(std::isfinite(sfd[static_cast<std::size_t>(v)]));
    }
    EXPECT_LE(max_abs, 1e-3)
        << "an=" << san[0] << "," << san[1] << "," << san[2] << " fd=" << sfd[0]
        << "," << sfd[1] << "," << sfd[2] << " max|Δ|=" << max_abs;
}
