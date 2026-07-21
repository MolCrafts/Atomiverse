/// @file test_pw_components.cpp
/// @brief Unit tests for the plane-wave internal step helpers.
///
/// The helpers live in `cpu::pw::internal::` and are owned by
/// `cpu::pw::PWSolver` — tests may include the internal headers directly
/// (same policy as `cpu/scf/internal/`). They must NOT be
/// `cpu::Component` subclasses (static-asserted below).
#include "atomiverse/core/box.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/pw/internal/convergence.h"
#include "atomiverse/cpu/pw/internal/davidson.h"
#include "atomiverse/cpu/pw/internal/density_builder.h"
#include "atomiverse/cpu/pw/internal/g_vector.h"
#include "atomiverse/cpu/pw/internal/poisson.h"
#include "atomiverse/cpu/pw/internal/pw_workspace.h"
#include "atomiverse/cpu/pw/internal/veff_builder.h"
#include "atomiverse/cpu/pw/internal/xc_builder.h"
#include <gtest/gtest.h>
#include <type_traits>

using namespace ATV;
using namespace ATV::cpu::pw::internal;

// ── Static-assert: PW step helpers are not Plan components ──────────

static_assert(!std::is_base_of_v<cpu::Component, PoissonSolver>,
              "internal::PoissonSolver must not inherit cpu::Component");
static_assert(!std::is_base_of_v<cpu::Component, XCBuilder>,
              "internal::XCBuilder must not inherit cpu::Component");
static_assert(!std::is_base_of_v<cpu::Component, VeffBuilder>,
              "internal::VeffBuilder must not inherit cpu::Component");
static_assert(!std::is_base_of_v<cpu::Component, DavidsonSolver>,
              "internal::DavidsonSolver must not inherit cpu::Component");
static_assert(!std::is_base_of_v<cpu::Component, DensityBuilder>,
              "internal::DensityBuilder must not inherit cpu::Component");
static_assert(!std::is_base_of_v<cpu::Component, ConvergenceCheck>,
              "internal::ConvergenceCheck must not inherit cpu::Component");

namespace {

// Helper: create a minimal PW test fixture (Si diamond primitive cell)
struct PWFixture {
    PWWorkspace ws;

    PWFixture() {
        double h = 5.43 / 2.0;
        Mat3 cell;
        cell(0, 0) = 0;
        cell(0, 1) = h;
        cell(0, 2) = h;
        cell(1, 0) = h;
        cell(1, 1) = 0;
        cell(1, 2) = h;
        cell(2, 0) = h;
        cell(2, 1) = h;
        cell(2, 2) = 0;
        ::ATV::Box box;
        box.cell = cell;
        ws.gvec0 = enumerate_g_vectors(box, AccReal(15), Vec3::zeros());
        auto& gv = ws.gvec0;

        ws.n_g = gv.count();
        ws.gvec = &gv;
        ws.fft_nx = gv.nx;
        ws.fft_ny = gv.ny;
        ws.fft_nz = gv.nz;
        ws.nr = gv.nx * gv.ny * gv.nz;
        ws.nc = (gv.nx / 2 + 1) * gv.ny * gv.nz;
        ws.omega = box.volume();
        ws.fft = std::make_unique<FFTPlan3D>(gv.nx, gv.ny, gv.nz);

        auto ng = static_cast<std::size_t>(ws.n_g);
        auto nr_sz = static_cast<std::size_t>(ws.nr);
        ws.rho_G.resize({ng});
        ws.rho_G_old.resize({ng});
        ws.V_H_G.resize({ng});
        ws.V_xc_G.resize({ng});
        ws.V_eff_G.resize({ng});
        ws.V_loc_G.resize({ng});
        ws.rho_r.resize({nr_sz});
        ws.V_xc_r.resize({nr_sz});
        ws.V_eff_r.resize({nr_sz});
        ws.T_G.resize({ng});

        for (int i = 0; i < ws.n_g; ++i) {
            ws.rho_G[i] =
                (i == 0) ? std::complex<AccReal>(AccReal(4) / box.volume(), 0)
                         : std::complex<AccReal>(0, 0);
            ws.V_loc_G[i] = (i == 0) ? AccReal(-0.5) : AccReal(0);
        }
        for (int i = 0; i < ws.nr; ++i)
            ws.V_eff_r[i] = AccReal(0);

        // kb_proj left empty -- exercising local Hamiltonian only; nonlocal
        // KB path is covered by the integration tests in test_pw_si.
        ws.atom_pos_bohr.clear();
    }
};

} // namespace

TEST(PWInternal, Poisson) {
    PWFixture f;
    PoissonSolver p;
    p.step(f.ws);
    EXPECT_NEAR(std::abs(f.ws.V_H_G[0]), 0.0, 1e-12); // G=0 zeroed
}

TEST(PWInternal, VeffAssembly) {
    PWFixture f;
    for (int i = 0; i < f.ws.n_g; ++i) {
        f.ws.V_H_G[i] = std::complex<AccReal>(AccReal(0.1), 0);
        f.ws.V_xc_G[i] = std::complex<AccReal>(AccReal(-0.05), 0);
    }
    VeffBuilder v;
    v.prepare(f.ws);
    v.step(f.ws);
    for (int i = 0; i < f.ws.n_g; ++i) {
        AccReal expected =
            f.ws.V_loc_G[i].real() + AccReal(0.1) + AccReal(-0.05);
        EXPECT_NEAR(f.ws.V_eff_G[i].real(), expected, AccReal(1e-12));
    }
}

TEST(PWInternal, KineticDiagonalNonNegative) {
    PWFixture f;
    // The solver fills T_G from the G-sphere kinetic energies per k-point.
    for (int i = 0; i < f.ws.n_g; ++i)
        f.ws.T_G[i] = f.ws.gvec->ekin[i];
    for (int i = 0; i < f.ws.n_g; ++i)
        EXPECT_GE(static_cast<double>(f.ws.T_G[i]), 0.0);
}

TEST(PWInternal, DavidsonEigenvalues) {
    PWFixture f;
    for (int i = 0; i < f.ws.n_g; ++i)
        f.ws.T_G[i] = f.ws.gvec->ekin[i];
    VeffBuilder v;
    v.prepare(f.ws);
    v.step(f.ws);

    DavidsonSolver d(4, AccReal(1e-8), 20);
    d.prepare(f.ws);
    d.step(f.ws);
    EXPECT_EQ(f.ws.eigenvalues.size(), 4u);
    for (size_t i = 1; i < f.ws.eigenvalues.size(); ++i)
        EXPECT_LE(static_cast<double>(f.ws.eigenvalues[i - 1]),
                  static_cast<double>(f.ws.eigenvalues[i]) + 1e-10);
}

TEST(PWInternal, ConvergenceCheck) {
    PWFixture f;
    ConvergenceCheck c(AccReal(1e-8), 0);

    for (int i = 0; i < f.ws.n_g; ++i) {
        f.ws.rho_G[i] =
            std::complex<AccReal>(AccReal(1e-7) * AccReal(i + 1), 0);
        f.ws.rho_G_old[i] = std::complex<AccReal>(0, 0);
    }
    EXPECT_FALSE(c.step(f.ws));

    for (int i = 0; i < f.ws.n_g; ++i) {
        f.ws.rho_G[i] = std::complex<AccReal>(AccReal(1e-12), 0);
        f.ws.rho_G_old[i] = std::complex<AccReal>(0, 0);
    }
    EXPECT_TRUE(c.step(f.ws));
}

TEST(PWInternal, WarmupSuppression) {
    PWFixture f;
    ConvergenceCheck c(AccReal(1e-20), 3);
    for (int i = 0; i < f.ws.n_g; ++i) {
        f.ws.rho_G[i] = f.ws.rho_G_old[i] = std::complex<AccReal>(0, 0);
    }
    for (int step = 0; step < 3; ++step)
        EXPECT_FALSE(c.step(f.ws)) << "warmup step " << step;
}

TEST(PWInternal, DensityBuilderConservesElectrons) {
    PWFixture f;
    // One doubly-occupied band with the Davidson normalization
    // Σ_G |ψ(G)|² = 1: ψ(G) = δ_{G,0}. Then ψ_phys(r) = 1/√Ω, so
    // ρ(r) = 2/Ω and the electron count ∫ρ d³r = ρ(G=0)·Ω must be 2.
    // mix_beta = 1 turns the first-iteration linear mixing into a pure
    // replacement, exposing the raw built density.
    DensityBuilder db(AccReal(1.0), 5);
    db.prepare(f.ws);
    db.reset(f.ws);

    int nb = 1;
    f.ws.wfc.resize(
        {static_cast<std::size_t>(nb), static_cast<std::size_t>(f.ws.n_g)});
    f.ws.eigenvalues.resize({static_cast<std::size_t>(nb)});
    f.ws.eigenvalues[0] = AccReal(0);
    for (int ig = 0; ig < f.ws.n_g; ++ig)
        f.ws.wfc(0, ig) = (ig == 0) ? std::complex<AccReal>(1, 0)
                                    : std::complex<AccReal>(0, 0);

    db.begin(f.ws);
    db.accumulate_kpoint(f.ws, AccReal(1.0), nb);
    db.finish(f.ws);

    // 2 electrons per band × weight 1.0
    double n_elec = static_cast<double>(f.ws.rho_G[0].real() * f.ws.omega);
    EXPECT_NEAR(n_elec, 2.0, 1e-6);
}
