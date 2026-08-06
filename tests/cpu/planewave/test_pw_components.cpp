/// @file test_pw_components.cpp
#include <algorithm>
/// @brief Unit tests for the plane-wave internal step helpers.
///
/// The helpers live in `cpu::pw::internal::` and are owned by
/// `cpu::pw::PW` — tests may include the internal headers directly
/// (same policy as `cpu/scf/internal/`). They must NOT be
/// `cpu::Component` subclasses (static-asserted below).
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/internal/convergence.h"
#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/density_builder.h"
#include "atomiverse/cpu/planewave/internal/g_vector.h"
#include "atomiverse/cpu/planewave/internal/poisson.h"
#include "atomiverse/cpu/planewave/internal/pw_workspace.h"
#include "atomiverse/cpu/planewave/internal/veff_builder.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <gtest/gtest.h>
#include <type_traits>
#include <vector>

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
        cpu::Box box(cell);
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

// A3: uniform density → GGA PBE E_xc equals PBE@σ=0 (LDA limit of PBE).
TEST(PWInternal, XcGgaUniformMatchesSigmaZero) {
    PWFixture f;
    // Uniform ρ: only G=0 component. |∇ρ|=0 ⇒ true GGA collapses to σ=0.
    const AccReal rho0 = AccReal(4) / f.ws.omega;
    for (int i = 0; i < f.ws.n_g; ++i)
        f.ws.rho_G[i] = (i == 0) ? std::complex<AccReal>(rho0, 0)
                                 : std::complex<AccReal>(0, 0);

    XCBuilder xc;
    xc.prepare(f.ws);
    xc.step(f.ws, cpu::scf::XCType::PBE);
    const AccReal e_gga = f.ws.E_xc;

    // Pointwise PBE@σ=0 on the same uniform density (analytic check).
    const auto out = cpu::scf::evaluate_xc(cpu::scf::XCType::PBE,
                                           static_cast<double>(rho0) / 2.0,
                                           static_cast<double>(rho0) / 2.0);
    const AccReal e_ref =
        rho0 * static_cast<AccReal>(out.exc) * f.ws.omega; // ∫ ρ ε dV
    EXPECT_NEAR(static_cast<double>(e_gga), static_cast<double>(e_ref), 1e-10)
        << "uniform GGA must match PBE@σ=0";
}

// A2: nonuniform density → true GGA E_xc differs from PBE@σ=0 on the same ρ.
TEST(PWInternal, XcGgaNonuniformDiffersFromSigmaZero) {
    PWFixture f;
    // ρ(G=0) + small G≠0 modulation so |∇ρ| ≠ 0 on the grid.
    const AccReal rho0 = AccReal(4) / f.ws.omega;
    for (int i = 0; i < f.ws.n_g; ++i)
        f.ws.rho_G[i] = std::complex<AccReal>(0, 0);
    f.ws.rho_G[0] = std::complex<AccReal>(rho0, 0);
    if (f.ws.n_g > 1) {
        // First nonzero G mode (sorted by |G|).
        f.ws.rho_G[1] = std::complex<AccReal>(rho0 * AccReal(0.15), 0);
    }

    XCBuilder xc;
    xc.prepare(f.ws);

    // True GGA path.
    xc.step(f.ws, cpu::scf::XCType::PBE);
    const AccReal e_gga = f.ws.E_xc;

    // PBE@σ=0 on the same ρ(r) reconstructed from ρ(G).
    const int nr = f.ws.nr;
    const int ng = f.ws.n_g;
    const int nx = f.ws.fft_nx, ny = f.ws.fft_ny, nz = f.ws.fft_nz;
    const int nxc = nx / 2 + 1;
    const int nc = f.ws.nc;
    // fftw_complex is double[2] — not a valid std::vector element on libc++.
    std::vector<double> half_reim(static_cast<std::size_t>(2 * nc), 0.0);
    auto* half = reinterpret_cast<fftw_complex*>(half_reim.data());
    std::vector<AccReal> rho_r(static_cast<std::size_t>(nr));
    auto& gv = *f.ws.gvec;
    for (int ig = 0; ig < ng; ++ig) {
        int h = gv.h[ig], k = gv.k[ig], l = gv.l[ig];
        int kx = ((h % nx) + nx) % nx;
        int ky = ((k % ny) + ny) % ny;
        int kz = ((l % nz) + nz) % nz;
        bool conj = false;
        if (kx > nx / 2) {
            kx = nx - kx;
            ky = (ny - ky) % ny;
            kz = (nz - kz) % nz;
            conj = true;
        }
        const int idx = kx + nxc * (ky + ny * kz);
        if (idx >= 0 && idx < nc) {
            half[idx][0] = static_cast<double>(f.ws.rho_G[ig].real());
            half[idx][1] = static_cast<double>(conj ? -f.ws.rho_G[ig].imag()
                                                    : f.ws.rho_G[ig].imag());
        }
    }
    f.ws.fft->c2r(half, rho_r.data());
    AccReal e_sigma0 = 0;
    const AccReal dV = f.ws.omega / static_cast<AccReal>(nr);
    for (int i = 0; i < nr; ++i) {
        const AccReal rho = rho_r[i];
        if (rho < AccReal(1e-12))
            continue;
        const auto out = cpu::scf::evaluate_xc(
            cpu::scf::XCType::PBE, static_cast<double>(rho) / 2.0,
            static_cast<double>(rho) / 2.0); // σ defaults to 0
        e_sigma0 += rho * static_cast<AccReal>(out.exc) * dV;
    }

    const double de = std::abs(static_cast<double>(e_gga - e_sigma0));
    EXPECT_GT(de, 1e-8) << "nonuniform GGA must differ from PBE@σ=0; |ΔE_xc|="
                        << de << " E_gga=" << e_gga << " E_σ0=" << e_sigma0;
}

// Collinear spin-GGA: closed-shell collapse ρ↑=ρ↓=ρ/2 must match restricted
// true GGA on total ρ (same as GPW SpinCollapseMatchesClosedShell).
TEST(PWInternal, XcSpinGgaClosedShellCollapse) {
    PWFixture f;
    const AccReal rho0 = AccReal(4) / f.ws.omega;
    for (int i = 0; i < f.ws.n_g; ++i)
        f.ws.rho_G[i] = std::complex<AccReal>(0, 0);
    f.ws.rho_G[0] = std::complex<AccReal>(rho0, 0);
    if (f.ws.n_g > 1)
        f.ws.rho_G[1] = std::complex<AccReal>(rho0 * AccReal(0.12), 0);

    XCBuilder xc;
    f.ws.nspin = 1;
    xc.prepare(f.ws);
    xc.step(f.ws, cpu::scf::XCType::PBE);
    const AccReal e_rks = f.ws.E_xc;
    std::vector<std::complex<AccReal>> v_rks(
        static_cast<std::size_t>(f.ws.n_g));
    for (int ig = 0; ig < f.ws.n_g; ++ig)
        v_rks[static_cast<std::size_t>(ig)] = f.ws.V_xc_G[ig];

    // Spin channels: half the total density each.
    f.ws.nspin = 2;
    f.ws.rho_spin_G.resize(2);
    f.ws.rho_spin_G[0].resize({static_cast<std::size_t>(f.ws.n_g)});
    f.ws.rho_spin_G[1].resize({static_cast<std::size_t>(f.ws.n_g)});
    f.ws.V_xc_spin_G.resize(2);
    f.ws.V_xc_spin_G[0].resize({static_cast<std::size_t>(f.ws.n_g)});
    f.ws.V_xc_spin_G[1].resize({static_cast<std::size_t>(f.ws.n_g)});
    for (int ig = 0; ig < f.ws.n_g; ++ig) {
        f.ws.rho_spin_G[0][ig] = AccReal(0.5) * f.ws.rho_G[ig];
        f.ws.rho_spin_G[1][ig] = AccReal(0.5) * f.ws.rho_G[ig];
    }
    xc.prepare(f.ws); // size spin-GGA scratch
    xc.step_spin(f.ws, cpu::scf::XCType::PBE);

    EXPECT_NEAR(static_cast<double>(f.ws.E_xc), static_cast<double>(e_rks),
                1e-9 * std::max(1.0, std::abs(static_cast<double>(e_rks))))
        << "spin-GGA closed shell E_xc must match restricted GGA";
    for (int ig = 0; ig < f.ws.n_g; ++ig) {
        EXPECT_NEAR(f.ws.V_xc_spin_G[0][ig].real(),
                    v_rks[static_cast<std::size_t>(ig)].real(), 1e-8)
            << "ig=" << ig << " V↑ vs V_rks";
        EXPECT_NEAR(f.ws.V_xc_spin_G[0][ig].real(),
                    f.ws.V_xc_spin_G[1][ig].real(), 1e-12)
            << "ig=" << ig << " V↑ vs V↓";
    }
}

// Collinear spin-GGA with polarized nonuniform ρ must differ from σ=0 LSDA-limit.
TEST(PWInternal, XcSpinGgaPolarizedDiffersFromSigmaZero) {
    PWFixture f;
    const AccReal rho0 = AccReal(4) / f.ws.omega;
    f.ws.nspin = 2;
    f.ws.rho_spin_G.resize(2);
    f.ws.rho_spin_G[0].resize({static_cast<std::size_t>(f.ws.n_g)});
    f.ws.rho_spin_G[1].resize({static_cast<std::size_t>(f.ws.n_g)});
    f.ws.V_xc_spin_G.resize(2);
    f.ws.V_xc_spin_G[0].resize({static_cast<std::size_t>(f.ws.n_g)});
    f.ws.V_xc_spin_G[1].resize({static_cast<std::size_t>(f.ws.n_g)});
    for (int i = 0; i < f.ws.n_g; ++i) {
        f.ws.rho_spin_G[0][i] = std::complex<AccReal>(0, 0);
        f.ws.rho_spin_G[1][i] = std::complex<AccReal>(0, 0);
    }
    // Polarized: more ↑ than ↓, plus a G≠0 modulation on ↑ only.
    f.ws.rho_spin_G[0][0] = std::complex<AccReal>(rho0 * AccReal(0.7), 0);
    f.ws.rho_spin_G[1][0] = std::complex<AccReal>(rho0 * AccReal(0.3), 0);
    if (f.ws.n_g > 1)
        f.ws.rho_spin_G[0][1] = std::complex<AccReal>(rho0 * AccReal(0.10), 0);

    XCBuilder xc;
    xc.prepare(f.ws);
    xc.step_spin(f.ws, cpu::scf::XCType::PBE);
    const AccReal e_gga = f.ws.E_xc;

    // Reconstruct ρ↑(r), ρ↓(r) and evaluate PBE@σ=0 pointwise.
    const int nr = f.ws.nr;
    const int ng = f.ws.n_g;
    const int nx = f.ws.fft_nx, ny = f.ws.fft_ny, nz = f.ws.fft_nz;
    const int nxc = nx / 2 + 1;
    const int nc = f.ws.nc;
    std::vector<double> half_reim(static_cast<std::size_t>(2 * nc), 0.0);
    auto* half = reinterpret_cast<fftw_complex*>(half_reim.data());
    std::vector<AccReal> ra(static_cast<std::size_t>(nr));
    std::vector<AccReal> rb(static_cast<std::size_t>(nr));
    auto& gv = *f.ws.gvec;
    auto fft_spin = [&](const xt::xarray<std::complex<AccReal>>& rhoG,
                        std::vector<AccReal>& out) {
        std::fill(half_reim.begin(), half_reim.end(), 0.0);
        for (int ig = 0; ig < ng; ++ig) {
            int h = gv.h[ig], k = gv.k[ig], l = gv.l[ig];
            int kx = ((h % nx) + nx) % nx;
            int ky = ((k % ny) + ny) % ny;
            int kz = ((l % nz) + nz) % nz;
            bool conj = false;
            if (kx > nx / 2) {
                kx = nx - kx;
                ky = (ny - ky) % ny;
                kz = (nz - kz) % nz;
                conj = true;
            }
            const int idx = kx + nxc * (ky + ny * kz);
            if (idx >= 0 && idx < nc) {
                half[static_cast<std::size_t>(idx)][0] =
                    static_cast<double>(rhoG[ig].real());
                half[static_cast<std::size_t>(idx)][1] = static_cast<double>(
                    conj ? -rhoG[ig].imag() : rhoG[ig].imag());
            }
        }
        f.ws.fft->c2r(half, out.data());
    };
    fft_spin(f.ws.rho_spin_G[0], ra);
    fft_spin(f.ws.rho_spin_G[1], rb);

    AccReal e_sigma0 = 0;
    const AccReal dV = f.ws.omega / static_cast<AccReal>(nr);
    for (int i = 0; i < nr; ++i) {
        AccReal a = ra[static_cast<std::size_t>(i)];
        AccReal b = rb[static_cast<std::size_t>(i)];
        if (a < AccReal(0))
            a = AccReal(0);
        if (b < AccReal(0))
            b = AccReal(0);
        const AccReal rtot = a + b;
        if (rtot < AccReal(1e-12))
            continue;
        const auto out =
            cpu::scf::evaluate_xc(cpu::scf::XCType::PBE, static_cast<double>(a),
                                  static_cast<double>(b));
        e_sigma0 += rtot * static_cast<AccReal>(out.exc) * dV;
    }

    const double de = std::abs(static_cast<double>(e_gga - e_sigma0));
    EXPECT_GT(de, 1e-8)
        << "polarized nonuniform spin-GGA must differ from PBE@σ=0; |ΔE_xc|="
        << de << " E_gga=" << e_gga << " E_σ0=" << e_sigma0;
}

TEST(PWInternal, DensityBuilderConservesElectrons) {
    PWFixture f;
    // One doubly-occupied band with the Davidson normalization
    // Σ_G |ψ(G)|² = 1: ψ(G) = δ_{G,0}. Then ψ_phys(r) = 1/√Ω, so
    // ρ(r) = 2/Ω and the electron count ∫ρ d³r = ρ(G=0)·Ω must be 2.
    // mix_beta = 1 turns the first-iteration linear mixing into a pure
    // replacement, exposing the raw built density. The builder reads the
    // LIVE workspace values — it deliberately keeps no constructor snapshot,
    // so mixing parameters set here are the ones finish() honours.
    f.ws.mix_beta = AccReal(1.0);
    f.ws.pulay_depth = 5;
    DensityBuilder db;
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
