/// @file regressions/md/const-pot-cg-large-electrode/reg_const-pot-cg-large-electrode.cpp
/// @brief E2E: ConstPotCG matches dense ConstPotBO on a multi-site plate.
///
/// Oracle: MetalWalls `c8ad175d`, run offline. The pinned primary-cell
/// \(Q_{\mathrm{top}}\) below reproduces MetalWalls to 3.1e-9 relative — the
/// primary system is 4 particles on one \((x,y)\) line, so ATV's
/// `xy_image_range = 0` min-image matrix is *exactly* the isolated cluster
/// MetalWalls converges to as \(L_{xy},L_z\to\infty\).
///
/// The large-plate numbers printed below are a CG-vs-dense-BO equivalence
/// check on a **shared** truncated \(A\); they are NOT MetalWalls-validated.
/// With `xy_image_range = 0` the 4x4 plate sits 20.9% above MetalWalls' full
/// 2D-Ewald answer (0.365146 vs 0.301964 e) — see REFERENCE.md "Known gap".
/// Public API only.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/const_pot_cg.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"

using namespace ATV;

namespace {

/// nx×ny electrode sites per wall + two electrolyte ions.
cpu::Frame multi_site_plate(int nx, int ny) {
    const int n_wall = nx * ny;
    const int n = 2 * n_wall + 2;
    cpu::Frame f(n);
    const double Lx = static_cast<double>(nx);
    const double Ly = static_cast<double>(ny);
    const double Lz = 20.0;
    const double z_bot = 8.0;
    const double z_top = 12.0;
    int a = 0;
    auto place_wall = [&](int type, double z) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                f.span(kAtom_x)[a] = (ix + 0.5) * (Lx / nx);
                f.span(kAtom_y)[a] = (iy + 0.5) * (Ly / ny);
                f.span(kAtom_z)[a] = z;
                f.span(kAtom_charge)[a] = 0.0;
                f.span(kAtom_type)[a] = type;
                f.span(kAtom_mass)[a] = 196.966553;
                f.span(kAtom_tag)[a] = a + 1;
                f.span(kAtom_vx)[a] = f.span(kAtom_vy)[a] =
                    f.span(kAtom_vz)[a] = 0.0;
                ++a;
            }
        }
    };
    place_wall(1, z_bot);
    place_wall(2, z_top);
    f.span(kAtom_x)[a] = 0.5 * Lx;
    f.span(kAtom_y)[a] = 0.5 * Ly;
    f.span(kAtom_z)[a] = 9.0;
    f.span(kAtom_charge)[a] = 0.5;
    f.span(kAtom_type)[a] = 3;
    f.span(kAtom_mass)[a] = 1.0;
    f.span(kAtom_tag)[a] = a + 1;
    f.span(kAtom_vx)[a] = f.span(kAtom_vy)[a] = f.span(kAtom_vz)[a] = 0.0;
    ++a;
    f.span(kAtom_x)[a] = 0.5 * Lx;
    f.span(kAtom_y)[a] = 0.5 * Ly;
    f.span(kAtom_z)[a] = 11.0;
    f.span(kAtom_charge)[a] = -0.5;
    f.span(kAtom_type)[a] = 3;
    f.span(kAtom_mass)[a] = 1.0;
    f.span(kAtom_tag)[a] = a + 1;
    f.span(kAtom_vx)[a] = f.span(kAtom_vy)[a] = f.span(kAtom_vz)[a] = 0.0;

    f.box = cpu::Box::orthorhombic(Lx, Ly, Lz);
    return f;
}

cpu::Frame madelung_primary() {
    cpu::Frame frame(4);
    const double z[4] = {8.0, 12.0, 9.0, 11.0};
    const double q0[4] = {0.0, 0.0, 0.5, -0.5};
    const int typ[4] = {1, 2, 3, 3};
    for (int i = 0; i < 4; ++i) {
        frame.span(kAtom_x)[i] = 0.0;
        frame.span(kAtom_y)[i] = 0.0;
        frame.span(kAtom_z)[i] = z[i];
        frame.span(kAtom_charge)[i] = q0[i];
        frame.span(kAtom_type)[i] = typ[i];
        frame.span(kAtom_mass)[i] = (typ[i] < 3) ? 196.966553 : 1.0;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_vx)[i] = frame.span(kAtom_vy)[i] =
            frame.span(kAtom_vz)[i] = 0.0;
    }
    frame.box = cpu::Box::orthorhombic(1.0, 1.0, 20.0);
    return frame;
}

template <typename Solver>
void solve_once(Solver& s, cpu::Frame& f) {
    cpu::md::MD sim;
    sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/12.0, /*skin=*/0.5});
    sim.set_evaluator(cpu::PairCoulCut{/*cutoff=*/12.0});
    sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.0});
    sim.set_observer_ref(s);
    sim.build();
    sim.run(f, 1);
}

} // namespace

int main() {
    constexpr double kEta = 2.0;
    constexpr double kPsiBot = 0.0;
    constexpr double kPsiTop = 1.0;
    constexpr double kSiteTol = 1e-8; // [e]
    // MetalWalls c8ad175d, isolated limit: 0.2717535549687 e (REFERENCE.md).
    constexpr double kQTopPrimary = 0.27175356;
    constexpr int nx = 4, ny = 4; // 32 electrode sites
    int nfail = 0;

    std::printf("const-pot-cg-large-electrode\n");

    // ── Small madelung primary-cell (sanity vs dense BO pin) ──────────
    {
        auto f = madelung_primary();
        cpu::md::ConstPotCG cg(kEta);
        cg.add_electrode(1, kPsiBot);
        cg.add_electrode(2, kPsiTop);
        cg.electroneutrality = true;
        cg.xy_image_range = 0;
        cg.cg_tol = 1e-14;
        solve_once(cg, f);
        const double q_top = cg.total_charge(1);
        const double neut = cg.neutrality_residual();
        std::printf(
            "  primary Q_top = % .12e  (pin %.8f)  |sumQ|=% .3e  cg_it=%d\n",
            q_top, kQTopPrimary, neut, cg.cg_iterations());
        if (!(neut <= 1e-10)) {
            std::fprintf(stderr, "FAIL primary neutrality %g\n", neut);
            ++nfail;
        }
        if (!(std::abs(q_top - kQTopPrimary) <= 1e-6)) {
            std::fprintf(stderr, "FAIL primary Q_top %g vs pin %g\n", q_top,
                         kQTopPrimary);
            ++nfail;
        }
    }

    // ── Large multi-site plate: CG vs dense BO ────────────────────────
    auto f_cg = multi_site_plate(nx, ny);
    auto f_bo = multi_site_plate(nx, ny);

    cpu::md::ConstPotCG cg(kEta);
    cg.add_electrode(1, kPsiBot);
    cg.add_electrode(2, kPsiTop);
    cg.electroneutrality = true;
    cg.xy_image_range = 0;
    cg.cg_tol = 1e-14;
    cg.max_cg_iter = 2000;

    cpu::md::ConstPotBO bo(kEta);
    bo.add_electrode(1, kPsiBot);
    bo.add_electrode(2, kPsiTop);
    bo.electroneutrality = true;
    bo.xy_image_range = 0;

    const auto t0_bo = std::chrono::steady_clock::now();
    solve_once(bo, f_bo);
    const auto t1_bo = std::chrono::steady_clock::now();

    const auto t0_cg = std::chrono::steady_clock::now();
    solve_once(cg, f_cg);
    const auto t1_cg = std::chrono::steady_clock::now();

    const double ms_bo =
        std::chrono::duration<double, std::milli>(t1_bo - t0_bo).count();
    const double ms_cg =
        std::chrono::duration<double, std::milli>(t1_cg - t0_cg).count();

    const int n_elec = cg.n_electrode();
    const double neut = cg.neutrality_residual();
    const double dq_bot = std::abs(cg.total_charge(0) - bo.total_charge(0));
    const double dq_top = std::abs(cg.total_charge(1) - bo.total_charge(1));

    double max_dq = 0.0;
    for (int i = 0; i < n_elec; ++i) {
        max_dq = std::max(
            max_dq,
            std::abs(static_cast<double>(f_cg.span(kAtom_charge)[i])
                     - static_cast<double>(f_bo.span(kAtom_charge)[i])));
    }

    std::printf("  large plate %dx%d  N_elec=%d\n", nx, ny, n_elec);
    std::printf("  Q_bot CG=% .12e  BO=% .12e  |d|=% .3e\n", cg.total_charge(0),
                bo.total_charge(0), dq_bot);
    std::printf("  Q_top CG=% .12e  BO=% .12e  |d|=% .3e\n", cg.total_charge(1),
                bo.total_charge(1), dq_top);
    std::printf("  max|Qi_CG-Qi_BO|=% .3e e  |sumQ|=% .3e  cg_it=%d\n", max_dq,
                neut, cg.cg_iterations());
    std::printf("  wall_ms CG=%.3f  BO=%.3f  (internal twin, not MW)\n", ms_cg,
                ms_bo);

    if (n_elec < 18) {
        std::fprintf(stderr, "FAIL N_elec=%d < 18\n", n_elec);
        ++nfail;
    }
    if (!(neut <= 1e-10)) {
        std::fprintf(stderr, "FAIL large neutrality %g\n", neut);
        ++nfail;
    }
    if (!(max_dq <= kSiteTol)) {
        std::fprintf(stderr, "FAIL max site |dQ|=%g > %g\n", max_dq, kSiteTol);
        ++nfail;
    }
    if (!(dq_bot <= kSiteTol && dq_top <= kSiteTol)) {
        std::fprintf(stderr, "FAIL group |dQ| bot=%g top=%g\n", dq_bot, dq_top);
        ++nfail;
    }
    // Electrolyte untouched
    const int i_elyt0 = n_elec;
    if (std::abs(static_cast<double>(f_cg.span(kAtom_charge)[i_elyt0]) - 0.5)
            > 1e-14
        || std::abs(static_cast<double>(f_cg.span(kAtom_charge)[i_elyt0 + 1])
                    + 0.5)
               > 1e-14) {
        std::fprintf(stderr, "FAIL electrolyte charges mutated\n");
        ++nfail;
    }

    // ── MD binding path (one step) ────────────────────────────────────
    {
        auto f2 = multi_site_plate(nx, ny);
        cpu::md::ConstPotCG conp2(kEta);
        conp2.add_electrode(1, kPsiBot);
        conp2.add_electrode(2, kPsiTop);
        conp2.electroneutrality = true;
        conp2.xy_image_range = 0;
        conp2.cg_tol = 1e-12;
        conp2.rebuild_matrix_every_step = true;

        cpu::md::MD sim;
        sim.set_neighbor(
            cpu::md::NeighborList<>{/*cutoff=*/12.0, /*skin=*/0.5});
        sim.set_evaluator(cpu::PairCoulCut{/*cutoff=*/12.0});
        sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
        sim.set_observer_ref(conp2);
        sim.build();
        sim.run(f2, 1);

        const double d_top =
            std::abs(conp2.total_charge(1) - bo.total_charge(1));
        std::printf("  MD path Q_top=% .12e  |d vs BO|=% .3e  |sumQ|=% .3e\n",
                    conp2.total_charge(1), d_top, conp2.neutrality_residual());
        if (!(conp2.neutrality_residual() <= 1e-10) || !(d_top <= 1e-6)) {
            std::fprintf(stderr, "FAIL MD-bound ConstPotCG d_top=%g neut=%g\n",
                         d_top, conp2.neutrality_residual());
            ++nfail;
        }
    }

    if (nfail) {
        std::fprintf(stderr, "%d check(s) failed\n", nfail);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
