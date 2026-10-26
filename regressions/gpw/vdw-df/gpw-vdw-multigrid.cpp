/// @file regressions/gpw/gpw-vdw-multigrid/reg_gpw-vdw-multigrid.cpp
/// @brief Γ GPW PBE + vdW-DF (DRSLL) on the REL_CUTOFF multigrid (NGRIDS 4).
///
/// Out-of-library consumer. Self-consistent nonlocal V_nl rides the CP2K-style
/// multigrid product path (ngrids > 1 && rel_cutoff > 0): the fine-grid V_nl
/// is added to veff before the per-level restriction, so the SCF stays
/// variational. Gates (REFERENCE.md):
///   1. internal: |E_tot(mg) − E_tot(sg)| — multigrid is a cost optimization,
///      not a physics change;
///   2. pin: E_disp(mg) vs offline CP2K 2026.1 NGRIDS 4 + NON_LOCAL DRSLL;
///   3. pin: E_tot(mg) vs the same CP2K run.
/// No live oracle in ctest.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molpy.hpp"
#include "harness/registry.h"

namespace atv_reg_gpw_gpw_vdw_multigrid {

namespace {

// Same water dimer as gpw-vdw-df-ice / gpw-vdw-scf-potential [Å].
const double kAng[6][3] = {
    {0.000000, 0.000000, 0.000000},  // O
    {0.758602, 0.000000, 0.504284},  // H
    {0.758602, 0.000000, -0.504284}, // H
    {3.000000, 0.000000, 0.000000},  // O
    {3.758602, 0.000000, 0.504284},  // H
    {3.758602, 0.000000, -0.504284}, // H
};
const int kZ[6] = {8, 1, 1, 8, 1, 1};

ATV::cpu::Frame make_frame() {
    using namespace ATV;
    cpu::Frame frame(6);
    for (int i = 0; i < 6; ++i) {
        frame.span(kAtom_type)[i] = kZ[i];
        frame.span(kAtom_x)[i] = kAng[i][0];
        frame.span(kAtom_y)[i] = kAng[i][1];
        frame.span(kAtom_z)[i] = kAng[i][2];
    }
    frame.box = cpu::Box::cubic(12.0);
    return frame;
}

std::vector<ATV::cpu::gpw::Kind> water_kinds() {
    using namespace ATV::cpu::gpw;
    return {Kind{molrs::Element::O,
                 GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
                 GthPotential(molrs::Element::O, "GTH-PBE-q6")},
            Kind{molrs::Element::H,
                 GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
                 GthPotential(molrs::Element::H, "GTH-PBE-q1")}};
}

ATV::cpu::gpw::GpwDft make_gpw(int ngrids, double rel_cutoff) {
    ATV::cpu::gpw::GpwDft gpw(water_kinds(), 200.0, ATV::cpu::gpw::XcKind::PBE);
    gpw.ngrids = ngrids;
    gpw.rel_cutoff = rel_cutoff;
    gpw.n_occ = 8;
    gpw.max_iter = 60;
    gpw.tol = 1e-7;
    gpw.nonperiodic = false;
    gpw.vdw_df = true;
    return gpw;
}

} // namespace

int run() {
    using namespace ATV::cpu::gpw;

    // Offline CP2K 2026.1 (cp2k.psmp), NGRIDS 4 / REL_CUTOFF 40 + NON_LOCAL
    // DRSLL — see REFERENCE.md. CP2K's own NGRIDS 4 ↔ 1 gap is 6.0e-7 Ha.
    constexpr double kEDispCp2kMg = 1.5193559412669e-01; // Ha
    constexpr double kETotCp2kMg = -34.186333524093456;  // Ha
    constexpr double kEDispTol = 1.0e-4;                 // Ha
    // ATV ↔ CP2K absolute E_tot on this Γ dimer twin family carries a known
    // ~7.9e-3 Ha systematic that is IDENTICAL on the single-grid and
    // multigrid paths (measured Δ_sg = 7.922e-3, Δ_mg = 7.926e-3, and
    // eps_default 1e-10 moves it by <1e-8): the gate pins that the multigrid
    // total stays inside that pre-existing single-grid envelope.
    constexpr double kETotTol = 1.0e-2; // Ha
    // Internal single-grid ↔ multigrid consistency (measured 4.28e-6 Ha;
    // CP2K's own NGRIDS 4 ↔ 1 gap is 6.0e-7 Ha).
    constexpr double kEConsistencyTol = 2.0e-5; // Ha

    ATV::cpu::Frame frame = make_frame();

    GpwDft sg = make_gpw(/*ngrids=*/1, /*rel_cutoff=*/0.0);
    sg.build(frame);
    const GpwResult& rs = sg.run(frame);
    if (!rs.converged) {
        std::fprintf(stderr, "FAIL: single-grid SCF did not converge (%d)\n",
                     rs.iters);
        return 1;
    }
    const double e_tot_sg = rs.e_total;
    const double e_disp_sg = rs.e_disp;

    GpwDft mg = make_gpw(/*ngrids=*/4, /*rel_cutoff=*/40.0);
    mg.build(frame);
    const GpwResult& rm = mg.run(frame);
    if (!rm.converged) {
        std::fprintf(stderr, "FAIL: multigrid SCF did not converge (%d)\n",
                     rm.iters);
        return 1;
    }

    std::printf("E_tot  sg=%.12f mg=%.12f  |d|=%.3e Ha\n", e_tot_sg, rm.e_total,
                std::abs(rm.e_total - e_tot_sg));
    std::printf("E_disp sg=%.12f mg=%.12f  (CP2K mg %.12f)\n", e_disp_sg,
                rm.e_disp, kEDispCp2kMg);

    if (std::abs(rm.e_total - e_tot_sg) > kEConsistencyTol) {
        std::fprintf(stderr, "FAIL: |E_tot(mg) - E_tot(sg)| = %.3e Ha > %.1e\n",
                     std::abs(rm.e_total - e_tot_sg), kEConsistencyTol);
        return 1;
    }
    if (!std::isfinite(rm.e_disp)
        || std::abs(rm.e_disp - kEDispCp2kMg) > kEDispTol) {
        std::fprintf(stderr, "FAIL: E_disp(mg) got=%.14e want=%.14e ±%.1e Ha\n",
                     rm.e_disp, kEDispCp2kMg, kEDispTol);
        return 1;
    }
    if (std::abs(rm.e_total - kETotCp2kMg) > kETotTol) {
        std::fprintf(stderr, "FAIL: E_tot(mg) got=%.14f want=%.14f ±%.1e Ha\n",
                     rm.e_total, kETotCp2kMg, kETotTol);
        return 1;
    }

    std::printf("PASS gpw-vdw-multigrid\n");
    return 0;
}

} // namespace atv_reg_gpw_gpw_vdw_multigrid

ATV_REGRESSION("gpw/vdw-df", "gpw-vdw-multigrid",
               atv_reg_gpw_gpw_vdw_multigrid::run);
