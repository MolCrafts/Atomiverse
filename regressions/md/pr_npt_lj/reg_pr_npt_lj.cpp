/// @file regressions/md/pr_npt_lj/reg_pr_npt_lj.cpp
/// @brief END-TO-END: Parrinello–Rahman NPT on compressed LJ argon.
///
/// Checks: finite positive volume, pressure relief (expansion from a
/// compressed cubic start), bounded shear on an isotropic target, and a
/// finite MTK extended Hamiltonian. See REFERENCE.md.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/parrinello_rahman_npt.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    // Compressed FCC argon (a = 5.1 Å < equilibrium).
    const double eps = 0.0104; // [eV]
    const double sigma = 3.4;
    const double cutoff = 6.8;
    const double skin = 0.5;
    const double a_lat = 5.1; // [Å]
    const int n_cells = 3;    // 108 atoms
    const double T = 100.0;
    const double P_target = 0.0;
    const double dt = 1.0;
    const double tau_T = 100.0;
    const double tau_P = 1000.0;
    const int n_steps = 800;

    const int n = n_cells * n_cells * n_cells * 4;
    cpu::Frame frame(n);
    const double basis[4][3] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
    int idx = 0;
    for (int ix = 0; ix < n_cells; ++ix)
        for (int iy = 0; iy < n_cells; ++iy)
            for (int iz = 0; iz < n_cells; ++iz)
                for (int b = 0; b < 4; ++b) {
                    frame.span(ATV::kAtom_x)[idx] = (ix + basis[b][0]) * a_lat;
                    frame.span(ATV::kAtom_y)[idx] = (iy + basis[b][1]) * a_lat;
                    frame.span(ATV::kAtom_z)[idx] = (iz + basis[b][2]) * a_lat;
                    frame.span(ATV::kAtom_mass)[idx] = 39.948;
                    frame.span(ATV::kAtom_type)[idx] = 18;
                    ++idx;
                }
    frame.box = cpu::Box::cubic(n_cells * a_lat);
    const double V0 = frame.box.volume();

    cpu::md::MD sim;
    sim.set_evaluator(cpu::PairLJCut{eps, sigma, cutoff});
    sim.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
    sim.set_integrator(
        cpu::md::ParrinelloRahmanNPT{T, P_target, tau_T, tau_P, dt});
    sim.set_observer(cpu::md::InitVelocity{T, 42u});
    sim.set_observer(cpu::md::Virial{});
    sim.enable_timing(false);
    sim.build();

    sim.run(frame, n_steps);

    const double V1 = frame.box.volume();
    const Mat3& h = frame.box.cell();
    const double edge =
        std::max({std::fabs(h(0, 0)), std::fabs(h(1, 1)), std::fabs(h(2, 2))});
    const double shear = std::fabs(h(0, 1)) + std::fabs(h(0, 2))
                         + std::fabs(h(1, 0)) + std::fabs(h(1, 2))
                         + std::fabs(h(2, 0)) + std::fabs(h(2, 1));
    const double H = frame.meta.get(ATV::kConserved);

    std::printf("PR-NPT LJ FCC, N=%d, T=%.0f K, a=%.1f A, P=0, dt=%.1f fs\n", n,
                T, a_lat, dt);
    std::printf("  V0            = %.3f A^3\n", V0);
    std::printf("  V1            = %.3f A^3   (want V1 > V0)\n", V1);
    std::printf("  shear/edge    = %.3e   (want < 0.20)\n", shear / edge);
    std::printf("  conserved H   = %.6f eV\n", H);

    bool ok = true;
    if (!(V1 > V0)) {
        std::printf("FAIL: compressed cell did not expand\n");
        ok = false;
    }
    if (!(V1 < 5.0 * V0)) {
        std::printf("FAIL: cell expanded unboundedly (V1/V0=%.2f)\n", V1 / V0);
        ok = false;
    }
    if (!(shear / edge < 0.20)) {
        std::printf("FAIL: shear DOF unbounded on isotropic target\n");
        ok = false;
    }
    if (!(std::isfinite(H) && std::isfinite(V1) && V1 > 0.0)) {
        std::printf("FAIL: non-finite final state\n");
        ok = false;
    }

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
