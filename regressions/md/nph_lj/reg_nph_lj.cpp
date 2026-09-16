/// @file regressions/md/nph_lj/reg_nph_lj.cpp
/// @brief END-TO-END: isotropic NPH on LJ argon — relative drift of H.
///
/// See REFERENCE.md. Out-of-library consumer; exits non-zero on failure.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nph_integrator.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    // Compressed FCC argon — dense start so the barostat moves volume
    // without free expansion to vacuum at P_target = 0.
    const double eps = 0.0104;
    const double sigma = 3.4;
    const double cutoff = 6.8;
    const double skin = 0.5;
    const double a_lat = 5.2;
    const int n_cells = 3; // 108 atoms
    const double T_init = 100.0;
    const double P_target = 0.0;
    const double dt = 1.0;
    const double tau_P = 1000.0;
    const int n_warm = 100;
    const int n_prod = 600;
    const int n_chunks = 20;

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
    sim.set_integrator(cpu::md::NphIntegrator{P_target, tau_P, dt});
    sim.set_observer(cpu::md::InitVelocity{T_init, 99u});
    sim.set_observer(cpu::md::Virial{});
    sim.enable_timing(false);
    sim.build();

    sim.run(frame, n_warm);
    const double H0 = frame.meta.get(ATV::kConserved);
    if (!std::isfinite(H0)) {
        std::printf("FAIL: H0 not finite (%.6e)\n", H0);
        return 1;
    }

    double worst_rel = 0.0;
    const int chunk = n_prod / n_chunks;
    for (int c = 0; c < n_chunks; ++c) {
        sim.run(frame, chunk);
        const double H = frame.meta.get(ATV::kConserved);
        if (!std::isfinite(H) || !(frame.box.volume() > 0.0)) {
            std::printf("FAIL: non-finite H or volume at chunk %d\n", c);
            return 1;
        }
        const double denom = std::max(1.0, std::fabs(H0));
        const double rel = std::fabs(H - H0) / denom;
        if (rel > worst_rel)
            worst_rel = rel;
    }

    const double rel_bound = 1.5e-1;
    const double V1 = frame.box.volume();

    std::printf("NPH LJ FCC, N=%d, P_target=%.2e, dt=%.1f fs, tau_P=%.0f\n", n,
                P_target, dt, tau_P);
    std::printf("  H0              = %.6f eV\n", H0);
    std::printf("  worst |dH|/max(1,|H0|) = %.3e   want <= %.1e\n", worst_rel,
                rel_bound);
    std::printf("  V0              = %.3f A^3\n", V0);
    std::printf("  final V         = %.3f A^3\n", V1);

    bool ok = true;
    if (!(worst_rel <= rel_bound)) {
        std::printf("FAIL: relative H drift exceeds gate\n");
        ok = false;
    }
    if (!(V1 > 0.0) || !(V1 < 10.0 * V0)) {
        std::printf("FAIL: volume not in safe window\n");
        ok = false;
    }

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
