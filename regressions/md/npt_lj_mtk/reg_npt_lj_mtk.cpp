/// @file regressions/md/npt_lj_mtk/reg_npt_lj_mtk.cpp
/// @brief CPU-only MTK NPT on 64-atom LJ (CUDA builds use the .cu twin).
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

int main() {
    constexpr int n_side = 4;
    const int n = n_side * n_side * n_side;
    const double density = 0.8;
    const double box_len = std::cbrt(double(n) / density);
    const double spacing = box_len / double(n_side);
    cpu::Frame frame(n);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                frame.span(kAtom_x)[idx] = (ix + 0.5) * spacing;
                frame.span(kAtom_y)[idx] = (iy + 0.5) * spacing;
                frame.span(kAtom_z)[idx] = (iz + 0.5) * spacing;
                frame.span(kAtom_mass)[idx] = 1.0;
                frame.span(kAtom_type)[idx] = 18;
                frame.span(kAtom_vx)[idx] = 0.02 * ((ix + iy + iz) % 3 - 1);
                frame.span(kAtom_vy)[idx] = 0.02 * ((ix + 2 * iy) % 3 - 1);
                frame.span(kAtom_vz)[idx] = 0.02 * ((2 * ix + iz) % 3 - 1);
                ++idx;
            }
    frame.box = cpu::Box::cubic(box_len);
    const double V0 = frame.box.volume();

    cpu::md::MD sim;
    sim.set_neighbor(cpu::md::NeighborList<>{2.5, 0.3});
    sim.set_evaluator(cpu::PairLJCut{1.0, 1.0, 2.5});
    sim.set_integrator(cpu::md::NoseHooverNPT{1.0, 0.1, 0.0, 1.0, 0.005});
    sim.set_observer(cpu::md::Virial{});
    sim.build();
    sim.run(frame, 80);

    const double V = frame.box.volume();
    const double H = static_cast<double>(frame.meta.get(kConserved));
    std::printf("[cpu] V0=%.6f V=%.6f H=%.6e dV/V0=%.4e\n", V0, V, H,
                (V - V0) / V0);
    if (!std::isfinite(H) || !std::isfinite(V) || V <= 0) {
        std::fprintf(stderr, "FAIL: non-finite H or V\n");
        return 1;
    }
    std::printf("PASS: npt_lj MTK NoseHooverNPT\n");
    return 0;
}
