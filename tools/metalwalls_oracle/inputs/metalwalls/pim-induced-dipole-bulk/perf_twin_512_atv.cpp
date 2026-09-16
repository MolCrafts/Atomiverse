// Standalone wall-clock twin of the MetalWalls 512-ion bulk-PIM single point.
// Public Atomiverse headers only, same shape as a regressions/ consumer.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/potentials/pim_induced_dipole.h"

using namespace ATV;

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "geom.txt";
    std::ifstream in(path);
    int n = 0;
    double L = 0;
    in >> n >> L;
    std::vector<int> ty(n);
    std::vector<double> q(n), X(n), Y(n), Z(n);
    for (int i = 0; i < n; ++i)
        in >> ty[i] >> q[i] >> X[i] >> Y[i] >> Z[i];
    std::printf("n=%d L=%g\n", n, L);

    double best = 1e30;
    int iters = 0;
    double epol = 0;
    for (int rep = 0; rep < 7; ++rep) {
        cpu::Frame frame(n);
        for (int i = 0; i < n; ++i) {
            frame.span(kAtom_x)[i] = X[i];
            frame.span(kAtom_y)[i] = Y[i];
            frame.span(kAtom_z)[i] = Z[i];
            frame.span(kAtom_charge)[i] = q[i];
            frame.span(kAtom_type)[i] = ty[i];
            frame.span(kAtom_mass)[i] = 1.0;
            frame.span(kAtom_tag)[i] = i + 1;
            frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
                frame.span(kAtom_fz)[i] = 0;
        }
        frame.box = cpu::Box::cubic(L);
        frame.meta.get(kEnergy) = 0;

        cpu::PimInducedDipole pim(11.0, {0.0, 1.0, 1.5}, /*thole_a=*/0.0, 1e-10,
                                  500);
        cpu::Evaluate eval;
        eval.on(evaluate::Phase::evaluate).add_ref(pim);
        eval.build();
        auto t0 = std::chrono::steady_clock::now();
        eval.run(frame);
        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        if (dt < best)
            best = dt;
        iters = pim.last_pol_iters();
        epol = pim.last_pol_energy();
    }
    std::printf("ATV PIM single point: best %.4f s over 7 reps, %d SCC iters, "
                "E_pol = %.12e eV\n",
                best, iters, epol);
    return 0;
}
