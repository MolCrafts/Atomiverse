/// @file regressions/sampling/metad_2d/reg_metad_2d.cpp
/// @brief 2D well-tempered metadynamics on two DistanceCVs (textbook gate).
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/bias/distance_cv.h"
#include "atomiverse/cpu/bias/metadynamics.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

int main() {
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 1.4;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_x)[2] = 0;
    f.span(kAtom_y)[2] = 1.2;
    f.span(kAtom_z)[2] = 0;
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20);

    std::vector<std::unique_ptr<cpu::bias::CollectiveVariable>> cvs;
    cvs.push_back(std::make_unique<cpu::bias::DistanceCV>(1, 2));
    cvs.push_back(std::make_unique<cpu::bias::DistanceCV>(1, 3));
    cpu::bias::Metadynamics meta(std::move(cvs), 0.2, {0.15, 0.15}, 300.0, 8.0,
                                 1);

    // Continuable MD with dt=0 keeps Metadynamics state across launches while
    // the host walk visits different CV values between steps (Evaluate rebuilds
    // every run and would wipe hills).
    cpu::md::MD sim;
    sim.set_integrator(cpu::md::VelocityVerlet{0.0});
    sim.set_evaluator_ref(meta);
    sim.build();

    int hills = 0;
    double last_u = 0;
    for (int i = 0; i < 25; ++i) {
        f.span(kAtom_x)[1] = 1.2 + 0.02 * i;
        f.span(kAtom_y)[2] = 1.0 + 0.01 * i;
        sim.run(f, 1);
        hills = meta.n_hills();
        last_u = meta.last_bias_energy();
    }
    std::printf("metad_2d hills=%d last_u=%.6e n_cv=%d\n", hills, last_u,
                meta.n_cv());
    if (hills < 20) {
        std::printf("FAIL: expected >=20 hills\n");
        return 1;
    }
    if (!(last_u > 0.0) || !std::isfinite(last_u)) {
        std::printf("FAIL: bias energy\n");
        return 1;
    }
    if (meta.n_cv() != 2 || static_cast<int>(meta.last_cv().size()) != 2) {
        std::printf("FAIL: CV dimension\n");
        return 1;
    }
    double f2 = 0;
    for (int a = 0; a < 3; ++a)
        f2 += f.span(kAtom_fx)[a] * f.span(kAtom_fx)[a];
    if (!(f2 > 0.0) || !std::isfinite(f2)) {
        std::printf("FAIL: forces not finite/nonzero\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
