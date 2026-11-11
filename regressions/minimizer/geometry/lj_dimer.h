#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

namespace lj_dimer_case {
using namespace ATV;
constexpr double epsilon = 0.0104; // eV
constexpr double sigma = 3.4;      // Angstrom
constexpr double cutoff = 2.5 * sigma;
inline const double r_star = sigma * std::pow(2.0, 1.0 / 6.0);
inline const double r_start = 1.2 * r_star;

inline cpu::Frame initial_frame() {
    cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_x)[i] = i * r_start;
        frame.span(kAtom_y)[i] = 0;
        frame.span(kAtom_z)[i] = 0;
        frame.span(kAtom_mass)[i] = 39.948;
        frame.span(kAtom_type)[i] = 0;
        frame.span(kAtom_tag)[i] = i + 1;
        // Initial preparation must replace deliberately wrong force values.
        frame.span(kAtom_fx)[i] = 1;
        frame.span(kAtom_fy)[i] = 2;
        frame.span(kAtom_fz)[i] = 3;
    }
    frame.box = cpu::Box::cubic(40.0);
    return frame;
}

inline bool check_initial(const cpu::Frame& frame, const char* device,
                          double arithmetic_epsilon) {
    const double sr6 = std::pow(sigma / r_start, 6);
    const double sc6 = std::pow(sigma / cutoff, 6);
    const double expected_energy =
        4 * epsilon * (sr6 * sr6 - sr6 - sc6 * sc6 + sc6);
    const double expected_force =
        24 * epsilon * (sr6 - 2 * sr6 * sr6) / r_start;
    // Cover rounding in the 12-6 expression and boundary conversions, including
    // FP32 classical kernels. This is separate from the existing minimum gates.
    const double tol = std::max(1e-10, 64 * arithmetic_epsilon * epsilon);
    const double pos_tol = std::max(1e-12, 32 * arithmetic_epsilon * r_start);
    const double energy = frame.meta.get(kEnergy);
    if (!(std::abs(energy - expected_energy) <= tol)) {
        std::printf(
            "FAIL [%s] initial LJ energy got=%.16g want=%.16g tol=%.3g eV\n",
            device, energy, expected_energy, tol);
        return false;
    }
    for (int i = 0; i < 2; ++i) {
        const double position[3] = {frame.span(kAtom_x)[i],
                                    frame.span(kAtom_y)[i],
                                    frame.span(kAtom_z)[i]};
        const double force[3] = {frame.span(kAtom_fx)[i],
                                 frame.span(kAtom_fy)[i],
                                 frame.span(kAtom_fz)[i]};
        for (int d = 0; d < 3; ++d) {
            const double want_x = d == 0 ? i * r_start : 0;
            const double want_f =
                d == 0 ? (i == 0 ? expected_force : -expected_force) : 0;
            if (!(std::abs(position[d] - want_x) <= pos_tol)) {
                std::printf("FAIL [%s] zero-budget position atom=%d axis=%d "
                            "got=%.16g want=%.16g tol=%.3g A\n",
                            device, i, d, position[d], want_x, pos_tol);
                return false;
            }
            if (!(std::abs(force[d] - want_f) <= tol)) {
                std::printf("FAIL [%s] initial LJ force atom=%d axis=%d "
                            "got=%.16g want=%.16g tol=%.3g eV/A\n",
                            device, i, d, force[d], want_f, tol);
                return false;
            }
        }
    }
    return true;
}

template <class Minimize>
int check_minimum(Minimize& opt, const char* device,
                  double arithmetic_epsilon) {
    auto frame = initial_frame();
    const auto prepared = opt.run(frame, 0);
    if (prepared != 0) {
        std::printf("FAIL [%s] zero-budget iterations got=%lld want=0 tol=0\n",
                    device, static_cast<long long>(prepared));
        return 1;
    }
    if (!check_initial(frame, device, arithmetic_epsilon))
        return 1;
    const auto iterations = opt.run(frame, 200);
    double dx = frame.span(kAtom_x)[1] - frame.span(kAtom_x)[0];
    double dy = frame.span(kAtom_y)[1] - frame.span(kAtom_y)[0];
    double dz = frame.span(kAtom_z)[1] - frame.span(kAtom_z)[0];
    // A minimizer may wrap either atom independently at the periodic boundary.
    frame.box.min_image(dx, dy, dz);
    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double energy = frame.meta.get(kEnergy);
    const double fmax =
        std::max(std::hypot(frame.span(kAtom_fx)[0], frame.span(kAtom_fy)[0],
                            frame.span(kAtom_fz)[0]),
                 std::hypot(frame.span(kAtom_fx)[1], frame.span(kAtom_fy)[1],
                            frame.span(kAtom_fz)[1]));
    const std::string reason = opt.stop_reason();
    std::printf("[%s] LJ dimer: iterations=%lld r=%.9f want=%.9f "
                "U=%.12f fmax=%.6g converged=%d reason=%s\n",
                device, static_cast<long long>(iterations), r, r_star, energy,
                fmax, int(opt.converged()), reason.c_str());
    bool ok = true;
    if (!(std::abs(r - r_star) < 0.01)) {
        std::printf(
            "FAIL [%s] LJ minimum distance got=%.16g want=%.16g tol=0.01 A\n",
            device, r, r_star);
        ok = false;
    }
    // Retain the existing shifted-LJ well interval and force thresholds.
    if (!(energy < -0.9 * epsilon && energy > -1.1 * epsilon)) {
        std::printf(
            "FAIL [%s] LJ minimum energy got=%.16g want=%.16g tol=%.16g eV\n",
            device, energy, -epsilon, 0.1 * epsilon);
        ok = false;
    }
    if (!(fmax < 0.05)) {
        std::printf(
            "FAIL [%s] LJ residual force got=%.16g want=0 tol=0.05 eV/A\n",
            device, fmax);
        ok = false;
    }
    if (!opt.converged()
        || reason.find("force_converged") == std::string::npos) {
        std::printf("FAIL [%s] convergence got=%d reason=%s "
                    "want=1/force_converged tol=0\n",
                    device, int(opt.converged()), reason.c_str());
        ok = false;
    }
    return ok ? 0 : 1;
}

inline int run_cpu() {
    cpu::minimizer::Minimize opt;
    opt.set_optimizer(cpu::minimizer::LBFGS{10, 0.05, 20});
    opt.add_criterion(cpu::ForceConverged{0.01, 0.005, 0.05, 0.02, 1e-6});
    opt.set_evaluator(cpu::PairLJCut{epsilon, sigma, cutoff});
    opt.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3});
    opt.build();
    std::printf(
        "[cpu] LBFGS: memory=10 alpha0=0.05 StrongWolfe max_iter=200\n");
    return check_minimum(opt, "cpu", 0);
}
} // namespace lj_dimer_case
