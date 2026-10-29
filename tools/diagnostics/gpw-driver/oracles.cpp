// Existing independent CP2K water energy pin, exercised through public methods.
// Opt-in complete solves; never registered with CTest.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace ATV;
namespace {
// Retained from regressions/gpw/orbital-transformation/REFERENCE.md and
// ot-minimizer-water.cpp: actual CP2K 2026.1 OT output at 250 Ry.
constexpr double kCp2kEnergyHa = -17.212848291913666;
constexpr double kCp2kToleranceHa = 1.5e-3;
constexpr double kAlgorithmToleranceHa = 1e-4;
void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}
cpu::Frame water() {
    cpu::Frame frame(3);
    frame.box = cpu::Box::cubic(10.0);
    constexpr int z[] = {8, 1, 1};
    constexpr double position[3][3] = {{5.0000, 5.0000, 5.0000},
                                       {5.7586, 5.0000, 5.5043},
                                       {4.2414, 5.0000, 5.5043}};
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_type)[i] = z[i];
        frame.span(kAtom_x)[i] = position[i][0];
        frame.span(kAtom_y)[i] = position[i][1];
        frame.span(kAtom_z)[i] = position[i][2];
    }
    return frame;
}
double solve(bool ot, double ot_energy) {
    const auto start = std::chrono::steady_clock::now();
    auto frame = water();
    cpu::gpw::GpwDft method(
        {{molrs::Element::O,
          cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
          cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")},
         {molrs::Element::H,
          cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
          cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")}},
        250.0, cpu::xc::XCType::PBE);
    method.max_iter = 200;
    method.tol = 1e-7;
    method.mix = .3;
    method.nonperiodic = true;
    method.d3 = false;
    method.use_ot = ot;
    method.build(frame);
    const auto result = method.run(frame);
    require(result.converged && result.iters > 0 && result.iters <= 200,
            "water reference must converge within its original budget");
    require(std::isfinite(result.e_total)
                && std::abs(result.e_total - kCp2kEnergyHa) <= kCp2kToleranceHa,
            "water energy exceeds original CP2K 250-Ry gate");
    require(
        std::abs(frame.meta.get(kEnergy) - result.e_total * units::kHartreeInEv)
            <= 1e-10,
        "water Frame energy publication");
    if (!ot)
        require(std::abs(result.e_total - ot_energy) <= kAlgorithmToleranceHa,
                "water OT/DIIS energy exceeds original algorithm gate");
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::printf("PASS GPW_ORACLE WATER_%s E_Ha=%.15g dCP2K_Ha=%.8g "
                "iterations=%d diagnostic_wall_s=%.6g\n",
                ot ? "OT" : "DIIS", result.e_total,
                std::abs(result.e_total - kCp2kEnergyHa), result.iters,
                seconds);
    std::fflush(stdout);
    return result.e_total;
}
} // namespace
int main() {
    try {
        const double ot = solve(true, 0);
        solve(false, ot);
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "FAIL GPW_ORACLE %s\n", failure.what());
        return 1;
    }
}
