// Public GPW results compared with actual independent CP2K output.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "cp2k_pins.h"
#ifdef ATV_HAS_CUDA
#include "atomiverse/cuda/gpw/gpw_solver.cuh"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
using namespace ATV;
namespace {
cpu::Frame hydrogen() {
    cpu::Frame frame(2);
    constexpr double a = units::kBohrInAngstrom;
    frame.box = cpu::Box::orthorhombic(12 * a, 11 * a, 10 * a);
    constexpr double p[2][3] = {{5.45, 5.15, 4.4}, {6.55, 5.65, 4.8}};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_mass)[i] = 1.008;
        frame.span(kAtom_x)[i] = p[i][0] * a;
        frame.span(kAtom_y)[i] = p[i][1] * a;
        frame.span(kAtom_z)[i] = p[i][2] * a;
    }
    return frame;
}
std::vector<cpu::gpw::Kind> kinds() {
    return {{molrs::Element::H,
             cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
             cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")}};
}
bool finite_close(double a, double b, double tol) {
    return std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tol;
}
template <class Method>
bool compare(const pbe_periodic_cp2k::Pin& pin, const char* backend) {
    constexpr cpu::xc::XCType types[] = {cpu::xc::XCType::PBEsol,
                                         cpu::xc::XCType::RevPBE,
                                         cpu::xc::XCType::RPBE};
    auto frame = hydrogen();
    Method method{kinds(), 80, types[pin.functional]};
    method.n_occ = 1;
    method.max_iter = 100;
    method.tol = 1e-10;
    method.diis_depth = 6;
    method.eps_default = 1e-10;
    method.nonperiodic = false;
    method.local_pp_on_grid = true;
    method.use_ot = false;
    method.ngrids = 1;
    method.stress_strain_h = .002;
    method.stress_axes = {true, true, true};
    method.build(frame);
    const auto result = method.forces(frame);
    if (!result.converged)
        throw std::runtime_error("public GPW force SCF did not converge");
    const double energy_error = std::abs(result.e_total - pin.energy_ha);
    bool pass = finite_close(result.e_total, pin.energy_ha,
                             pbe_periodic_cp2k::energy_tolerance_ha);
    double force_error = 0, stress_error = 0;
    int failed_force_lanes = 0, failed_stress_lanes = 0;
    for (int atom = 0; atom < 2; ++atom) {
        const double f[] = {frame.span(kAtom_fx)[atom],
                            frame.span(kAtom_fy)[atom],
                            frame.span(kAtom_fz)[atom]};
        std::printf("GPW_CP2K_FORCE %s %s atom=%d Fx_Ha_Bohr=%.15g "
                    "Fy_Ha_Bohr=%.15g Fz_Ha_Bohr=%.15g\n",
                    pin.id, backend, atom,
                    f[0] / units::kHartreePerBohrInEvPerAngstrom,
                    f[1] / units::kHartreePerBohrInEvPerAngstrom,
                    f[2] / units::kHartreePerBohrInEvPerAngstrom);
        for (int axis = 0; axis < 3; ++axis) {
            const double got = f[axis] / units::kHartreePerBohrInEvPerAngstrom;
            const double error = std::abs(got - pin.forces_ha_bohr[atom][axis]);
            force_error = std::max(force_error, error);
            const bool lane =
                finite_close(got, pin.forces_ha_bohr[atom][axis],
                             pbe_periodic_cp2k::force_tolerance_ha_bohr);
            pass = pass && lane;
            failed_force_lanes += !lane;
        }
    }
    pass = pass
           && finite_close(frame.meta.get(kEnergy),
                           result.e_total * units::kHartreeInEv, 1e-10);
    const auto sigma = method.stress(frame);
    if (!method.last_result().converged)
        throw std::runtime_error("public GPW stress SCF did not converge");
    pass = pass
           && finite_close(method.last_result().e_total, result.e_total, 1e-8);
    std::printf("GPW_CP2K_DIAG_STRESS %s %s xx_Ha_Bohr3=%.15g "
                "yy_Ha_Bohr3=%.15g zz_Ha_Bohr3=%.15g\n",
                pin.id, backend,
                sigma[0] / units::kHartreePerBohr3InEvPerAngstrom3,
                sigma[1] / units::kHartreePerBohr3InEvPerAngstrom3,
                sigma[2] / units::kHartreePerBohr3InEvPerAngstrom3);
    for (int axis = 0; axis < 3; ++axis) {
        // 1 eV/A^3 = 160.2176634 GPa (exact SI elementary charge).
        const double expected = pin.diagonal_stress_gpa[axis] / 160.2176634
                                / units::kHartreePerBohr3InEvPerAngstrom3;
        const double got =
            sigma[axis] / units::kHartreePerBohr3InEvPerAngstrom3;
        stress_error = std::max(stress_error, std::abs(got - expected));
        const bool lane =
            finite_close(got, expected,
                         pbe_periodic_cp2k::diagonal_stress_tolerance_ha_bohr3);
        pass = pass && lane;
        failed_stress_lanes += !lane;
    }
    for (int i = 0; i < 6; ++i) {
        pass = pass
               && finite_close(frame.meta.get(kStress)[i],
                               sigma[i] * frame.box.volume(), 1e-10);
        if (i >= 3)
            pass = pass && finite_close(sigma[i], 0, 0);
    }
    std::printf("%s PBE_PERIODIC_CP2K %s %s E_Ha=%.15g dE_Ha=%.9g "
                "dF_Ha_Bohr=%.9g dDiagStress_Ha_Bohr3=%.9g "
                "failed_force_lanes=%d failed_stress_lanes=%d iterations=%d\n",
                pass ? "PASS" : "FAIL", pin.id, backend, result.e_total,
                energy_error, force_error, stress_error, failed_force_lanes,
                failed_stress_lanes, result.iters);
    std::fflush(stdout);
    return pass;
}
} // namespace
int main(int argc, char** argv) {
    std::string backend = "cpu";
#ifdef ATV_HAS_CUDA
    backend = "both";
#endif
    if (argc == 3 && std::string(argv[1]) == "--backend")
        backend = argv[2];
    else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--backend cpu|cuda|both]\n", argv[0]);
        return 2;
    }
    if (backend != "cpu" && backend != "cuda" && backend != "both")
        return 2;
#ifndef ATV_HAS_CUDA
    if (backend != "cpu")
        return 2;
#endif
    bool pass = true;
    for (const auto& pin : pbe_periodic_cp2k::pins) {
        try {
            if (backend != "cuda")
                pass = compare<cpu::gpw::GpwDft>(pin, "CPU") && pass;
#ifdef ATV_HAS_CUDA
            if (backend != "cpu")
                pass = compare<cuda::gpw::GpwDft>(pin, "CUDA") && pass;
#endif
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FAIL PBE_PERIODIC_CP2K %s %s %s\n", pin.id,
                         backend.c_str(), e.what());
            pass = false;
        }
    }
    return pass ? 0 : 1;
}
