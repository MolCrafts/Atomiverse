// Public GPW results compared with actual independent CP2K output.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "uks_fixture.h"
#include "uks_pins.h"
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
bool finite_close(double a, double b, double tol) {
    return std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tol;
}
template <class Method>
bool compare(const pbe_periodic_cp2k_uks::Pin& pin, const char* backend) {
    constexpr cpu::xc::XCType types[] = {
        cpu::xc::XCType::PBE, cpu::xc::XCType::PBEsol, cpu::xc::XCType::RevPBE,
        cpu::xc::XCType::RPBE};
    auto frame = pbe_periodic_uks_fixture::h3();
    Method method{pbe_periodic_uks_fixture::kinds(),
                  pbe_periodic_uks_fixture::cutoff_ry, types[pin.functional]};
    pbe_periodic_uks_fixture::configure(method);
    method.build(frame);
    const auto result = method.forces(frame);
    if (!result.converged)
        throw std::runtime_error("public GPW force SCF did not converge");
    if (result.P.size() == 0 || result.P_beta.size() == 0
        || result.S.size() == 0 || result.P.shape() != result.S.shape()
        || result.P_beta.shape() != result.S.shape()
        || result.S.shape()[0] != result.S.shape()[1])
        throw std::runtime_error(
            "converged Gamma UKS P/Pbeta/S payload missing or inconsistent");
    double alpha = 0, beta = 0;
    for (std::size_t i = 0; i < result.S.shape()[0]; ++i)
        for (std::size_t j = 0; j < result.S.shape()[1]; ++j) {
            alpha += result.P(i, j) * result.S(j, i);
            beta += result.P_beta(i, j) * result.S(j, i);
        }
    const double spin = alpha - beta;
    if (!finite_close(alpha, 2., 1e-8) || !finite_close(beta, 1., 1e-8)
        || result.e_entropy != 0 || method.n_alpha != 2 || method.n_beta != 1)
        throw std::runtime_error(
            "Gamma UKS integer Tr(Palpha S)=2/Tr(Pbeta S)=1 contract failed");
    if (!result.mulliken_spin.empty()) {
        double mulliken = 0;
        for (double moment : result.mulliken_spin)
            mulliken += moment;
        if (result.mulliken_spin.size() != 3
            || !finite_close(mulliken, spin, 1e-8))
            throw std::runtime_error("optional Mulliken populations disagree "
                                     "with overlap spin-density traces");
    }
    const double energy_error = std::abs(result.e_total - pin.energy_ha);
    bool pass = finite_close(result.e_total, pin.energy_ha,
                             pbe_periodic_cp2k_uks::energy_tolerance_ha);
    double force_error = 0, stress_error = 0;
    int failed_force_lanes = 0, failed_stress_lanes = 0;
    for (int atom = 0; atom < 3; ++atom) {
        const double f[] = {frame.span(kAtom_fx)[atom],
                            frame.span(kAtom_fy)[atom],
                            frame.span(kAtom_fz)[atom]};
        std::printf("GPW_CP2K_UKS_FORCE %s %s atom=%d Fx_Ha_Bohr=%.15g "
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
                             pbe_periodic_cp2k_uks::force_tolerance_ha_bohr);
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
    std::printf("GPW_CP2K_UKS_DIAG_STRESS %s %s xx_Ha_Bohr3=%.15g "
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
        const bool lane = finite_close(
            got, expected,
            pbe_periodic_cp2k_uks::diagonal_stress_tolerance_ha_bohr3);
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
    std::printf("%s PBE_PERIODIC_CP2K_UKS %s %s E_Ha=%.15g dE_Ha=%.9g "
                "dF_Ha_Bohr=%.9g dDiagStress_Ha_Bohr3=%.9g "
                "failed_force_lanes=%d failed_stress_lanes=%d iterations=%d "
                "n_alpha=%.12g n_beta=%.12g spin=%.12g e_entropy_Ha=%.12g\n",
                pass ? "PASS" : "FAIL", pin.id, backend, result.e_total,
                energy_error, force_error, stress_error, failed_force_lanes,
                failed_stress_lanes, result.iters, alpha, beta, spin,
                result.e_entropy);
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
    for (const auto& pin : pbe_periodic_cp2k_uks::pins) {
        try {
            if (backend != "cuda")
                pass = compare<cpu::gpw::GpwDft>(pin, "CPU") && pass;
#ifdef ATV_HAS_CUDA
            if (backend != "cpu")
                pass = compare<cuda::gpw::GpwDft>(pin, "CUDA") && pass;
#endif
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FAIL PBE_PERIODIC_CP2K_UKS %s %s %s\n",
                         pin.id, backend.c_str(), e.what());
            pass = false;
        }
    }
    return pass ? 0 : 1;
}
