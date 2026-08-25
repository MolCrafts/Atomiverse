/// @file regressions/semiempirical/ipea-xtb-analytic-hessian/reg_ipea-xtb-analytic-hessian.cpp
/// @brief CH4 IPEA-xTB FD Hessian via `cpu::vib::Hessian` vs xtb 6.7.1 `--hess`.
///
/// oracle: xtb  oracle_version: 6.7.1
/// Public API only. Exit non-zero on failure. <= 60 s.
/// Requires ATV_BUILD_VIB=ON and ATV_BUILD_SEMIEMPIRICAL=ON.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/ipea.h"
#include "atomiverse/cpu/vibration/hessian.h"

using namespace ATV;

// Offline xtb 6.7.1 IPEA `--chrg 0 --uhf 0 --acc 0.01 --hess` on mol.xyz
// (tools/tb_oracle/inputs/xtb/ch4_ipea_hess/; param_ipea as GFN1).
static constexpr double kE_pin_Ha = -4.670476737294;
static constexpr double k_omega_pin[9] = {1541.68, 1541.68, 1541.68,
                                          1705.89, 1705.89, 3548.78,
                                          3576.13, 3576.13, 3576.13};
static constexpr double kH_Cxx_HaBohr2 = 0.6813937129;
static constexpr double kE_tol_eV = 5.0e-2;
static constexpr double k_omega_tol = 10.0;
static constexpr double kH_tol_eVA2 = 0.2;
static constexpr double k_xtb_wall_s = 0.17;

static cpu::Frame make_ch4() {
    const double s = 1.09 / std::sqrt(3.0);
    cpu::Frame f(5);
    const double x[5] = {0, s, s, -s, -s};
    const double y[5] = {0, s, -s, s, -s};
    const double z[5] = {0, s, -s, -s, s};
    const int Z[5] = {6, 1, 1, 1, 1};
    for (int i = 0; i < 5; ++i) {
        f.span(kAtom_type)[i] = Z[i];
        f.span(kAtom_x)[i] = x[i];
        f.span(kAtom_y)[i] = y[i];
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_mass)[i] = (i == 0) ? 12.011 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

int main() {
    const double a0 = units::kBohrInAngstrom;
    const double hfac = units::kHartreeInEv / (a0 * a0);
    const double s = 1.09 / std::sqrt(3.0);

    cpu::Frame f = make_ch4();
    cpu::semiempirical::IPEA ipea;
    ipea.with_forces = true;
    cpu::vib::Hessian hess(298.15, 101325.0, 1, 1, "", "");
    hess.on(vib::Phase::force).add(std::move(ipea));
    hess.build();
    const auto t0 = std::chrono::steady_clock::now();
    const cpu::vib::HessianResult res = hess.run(f);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (res.n_atoms != 5) {
        std::printf("FAIL n_atoms=%d\n", res.n_atoms);
        return 1;
    }
    if (res.hessian.shape(0) != 15 || res.hessian.shape(1) != 15) {
        std::printf("FAIL hessian shape\n");
        return 1;
    }
    double max_asym = 0.0;
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 15; ++j) {
            const double a = res.hessian(i, j);
            const double b = res.hessian(j, i);
            if (!std::isfinite(a)) {
                std::printf("FAIL non-finite H(%d,%d)\n", i, j);
                return 1;
            }
            max_asym = std::max(max_asym, std::fabs(a - b));
        }
    }
    if (max_asym > 1e-8) {
        std::printf("FAIL Hessian asymmetry %.3e eV/A^2\n", max_asym);
        return 1;
    }
    if (res.is_linear) {
        std::printf("FAIL CH4 classified linear\n");
        return 1;
    }
    if (res.frequencies.size() != 9) {
        std::printf("FAIL expected 9 vib modes for CH4, got %zu\n",
                    res.frequencies.size());
        return 1;
    }
    if (res.n_imaginary != 0) {
        std::printf("FAIL n_imaginary=%d at Td CH4\n", res.n_imaginary);
        return 1;
    }
    std::printf("  max|H-H^T|=%.3e  n_imag=%d  linear=%d\n", max_asym,
                res.n_imaginary, static_cast<int>(res.is_linear));

    if (std::fabs(static_cast<double>(f.span(kAtom_x)[0])) > 1e-12
        || std::fabs(static_cast<double>(f.span(kAtom_x)[1]) - s) > 1e-12
        || std::fabs(static_cast<double>(f.span(kAtom_z)[4]) - s) > 1e-12) {
        std::printf("FAIL geometry not restored after Hessian::run\n");
        return 1;
    }

    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < 5; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    cpu::Evaluate ev;
    cpu::semiempirical::IPEA eonly;
    ev.on(evaluate::Phase::evaluate).add(std::move(eonly));
    ev.build();
    ev.run(f);

    const double h_cxx_pin = kH_Cxx_HaBohr2 * hfac;
    const double h_cxx = res.hessian(0, 0);
    const double e_ev = f.meta.get(kEnergy);
    const double e_ha = e_ev / units::kHartreeInEv;
    const double pin_ev = kE_pin_Ha * units::kHartreeInEv;

    std::printf(
        "IPEA CH4 Hessian: E_ATV=%.12f Eh  E_pin=%.12f Eh  |dE|=%.3e eV\n",
        e_ha, kE_pin_Ha, std::fabs(e_ev - pin_ev));
    std::printf("  H_Cxx=%.12f eV/A^2  pin=%.12f  |d|=%.3e\n", h_cxx, h_cxx_pin,
                std::fabs(h_cxx - h_cxx_pin));
    for (int k = 0; k < 9; ++k) {
        const double w = res.frequencies(k);
        std::printf("  omega[%d]=%.4f cm^-1  pin=%.2f  |d|=%.3e\n", k, w,
                    k_omega_pin[k], std::fabs(w - k_omega_pin[k]));
        if (!(w > 100.0) || !std::isfinite(w)) {
            std::printf("FAIL frequency not positive/physical: %g\n", w);
            return 1;
        }
        if (std::fabs(w - k_omega_pin[k]) > k_omega_tol) {
            std::printf("FAIL omega[%d] vs xtb (tol %.1f cm^-1)\n", k,
                        k_omega_tol);
            return 1;
        }
    }
    if (!std::isfinite(e_ev) || std::fabs(e_ev - pin_ev) > kE_tol_eV) {
        std::printf("FAIL energy pin\n");
        return 1;
    }
    if (std::fabs(h_cxx - h_cxx_pin) > kH_tol_eVA2) {
        std::printf("FAIL H_Cxx vs xtb (tol %.1f eV/A^2)\n", kH_tol_eVA2);
        return 1;
    }
    const double ratio = (ms / 1000.0) / k_xtb_wall_s;
    std::printf("  wall=%.3f ms  xtb=%.3f s  ratio=%.3f\n", ms, k_xtb_wall_s,
                ratio);
    if (ratio > 10.0) {
        std::printf("FAIL perf ratio %.3f > 10\n", ratio);
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
