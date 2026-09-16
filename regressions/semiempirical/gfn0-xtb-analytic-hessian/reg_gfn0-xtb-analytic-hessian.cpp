/// @file regressions/semiempirical/gfn0-xtb-analytic-hessian/reg_gfn0-xtb-analytic-hessian.cpp
/// @brief Water GFN0-xTB FD Hessian via `cpu::vib::Hessian` vs xtb 6.7.1 `--hess`.
///
/// oracle: xtb  oracle_version: 6.7.1
/// Public API only. Exit non-zero on failure. <= 60 s.
/// Requires ATV_BUILD_VIB=ON and ATV_BUILD_SEMIEMPIRICAL=ON.
#include <chrono>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/basis_data/gfn0_tables.h"
#include "atomiverse/cpu/semiempirical/gfn0.h"
#include "atomiverse/cpu/vibration/hessian.h"

using namespace ATV;

// Offline xtb 6.7.1 `--gfn 0 --chrg 0 --uhf 0 --acc 1.0 --hess` on mol.xyz
// (tools/tb_oracle/inputs/xtb/h2o_gfn0_hess/).
static constexpr double kE_pin_Ha = -4.366816412151;
static constexpr double kE_tol_eV = 1.0e-3;
static constexpr double kOmega_pin[3] = {1563.14, 3497.87, 3505.03};
static constexpr double kOmega_tol = 10.0;
static constexpr double kH_Oxx_HaBohr2 = 0.4793058695;
static constexpr double kH_tol_eVA2 = 0.2;
static constexpr double kXtB_wall_s = 0.02;

static cpu::Frame make_water() {
    cpu::Frame f(3);
    const double xyz[3][3] = {
        {0.00, 0.00, 0.00},
        {0.96, 0.00, 0.00},
        {-0.24, 0.93, 0.00},
    };
    const int Z[3] = {8, 1, 1};
    const double mass[3] = {15.999, 1.008, 1.008};
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_type)[i] = Z[i];
        f.span(kAtom_x)[i] = xyz[i][0];
        f.span(kAtom_y)[i] = xyz[i][1];
        f.span(kAtom_z)[i] = xyz[i][2];
        f.span(kAtom_mass)[i] = mass[i];
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

int main() {
    cpu::Frame f = make_water();

    cpu::semiempirical::GFN0 gfn0;
    gfn0.with_forces = true;

    // Empty recorder paths: do not litter CWD with frequencies.tsv / mode_*.xyz.
    cpu::vib::Hessian hess(298.15, 101325.0, 1, 1, "", "");
    hess.on(vib::Phase::force).add(std::move(gfn0));
    hess.build();
    const auto t0 = std::chrono::steady_clock::now();
    const cpu::vib::HessianResult res = hess.run(f);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (res.n_atoms != 3) {
        std::printf("FAIL n_atoms=%d\n", res.n_atoms);
        return 1;
    }
    if (res.hessian.shape(0) != 9 || res.hessian.shape(1) != 9) {
        std::printf("FAIL hessian shape\n");
        return 1;
    }

    double max_asym = 0.0;
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
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
        std::printf("FAIL H2O classified linear\n");
        return 1;
    }
    if (res.frequencies.size() != 3) {
        std::printf("FAIL expected 3 vib modes for H2O, got %zu\n",
                    res.frequencies.size());
        return 1;
    }
    if (res.n_imaginary != 0) {
        std::printf("FAIL n_imaginary=%d at near-eq H2O\n", res.n_imaginary);
        return 1;
    }

    // Hessian::run restores R0; leftover kEnergy is from the last displacement.
    if (std::fabs(static_cast<double>(f.span(kAtom_x)[0])) > 1e-12
        || std::fabs(static_cast<double>(f.span(kAtom_x)[1]) - 0.96) > 1e-12
        || std::fabs(static_cast<double>(f.span(kAtom_y)[2]) - 0.93) > 1e-12) {
        std::printf("FAIL geometry not restored after Hessian::run\n");
        return 1;
    }
    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < 3; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    cpu::Evaluate ev;
    cpu::semiempirical::GFN0 eonly;
    ev.on(evaluate::Phase::evaluate).add(std::move(eonly));
    ev.build();
    ev.run(f);

    const double a0 = cpu::semiempirical::basis_data::kGfn0AutoAA;
    const double hfac = units::kHartreeInEv / (a0 * a0);
    const double h_oxx_pin = kH_Oxx_HaBohr2 * hfac;
    const double h_oxx = res.hessian(0, 0);
    const double e_ev = f.meta.get(kEnergy);
    const double e_ha = e_ev / units::kHartreeInEv;
    const double pin_ev = kE_pin_Ha * units::kHartreeInEv;

    std::printf(
        "GFN0 H2O Hessian: E_ATV=%.12f Eh  E_pin=%.12f Eh  |dE|=%.3e eV\n",
        e_ha, kE_pin_Ha, std::fabs(e_ev - pin_ev));
    std::printf("  H_Oxx=%.12f eV/A^2  pin=%.12f  |d|=%.3e\n", h_oxx, h_oxx_pin,
                std::fabs(h_oxx - h_oxx_pin));
    std::printf("  max|H-H^T|=%.3e  n_imag=%d  linear=%d\n", max_asym,
                res.n_imaginary, static_cast<int>(res.is_linear));
    for (int k = 0; k < 3; ++k) {
        const double w = res.frequencies(k);
        std::printf("  omega[%d]=%.4f cm^-1  pin=%.2f  |d|=%.3e\n", k, w,
                    kOmega_pin[k], std::fabs(w - kOmega_pin[k]));
        if (!(w > 100.0) || !std::isfinite(w)) {
            std::printf("FAIL frequency not positive/physical: %g\n", w);
            return 1;
        }
        if (std::fabs(w - kOmega_pin[k]) > kOmega_tol) {
            std::printf("FAIL omega[%d] vs xtb (tol %.1f cm^-1)\n", k,
                        kOmega_tol);
            return 1;
        }
    }

    if (!std::isfinite(e_ev) || std::fabs(e_ev - pin_ev) > kE_tol_eV) {
        std::printf("FAIL energy pin\n");
        return 1;
    }
    if (std::fabs(h_oxx - h_oxx_pin) > kH_tol_eVA2) {
        std::printf("FAIL H_Oxx vs xtb (tol %.1f eV/A^2)\n", kH_tol_eVA2);
        return 1;
    }

    const double ratio = (ms / 1000.0) / kXtB_wall_s;
    std::printf("  wall=%.3f ms  xtb=%.3f s  ratio=%.3f\n", ms, kXtB_wall_s,
                ratio);
    if (ratio > 10.0) {
        std::printf("FAIL perf ratio %.3f > 10\n", ratio);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
