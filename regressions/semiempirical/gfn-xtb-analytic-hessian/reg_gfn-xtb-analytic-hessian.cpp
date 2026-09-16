/// @file regressions/semiempirical/gfn-xtb-analytic-hessian/reg_gfn-xtb-analytic-hessian.cpp
/// @brief Water GFN1/GFN2-xTB FD Hessian via `cpu::vib::Hessian` vs xtb 6.7.1 `--hess`.
///
/// oracle: xtb  oracle_version: 6.7.1
/// Public API only. Exit non-zero on failure. <= 60 s.
/// Requires ATV_BUILD_VIB=ON and ATV_BUILD_SEMIEMPIRICAL=ON.
/// Primary pin: GFN1. GFN2 carries the documented AES-in-Fock hole.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <utility>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/gfn1.h"
#include "atomiverse/cpu/semiempirical/gfn2.h"
#include "atomiverse/cpu/vibration/hessian.h"

using namespace ATV;

// Offline xtb 6.7.1 `--gfn 1 --chrg 0 --uhf 0 --acc 0.01 --hess` on mol.xyz
// (tools/tb_oracle/inputs/xtb/h2o_gfn1_hess/).
static constexpr double kGfn1_E_pin_Ha = -5.768641651774;
static constexpr double kGfn1_omega_pin[3] = {1474.96, 3657.78, 3761.56};
static constexpr double kGfn1_H_Oxx_HaBohr2 = 0.5466593694;
static constexpr double kGfn1_E_tol_eV = 5.0e-3;
static constexpr double kGfn1_omega_tol = 10.0;
static constexpr double kGfn1_H_tol_eVA2 = 0.2;
static constexpr double kGfn1_xtb_wall_s = 0.03;

// Offline xtb 6.7.1 `--gfn 2 --chrg 0 --uhf 0 --acc 0.01 --hess` on the same
// 0.96 Å water (tools/tb_oracle/inputs/xtb/h2o_gfn2_hess/). AES is still
// post-SCC in ATV; ω / Cartesian-H tols absorb that Fock hole.
static constexpr double kGfn2_E_pin_Ha = -5.070383095219;
static constexpr double kGfn2_omega_pin[3] = {1565.10, 3634.95, 3643.62};
static constexpr double kGfn2_H_Oxx_HaBohr2 = 0.5210291876;
static constexpr double kGfn2_E_tol_eV = 1.0e-3;
static constexpr double kGfn2_omega_tol = 150.0;
static constexpr double kGfn2_H_tol_eVA2 = 1.0;
static constexpr double kGfn2_xtb_wall_s = 0.02;

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

static int check_hessian_shape(const char* tag,
                               const cpu::vib::HessianResult& res) {
    if (res.n_atoms != 3) {
        std::printf("FAIL %s n_atoms=%d\n", tag, res.n_atoms);
        return 1;
    }
    if (res.hessian.shape(0) != 9 || res.hessian.shape(1) != 9) {
        std::printf("FAIL %s hessian shape\n", tag);
        return 1;
    }
    double max_asym = 0.0;
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            const double a = res.hessian(i, j);
            const double b = res.hessian(j, i);
            if (!std::isfinite(a)) {
                std::printf("FAIL %s non-finite H(%d,%d)\n", tag, i, j);
                return 1;
            }
            max_asym = std::max(max_asym, std::fabs(a - b));
        }
    }
    if (max_asym > 1e-8) {
        std::printf("FAIL %s Hessian asymmetry %.3e eV/A^2\n", tag, max_asym);
        return 1;
    }
    if (res.is_linear) {
        std::printf("FAIL %s H2O classified linear\n", tag);
        return 1;
    }
    if (res.frequencies.size() != 3) {
        std::printf("FAIL %s expected 3 vib modes for H2O, got %zu\n", tag,
                    res.frequencies.size());
        return 1;
    }
    if (res.n_imaginary != 0) {
        std::printf("FAIL %s n_imaginary=%d at near-eq H2O\n", tag,
                    res.n_imaginary);
        return 1;
    }
    std::printf("  %s max|H-H^T|=%.3e  n_imag=%d  linear=%d\n", tag, max_asym,
                res.n_imaginary, static_cast<int>(res.is_linear));
    return 0;
}

static int check_geometry_restored(const char* tag, const cpu::Frame& f) {
    if (std::fabs(static_cast<double>(f.span(kAtom_x)[0])) > 1e-12
        || std::fabs(static_cast<double>(f.span(kAtom_x)[1]) - 0.96) > 1e-12
        || std::fabs(static_cast<double>(f.span(kAtom_y)[2]) - 0.93) > 1e-12) {
        std::printf("FAIL %s geometry not restored after Hessian::run\n", tag);
        return 1;
    }
    return 0;
}

int main() {
    const double a0 = units::kBohrInAngstrom;
    const double hfac = units::kHartreeInEv / (a0 * a0);

    // ── GFN1 (primary pin) ──────────────────────────────────────────
    {
        cpu::Frame f = make_water();
        cpu::semiempirical::GFN1 gfn1;
        gfn1.with_forces = true;
        cpu::vib::Hessian hess(298.15, 101325.0, 1, 1, "", "");
        hess.on(vib::Phase::force).add(std::move(gfn1));
        hess.build();
        const auto t0 = std::chrono::steady_clock::now();
        const cpu::vib::HessianResult res = hess.run(f);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (check_hessian_shape("GFN1", res))
            return 1;
        if (check_geometry_restored("GFN1", f))
            return 1;

        f.meta.get(kEnergy) = 0;
        for (int i = 0; i < 3; ++i)
            f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
        cpu::Evaluate ev;
        cpu::semiempirical::GFN1 eonly;
        ev.on(evaluate::Phase::evaluate).add(std::move(eonly));
        ev.build();
        ev.run(f);

        const double h_oxx_pin = kGfn1_H_Oxx_HaBohr2 * hfac;
        const double h_oxx = res.hessian(0, 0);
        const double e_ev = f.meta.get(kEnergy);
        const double e_ha = e_ev / units::kHartreeInEv;
        const double pin_ev = kGfn1_E_pin_Ha * units::kHartreeInEv;

        std::printf(
            "GFN1 H2O Hessian: E_ATV=%.12f Eh  E_pin=%.12f Eh  |dE|=%.3e eV\n",
            e_ha, kGfn1_E_pin_Ha, std::fabs(e_ev - pin_ev));
        std::printf("  H_Oxx=%.12f eV/A^2  pin=%.12f  |d|=%.3e\n", h_oxx,
                    h_oxx_pin, std::fabs(h_oxx - h_oxx_pin));
        for (int k = 0; k < 3; ++k) {
            const double w = res.frequencies(k);
            std::printf("  omega[%d]=%.4f cm^-1  pin=%.2f  |d|=%.3e\n", k, w,
                        kGfn1_omega_pin[k], std::fabs(w - kGfn1_omega_pin[k]));
            if (!(w > 100.0) || !std::isfinite(w)) {
                std::printf("FAIL GFN1 frequency not positive/physical: %g\n",
                            w);
                return 1;
            }
            if (std::fabs(w - kGfn1_omega_pin[k]) > kGfn1_omega_tol) {
                std::printf("FAIL GFN1 omega[%d] vs xtb (tol %.1f cm^-1)\n", k,
                            kGfn1_omega_tol);
                return 1;
            }
        }
        if (!std::isfinite(e_ev) || std::fabs(e_ev - pin_ev) > kGfn1_E_tol_eV) {
            std::printf("FAIL GFN1 energy pin\n");
            return 1;
        }
        if (std::fabs(h_oxx - h_oxx_pin) > kGfn1_H_tol_eVA2) {
            std::printf("FAIL GFN1 H_Oxx vs xtb (tol %.1f eV/A^2)\n",
                        kGfn1_H_tol_eVA2);
            return 1;
        }
        const double ratio = (ms / 1000.0) / kGfn1_xtb_wall_s;
        std::printf("  wall=%.3f ms  xtb=%.3f s  ratio=%.3f\n", ms,
                    kGfn1_xtb_wall_s, ratio);
        if (ratio > 10.0) {
            std::printf("FAIL GFN1 perf ratio %.3f > 10\n", ratio);
            return 1;
        }
    }

    // ── GFN2 (family close; AES still post-SCC) ─────────────────────
    {
        cpu::Frame f = make_water();
        cpu::semiempirical::GFN2 gfn2;
        gfn2.with_forces = true;
        cpu::vib::Hessian hess(298.15, 101325.0, 1, 1, "", "");
        hess.on(vib::Phase::force).add(std::move(gfn2));
        hess.build();
        const auto t0 = std::chrono::steady_clock::now();
        const cpu::vib::HessianResult res = hess.run(f);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (check_hessian_shape("GFN2", res))
            return 1;
        if (check_geometry_restored("GFN2", f))
            return 1;

        f.meta.get(kEnergy) = 0;
        for (int i = 0; i < 3; ++i)
            f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
        cpu::Evaluate ev;
        cpu::semiempirical::GFN2 eonly;
        ev.on(evaluate::Phase::evaluate).add(std::move(eonly));
        ev.build();
        ev.run(f);

        const double h_oxx_pin = kGfn2_H_Oxx_HaBohr2 * hfac;
        const double h_oxx = res.hessian(0, 0);
        const double e_ev = f.meta.get(kEnergy);
        const double e_ha = e_ev / units::kHartreeInEv;
        const double pin_ev = kGfn2_E_pin_Ha * units::kHartreeInEv;

        std::printf(
            "GFN2 H2O Hessian: E_ATV=%.12f Eh  E_pin=%.12f Eh  |dE|=%.3e eV\n",
            e_ha, kGfn2_E_pin_Ha, std::fabs(e_ev - pin_ev));
        std::printf("  H_Oxx=%.12f eV/A^2  pin=%.12f  |d|=%.3e\n", h_oxx,
                    h_oxx_pin, std::fabs(h_oxx - h_oxx_pin));
        for (int k = 0; k < 3; ++k) {
            const double w = res.frequencies(k);
            std::printf("  omega[%d]=%.4f cm^-1  pin=%.2f  |d|=%.3e\n", k, w,
                        kGfn2_omega_pin[k], std::fabs(w - kGfn2_omega_pin[k]));
            if (!(w > 100.0) || !std::isfinite(w)) {
                std::printf("FAIL GFN2 frequency not positive/physical: %g\n",
                            w);
                return 1;
            }
            if (std::fabs(w - kGfn2_omega_pin[k]) > kGfn2_omega_tol) {
                std::printf("FAIL GFN2 omega[%d] vs xtb (tol %.1f cm^-1)\n", k,
                            kGfn2_omega_tol);
                return 1;
            }
        }
        if (!std::isfinite(e_ev) || std::fabs(e_ev - pin_ev) > kGfn2_E_tol_eV) {
            std::printf("FAIL GFN2 energy pin\n");
            return 1;
        }
        if (std::fabs(h_oxx - h_oxx_pin) > kGfn2_H_tol_eVA2) {
            std::printf("FAIL GFN2 H_Oxx vs xtb (tol %.1f eV/A^2)\n",
                        kGfn2_H_tol_eVA2);
            return 1;
        }
        const double ratio = (ms / 1000.0) / kGfn2_xtb_wall_s;
        std::printf("  wall=%.3f ms  xtb=%.3f s  ratio=%.3f\n", ms,
                    kGfn2_xtb_wall_s, ratio);
        if (ratio > 10.0) {
            std::printf("FAIL GFN2 perf ratio %.3f > 10\n", ratio);
            return 1;
        }
    }

    std::printf("PASS\n");
    return 0;
}
