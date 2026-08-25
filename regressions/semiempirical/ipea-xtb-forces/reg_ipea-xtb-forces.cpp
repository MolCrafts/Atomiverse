/// @file regressions/semiempirical/ipea-xtb-forces/reg_ipea-xtb-forces.cpp
/// @brief Off-eq CH4 IPEA-xTB analytic forces vs offline xtb 6.7.1 IPEA `--grad`.
///
/// oracle: xtb  oracle_version: 6.7.1
/// Public API only. Exit non-zero on failure. <= 60 s.
#include <chrono>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/ipea.h"

using namespace ATV;

// Offline xtb 6.7.1 IPEA (`param_ipea` as GFN1) `--chrg 0 --uhf 0 --acc 0.01
// --grad` on mol.xyz (tools/tb_oracle/inputs/xtb/ch4_ipea_offeq_grad/).
// Tetrahedral CH4 with one C–H stretched 1.09 → 1.20 Å.
static constexpr double kE_pin_Ha = -4.662477307157;
static constexpr double kE_tol_eV = 5.0e-2; // same floor as reg_ipea-xtb-small
static constexpr double kG_C_x_HaBohr = -3.5026444745746e-2;
static constexpr double kG_H1_x_HaBohr = 4.0716066407528e-2;
static constexpr double kF_tol_eVA = 5.0e-3;
static constexpr double kXtB_wall_s = 0.004;
static constexpr int kNat = 5;

static cpu::Frame make_ch4() {
    cpu::Frame f(kNat);
    const double s = 0.629311793417;
    const double h1 = s * (1.20 / 1.09);
    const double xyz[kNat][3] = {
        {0.0, 0.0, 0.0},
        {h1, h1, h1},
        {s, -s, -s},
        {-s, s, -s},
        {-s, -s, s},
    };
    const int Z[kNat] = {6, 1, 1, 1, 1};
    const double mass[kNat] = {12.011, 1.008, 1.008, 1.008, 1.008};
    for (int i = 0; i < kNat; ++i) {
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

static double energy_only(cpu::Frame g) {
    g.meta.get(kEnergy) = 0;
    for (int i = 0; i < kNat; ++i)
        g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
    cpu::Evaluate ev;
    cpu::semiempirical::IPEA m;
    m.with_forces = false;
    ev.on(evaluate::Phase::evaluate).add(std::move(m));
    ev.build();
    ev.run(g);
    return g.meta.get(kEnergy);
}

static double fd_force(int atom, int k, double h) {
    auto shifted = [&](double s) {
        cpu::Frame g = make_ch4();
        StorageReal* col = (k == 0)   ? g.span(kAtom_x)
                           : (k == 1) ? g.span(kAtom_y)
                                      : g.span(kAtom_z);
        col[atom] += static_cast<StorageReal>(s);
        return energy_only(g);
    };
    return -(shifted(h) - shifted(-h)) / (2.0 * h);
}

int main() {
    cpu::Frame f = make_ch4();
    cpu::semiempirical::IPEA ipea;
    ipea.with_forces = true;
    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(ipea);
    eval.build();
    const auto t0 = std::chrono::steady_clock::now();
    eval.run(f);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    const double e_ev = f.meta.get(kEnergy);
    const double e_ha = e_ev / units::kHartreeInEv;
    const double pin_ev = kE_pin_Ha * units::kHartreeInEv;
    const double fac = units::kHartreePerBohrInEvPerAngstrom;
    const double fCx_pin = -kG_C_x_HaBohr * fac;
    const double fH1x_pin = -kG_H1_x_HaBohr * fac;
    const double fCx = static_cast<double>(f.span(kAtom_fx)[0]);
    const double fH1x = static_cast<double>(f.span(kAtom_fx)[1]);

    std::printf("IPEA CH4 forces: E_ATV=%.12f Eh  E_pin=%.12f Eh  |dE|=%.3e eV\n",
                e_ha, kE_pin_Ha, std::fabs(e_ev - pin_ev));
    std::printf("  F_C_x  =%.12f eV/A  pin=%.12f  |d|=%.3e\n", fCx, fCx_pin,
                std::fabs(fCx - fCx_pin));
    std::printf("  F_H1_x =%.12f eV/A  pin=%.12f  |d|=%.3e\n", fH1x, fH1x_pin,
                std::fabs(fH1x - fH1x_pin));
    std::printf("  wall=%.3f ms  xtb=%.3f s  ratio=%.3f\n", ms, kXtB_wall_s,
                (ms / 1000.0) / kXtB_wall_s);

    if (!std::isfinite(e_ev) || std::fabs(e_ev - pin_ev) > kE_tol_eV) {
        std::printf("FAIL energy pin\n");
        return 1;
    }

    double net[3] = {0, 0, 0};
    for (int i = 0; i < kNat; ++i) {
        net[0] += static_cast<double>(f.span(kAtom_fx)[i]);
        net[1] += static_cast<double>(f.span(kAtom_fy)[i]);
        net[2] += static_cast<double>(f.span(kAtom_fz)[i]);
        std::printf("  F[%d]=(%.12f, %.12f, %.12f) eV/A\n", i,
                    static_cast<double>(f.span(kAtom_fx)[i]),
                    static_cast<double>(f.span(kAtom_fy)[i]),
                    static_cast<double>(f.span(kAtom_fz)[i]));
    }
    for (int d = 0; d < 3; ++d) {
        if (std::fabs(net[d]) > 1e-9) {
            std::printf("FAIL net force[%d]=%.3e eV/A\n", d, net[d]);
            return 1;
        }
    }

    if (std::fabs(fCx - fCx_pin) > kF_tol_eVA
        || std::fabs(fH1x - fH1x_pin) > kF_tol_eVA) {
        std::printf("FAIL xtb force pin (tol %.1e eV/A)\n", kF_tol_eVA);
        return 1;
    }

    constexpr double h = 1.0e-4;
    constexpr double fd_tol = 3e-5;
    const double fd_cx = fd_force(0, 0, h);
    const double fd_h1x = fd_force(1, 0, h);
    std::printf("  FD F_C_x  =%.12f  |analytic-FD|=%.3e\n", fd_cx,
                std::fabs(fCx - fd_cx));
    std::printf("  FD F_H1_x =%.12f  |analytic-FD|=%.3e\n", fd_h1x,
                std::fabs(fH1x - fd_h1x));
    if (std::fabs(fCx - fd_cx) > fd_tol || std::fabs(fH1x - fd_h1x) > fd_tol) {
        std::printf("FAIL analytic vs energy FD (tol %.1e eV/A)\n", fd_tol);
        return 1;
    }

    const double ratio = (ms / 1000.0) / kXtB_wall_s;
    if (ratio > 10.0) {
        std::printf("FAIL perf ratio %.3f > 10\n", ratio);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
