/// @file regressions/semiempirical/dftb-plus-u/reg_dftb-plus-u.cpp
/// @brief DFTB2+U: Dudarev E_U identity + analytic FD forces vs energy FD.
///
/// Domain: Dudarev et al., Phys. Rev. B 57, 1505 (1998); design
/// `.claude/notes/dftb-plus-u.md`. Driven via `cpu::Evaluate` only (Frame is
/// the sole I/O token). Exit non-zero on failure. Public API only. <= 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

cpu::Frame make_water() {
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 0.96;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_x)[2] = -0.24;
    f.span(kAtom_y)[2] = 0.93;
    f.span(kAtom_z)[2] = 0.0;
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_mass)[i] = (i == 0) ? 15.999 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

/// @brief Single-point DFTB2(+U) energy via Evaluate (Frame only).
double energy_only(cpu::Frame g, double u_o) {
    cpu::semiempirical::DFTB2 d;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    d.boundary = cpu::semiempirical::DftbBoundary::cluster;
    d.with_forces = false;
    d.scc_tol = 1e-10;
    if (u_o > 0.0)
        d.u_eff[8] = u_o;
    cpu::Evaluate eval;
    eval.set_evaluator_ref(d);
    eval.build();
    eval.run(g);
    return d.last_energy();
}

} // namespace

int main() {
    constexpr double kU = 4.0;
    constexpr double kTol = 1e-9;

    auto f0 = make_water();
    cpu::semiempirical::DFTB2 plain;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    plain.boundary = cpu::semiempirical::DftbBoundary::cluster;
    plain.with_forces = false;
    plain.scc_tol = 1e-10;
    {
        cpu::Evaluate eval;
        eval.set_evaluator_ref(plain);
        eval.build();
        eval.run(f0);
    }
    const double e0 = plain.last_energy();

    auto f = make_water();
    cpu::semiempirical::DFTB2 dftb;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    dftb.boundary = cpu::semiempirical::DftbBoundary::cluster;
    dftb.with_forces = true;
    dftb.scc_tol = 1e-10;
    dftb.u_eff[8] = kU;
    {
        cpu::Evaluate eval;
        eval.set_evaluator_ref(dftb);
        eval.build();
        eval.run(f);
    }
    const double e_tot = dftb.last_energy();
    const double e_u = dftb.last_e_u();
    const auto& dq = dftb.last_mulliken_dq();
    if (dq.size() != 3) {
        std::printf("FAIL dq size\n");
        return 1;
    }
    const double n_o = 6.0 - dq[0];
    const double N_o = 2.0 * 4.0;
    const double e_u_formula = 0.5 * kU * n_o * (1.0 - n_o / N_o);

    std::printf(
        "DFTB2+U H2O: E0=%.12f  E(U)=%.12f  E_U=%.12f  formula=%.12f eV\n", e0,
        e_tot, e_u, e_u_formula);

    if (!std::isfinite(e0) || !std::isfinite(e_tot) || !std::isfinite(e_u)) {
        std::printf("FAIL non-finite energies\n");
        return 1;
    }
    if (std::fabs(e_u - e_u_formula) > kTol) {
        std::printf("FAIL E_U vs formula |d|=%.3e\n",
                    std::fabs(e_u - e_u_formula));
        return 1;
    }
    if (std::fabs((e_tot - e0) - e_u) > 1e-8) {
        std::printf("FAIL E(U)-E0 != E_U  d=%.3e\n", (e_tot - e0) - e_u);
        return 1;
    }
    if (std::fabs(e_u) < 1e-6) {
        std::printf("FAIL E_U unexpectedly ~0\n");
        return 1;
    }

    // FD force gate: F_O_x (from Evaluate+with_forces) vs −∂E/∂x_O on the
    // same +U surface (energy-only Evaluate at ±h).
    constexpr double h = 1e-4;
    auto fp = make_water();
    auto fm = make_water();
    fp.span(kAtom_x)[0] =
        static_cast<StorageReal>(static_cast<double>(fp.span(kAtom_x)[0]) + h);
    fm.span(kAtom_x)[0] =
        static_cast<StorageReal>(static_cast<double>(fm.span(kAtom_x)[0]) - h);
    const double fd =
        -(energy_only(std::move(fp), kU) - energy_only(std::move(fm), kU))
        / (2.0 * h);
    const double fox = static_cast<double>(f.span(kAtom_fx)[0]);
    std::printf("  F_O_x analytic/FD-path=%.12f  fd=%.12f  |d|=%.3e eV/A\n",
                fox, fd, std::fabs(fox - fd));
    if (std::fabs(fox - fd) > 5e-4) {
        std::printf("FAIL force vs energy FD (tol 5e-4 eV/A)\n");
        return 1;
    }
    if (std::fabs(fox) < 1e-4) {
        std::printf("FAIL F_O_x ~ 0\n");
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
