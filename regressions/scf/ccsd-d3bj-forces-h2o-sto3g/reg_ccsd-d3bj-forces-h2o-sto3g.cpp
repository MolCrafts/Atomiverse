/// @file regressions/scf/ccsd-d3bj-forces-h2o-sto3g/reg_ccsd-d3bj-forces-h2o-sto3g.cpp
/// @brief Water-dimer / STO-3G RHF-CCSD + D3(BJ): dispersion leg of
///        `ccsd_forces` matches bare D3 forces (wiring gate).
///
/// Electronic CCSD nuclear gradients are already FD of the bare energy
/// (`ccsd-grad-h2o-sto3g`). This case pins the product composition:
///
///   F(d3bj=true) − F(d3bj=false) = F_D3BJ
///
/// on a water dimer so |F_D3| is material. Public API only. ≤ 60 s.

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/dftd3.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

using namespace ATV;

cpu::Frame water_dimer() {
    cpu::Frame f(6);
    const double xyz[6][3] = {
        {0.000, 0.000, 0.000}, {0.757, 0.000, 0.586}, {-0.757, 0.000, 0.586},
        {0.000, 0.000, 2.900}, {0.757, 0.000, 3.486}, {-0.757, 0.000, 3.486},
    };
    const int Z[6] = {8, 1, 1, 8, 1, 1};
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_type)[i] = Z[i];
        f.span(kAtom_x)[i] = static_cast<StorageReal>(xyz[i][0]);
        f.span(kAtom_y)[i] = static_cast<StorageReal>(xyz[i][1]);
        f.span(kAtom_z)[i] = static_cast<StorageReal>(xyz[i][2]);
        f.span(kAtom_mass)[i] = (Z[i] == 8) ? 15.999 : 1.008;
        f.span(kAtom_tag)[i] = static_cast<TagInt>(i + 1);
    }
    f.box = cpu::Box::cubic(20.0);
    return f;
}

void configure(cpu::scf::RHF& rhf, bool d3) {
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-10;
    rhf.max_iter = 120;
    rhf.d3bj = d3;
}

} // namespace

int main() {
    using namespace ATV;

    const cpu::Frame frame = water_dimer();
    const int n = frame.nlocal();

    cpu::scf::RHF rhf_off;
    configure(rhf_off, /*d3=*/false);
    cpu::Frame f_off = frame;
    const auto g_off = rhf_off.ccsd_forces(f_off);

    cpu::scf::RHF rhf_on;
    configure(rhf_on, /*d3=*/true);
    cpu::Frame f_on = frame;
    const auto g_on = rhf_on.ccsd_forces(f_on);

    const auto f_d3 = cpu::scf::D3BJ{}.forces(frame); // Ha/Bohr
    double d3_max = 0.0;
    double max_err = 0.0;
    double net[3] = {0.0, 0.0, 0.0};
    for (int A = 0; A < n; ++A) {
        for (int d = 0; d < 3; ++d) {
            const double f_disp_ev = units::convert(
                f_d3[static_cast<std::size_t>(A)][static_cast<std::size_t>(d)],
                units::Force::HartreePerBohr, units::Force::eVPerAngstrom);
            d3_max = std::max(d3_max, std::fabs(f_disp_ev));
            const double df = g_on.forces(A, d) - g_off.forces(A, d); // eV/Å
            const double err = std::fabs(df - f_disp_ev);
            max_err = std::max(max_err, err);
            net[d] += g_on.forces(A, d);
            if (A == 0 || A == 3)
                std::printf("A=%d d=%d ΔF=%.6e F_D3=%.6e |err|=%.2e eV/A\n", A,
                            d, df, f_disp_ev, err);
        }
    }

    constexpr double kMaterial = 1e-3; // eV/Å
    constexpr double kTol = 5e-4;      // eV/Å
    constexpr double kTrans = 1e-5;
    const double net_max =
        std::max({std::fabs(net[0]), std::fabs(net[1]), std::fabs(net[2])});

    std::printf("D3 |F|_max=%.6e  max|ΔF−F_D3|=%.3e  |ΣF|_max=%.3e\n", d3_max,
                max_err, net_max);
    std::printf("E_CCSD+D3=%.12f eV  E_CCSD=%.12f eV\n", g_on.energy,
                g_off.energy);

    if (!(d3_max > kMaterial)) {
        std::printf("FAIL: D3 not material\n");
        return 1;
    }
    if (!(max_err < kTol)) {
        std::printf("FAIL: ccsd_forces d3 leg != D3BJ forces\n");
        return 1;
    }
    if (!(net_max < kTrans)) {
        std::printf("FAIL: total force not translationally invariant\n");
        return 1;
    }
    // Energy additivity of the dispersion leg.
    const double de = g_on.energy - g_off.energy;
    const double e_disp_ev =
        units::convert(cpu::scf::D3BJ{}.energy(frame), units::Energy::Hartree,
                       units::Energy::eV);
    if (std::fabs(de - e_disp_ev) > 1e-6) {
        std::printf("FAIL: ΔE != E_D3 (%.12e vs %.12e eV)\n", de, e_disp_ev);
        return 1;
    }

    std::printf("PASS: ccsd-d3bj-forces-h2o-sto3g\n");
    return 0;
}
