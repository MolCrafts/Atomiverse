/// @file regressions/scf/mp2-d3bj-forces-h2o-sto3g/reg_mp2-d3bj-forces-h2o-sto3g.cpp
/// @brief Water-dimer / STO-3G RMP2 + DFT-D3(BJ) analytic forces vs central
///        FD of E_MP2 + E_D3(BJ). Wiring gate (not a literature absolute).
///
/// A monomer H2O has |F_D3| ~ 7e-5 eV/Å — smaller than a comfortable FD
/// tolerance — so a missing dispersion leg is invisible. The dimer geometry
/// below carries a material intermolecular D3 force (~0.01 eV/Å class), so
/// dropping `add_dispersion_forces` fails the FD gate by that amount.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Checks (REFERENCE.md):
///   1. |F_analytic − F_FD| ≤ tol (eV/Å) on E_tot = E_MP2 + E_D3(BJ)
///   2. D3 force contribution is material (|F_D3|_max above noise)
///   3. translational invariance |Σ_A F_A| small
///
/// Exit non-zero on mismatch. ≤ 60 s, single CPU.

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

/// @brief Water dimer (Å) — same geometry as dftb2-dispersion-d3 / s-dftd3 pin.
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

void configure(cpu::scf::RHF& rhf) {
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-10;
    rhf.max_iter = 200;
    rhf.d3bj = true;
}

/// @brief Total MP2+D3(BJ) energy [Hartree] at a geometry.
double e_mp2_d3bj(const cpu::Frame& frame) {
    cpu::scf::RHF rhf;
    configure(rhf);
    cpu::Frame f = frame;
    return rhf.mp2(f).e_tot;
}

cpu::Frame displaced(const cpu::Frame& frame, int A, int d, double dx_ang) {
    cpu::Frame f = frame;
    if (d == 0)
        f.span(kAtom_x)[A] += static_cast<StorageReal>(dx_ang);
    else if (d == 1)
        f.span(kAtom_y)[A] += static_cast<StorageReal>(dx_ang);
    else
        f.span(kAtom_z)[A] += static_cast<StorageReal>(dx_ang);
    return f;
}

} // namespace

int main() {
    using namespace ATV;

    const cpu::Frame frame = water_dimer();
    const int n = frame.nlocal();

    // Analytic forces on E_MP2 + E_D3(BJ) [eV/Å].
    cpu::scf::RHF rhf;
    configure(rhf);
    cpu::Frame ff = frame;
    const cpu::scf::GradientResult res = rhf.mp2_forces(ff);

    // Bare D3(BJ) forces for materiality [Ha/Bohr] → eV/Å.
    const auto d3_f_au = cpu::scf::D3BJ{}.forces(frame);
    double d3_max = 0.0;
    for (int A = 0; A < n; ++A)
        for (int d = 0; d < 3; ++d) {
            const double f_ev = units::convert(
                d3_f_au[static_cast<std::size_t>(A)]
                       [static_cast<std::size_t>(d)],
                units::Force::HartreePerBohr, units::Force::eVPerAngstrom);
            d3_max = std::max(d3_max, std::fabs(f_ev));
        }
    // Intermolecular D3 on this dimer is ~0.01 eV/Å; a missing leg must fail.
    constexpr double kD3Material = 1e-3; // eV/Å
    if (!(d3_max > kD3Material)) {
        std::printf("FAIL: D3(BJ) force max=%.3e eV/A not material (<%.0e)\n",
                    d3_max, kD3Material);
        return 1;
    }
    std::printf("D3(BJ) |F|_max = %.6e eV/A (material > %.0e)\n", d3_max,
                kD3Material);

    // Central FD of E_MP2+D3(BJ) [Ha] → force [eV/Å].
    // Sample only O atoms of each monomer (material D3, fewer SCFs).
    const int atoms[2] = {0, 3};
    const double h_bohr = 1e-3;
    const double h_ang = h_bohr * units::kBohrInAngstrom;
    constexpr double kTol = 5e-4; // eV/Å

    double max_err = 0.0;
    double net[3] = {0.0, 0.0, 0.0};
    for (int A = 0; A < n; ++A)
        for (int d = 0; d < 3; ++d)
            net[d] += res.forces(A, d);

    for (int a_idx = 0; a_idx < 2; ++a_idx) {
        const int A = atoms[a_idx];
        for (int d = 0; d < 3; ++d) {
            const double e_p = e_mp2_d3bj(displaced(frame, A, d, +h_ang));
            const double e_m = e_mp2_d3bj(displaced(frame, A, d, -h_ang));
            const double dE_dR_ha_bohr = (e_p - e_m) / (2.0 * h_bohr);
            const double f_fd =
                -units::convert(dE_dR_ha_bohr, units::Force::HartreePerBohr,
                                units::Force::eVPerAngstrom);
            const double f_an = res.forces(A, d);
            const double err = std::fabs(f_an - f_fd);
            max_err = std::max(max_err, err);
            std::printf("F[%d][%d] analytic=% .10e fd=% .10e |err|=%.2e eV/A\n",
                        A, d, f_an, f_fd, err);
        }
    }

    const double net_max =
        std::max({std::fabs(net[0]), std::fabs(net[1]), std::fabs(net[2])});
    std::printf("E_MP2+D3 = %.12f eV\n", res.energy);
    std::printf("max_elementwise_err=%.3e tol=%.0e eV/A  |Σ_A F_A|_max=%.3e\n",
                max_err, kTol, net_max);

    if (!(max_err < kTol)) {
        std::printf("FAIL: mp2+d3bj forces != FD of E_MP2+E_D3(BJ)\n");
        return 1;
    }
    constexpr double kTransTol = 1e-5; // eV/Å
    if (!(net_max < kTransTol)) {
        std::printf("FAIL: force not translationally invariant "
                    "(|Σ F|=%.3e > %.0e eV/A)\n",
                    net_max, kTransTol);
        return 1;
    }

    std::printf("PASS: mp2-d3bj-forces-h2o-sto3g (water dimer)\n");
    return 0;
}
