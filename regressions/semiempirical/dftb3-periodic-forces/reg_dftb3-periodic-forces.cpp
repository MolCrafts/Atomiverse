/// @file regressions/semiempirical/dftb3-periodic-forces/reg_dftb3-periodic-forces.cpp
/// @brief Γ-PBC SCC-DFTB3 analytic nuclear forces vs re-SCC FD (lattice images).
///
/// Domain: Elstner et al., Phys. Rev. B 58, 7260 (1998); DFTB+ force module
/// boundary (Hourahine et al., J. Chem. Phys. 152, 124101 (2020)). Off-eq 1-D
/// C-chain cell so left/right lattice shells differ. In-tree mio-style SK
/// tables — not absolute mio-1-1 DFTB+ numbers. Public API only. <= 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb3.h"

using namespace ATV;

namespace {

// Off-eq 1-D C dimer: r=1.20 Å, Lx=2.8 Å → images at 1.20 and 1.60 Å (etc.).
cpu::Frame make_c_chain_offeq() {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 6;
    f.span(kAtom_type)[1] = 6;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 1.20;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 12.011;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::orthorhombic(2.8, 12.0, 12.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

double energy_only(cpu::Frame g) {
    g.meta.get(kEnergy) = 0;
    for (int i = 0; i < g.nlocal(); ++i)
        g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;
    cpu::Evaluate ev;
    cpu::semiempirical::DFTB3 d;
    d.sk_set = cpu::semiempirical::DftbSkSet::MioSketch;
    d.hubbard_du_ha = cpu::semiempirical::basis_data::k3obHubbardDerivsHa;
    d.with_forces = false;
    d.scc_tol = 1e-10;
    ev.on(evaluate::Phase::evaluate).add(std::move(d));
    ev.build();
    ev.run(g);
    return g.meta.get(kEnergy);
}

} // namespace

int main() {
    auto frame = make_c_chain_offeq();

    cpu::Evaluate eval;
    cpu::semiempirical::DFTB3 dftb;
    dftb.sk_set = cpu::semiempirical::DftbSkSet::MioSketch;
    dftb.hubbard_du_ha = cpu::semiempirical::basis_data::k3obHubbardDerivsHa;
    dftb.with_forces = true;
    dftb.scc_tol = 1e-10;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
    eval.build();
    eval.run(frame);

    const double e = frame.meta.get(kEnergy);
    std::printf("DFTB3 Γ-PBC C-chain forces: E=%.10f eV\n", e);
    if (!std::isfinite(e) || std::fabs(e) < 1e-3) {
        std::printf("FAIL energy not finite/bound\n");
        return 1;
    }

    double f2 = 0;
    double net[3] = {0, 0, 0};
    for (int i = 0; i < 2; ++i) {
        const double fxi = static_cast<double>(frame.span(kAtom_fx)[i]);
        const double fyi = static_cast<double>(frame.span(kAtom_fy)[i]);
        const double fzi = static_cast<double>(frame.span(kAtom_fz)[i]);
        std::printf("  F[%d]=(%.10f, %.10f, %.10f) eV/A\n", i, fxi, fyi, fzi);
        f2 += fxi * fxi + fyi * fyi + fzi * fzi;
        net[0] += fxi;
        net[1] += fyi;
        net[2] += fzi;
        if (!std::isfinite(fxi) || !std::isfinite(fyi) || !std::isfinite(fzi)) {
            std::printf("FAIL non-finite force\n");
            return 1;
        }
    }
    if (f2 < 1e-8) {
        std::printf("FAIL all forces ~0 at off-eq geometry\n");
        return 1;
    }
    for (int k = 0; k < 3; ++k) {
        if (std::fabs(net[k]) > 1e-6) {
            std::printf("FAIL net force[%d]=%.3e > 1e-6 eV/A\n", k, net[k]);
            return 1;
        }
    }

    // Analytic F vs re-SCC central FD of lattice-summed E.
    constexpr double h = 1.0e-4;
    constexpr double kFdTol = 1.0e-5; // eV/Å
    const double fx0 = static_cast<double>(frame.span(kAtom_fx)[0]);
    const double fx1 = static_cast<double>(frame.span(kAtom_fx)[1]);
    {
        auto gp = make_c_chain_offeq();
        auto gm = make_c_chain_offeq();
        gp.span(kAtom_x)[0] += static_cast<StorageReal>(h);
        gm.span(kAtom_x)[0] -= static_cast<StorageReal>(h);
        const double fd = -(energy_only(gp) - energy_only(gm)) / (2.0 * h);
        std::printf("  F0x analytic=%.10f  fd=%.10f  |d|=%.3e\n", fx0, fd,
                    std::fabs(fx0 - fd));
        if (std::fabs(fx0 - fd) > kFdTol) {
            std::printf("FAIL F0x vs FD: |d|=%.3e > %.1e eV/A\n",
                        std::fabs(fx0 - fd), kFdTol);
            return 1;
        }
    }
    {
        auto gp = make_c_chain_offeq();
        auto gm = make_c_chain_offeq();
        gp.span(kAtom_x)[1] += static_cast<StorageReal>(h);
        gm.span(kAtom_x)[1] -= static_cast<StorageReal>(h);
        const double fd = -(energy_only(gp) - energy_only(gm)) / (2.0 * h);
        std::printf("  F1x analytic=%.10f  fd=%.10f  |d|=%.3e\n", fx1, fd,
                    std::fabs(fx1 - fd));
        if (std::fabs(fx1 - fd) > kFdTol) {
            std::printf("FAIL F1x vs FD: |d|=%.3e > %.1e eV/A\n",
                        std::fabs(fx1 - fd), kFdTol);
            return 1;
        }
    }

    // Lattice images matter vs isolated dimer.
    auto isol = make_c_chain_offeq();
    isol.box = cpu::Box::cubic(20.0);
    isol.meta.get(kEnergy) = 0;
    const double e_mol = energy_only(std::move(isol));
    std::printf("  isolated dimer (20 A cubic): E=%.10f  Δ=%.6f eV\n", e_mol,
                e - e_mol);
    if (!std::isfinite(e_mol) || std::fabs(e - e_mol) < 0.05) {
        std::printf("FAIL PBC energy does not separate from molecular dimer\n");
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
