/// @file regressions/md/gaff_protocol_smoke/reg_gaff_protocol_smoke.cpp
/// @brief Short min → NVT → NPT smoke on a charged molecular fragment.
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/lammps_real.h"
#include "atomiverse/cpu/md/init.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/nose_hoover.h"
#include "atomiverse/cpu/md/nose_hoover_npt.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/minimizer/lbfgs.h"
#include "atomiverse/cpu/minimizer/minimize.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/facade/stage.h"

using namespace ATV;

namespace {

cpu::Frame make_fragment() {
    const int n = 6;
    cpu::Frame f(n);
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] = 1.4 * i;
        f.span(kAtom_y)[i] = 0.1 * (i % 2);
        f.span(kAtom_z)[i] = 0;
        f.span(kAtom_type)[i] = (i % 2 == 0) ? 0 : 1;
        f.span(kAtom_mass)[i] = (i % 2 == 0) ? 12.0 : 16.0;
        f.span(kAtom_charge)[i] = (i % 2 == 0) ? 0.2 : -0.2;
        f.span(kAtom_tag)[i] = i;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
    }
    f.simbox = cpu::Box::cubic(20.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    for (int i = 0; i < n - 1; ++i)
        f.add_bond(i, i + 1, 0);
    f.add_angle(0, 1, 2, 0);
    f.build_special_list();
    return f;
}

template <typename Driver, typename Phase>
void install_ff(Driver& d, Phase force_phase) {
    using cpu::kcal_per_mol_to_ev;
    const double eps0 = kcal_per_mol_to_ev(0.1);
    const double eps1 = kcal_per_mol_to_ev(0.09);
    d.set_neighbor(cpu::md::NeighborList<>{8.0, 0.5});
    d.add_potential(
        cpu::PairLJCut{{{0, 0, eps0, 3.4}, {1, 1, eps1, 3.2}}, 8.0}
            .set_special_bonds(0.0, 0.0, 0.5));
    d.on(force_phase)
        .add(cpu::PairCoulCut{8.0}.set_special_bonds(0.0, 0.0, 0.5));
    d.add_potential(cpu::BondHarmonic{{kcal_per_mol_to_ev(200.0)}, {1.5}});
    d.add_potential(
        cpu::AngleHarmonic{{kcal_per_mol_to_ev(50.0)}, {1.9}});
}

} // namespace

int main() {
    auto f = make_fragment();

    // Stage 0: minimize
    {
        cpu::minimizer::Minimize opt;
        opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.02, 8});
        install_ff(opt, minimizer::Phase::force);
        opt.build();
        opt.run(f, 30);
    }
    const double e_min = f.meta.get(kEnergy);
    std::printf("after min E=%.6f eV\n", e_min);

    // Stage 1: short NVT
    {
        cpu::md::MD sim;
        install_ff(sim, md::Phase::force);
        sim.set_integrator(cpu::md::VelocityVerlet{1.0});
        sim.set_thermostat(cpu::md::NoseHoover{400.0, 100.0, 1.0});
        sim.add(cpu::md::InitVelocity{400.0, 1});
        sim.build();
        sim.run(f, 50);
    }
    const double e_nvt = f.meta.get(kEnergy);
    std::printf("after NVT E=%.6f\n", e_nvt);

    // Stage 2: short NPT with T ramp
    {
        cpu::md::MD sim;
        install_ff(sim, md::Phase::force);
        cpu::md::NoseHooverNPT::Config cfg;
        cfg.T_target = 400.0;
        cfg.T_stop = 500.0;
        cfg.tau_T = 100.0;
        cfg.P_target = 0.0;
        cfg.tau_P = 1000.0;
        cfg.dt = 1.0;
        auto npt = cpu::md::NoseHooverNPT{cfg};
        npt.set_run_span(0, 50);
        sim.set_integrator(std::move(npt));
        sim.build();
        sim.run(f, 50);
    }
    const double e_npt = f.meta.get(kEnergy);
    const double vol = f.simbox.volume();
    std::printf("after NPT E=%.6f V=%.6f\n", e_npt, vol);

    if (!std::isfinite(e_min) || !std::isfinite(e_nvt) || !std::isfinite(e_npt)
        || !std::isfinite(vol) || vol <= 0.0) {
        std::printf("FAIL: non-finite thermo or bad volume\n");
        return 1;
    }
    std::printf("PASS: gaff_protocol_smoke min+NVT+NPT\n");
    return 0;
}
