/// @file tests/cpu/io/test_compiled_forcefield.cpp
/// @brief Potential components compose into Evaluate / MD / Minimize.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "atomiverse/cpu/driver/evaluate.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/minimizer/lbfgs.h"
#include "atomiverse/cpu/minimizer/minimize.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {

cpu::Frame two_water() {
    cpu::Frame f(6);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 0.9572;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_x)[2] = -0.24;
    f.span(kAtom_y)[2] = 0.93;
    f.span(kAtom_z)[2] = 0;
    f.span(kAtom_x)[3] = 3.5;
    f.span(kAtom_y)[3] = 0;
    f.span(kAtom_z)[3] = 0;
    f.span(kAtom_x)[4] = 4.45;
    f.span(kAtom_y)[4] = 0;
    f.span(kAtom_z)[4] = 0;
    f.span(kAtom_x)[5] = 3.26;
    f.span(kAtom_y)[5] = 0.93;
    f.span(kAtom_z)[5] = 0;
    const int types[6] = {0, 1, 1, 0, 1, 1};
    const double q[6] = {-0.834, 0.417, 0.417, -0.834, 0.417, 0.417};
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_type)[i] = types[i];
        f.span(kAtom_mass)[i] = (types[i] == 0) ? 15.999 : 1.008;
        f.span(kAtom_charge)[i] = q[i];
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.simbox = cpu::Box::cubic(20.0);
    f.clear_tag_index();
    f.rebuild_tag_index(0, f.nlocal());
    f.add_bond(1, 2, 0);
    f.add_bond(1, 3, 0);
    f.add_bond(4, 5, 0);
    f.add_bond(4, 6, 0);
    f.add_angle(2, 1, 3, 0);
    f.add_angle(5, 4, 6, 0);
    f.build_special_list();
    return f;
}

template <typename Driver, typename Phase>
void install_tip3p(Driver& d, Phase force_phase) {
    d.set_neighbor(cpu::md::NeighborList<>{8.0, 0.3});
    d.add_potential(
        cpu::PairLJCut{0.0067, 3.15, 8.0}.set_special_bonds(0.0, 0.0, 0.0));
    d.on(force_phase)
        .add(cpu::PairCoulCut{8.0}.set_special_bonds(0.0, 0.0, 0.0));
    d.add_potential(cpu::BondHarmonic{std::vector<double>{25.0},
                                      std::vector<double>{0.9572}});
    d.add_potential(cpu::AngleHarmonic{std::vector<double>{5.0},
                                       std::vector<double>{1.824218}});
}

class StopAfter : public cpu::Criterion {
  public:
    explicit StopAfter(int n) : n_(n) {}
    std::string name() const override { return "stop_after"; }
    void launch(const cpu::LaunchContext&) override {} // verdict-only
    ATV::Verdict verdict(const cpu::LoopState& s) override {
        return (s.iteration + 1 >= n_) ? ATV::Verdict::Stop
                                       : ATV::Verdict::Continue;
    }

  private:
    int n_;
};

} // namespace

TEST(PotentialCompose, EvaluateMDMinimize) {
    {
        auto f = two_water();
        cpu::Evaluate eval;
        install_tip3p(eval, evaluate::Phase::evaluate);
        eval.build();
        eval.run(f);
        EXPECT_TRUE(std::isfinite(f.meta.get(kEnergy)));
        EXPECT_LT(std::fabs(f.meta.get(kEnergy)), 50.0);
    }
    {
        auto f = two_water();
        for (int i = 0; i < 6; ++i)
            f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
        cpu::md::MD md;
        md.set_integrator(cpu::md::VelocityVerlet{0.5});
        install_tip3p(md, md::Phase::force);
        md.build();
        md.run(f, 5);
        EXPECT_EQ(f.nlocal(), 6);
        EXPECT_TRUE(std::isfinite(f.meta.get(kEnergy)));
    }
    {
        auto f = two_water();
        f.span(kAtom_x)[1] = 1.2;
        cpu::minimizer::Minimize opt;
        opt.set_optimizer(cpu::minimizer::LBFGS{5, 0.01, 5});
        opt.add_criterion(StopAfter{3});
        install_tip3p(opt, minimizer::Phase::force);
        opt.build();
        opt.run(f, 10);
        EXPECT_TRUE(std::isfinite(f.meta.get(kEnergy)));
        EXPECT_EQ(f.nlocal(), 6);
    }
}
