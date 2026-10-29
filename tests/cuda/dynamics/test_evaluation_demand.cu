/// @file tests/cuda/dynamics/test_evaluation_demand.cu
/// @brief Per-step energy/virial demand (LAMMPS `ev_set`) in `cuda::md::MD`:
///        a step evaluates what a consumer's cadence asks for, the reported
///        final step always evaluates the energy, and the retrieved energy
///        is that of the final state.

#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/component/observer.cuh"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"

using namespace ATV;

namespace {

/// Host-only probe: records the step's evaluation flags; demands what the
/// test configures.
class DemandProbe : public cuda::Observer {
  public:
    struct Seen {
        BigInt step;
        bool energy, virial;
    };
    DemandProbe(cuda::EvaluationDemand demand,
                std::shared_ptr<std::vector<Seen>> seen)
        : demand_(demand), seen_(std::move(seen)) {}
    std::string name() const override { return "demand_probe"; }
    cuda::EvaluationDemand demand() const override { return demand_; }
    void launch(const cuda::LaunchContext& ctx) override {
        seen_->push_back(
            {ctx.step, ctx.energy_requested, ctx.stress_requested});
    }

  private:
    cuda::EvaluationDemand demand_;
    std::shared_ptr<std::vector<Seen>> seen_;
};

cpu::Frame make_lattice() {
    const int side = 6, n = side * side * side;
    const double spacing = 3.9;
    cpu::Frame frame(n);
    int i = 0;
    for (int x = 0; x < side; ++x)
        for (int y = 0; y < side; ++y)
            for (int z = 0; z < side; ++z, ++i) {
                frame.span(kAtom_x)[i] = (x + 0.5) * spacing;
                frame.span(kAtom_y)[i] = (y + 0.5) * spacing;
                frame.span(kAtom_z)[i] = (z + 0.5) * spacing + 0.1 * (x % 2);
                frame.span(kAtom_mass)[i] = 39.948;
                frame.span(kAtom_type)[i] = 0;
                frame.span(kAtom_tag)[i] = i;
                frame.span(kAtom_vx)[i] = 0.002 * ((i * 7) % 5 - 2);
                frame.span(kAtom_vy)[i] = 0.002 * ((i * 3) % 5 - 2);
                frame.span(kAtom_vz)[i] = 0.002 * ((i * 11) % 5 - 2);
            }
    frame.box = cpu::Box::cubic(side * spacing);
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    return frame;
}

constexpr double kEps = 0.0104, kSigma = 3.4, kCutoff = 8.5;

} // namespace

TEST(CudaEvaluationDemand, StepsFollowCadenceAndTheReportStep) {
    auto frame = make_lattice();
    auto seen = std::make_shared<std::vector<DemandProbe::Seen>>();
    cuda::md::MD sim;
    sim.set_neighbor(cuda::NeighborList{kCutoff});
    sim.set_evaluator(cuda::PairLJCut{kEps, kSigma, kCutoff});
    sim.set_integrator(cuda::md::VelocityVerlet{1.0});
    // Energy every 3 steps; a second consumer wants the virial every 4.
    sim.on(md::Phase::end_of_step)
        .add(DemandProbe{{true, false, 3}, seen}, 50)
        .add(DemandProbe{{false, true, 4},
                         std::make_shared<std::vector<DemandProbe::Seen>>()},
             51);
    sim.build();
    sim.run(frame, 10); // steps 0..9; 9 is the report step
    ASSERT_EQ(seen->size(), 10u);
    for (const auto& s : *seen) {
        EXPECT_EQ(s.energy, s.step % 3 == 0 || s.step == 9) << s.step;
        // The report step also evaluates the virial: someone demands it.
        EXPECT_EQ(s.virial, s.step % 4 == 0 || s.step == 9) << s.step;
    }
}

TEST(CudaEvaluationDemand, RetrievedEnergyIsTheFinalState) {
    // No thermo, no barostat: no step but the last evaluates the energy.
    auto frame = make_lattice();
    cuda::md::MD sim;
    sim.set_neighbor(cuda::NeighborList{kCutoff});
    sim.set_evaluator(cuda::PairLJCut{kEps, kSigma, kCutoff});
    sim.set_integrator(cuda::md::VelocityVerlet{1.0});
    sim.build();
    sim.run(frame, 7);
    const double reported = frame.meta.get(kEnergy);

    auto probe = frame;
    cuda::Evaluate eval;
    eval.set_neighbor(cuda::NeighborList{kCutoff});
    eval.set_evaluator(cuda::PairLJCut{kEps, kSigma, kCutoff});
    eval.build();
    eval.run(probe);
    const double fresh = probe.meta.get(kEnergy);
    ASSERT_TRUE(std::isfinite(reported));
    EXPECT_NEAR(reported, fresh, 1e-10 * std::abs(fresh));
}
