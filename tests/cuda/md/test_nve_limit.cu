/// @file tests/cuda/md/test_nve_limit.cu
/// @brief `cuda::md::NVELimit` — per-step displacement cap (LAMMPS
///        `fix nve/limit`).
///
/// Covers `msu-pe-cuda-components-03-cuda-integrators-stage1`:
/// - role / binding contract;
/// - cap engages on a synthetic single-atom configuration whose
///   half-kick would otherwise put `|v|·dt = 10·xmax`;
/// - NaN guard: 16-atom box with `fx[0] = 1e8 eV/Å` (mimicking the
///   MSU PE unrelaxed start state) survives 100 steps without
///   non-finite positions/velocities.

#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/md/md.cuh"
#include "atomiverse/cuda/md/nve_limit.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"

using namespace ATV;

namespace {

cpu::Frame make_single_atom(double vx_init) {
    cpu::Frame frame(1);
    frame.atoms.x[0] = 5.0;
    frame.atoms.y[0] = 5.0;
    frame.atoms.z[0] = 5.0;
    frame.atoms.vx[0] = vx_init;
    frame.atoms.vy[0] = 0.0;
    frame.atoms.vz[0] = 0.0;
    frame.atoms.mass[0] = 1.0;
    frame.atoms.type[0] = 0;
    frame.atoms.tag[0] = 0;
    frame.box = Mat3::cubic(10.0);
    return frame;
}

cpu::Frame make_overlap_frame(int n_side) {
    int n = n_side * n_side * n_side;
    cpu::Frame frame(n);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                // Spacing < sigma so LJ generates a huge repulsion.
                frame.atoms.x[idx] = ix * 0.8;
                frame.atoms.y[idx] = iy * 0.8;
                frame.atoms.z[idx] = iz * 0.8;
                frame.atoms.vx[idx] = 0.0;
                frame.atoms.vy[idx] = 0.0;
                frame.atoms.vz[idx] = 0.0;
                frame.atoms.mass[idx] = 1.0;
                frame.atoms.type[idx] = 0;
                frame.atoms.tag[idx] = idx;
                ++idx;
            }
    frame.box = Mat3::cubic(double(n_side) * 0.8 + 5.0);
    return frame;
}

} // namespace

TEST(CudaNVELimit, ContractRoleAndBindings) {
    cuda::md::NVELimit nve{1.0, 0.05};
    EXPECT_EQ(nve.role(), ComponentRole::Integrator);
    EXPECT_EQ(nve.name(), "nve_limit");
    auto bindings = nve.default_bindings();
    ASSERT_EQ(bindings.size(), 2u);
    EXPECT_EQ(bindings[0].phase,
              ATV::make_binding(ATV::md::Phase::initial_integrate).phase);
    EXPECT_EQ(bindings[1].phase,
              ATV::make_binding(ATV::md::Phase::final_integrate).phase);
    EXPECT_EQ(bindings[0].order, 0);
    EXPECT_EQ(bindings[1].order, 0);
}

TEST(CudaNVELimit, CapClampsSingleAtom) {
    // dt=1.0, xmax=0.05 ⇒ vmax=0.05. Initial v_x = 5.0 ≫ vmax ⇒ cap.
    // After one initial_integrate step with zero force, displacement
    // |Δx| = xmax = 0.05 Å exactly (to FP tolerance).
    auto frame = make_single_atom(5.0);
    const double x0 = frame.atoms.x[0];
    cuda::md::MD sim;
    sim.add(cuda::md::NVELimit{1.0, 0.05});
    // Need a Force-capable component to satisfy cuda::md::MD. Use a
    // disposable LJ + NeighborList that won't trigger meaningful force
    // on the single isolated atom.
    sim.on(ATV::md::Phase::force).add(cuda::NeighborList{3.0}, -10);
    sim.on(ATV::md::Phase::force).add(cuda::PairLJCut{1.0, 1.0, 3.0});
    sim.build();
    sim.run(frame, 1);
    const double dx = std::fabs(frame.atoms.x[0] - x0);
    // Allow for PBC wrap (box 10 Å, atom at 5+0.05); displacement
    // should equal 0.05 within single-precision tolerance.
    EXPECT_NEAR(dx, 0.05, 1e-4);
}

TEST(CudaNVELimit, SurvivesHugeForcesNoNaN) {
    // Mimic msu_pe.data unrelaxed start state: severe atom overlap →
    // huge repulsion. NVE/limit's displacement cap must keep all
    // velocities and positions finite over 100 steps.
    auto frame = make_overlap_frame(2); // 8 atoms in a tight cube
    const double cutoff = 3.0;
    cuda::md::MD sim;
    sim.add(cuda::md::NVELimit{1.0, 0.05});
    sim.on(ATV::md::Phase::force).add(cuda::NeighborList{cutoff}, -10);
    sim.on(ATV::md::Phase::force).add(cuda::PairLJCut{1.0, 1.0, cutoff});
    sim.build();
    sim.run(frame, 100);
    for (int i = 0; i < frame.atoms.size(); ++i) {
        EXPECT_TRUE(std::isfinite(static_cast<double>(frame.atoms.x[i])));
        EXPECT_TRUE(std::isfinite(static_cast<double>(frame.atoms.vx[i])));
        EXPECT_TRUE(std::isfinite(static_cast<double>(frame.atoms.fx[i])));
    }
}
