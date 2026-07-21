/// @file tests/cpu/md/test_mpi_stubs.cpp
/// @brief RED tests for the four Phase-1 no-op MPI components.
///
/// Spec: .claude/specs/parallel-md-engine.md § A.4 (components table).
/// Expected bindings (Phase enum as int), LAMMPS-granular phases:
///   Migrate  → pre_exchange @ 0     (re-home atoms that left the rank)
///   HaloSend → pre_neighbor @ 0     (build ghosts before the neighbor list)
///   HaloRecv → post_force @ -1000   (reverse ghost-force comm)
///   Reduce   → end_of_step @ -50
///
/// In single-rank mode every launch() is a no-op: frame positions must be
/// byte-identical before and after.

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/comm/atom_migration.h"
#include "atomiverse/cpu/comm/halo_exchange.h"
#include "atomiverse/cpu/comm/reduce.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/facade/stage.h"

using namespace ATV;

namespace {

cpu::Frame make_fcc_argon(int n_cells, double a) {
    int n = n_cells * n_cells * n_cells * 4;
    cpu::Frame frame(n);
    double basis[4][3] = {
        {0.0, 0.0, 0.0},
        {0.5, 0.5, 0.0},
        {0.5, 0.0, 0.5},
        {0.0, 0.5, 0.5},
    };
    int idx = 0;
    for (int ix = 0; ix < n_cells; ++ix)
        for (int iy = 0; iy < n_cells; ++iy)
            for (int iz = 0; iz < n_cells; ++iz)
                for (int b = 0; b < 4; ++b) {
                    frame.span(ATV::kAtom_x)[idx] = (ix + basis[b][0]) * a;
                    frame.span(ATV::kAtom_y)[idx] = (iy + basis[b][1]) * a;
                    frame.span(ATV::kAtom_z)[idx] = (iz + basis[b][2]) * a;
                    frame.span(ATV::kAtom_mass)[idx] = 39.948;
                    frame.span(ATV::kAtom_type)[idx] = 18;
                    ++idx;
                }
    frame.simbox = ATV::cpu::Box::cubic(n_cells * a);
    return frame;
}

bool has_binding(const std::vector<DefaultBinding>& bindings,
                 const DefaultBinding& expected) {
    if (bindings.size() != 1)
        return false;
    return bindings[0].kind == expected.kind
           && bindings[0].phase == expected.phase
           && bindings[0].order == expected.order;
}

} // namespace

// ── bindings ─────────────────────────────────────────────────────────

TEST(MpiStubBindings, HaloSendBindings) {
    cpu::comm::HaloSend hx{};
    auto b = hx.default_bindings();
    EXPECT_TRUE(
        has_binding(b, ATV::make_binding(ATV::md::Phase::pre_neighbor, 0)))
        << "HaloSend must bind pre_neighbor @ 0";
}

TEST(MpiStubBindings, HaloRecvBindings) {
    cpu::comm::HaloRecv hx{};
    auto b = hx.default_bindings();
    EXPECT_TRUE(
        has_binding(b, ATV::make_binding(ATV::md::Phase::post_force, -1000)))
        << "HaloRecv must bind post_force @ -1000";
}

TEST(MpiStubBindings, MigrateBinding) {
    cpu::comm::Migrate am{};
    auto b = am.default_bindings();
    EXPECT_TRUE(
        has_binding(b, ATV::make_binding(ATV::md::Phase::pre_exchange, 0)))
        << "Migrate must bind pre_exchange @ 0";
}

TEST(MpiStubBindings, ReduceEnergyBinding) {
    cpu::md::mpi::Reduce re{};
    auto b = re.default_bindings();
    EXPECT_TRUE(
        has_binding(b, ATV::make_binding(ATV::md::Phase::end_of_step, -50)))
        << "Reduce must bind end_of_step @ -50";
}

// ── single-rank no-op: positions byte-identical ──────────────────────

TEST(MpiStubBindings, SingleRankLaunchNoOp) {
    auto frame = make_fcc_argon(3, 5.26);
    const int n = frame.nlocal();

    // Snapshot positions before.
    std::vector<double> x_before(n), y_before(n), z_before(n);
    std::memcpy(x_before.data(), frame.span(ATV::kAtom_x).data(),
                n * sizeof(double));
    std::memcpy(y_before.data(), frame.span(ATV::kAtom_y).data(),
                n * sizeof(double));
    std::memcpy(z_before.data(), frame.span(ATV::kAtom_z).data(),
                n * sizeof(double));

    // Wire a bare MD driver with the four stubs only (plus auto infra + a cheap
    // integrator to have a full step). Drive 10 steps, verify positions unchanged
    // *by the stubs* — since VelocityVerlet only moves if velocities are nonzero.
    //
    // The MD category requires a Capability::Force component; supply LJ with a
    // cutoff (3.0 Å) below the FCC nearest-neighbor distance (~3.72 Å at a=5.26)
    // so the pair force evaluates to exactly zero — matched NeighborList ditto.
    // This isolates the test to "do MPI stubs perturb positions?" without
    // pulling in a real interaction.
    {
        cpu::md::MD sim;
        const double zero_force_cut = 3.0;
        sim.set_integrator(cpu::md::VelocityVerlet{1e-6}); // negligible dt
        sim.add_potential(cpu::PairLJCut{1e-12, 3.4, zero_force_cut});
        sim.set_neighbor(cpu::md::NeighborList<>{zero_force_cut, /*skin=*/0.1});
        sim.add(cpu::comm::HaloSend{});
        sim.add(cpu::comm::HaloRecv{});
        sim.add(cpu::comm::Migrate{});
        sim.add(cpu::md::mpi::Reduce{});
        sim.build();
        sim.run(frame, 10);
    }

    // With no forces or velocities, the stubs must be bit-identical no-ops.
    EXPECT_EQ(0, std::memcmp(x_before.data(), frame.span(ATV::kAtom_x).data(),
                             n * sizeof(double)))
        << "MPI stubs perturbed frame.span(ATV::kAtom_x) in single-rank mode";
    EXPECT_EQ(0, std::memcmp(y_before.data(), frame.span(ATV::kAtom_y).data(),
                             n * sizeof(double)))
        << "MPI stubs perturbed frame.span(ATV::kAtom_y) in single-rank mode";
    EXPECT_EQ(0, std::memcmp(z_before.data(), frame.span(ATV::kAtom_z).data(),
                             n * sizeof(double)))
        << "MPI stubs perturbed frame.span(ATV::kAtom_z) in single-rank mode";
}

// ── class-name hygiene ───────────────────────────────────────────────

TEST(MpiStubBindings, ClassNamesAreDistinct) {
    cpu::comm::HaloSend fwd{};
    cpu::comm::HaloRecv rev{};
    const std::string n_fwd = fwd.name();
    const std::string n_rev = rev.name();
    EXPECT_NE(n_fwd, n_rev);
    EXPECT_NE(n_fwd.find("halo"), std::string::npos)
        << "HaloSend::name() should contain 'halo': got '" << n_fwd << "'";
    EXPECT_NE(n_rev.find("halo"), std::string::npos)
        << "HaloRecv::name() should contain 'halo': got '" << n_rev << "'";
}
