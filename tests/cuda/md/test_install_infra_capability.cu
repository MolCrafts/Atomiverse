/// @file tests/cuda/md/test_install_infra_capability.cu
/// @brief Capability-typed auto-injection tests for `cuda::md::MD::install_infra_`.
///
/// Covers `cuda-md-review-cleanup-01-component-capability` ac-006/ac-008,
/// migrated to the `NeighborConsumer`/`NeighborProvider` opt-in protocol by
/// `.claude/specs/arch-refactor-03-neighbor.md`:
/// - `cuda::md::MD::install_infra_` consults `reads()` / `capabilities()` /
///   the typed protocol accessors `as_neighbor_consumer()` /
///   `as_neighbor_provider()` instead of `name() == "..."` string proxies
///   and `dynamic_cast<const cuda::Pair*>`.
/// - A pending component carrying `(reads() & Capability::Stress) != None`
///   auto-injects `cuda::Virial`. This is the key behavior change
///   vs the old `role() == Barostat` proxy: `cuda::md::NoseHooverNPT`
///   reports `role() == Integrator` but consumes Stress, so the new
///   typed check picks it up where the old string/role proxy could not.
/// - A pending component exposing a non-null `as_neighbor_consumer()`
///   without a registered `cuda::NeighborList` throws at `build()`.
/// - `cuda::Pair` reports its interaction cutoff [Å] through
///   `NeighborConsumer::neighbor_cutoff()` — the CUDA side previously had
///   no cutoff report at all (the CPU-side hint had no mirror);
///   this closes the CPU/CUDA probe-surface gap.
///
/// These tests *cannot* introspect `pending_` directly (it's private to
/// the Backend). Instead they observe injection through externally
/// visible effects: a successful `build()` + `run()` for the NPT path
/// (would-throw if Virial wasn't auto-injected because NPT's
/// Stress-reading branch would fault) and a `runtime_error` for the
/// missing-NeighborList path.

#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/component/neighbor_protocol.cuh"
#include "atomiverse/cuda/md/md.cuh"
#include "atomiverse/cuda/md/nose_hoover_npt.cuh"
#include "atomiverse/cuda/md/velocity_verlet.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/observer/stress.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/facade/component_role.h"

using namespace ATV;

namespace {

/// Same builder as `test_cuda_md_components.cu::make_lj_system`.
cpu::Frame make_lj_system(int n_side, double density) {
    int n = n_side * n_side * n_side;
    double box_len = std::cbrt(double(n) / density);
    double spacing = box_len / double(n_side);
    cpu::Frame frame(n);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                frame.atoms.x[idx] = (double(ix) + 0.5) * spacing;
                frame.atoms.y[idx] = (double(iy) + 0.5) * spacing;
                frame.atoms.z[idx] = (double(iz) + 0.5) * spacing;
                frame.atoms.mass[idx] = 1.0;
                frame.atoms.type[idx] = 0;
                ++idx;
            }
    frame.box = Mat3::cubic(box_len);
    std::srand(42);
    for (int i = 0; i < n; ++i) {
        frame.atoms.vx[i] = 1e-3 * (double(std::rand()) / RAND_MAX - 0.5);
        frame.atoms.vy[i] = 1e-3 * (double(std::rand()) / RAND_MAX - 0.5);
        frame.atoms.vz[i] = 1e-3 * (double(std::rand()) / RAND_MAX - 0.5);
    }
    double mvx = 0, mvy = 0, mvz = 0;
    for (int i = 0; i < n; ++i) {
        mvx += frame.atoms.vx[i];
        mvy += frame.atoms.vy[i];
        mvz += frame.atoms.vz[i];
    }
    mvx /= n;
    mvy /= n;
    mvz /= n;
    for (int i = 0; i < n; ++i) {
        frame.atoms.vx[i] -= mvx;
        frame.atoms.vy[i] -= mvy;
        frame.atoms.vz[i] -= mvz;
    }
    return frame;
}

cuda::md::NoseHooverNPT::Config default_npt_cfg() {
    cuda::md::NoseHooverNPT::Config cfg{};
    cfg.T_target = 1.0;
    cfg.tau_T = 100.0;
    cfg.P_target = 0.0;
    cfg.tau_P = 1000.0;
    cfg.dt = 0.001;
    return cfg;
}

} // namespace

// ── ac-001 + ac-006: capability-typed virtuals exist and route ──────

TEST(InstallInfraCapability, ComponentBaseDefaultsAreNone) {
    // Smoke: a default-constructed bare Component subclass (here we
    // use cuda::md::VelocityVerlet, which doesn't override reads() or the
    // neighbor protocol accessors) returns the None / nullptr defaults.
    cuda::md::VelocityVerlet vv{0.001};
    EXPECT_EQ(vv.reads(), Capability::None);
    EXPECT_EQ(vv.as_neighbor_consumer(), nullptr);
    EXPECT_EQ(vv.as_neighbor_provider(), nullptr);
}

TEST(InstallInfraCapability, PairIsNeighborConsumerReportingCutoff) {
    // PairLJCut (cuda::PairLJCut) inherits from cuda::Pair, which
    // implements the NeighborConsumer protocol. Beyond the old boolean
    // missing-NL gate, cuda::Pair now ALSO reports its ctor
    // cutoff [Å] — the CUDA mirror of cpu::PairEvaluator::pair_cutoff()
    // forwarding, previously missing on this backend.
    cuda::PairLJCut lj{1.0, 1.0, 3.0};
    const cuda::NeighborConsumer* nc = lj.as_neighbor_consumer();
    ASSERT_NE(nc, nullptr)
        << "every cuda::Pair style must opt into NeighborConsumer";
    EXPECT_EQ(nc->neighbor_cutoff(), 3.0);
}

TEST(InstallInfraCapability, NeighborListIsNeighborProvider) {
    // Supply side: install_infra_ keys on as_neighbor_provider(), not on
    // name() == "neighbor_list" and not on the provider accessor.
    cuda::NeighborList nl{3.0};
    const cuda::NeighborProvider* np = nl.as_neighbor_provider();
    ASSERT_NE(np, nullptr);
    EXPECT_EQ(np, static_cast<const cuda::NeighborProvider*>(&nl));
}

// Detection idiom: the two legacy neighbor virtuals must be GONE from
// `cuda::Component` (replaced by the two typed protocol accessors;
// same base-vtable diet as cpu::Component, arch-refactor-03). The
// deletion of the legacy in-vtable neighbor trio is enforced by the
// acceptance grep gate (ac-001) plus every migrated call site compiling —
// an in-test detection idiom would trip the grep gate with its own tokens.

TEST(InstallInfraCapability, NPTDeclaresReadsStress) {
    // cuda::md::NoseHooverNPT is the canonical Integrator that *also*
    // acts as a barostat — it consumes Stress to drive ε̇ updates. With
    // the old install_infra_ proxy (role==Barostat) this was invisible
    // and required users to add cuda::Virial by hand. The new
    // reads() virtual exposes the dependency directly.
    cuda::md::NoseHooverNPT npt{default_npt_cfg()};
    EXPECT_EQ(npt.role(), ComponentRole::Integrator);
    EXPECT_NE((npt.reads() & Capability::Stress), Capability::None);
}

TEST(InstallInfraCapability, VirialDeclaresStressCapability) {
    cuda::Virial vs;
    EXPECT_EQ(vs.role(), ComponentRole::Observer);
    EXPECT_NE((vs.capabilities() & Capability::Stress), Capability::None);
}

// ── ac-008: end-to-end auto-injection for the NPT/Stress path ───────

TEST(InstallInfraCapability, NPTAutoInjectsVirial) {
    // Before the rewrite, `cuda::md::NoseHooverNPT` had role==Integrator
    // and install_infra_ checked role==Barostat — so this configuration
    // would NOT auto-inject Virial and the NPT stress-feedback
    // branch would fault. After the rewrite, install_infra_ checks
    // `reads() & Stress != None`, so NPT pulls in Virial
    // automatically.
    auto frame = make_lj_system(4, 0.8);
    const double cutoff = 3.0;
    cuda::md::MD sim;
    sim.set_neighbor(cuda::NeighborList{cutoff});
    sim.add_potential(cuda::PairLJCut{1.0, 1.0, cutoff});
    sim.set_integrator(cuda::md::NoseHooverNPT{default_npt_cfg()});
    EXPECT_NO_THROW(sim.build());
    EXPECT_NO_THROW(sim.run(frame, 1));
}

// ── ac-006: missing-NeighborList detection via the consumer protocol ──

TEST(InstallInfraCapability, MissingNeighborThrowsViaTypedQuery) {
    // The old install_infra_ identified pair evaluators via
    // dynamic_cast<const cuda::Pair*>, then via the boolean
    // legacy consumer virtual. It now probes
    // as_neighbor_consumer() != nullptr. Behavior is identical;
    // this test guards the typed contract.
    cuda::md::MD sim;
    sim.add_potential(cuda::PairLJCut{1.0, 1.0, 3.0});
    sim.set_integrator(cuda::md::VelocityVerlet{0.001});
    EXPECT_THROW(
        {
            try {
                sim.build();
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find("set_neighbor"),
                          std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}
