/// @file tests/cpu/test_md_no_alloc.cpp
/// @brief Phase 5 Track B.7 — assert no heap allocation occurs during the
///        MD inner loop after the warm-up steps have settled all arenas.
///
/// The architectural rule "no allocation in launch()" (CLAUDE.md +
/// `parallel-md-engine.md` Architectural rules) has been a code-review
/// convention. This test promotes it to a CI guarantee on Linux by
/// hooking glibc's `__malloc_hook` machinery via a thread-local
/// counter that increments on every `malloc/calloc/realloc/free`. It
/// runs `cpu::md::MD` for a warm-up burst (lets `NeighborList` and the
/// halo-buffer arenas reach steady state), arms the counter, runs a
/// further N steps, and asserts the counter stayed at zero.
///
/// macOS / non-glibc platforms cannot use `__malloc_hook`; they
/// `GTEST_SKIP` with a clear reason so local devs aren't blocked.
///
/// Acceptance: ac-020 in
/// `.claude/specs/parallel-md-engine.acceptance.md`.

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/init.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/nose_hoover.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

#if defined(__GLIBC__) && defined(__linux__)
#define ATV_TEST_HAVE_MALLOC_HOOK 1
#include <malloc.h>
#endif

#ifdef ATV_TEST_HAVE_MALLOC_HOOK

namespace {

/// Process-wide allocation counter. Atomic so OpenMP worker
/// allocations (if any happen) are still observed deterministically.
std::atomic<uint64_t>& alloc_count() {
    static std::atomic<uint64_t> c{0};
    return c;
}

bool& alloc_count_armed() {
    static bool armed = false;
    return armed;
}

extern "C" void* __libc_malloc(size_t);
extern "C" void* __libc_calloc(size_t, size_t);
extern "C" void* __libc_realloc(void*, size_t);
extern "C" void __libc_free(void*);

extern "C" void* malloc(size_t sz) {
    if (alloc_count_armed())
        alloc_count().fetch_add(1, std::memory_order_relaxed);
    return __libc_malloc(sz);
}
extern "C" void* calloc(size_t n, size_t sz) {
    if (alloc_count_armed())
        alloc_count().fetch_add(1, std::memory_order_relaxed);
    return __libc_calloc(n, sz);
}
extern "C" void* realloc(void* p, size_t sz) {
    if (alloc_count_armed())
        alloc_count().fetch_add(1, std::memory_order_relaxed);
    return __libc_realloc(p, sz);
}
extern "C" void free(void* p) {
    // Don't count free() — only allocation (the architectural rule is
    // "no allocation in launch()"; free of stack-built temporaries is
    // benign).
    __libc_free(p);
}

} // namespace

#endif // ATV_TEST_HAVE_MALLOC_HOOK

namespace {

constexpr double kKcalPerMolToEv = 1.0 / 23.060547830619026;
constexpr double kBondR0 = 1.53;
constexpr double kBondK_kcal = 350.0;

ATV::cpu::Frame make_small_chain(int n_carbons, double box_len) {
    using namespace ATV;
    cpu::Frame frame(n_carbons);
    for (int i = 0; i < n_carbons; ++i) {
        frame.span(ATV::kAtom_x)[i] = box_len / 2.0;
        frame.span(ATV::kAtom_y)[i] = box_len / 2.0 + i * kBondR0;
        frame.span(ATV::kAtom_z)[i] = box_len / 2.0;
        frame.span(ATV::kAtom_mass)[i] = 14.027;
        frame.span(ATV::kAtom_type)[i] = 0;
    }
    // Populate canonical topology Block columns after building the tag index,
    // so add_bond can resolve the owner row from its global tag.
    frame.clear_tag_index();
    frame.rebuild_tag_index(0, frame.nlocal());
    for (int k = 0; k < n_carbons - 1; ++k)
        frame.add_bond(k, k + 1, 0);
    frame.build_special_list();
    frame.simbox = ATV::cpu::Box::cubic(box_len);
    return frame;
}

} // namespace

TEST(MdNoAlloc, InnerLoopAfterWarmupAllocatesNothing) {
#ifndef ATV_TEST_HAVE_MALLOC_HOOK
    GTEST_SKIP() << "malloc-hook unavailable on this platform "
                 << "(__GLIBC__ + __linux__ required); the architectural "
                 << "rule is still verified by code review on macOS / "
                 << "non-glibc platforms.";
#else
    using namespace ATV;

    auto frame = make_small_chain(/*n_carbons=*/20, /*box_len=*/40.0);
    cpu::md::MD sim;
    sim.add(cpu::md::VelocityVerlet{0.5});
    sim.add(cpu::md::InitVelocity{300.0, /*seed=*/4242});
    sim.add(cpu::md::NoseHoover{300.0, 100.0, 0.5});
    sim.on(ATV::md::Phase::force).add(cpu::md::NeighborList<>{5.0}, -10);
    sim.on(ATV::md::Phase::force)
        .add(cpu::BondHarmonic{{kBondK_kcal * kKcalPerMolToEv}, {kBondR0}});
    sim.build();

    // Warm-up: lets NeighborList/halo arenas reach steady state.
    sim.run(frame, /*n_steps=*/100);

    // This test asserts the STEADY-STATE INNER LOOP allocates nothing. The
    // built-in end-of-run summary (time + memory breakdown) is a one-time
    // report at the end of run() and does allocate (report tables, MPI reduce
    // buffers) — exactly like LAMMPS's end-of-run Finish. Turn timing OFF so
    // the summary is skipped and only the hot loop is under the counter.
    // Done AFTER warm-up so the first run's setup_pass (which re-inits the
    // backend with timing on by default) cannot clobber it.
    sim.enable_timing(false);

    // Arm the counter.
    alloc_count().store(0, std::memory_order_relaxed);
    alloc_count_armed() = true;

    // Hot inner loop — the architectural rule says zero allocations.
    sim.run(frame, /*n_steps=*/100);

    alloc_count_armed() = false;
    const uint64_t observed = alloc_count().load(std::memory_order_relaxed);

    EXPECT_EQ(observed, 0u)
        << "Inner-loop allocations: " << observed
        << " (expected 0 — see CLAUDE.md 'no allocation in launch()').";
#endif
}
