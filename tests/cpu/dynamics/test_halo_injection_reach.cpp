/// @file tests/cpu/dynamics/test_halo_injection_reach.cpp
/// @brief Pin the auto-injected halo shell against the neighbor list's reach.
///
/// Mirrors `include/atomiverse/cpu/dynamics/md.h` — specifically the halo
/// auto-injection branch of `cpu::md::MD::install_defaults()`.
///
/// Contract under test (single concern): **the halo shell that
/// `install_defaults()` injects must be at least as wide as the total reach
/// the `NeighborList` builds its CSR out to, i.e. `cutoff + skin`.**
///
/// Why that is the contract and not merely a nicety: `NeighborList::build()`
/// initialises its spatial index with `cutoff_ + strategy_.skin()`
/// (`neighbor_list.h`), so a pair separated by anything up to `cutoff + skin`
/// at build time is *supposed* to be in the list — that skin band is the
/// entire reason the list is reusable between rebuilds. A ghost shell
/// narrower than that band means a cross-rank `j` sitting inside the band has
/// no ghost on `i`'s rank, so the pair is absent from the CSR rather than
/// merely inactive. It stays absent until the next full rebuild, while the
/// displacement trigger (`max_disp * 2 > skin`) is by construction not
/// watching for it. The pair can therefore drift from inside the band down to
/// well below the interaction cutoff and contribute exactly zero force the
/// whole way — a silent cross-rank correctness hole, not a truncation error.
///
/// Observable: the injected `comm::HaloSend` in the compiled Plan, probed
/// through the `HaloParticipant` protocol (same discipline as
/// `scan_pending_infra_()`) and read back via
/// `HaloSend::halo_cutoff()` (`halo_exchange.h`) — the value `launch()`
/// itself uses to decide which owned atoms ship as ghosts.

#include <gtest/gtest.h>

#include <cstddef>
#include <unordered_set>
#include <vector>

#include "atomiverse/core/plan.h"
#include "atomiverse/cpu/communication/halo_exchange.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/halo_participant.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

namespace {

/// @brief Pair-interaction cutoff of the single consumer [Angstrom].
constexpr double kCutoff = 9.0;

/// @brief Collect the halo send endpoints of a compiled Plan.
///
/// Deduplicates by component identity (a component bound to several phases
/// appears once per binding) and probes the typed protocol rather than
/// matching `name()`.
/// @param plan Compiled Plan to scan.
/// @return Every component whose halo endpoint is `Endpoint::send`.
std::vector<ATV::cpu::Component*>
collect_halo_sends(const ATV::Plan<ATV::cpu::Component>& plan) {
    std::vector<ATV::cpu::Component*> sends;
    std::unordered_set<const ATV::cpu::Component*> seen;
    for (const auto& slot : plan.slots) {
        for (const auto& op : slot.ops) {
            ATV::cpu::Component* c = op.component;
            if (!seen.insert(c).second)
                continue;
            ATV::cpu::HaloParticipant* p = c->halo_participant();
            if (p == nullptr)
                continue;
            if (p->halo_endpoint() == ATV::cpu::HaloParticipant::Endpoint::send)
                sends.push_back(c);
        }
    }
    return sends;
}

/// @brief Width [Angstrom] of the halo shell `MD::install_defaults()` injects
///        for one LJ consumer at `cutoff` alongside a list built with `skin`.
///
/// No frame is bound: the injection is a build-time decision taken from the
/// pending-component scan alone.
/// @param cutoff Pair cutoff for both the LJ consumer and the list [Angstrom].
/// @param skin Explicit Verlet skin for the NeighborList [Angstrom].
/// @return `HaloSend::halo_cutoff()` of the sole injected send endpoint [Angstrom].
double injected_halo_shell(double cutoff, double skin) {
    ATV::cpu::md::MD sim;
    sim.set_integrator(ATV::cpu::md::VelocityVerlet{1.0});
    // Single LJ type; only its cutoff matters here. It is the sole
    // NeighborConsumer, so it alone sizes the injected shell.
    sim.set_evaluator(ATV::cpu::PairLJCut{0.0104, 3.4, cutoff});
    sim.set_neighbor(ATV::cpu::md::NeighborList<>{cutoff, skin});
    sim.build();

    const std::vector<ATV::cpu::Component*> sends =
        collect_halo_sends(sim.plan());
    EXPECT_EQ(sends.size(), std::size_t{1})
        << "expected exactly one auto-injected HaloSend endpoint";
    if (sends.size() != 1)
        return -1.0;

    const auto* halo =
        dynamic_cast<const ATV::cpu::comm::HaloSend*>(sends.front());
    EXPECT_NE(halo, nullptr) << "halo send endpoint is not a comm::HaloSend";
    if (halo == nullptr)
        return -1.0;

    // No topology is bound, so bonded_extension() is 0 and halo_cutoff() is
    // exactly the injected shell width.
    return halo->halo_cutoff();
}

TEST(TestHaloInjectionReach, InjectedHaloCoversCutoffPlusSkin) {
#ifndef ATV_HAS_MPI
    GTEST_SKIP() << "halo auto-injection is compiled out without ATV_HAS_MPI";
#else
    // Two skins, one contract. A single (cutoff, skin) pair would also be
    // satisfied by merely enlarging the fixed multiplier; the relationship
    // only holds for both points if the shell is actually derived from the
    // list's skin.
    for (const double skin : {2.0, 6.0}) {
        const double reach = kCutoff + skin;
        const double shell = injected_halo_shell(kCutoff, skin);
        ASSERT_GT(shell, 0.0) << "no injected halo shell to measure";

        EXPECT_GE(shell, reach)
            << "auto-injected halo shell (" << shell
            << " A) is thinner than the neighbor list's reach (" << reach
            << " A = cutoff " << kCutoff << " + skin " << skin
            << "). Cross-rank pairs whose build-time separation falls in ("
            << shell << ", " << reach
            << ") A get no ghost, so they are absent from the CSR entirely "
               "and stay absent until the next full rebuild -- they can drift "
               "below "
            << kCutoff
            << " A contributing zero force, and the skin displacement trigger "
               "does not watch for it.";
    }
#endif
}

} // namespace
