/// @file cpu/md/md.h
/// @brief cpu::md::MD — the classical time-advance preset. Its termination is the
///        step budget: it runs every step it was asked for, unless a criterion stops it.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/asynchronous/snapshot_publisher.h"
#include "atomiverse/cpu/communication/atom_migration.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/communication/halo_exchange.h"
#include "atomiverse/cpu/communication/reduce.h"
#include "atomiverse/cpu/component/bias.h"
#include "atomiverse/cpu/component/gpw.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/component/integrator.h"
#include "atomiverse/cpu/component/pair_evaluator.h"
#include "atomiverse/cpu/component/path_coupling.h"
#include "atomiverse/cpu/component/plane_wave.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/component/timestep_protocol.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/driver/runtime.h"
#include "atomiverse/cpu/dynamics/fix_csvr.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/nvt.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#ifdef ATV_HAS_MOLRS
#include "atomiverse/cpu/frame/molrs_exchange.h"
#endif
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/record/timing_report.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief CPU MD simulation driver.
///
/// Convenience class that auto-adds always-on infrastructure components
/// in the constructor, exposes a LAMMPS-shaped categorical setter API
/// (set_evaluator, set_evaluator, set_neighbor, set_integrator, set_thermostat,
/// set_barostat, set_observer, set_record, set_thermo — the first six
/// are inherited from Driver), and runs `install_defaults()` at `build()`
/// time to derive opt-in infrastructure from declared component requirements:
///
///   - Stress capability dependency → auto-injects `Virial`
///     (deduped against an explicitly-registered `Virial`).
///   - `#ifdef ATV_HAS_MPI` and a pair evaluator was set →
///     auto-injects `HaloSend{cutoff}` and
///     `HaloRecv{}` at the documented orders.
///   - A `NeighborConsumer` (pair evaluator / real-space Coulomb) was
///     registered without a corresponding `set_neighbor()` →
///     `build()` throws naming the offending component.
///
/// Inheritance: MD → Driver<md::Phase> → Driver<md::Phase, cpu::device_tag>.
/// There is no `DynamicsDriver` category in between any more: "run a step budget" is a
/// loop bound, not a taxonomy.
/// Driver provides the shared categorical setter API (set_evaluator,
/// set_evaluator, set_angle, set_dihedral, set_neighbor, set_observer,
/// set_record, set_thermo) and the slot variables. MD adds
/// driver-specific setters (set_integrator, set_thermostat, set_barostat).
///
/// **No default criterion.** The step budget IS the termination, and time evolution has
/// no convergence success path. But MD is a loop like any other, so a user can now
/// attach one: `sim.add_criterion(cpu::StopOnNaN{})`, or their own `StopWhenTooHot{2000}` —
/// LAMMPS's `fix halt`, which was impossible to write when "stop or not" was a private
/// `if` inside DynamicsDriver.
///
/// `on(phase).add()` from the underlying `Driver<>` remains public — it
/// is the documented escape hatch for user-defined hooks (custom
/// observers, recorders, etc.) that do not yet have a categorical
/// setter. There is no bare `add()`: the phase is always explicit.
///
/// Phase skeleton (LAMMPS-granular; see
/// `.claude/notes/cpu-md-execution-model.md`): initial_integrate →
/// post_integrate → pre_exchange (Migrate) → pre_neighbor (HaloSend) →
/// neighbor (NeighborList) → pre_force (Zero{Forces,Energy,Stress}) → force
/// (pair + bonded) → post_force (HaloRecv, Virial) → final_integrate →
/// end_of_step. End-of-step ordering: ZeroKinetic(-1000) → KineticEnergy(-100)
/// → Reduce(-50) → Publish(+10) → user observers. Categorical thermo at
/// `set_thermo(N)` binds `cpu::Thermo{N}` at `end_of_step @ +20`.
///
/// Usage (LAMMPS-shaped):
/// @code{.cpp}
/// cpu::md::MD sim;
/// sim.set_evaluator(cpu::PairLJCut{0.0104, 3.4, 6.8});
/// sim.set_neighbor(cpu::md::NeighborList<>{6.8, 0.3});
/// sim.set_integrator(cpu::md::VelocityVerlet{0.001});
/// sim.set_thermo(100);
/// sim.build();
/// sim.run(frame, 1000);
/// @endcode
class MD : public Driver<ATV::md::Phase> {
  public:
    /// @brief Flush the deferred component `destroy()` of a live continuable run.
    ~MD() { this->flush_live_(); }

    /// @brief Advance the system for `n_steps` MD steps.
    ///
    /// **Continuable.** Consecutive calls extend one trajectory — the standard
    /// `run(burn_in); run(sample); …` pattern. @see Driver::begin_run_
    ///
    /// @param frame Host-resident system state, updated in place.
    /// @param n_steps Step budget.
    /// @return Number of iterations actually executed (may be below the
    ///         budget when a user criterion stops the loop).
    int run(cpu::Frame& frame, int n_steps) {
        this->begin_run_(frame);
        int iterations = 0;
        for (int s = 0; s < n_steps; ++s) {
            this->mutable_outer_status().reset();
            // Flag the final step so cadence-gated end-of-step collectives
            // (mpi::Reduce under set_scalar_reduce_every > 1) always publish
            // globally-reduced scalars before run() returns.
            this->backend_->context().md.is_last_step = (s == n_steps - 1);
            this->step_once(this->next_step_());
            ++iterations;
            if (this->poll_criteria_(s) != Verdict::Continue)
                break;
        }
        this->backend_->context().md.is_last_step = false;
        // Drain async-analysis workers so every snapshot from this segment reaches its
        // subscribers before `run()` returns — continuation defers full teardown, so the
        // barrier is made explicit here.
        this->backend_->drain_analysis();
        // Built-in end-of-run report (time + memory), like LAMMPS's loop summary.
        // Collective + rank-0 render; always emitted.
        this->emit_run_summary("md");
        return iterations;
    }

    /// @brief User-facing entry — advance the system stored in a molrs.Frame.
    ///
    /// Each call to this overload materializes a fresh `cpu::Frame`, so consecutive
    /// `run(molrs::FrameRef&, …)` calls are independent simulations (no trajectory
    /// continuation): every call hands its components the step indices 0 … n_steps-1.
    /// For a continued trajectory materialize once and use the `cpu::Frame&` overload.
    ///
    /// The independence is **structural**: the live run is torn down here, before the
    /// working frame dies, so `Driver::begin_run_` never compares the next call's frame
    /// address against a dangling pointer into this call's destroyed stack local. The
    /// teardown fires component `destroy()` and ends the live run, so the next call
    /// re-runs `setup_pass` from a zeroed step counter — no stateful integrator,
    /// thermostat ramp or recorder cadence carries over.
    ///
    /// @param frame molrs.Frame handle (input geometry + output sink).
    /// @param n_steps Step budget.
    /// @return Number of iterations actually executed.
#ifdef ATV_HAS_MOLRS
    int run(molrs::FrameRef& frame, int n_steps) {
        cpu::Frame work = cpu::from_molrs(frame);
        const int iterations = run(work, n_steps);
        cpu::to_molrs(work, frame);
        this->flush_live_();
        return iterations;
    }
#endif

    MD() {
        // Always-on zero / kinetic infrastructure. Phase-agnostic
        // Zero* classes are bound explicitly to MD phases.
        this->on(ATV::md::Phase::pre_force)
            .add(ZeroForces{}, -1000)
            .add(ZeroEnergy{}, -999)
            .add(ZeroStress{}, -998);
        this->on(ATV::md::Phase::end_of_step)
            .add(ZeroKinetic{}, -1000)
            .add(KineticEnergy{}, -100);

        // MPI infrastructure (single-rank no-ops in Phase 1). `Reduce` is
        // injected in install_defaults() so set_scalar_reduce_every() (which
        // runs after construction) can configure its cadence.
        this->on(ATV::md::Phase::pre_exchange).add(comm::Migrate{10});

        // Async-analysis publisher — shared infrastructure, explicitly
        // bound to *this driver's* end_of_step @ +10. Subscribers
        // inheriting `cpu::md::AsyncObserver` default to +20 on the same
        // phase and see this publisher's slot.
        this->on(ATV::md::Phase::end_of_step).add(cpu::Publish{}, 10);
    }

    // ─── MD-specific categorical setters ────────────────────────────
    //
    // set_evaluator, set_neighbor, set_observer, set_record, set_thermo
    // are inherited from Driver.

    /// @brief Configure the time integrator (LAMMPS `fix … nve|nvt|npt`).
    template <typename I>
        requires(!std::is_lvalue_reference_v<I>)
    MD& set_integrator(I&& integ) {
        this->set_singular_default_owned_(std::forward<I>(integ),
                                          integrator_set_, "set_integrator");
        return *this;
    }

    template <typename I>
    MD& set_integrator_ref(I& integ) {
        this->set_singular_default_ref_(integ, integrator_set_,
                                        "set_integrator_ref");
        return *this;
    }

    template <typename I>
    MD& set_integrator(I&) = delete;

    /// @brief Add a thermostat (LAMMPS `fix … langevin / temp/berendsen`).
    /// Plural: each call appends; insertion order preserved.
    template <typename T>
        requires(!std::is_lvalue_reference_v<T>)
    MD& set_thermostat(T&& t) {
        this->bind_defaults(std::forward<T>(t));
        return *this;
    }

    template <typename T>
    MD& set_thermostat_ref(T& t) {
        this->bind_defaults_ref(t);
        return *this;
    }

    template <typename T>
    MD& set_thermostat(T&) = delete;

    /// @brief Add a barostat (LAMMPS `fix … press/berendsen`). Plural.
    template <typename Bs>
        requires(!std::is_lvalue_reference_v<Bs>)
    MD& set_barostat(Bs&& b) {
        this->bind_defaults(std::forward<Bs>(b));
        return *this;
    }

    template <typename Bs>
    MD& set_barostat_ref(Bs& b) {
        this->bind_defaults_ref(b);
        return *this;
    }

    template <typename Bs>
    MD& set_barostat(Bs&) = delete;

    /// @brief Configure a canonical (NVT) ensemble in one call — LAMMPS
    /// `fix nvt` style: velocity-Verlet integration bundled with a selectable
    /// thermostat (Nose-Hoover or CSVR), instead of a separate
    /// `set_integrator` + `set_thermostat`.
    ///
    /// Expands the `NVT` config into the existing `VelocityVerlet` integrator
    /// plus the chosen thermostat, each registered with its own validated phase
    /// bindings (Nose-Hoover half-steps on initial/final integrate; CSVR rescale
    /// on end_of_step). Equivalent to the explicit two-call form, so it composes
    /// with `set_evaluator`/`set_neighbor`/`on(force).add(<DFT evaluator>)` exactly as
    /// before.
    ///
    /// @param cfg NVT configuration (dt, T, tau, thermostat kind, seed, ramp).
    MD& set_nvt(const NVT& cfg) {
        this->set_integrator(VelocityVerlet{cfg.dt});
        if (cfg.thermo == NVT::Thermostat::CSVR) {
            FixCSVR th{cfg.T, cfg.tau, cfg.dt, cfg.seed};
            if (cfg.has_ramp())
                th.set_T_stop(cfg.T_stop)
                    .set_run_span(cfg.ramp_first, cfg.ramp_last);
            this->bind_defaults(std::move(th));
        } else {
            NoseHoover th{cfg.T, cfg.tau, cfg.dt};
            if (cfg.has_ramp())
                th.set_T_stop(cfg.T_stop)
                    .set_run_span(cfg.ramp_first, cfg.ramp_last);
            this->bind_defaults(std::move(th));
        }
        return *this;
    }

    /// @brief Set the cadence of the global scalar `MPI_Allreduce` (energy /
    /// kinetic / stress). Default 1 = every step. A value > 1 fires the reduce
    /// only on step 0 and every Nth step thereafter — eliding the per-step
    /// latency-bound collective. ONLY valid when no per-step consumer reads the
    /// reduced scalars (NVE / per-atom Langevin, where the global energy/stress
    /// are pure diagnostics sampled every N steps). MUST stay 1 whenever a
    /// NoseHoover thermostat or barostat is active — those integrate the global
    /// temperature/pressure every step. Set N to align with the thermo /
    /// recorder sampling stride so sampled steps see globally-reduced scalars.
    MD& set_scalar_reduce_every(int every) {
        scalar_reduce_every_ = every < 1 ? 1 : every;
        return *this;
    }

    /// @brief Hand off the spatial `DomainDecomposition` from a previously-
    ///        torn-down MD instance.
    ///
    /// Call BEFORE `build()` to make the new MD reuse the partition state
    /// (`n_local`, `tag_to_local`, `sublo/subhi`, `cuts`) of the prior
    /// stage's DD. The companion `release_dd()` is invoked on the prior
    /// MD just before its destructor; the orchestrator (e.g.
    /// polyethylene_tension's per-stage loop) shuttles the unique_ptr
    /// through. Without this, every new MD ⇒ new Backend ⇒ new DD ⇒
    /// `first_partition=true` re-scan that misinterprets the previous
    /// stage's stale frame tail as owned atoms.
    void adopt_dd(std::unique_ptr<cpu::md::mpi::DomainDecomposition> dd) {
        using Base = ATV::Driver<ATV::md::Phase, cpu::device_tag>;
        if (!this->backend_)
            this->backend_ = std::make_unique<typename Base::BackendT>();
        this->backend_->adopt_dd(std::move(dd));
    }

    /// @brief Release the DD for the next stage to adopt.
    ///
    /// Call AFTER the last `run()` of this stage and BEFORE letting this
    /// MD destruct. Returns nullptr if no DD is owned.
    std::unique_ptr<cpu::md::mpi::DomainDecomposition> release_dd() {
        if (!this->backend_)
            return nullptr;
        return this->backend_->release_dd();
    }

    // enable_timing() / reset_timing() are inherited from Driver — the timing
    // registry is Backend infrastructure that every category owns, not an MD
    // privilege.

    /// @brief Raw access to the timing registry (null before the backend
    /// exists). @see core/timer.h
    const ATV::TimerRegistry* timing() const {
        return this->backend_ ? &this->backend_->timing() : nullptr;
    }

    /// @brief Cross-rank-reduced timing report for this run/stage.
    cpu::md::ReducedReport timing_report() const {
        return this->backend_ ? cpu::md::reduce_timing(this->backend_->timing())
                              : cpu::md::ReducedReport{};
    }

    /// @brief Emit the timing breakdown: human table (console, rank 0) and,
    /// if a Csv sink is given, a machine row (metrics file). Routed through
    /// the log module. @see cpu/md/timing_report.h
    void emit_timing(std::string_view stage,
                     ATV::report::Csv* csv = nullptr) const {
        if (!this->backend_)
            return;
        const auto rep = cpu::md::reduce_timing(this->backend_->timing());
        cpu::md::emit_timing_table(rep, stage);
        // Always print per-component (bond vs angle vs dihedral vs pair vs
        // pppm) so the LAMMPS "Bond" bucket can be un-lumped for diagnosis.
        // Set ATV_TIMER_VERBOSE=0 to suppress.
        const char* pv = std::getenv("ATV_TIMER_VERBOSE");
        if (!pv || pv[0] != '0')
            cpu::md::emit_per_component(this->backend_->timing(), stage);
        if (csv)
            cpu::md::emit_timing_csv(rep, *csv, stage);
    }

    /// @brief Built-in end-of-run summary: the LAMMPS-style time breakdown
    /// (Pair/Bond/Kspace/Neigh/Comm/Output/Modify/Other + Loop) PLUS a
    /// per-component table (harmonic_bond / harmonic_angle / charmm_dihedral /
    /// lj/cut/coul/long / pppm / …) so bonded terms are not opaque, AND a
    /// memory breakdown. Always emitted at the end of every `run()`; disable
    /// timing with ATV_TIMER=0. Collective; rendered on rank 0.
    void emit_run_summary(std::string_view stage) override {
        // On by default; ATV_TIMER=0 / enable_timing(false) opts out (and makes
        // the one-time report allocation-free for hot-loop micro-benchmarks).
        if (!this->backend_ || !this->backend_->timing().enabled())
            return;
        const auto t = cpu::md::reduce_timing(this->backend_->timing());
        cpu::md::emit_timing_table(t, stage);
        const char* pv = std::getenv("ATV_TIMER_VERBOSE");
        if (!pv || pv[0] != '0')
            cpu::md::emit_per_component(this->backend_->timing(), stage);
        const auto m =
            cpu::md::reduce_memory(this->backend_->collect_component_memory(),
                                   cpu::md::read_peak_rss_kb());
        cpu::md::emit_memory_table(m, stage);
    }

  protected:
    void validate_components() const override {
        this->require_integrator();
        this->require_force();
    }

    void install_defaults() override {
        const auto s = this->scan_pending_infra_();

        // Global scalar reduction (single-rank no-op). Same on().add path
        // as any plugin; skip if a previous build already mounted one.
        if (!this->pending_named_("reduce"))
            this->bind_defaults(mpi::Reduce{scalar_reduce_every_});

        if (s.needs_stress && !s.has_virial_stress)
            this->bind_defaults(Virial{});

#ifdef ATV_HAS_MPI
        if (s.neighbor_consumer_cutoff) {
            // Halo cutoff must include a SKIN beyond the pair cutoff so
            // ghost atoms stay registered as they drift in/out of the
            // pair-cutoff zone between full halo rebuilds. Without the
            // skin band, atoms slip out of `tag_to_local` at each
            // `every_=10` rebuild and any cross-rank bond/angle/dihedral
            // involving them is silently dropped — observed under MSU
            // PE stage-2 NPT as a +1-3 eV per-rebuild jump in the MTK
            // extended Hamiltonian (E_bond drops by ~2.5 eV at rebuild
            // boundaries because ~16 chain bonds lose their ghost
            // partner each cycle).
            //
            // The skin that matters is the one the neighbor list actually
            // uses: `NeighborList::build()` enumerates out to
            // `cutoff + skin`, so a pair anywhere in that band is supposed
            // to be in the CSR. A ghost shell thinner than the band leaves
            // a cross-rank `j` inside it with no ghost at all — the pair is
            // absent from the CSR rather than merely inactive, and stays
            // absent until the next full rebuild while the displacement
            // trigger (`max_disp * 2 > skin`) is by construction not
            // watching for it. Hence `neighbor_reach()`, taken straight
            // from the provider. The 10% multiplier stays as the floor for
            // consumers whose provider declares no reach (it matches the
            // ~0.1 Å typical 10-step drift at PE-thermalized 500 K); with a
            // provider present the wider of the two wins.
            //
            // A halo wider than half the local slab duplicates ghosts, so a
            // caller on a very small box should size the shell itself:
            // registering `HaloSend{halo_cut}` explicitly before the pair
            // evaluator suppresses the auto-inject entirely.
            // Skip the auto-inject when the caller already registered halo
            // components (custom cutoff / skin / reneighbor cadence). Auto-
            // injecting on top of an explicit halo would run the REVERSE
            // exchange (HaloRecv) twice, which double-applies the already-
            // reduced ghost forces to their owners — a per-step momentum leak
            // (Σ F ≠ 0) that pumps energy and blows the NVE trajectory up.
            // `reverse exchange must run exactly once` (see halo_exchange.h).
            double halo_cut =
                1.1 * static_cast<double>(*s.neighbor_consumer_cutoff);
            if (s.neighbor_provider_reach)
                halo_cut = std::max(
                    halo_cut, static_cast<double>(*s.neighbor_provider_reach));
            if (!s.has_halo_send)
                this->bind_defaults(comm::HaloSend{halo_cut});
            if (!s.has_halo_recv)
                this->bind_defaults(comm::HaloRecv{});
        }

        // A constraint needs ghost positions AFTER the drift and ghost velocities
        // AFTER the second half-kick. The built-in forward halo fires once, at
        // pre_neighbor, and carries positions only — so at the position stage the
        // ghost data is a step stale, and ghost velocities never exist at all.
        //
        // Auto-inject the two refreshes rather than documenting them. Requiring the
        // user to remember them would mean that forgetting one line produces
        // constraints that are silently wrong at rank boundaries — no throw, no crash,
        // just a molecule that is not rigid on one side of a cut. That is exactly the
        // failure mode this engine keeps trying to make impossible, and it is not a
        // thing to leave to a doc comment.
        if (s.has_constraint) {
            this->on(ATV::md::Phase::post_integrate)
                .add(comm::HaloRefresh{comm::HaloRefresh::Field::positions},
                     -100);
            this->on(ATV::md::Phase::final_integrate)
                .add(comm::HaloRefresh{comm::HaloRefresh::Field::velocities},
                     +150);
        }

        // A constant-potential charge solver rewrites owned charges at
        // pre_force (ConstPot* bind at -500/-50) — AFTER the step's forward
        // halo at pre_neighbor. Without a refresh, the force stages read
        // ghost charges one solve stale (zero at step 0): forces are silently
        // wrong exactly at rank boundaries. Same "make forgetting impossible"
        // rationale as the constraint refreshes above. Idempotent, so a
        // user-registered charge refresh alongside this one is harmless.
        // LAMMPS twin: fix electrode forward-communicates q after every conp
        // solve (Ahrens-Iwers et al., J. Chem. Phys. 157, 084801 (2022)).
        if (s.has_charge_updater) {
            this->on(ATV::md::Phase::pre_force)
                .add(comm::HaloRefresh{comm::HaloRefresh::Field::charges},
                     +900);
        }
#endif

        this->throw_if_missing_neighbor_list_(
            s, "MD",
            "Add `sim.set_neighbor(cpu::md::NeighborList<>{cutoff, skin})`.");

        this->push_timestep_();
    }

  private:
    /// @brief Push the integrator's timestep to every `TimestepConsumer`.
    ///
    /// Called LAST from `install_defaults()` — i.e. strictly before any
    /// component `build()`, so a consumer can validate the dt in its own build.
    ///
    /// The two checks below are gated differently, on purpose:
    ///   - **Conflicting providers** throw unconditionally, whether or not
    ///     a consumer is registered — two `TimestepProvider`s publishing
    ///     different dt is an ambiguous clock either way.
    ///   - **Dt validity** (must be finite and > 0) and the **push** itself
    ///     run only when at least one `TimestepConsumer` is registered. A
    ///     run with zero consumers never reads the value, so `dt = 0` stays
    ///     legal — e.g. the shipped energy-only MPI harnesses register
    ///     `VelocityVerlet{0.0}` purely to satisfy MD's Integrator
    ///     requirement and deliberately move nothing.
    void push_timestep_() {
        // ── The nuclear timestep is single-sourced from the integrator ────
        //
        // The integrator owns the clock. Anything that must advance on the
        // SAME clock — a Car-Parrinello electronic propagator, an
        // extended-Lagrangian density — used to re-declare `dt` as its own
        // member, and "must match the nuclear dt" was a docs-only contract: no
        // runtime check was even possible, so a typo bought a silently wrong
        // trajectory instead of an error. The dt is published by a
        // `TimestepProvider` and pushed here, at the END of install_defaults()
        // — i.e. strictly before any component `build()` — so a consumer can
        // already validate it in its own build.
        struct PublishedDt {
            std::string owner;
            double dt_fs;
        };
        std::vector<PublishedDt> published;
        std::unordered_set<const cpu::TimestepProvider*> seen_providers;
        for_each_pending([&](const cpu::Component& c) {
            const cpu::TimestepProvider* p = c.as_timestep_provider();
            // The walk visits BINDINGS: VelocityVerlet declares two
            // (initial_integrate + final_integrate), so dedup by identity.
            if (!p || !seen_providers.insert(p).second)
                return;
            published.push_back({c.name(), p->timestep()});
        });

        // Two integrators on different clocks is a broken setup whether or not
        // anything consumes the dt: one of them is wrong and the engine cannot
        // know which. Name BOTH — naming one leaves the user diffing the whole
        // component set for the other.
        for (std::size_t i = 1; i < published.size(); ++i) {
            const double a = published[0].dt_fs;
            const double b = published[i].dt_fs;
            const double diff = std::fabs(a - b);
            const double scale = std::max(std::fabs(a), std::fabs(b));
            const bool disagree =
                std::isfinite(diff) ? diff > 1e-12 * scale : !(a == b);
            if (disagree)
                throw std::runtime_error(
                    "MD::build(): conflicting MD timesteps — integrator '"
                    + published[0].owner + "' publishes " + std::to_string(a)
                    + " fs, integrator '" + published[i].owner + "' publishes "
                    + std::to_string(b)
                    + " fs; exactly one clock may own the run");
        }

        // Collect the consumers BEFORE validating anything: whether the
        // published dt must be *usable* is a question only a consumer can ask.
        // A run with no consumer never reads the value — the shipped
        // energy-only MPI harnesses register `VelocityVerlet{0.0}` purely to
        // satisfy MD's Integrator requirement and deliberately move nothing —
        // so rejecting dt = 0 there would fail a setup that is, for that run,
        // entirely well defined. The CONFLICT check above is different in kind
        // and stays unconditional: two clocks are ambiguous whether or not
        // anybody reads them.
        struct PendingConsumer {
            cpu::TimestepConsumer* consumer;
            std::string owner;
        };
        std::vector<PendingConsumer> consumers;
        std::unordered_set<const cpu::TimestepConsumer*> seen_consumers;
        auto collect_dt = [&](cpu::TimestepConsumer* tc,
                              const std::string& who) {
            // Dedup for the same reason as the providers: one consumer object
            // must be handed the dt exactly once, not once per binding.
            if (!tc || !seen_consumers.insert(tc).second)
                return;
            consumers.push_back({tc, who});
        };
        for_each_pending_kind_mut(
            [&](cpu::Component& c) {
                collect_dt(c.as_timestep_consumer(), c.name());
            },
            [&](cpu::DriverNode* d) {
                // A nested method driver (MeanFieldMethod) is a DriverNode, not
                // a Component, so it is reached by cross-cast — the same route
                // validate_components() uses for the `HF` force source.
                const auto* as_comp = dynamic_cast<const cpu::Component*>(d);
                collect_dt(dynamic_cast<cpu::TimestepConsumer*>(d),
                           as_comp ? as_comp->name()
                                   : std::string("nested method driver"));
            });

        if (!consumers.empty()) {
            // Validity is checked once, before the FIRST push: no consumer may
            // receive a dt the engine already knows is unusable.
            if (!published.empty()
                && (!(published[0].dt_fs > 0.0)
                    || !std::isfinite(published[0].dt_fs)))
                throw std::runtime_error("MD::build(): integrator '"
                                         + published[0].owner
                                         + "' publishes an unusable timestep "
                                         + std::to_string(published[0].dt_fs)
                                         + " fs (must be finite and > 0)");

            for (const PendingConsumer& pc : consumers) {
                if (published.empty()) {
                    if (pc.consumer->timestep_required())
                        throw std::runtime_error(
                            "MD::build(): '" + pc.owner
                            + "' requires the nuclear timestep, but no "
                              "registered component implements "
                              "TimestepProvider; add an MD integrator (e.g. "
                              "sim.set_integrator(cpu::md::VelocityVerlet{dt}"
                              "))");
                    // opt-in means opt-in: never push an unasked-for dt
                    continue;
                }
                pc.consumer->set_timestep(published[0].dt_fs);
            }
        }
    }

    std::optional<std::string> integrator_set_;
    int scalar_reduce_every_ = 1;
};

} // namespace md
} // namespace cpu
} // namespace ATV
