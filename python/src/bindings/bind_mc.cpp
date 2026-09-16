/// @file python/src/bindings/bind_mc.cpp
/// @brief Monte Carlo presets + movers (spec py-mc), aligned with the C++
///        categorical API of `cpu::mc::MC` / `NptMc` / `Gcmc`.
///
/// Public surface mirrors `regressions/mc/`:
///
/// @code
///   sim = MC(temperature, seed)      # temperature [K]
///   sim.set_evaluator(PairLJCut(eps, sigma, cutoff))
///   sim.set_neighbor(NeighborList(cutoff, skin))
///   sim.set_mover(TranslateMove(delta))
///   sim.run(frame, n_trials)          # continues ONE Markov chain
/// @endcode
///
/// Three things make this handle different from `MdHandle`:
///
///  1. **Chain continuation.** `McBase::run` has no `molrs::FrameRef` overload
///     and is continuable by contract (block sampling). The handle therefore
///     materializes its working frame ONCE and every later `run()` extends the
///     same chain; a fresh chain is a fresh handle object.
///  2. **Variable N.** μVT inserts and deletes atoms, and `cpu::to_molrs`
///     rejects a changed row count (molrs `RaggedAxis0`). When N changed the
///     binding rebuilds the molrs `atoms` block through the molrs *Python* API
///     before exporting — see `rebuild_atoms_block_`.
///  3. **Single rank.** There is no MC task in the launcher protocol, so a run
///     under `atomiverse-launcher` is refused instead of silently sampling N
///     independent chains.
///
/// Whole body gated on `ATV_HAS_MC` (emitted by `AtvProductModules.cmake` for
/// `ATV_BUILD_MC=ON`), matching the `bind.cpp` dispatch line.

#ifdef ATV_HAS_MC

#include "molrs_frame_bridge.h"
#include "owned_buffer.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/molrs_exchange.h"
#include "atomiverse/cpu/montecarlo/gcmc.h"
#include "atomiverse/cpu/montecarlo/insert_delete_move.h"
#include "atomiverse/cpu/montecarlo/mc.h"
#include "atomiverse/cpu/montecarlo/move_selector.h"
#include "atomiverse/cpu/montecarlo/npt_mc.h"
#include "atomiverse/cpu/montecarlo/transaction.h"
#include "atomiverse/cpu/montecarlo/translate_move.h"
#include "atomiverse/cpu/montecarlo/volume_move.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/task/launcher_gate.h"
#include "atomiverse/cpu/task/task_spec.h"

using namespace ATV;

namespace {

using Verb = ATV::cpu::mc::Transaction::Verb;

/// @brief The five trial verbs `McBase` tallies, in enum order.
///
/// `Verb::None` is a real, decided trial — a deletion drawn on an empty cell
/// or a volume draw that would take V <= 0 — that the policy always rejects.
/// It has no Python spelling (nobody proposes a "none" move on purpose) but
/// it IS counted in the verb-less aggregate, so `attempts()` is exactly the
/// number of trials the chain decided.
constexpr int kVerbCount = 5;

/// @brief Map a Python verb name onto a `Transaction::Verb`.
/// @param name One of `"displace"`, `"box"`, `"insert"`, `"delete"`.
/// @return The matching trial verb.
/// @throws std::invalid_argument (Python `ValueError`) on any other string —
///         the statistics accessors take a verb name, never free text.
Verb verb_from_name(std::string_view name) {
    if (name == "displace")
        return Verb::Displace;
    if (name == "insert")
        return Verb::Insert;
    if (name == "delete")
        return Verb::Delete;
    if (name == "box")
        return Verb::Box;
    throw std::invalid_argument(
        "unknown Monte Carlo trial verb '" + std::string(name)
        + "'; expected one of 'displace', 'box', 'insert', 'delete' (or no "
          "argument for the aggregate over every verb)");
}

/// @brief Copy one Frame column into a fresh, numpy-owned 1-D array.
///
/// The array owns its buffer through the shared `to_owned_array()` primitive
/// (`owned_buffer.h`): molrs copies the data into its own column storage on
/// `Block.insert`, and the engine's working frame keeps evolving afterwards,
/// so nothing may alias either side. This wrapper adds the engine-column
/// spelling (pointer + row count) and the dtype conversion the molrs schema
/// asks for — the one call site that needs a non-`double` element type, which
/// is why it cannot go through `result_array.h`.
///
/// @tparam Out numpy element type of the produced array (`double` /
///         `std::uint32_t`) — the dtype the molrs schema declares for that
///         column.
/// @tparam In Element type of the engine column being copied.
/// @param src First element of the contiguous engine column.
/// @param n Number of rows to copy.
/// @return A 1-D numpy array of length `n` holding its own buffer.
template <typename Out, typename In>
nb::object numpy_column(const In* src, std::size_t n) {
    using Column = nb::ndarray<nb::numpy, Out, nb::ndim<1>>;
    return to_owned_array<Column>(std::span<const In>(src, n), {n}).cast();
}

/// @brief One bound Monte Carlo preset: the driver, its working frame, and the
///        molrs write-back.
///
/// @tparam Preset `cpu::mc::MC`, `cpu::mc::NptMc` or `cpu::mc::Gcmc` — the
///         three presets differ only in the `OutcomePolicy` their constructor
///         attaches, so one handle serves all three and the statistics come
///         from the shared `McBase` tallies rather than from a preset's own
///         accessors (uniform across ensembles).
template <typename Preset>
struct McHandle {
    // Declared BEFORE `driver`: members are destroyed in reverse declaration
    // order, and `~McBase` flushes the deferred component `destroy()`, which
    // dereferences the frame the run was bound to (`DestroyContext{ctx,
    // *frame_}`). The working frame must therefore outlive the driver.
    cpu::Frame work_{0};
    Preset driver;

    /// @brief Construct the preset in place.
    /// @param args Preset constructor arguments (temperature, seed, ...).
    template <typename... Args>
    explicit McHandle(Args&&... args) : driver(std::forward<Args>(args)...) {}

    // A live chain owns a frame the driver points at and components that were
    // built against it; neither copies nor moves meaningfully, and nanobind
    // probes both traits. Deleting them keeps the probes honest.
    McHandle(const McHandle&) = delete;
    McHandle& operator=(const McHandle&) = delete;
    McHandle(McHandle&&) = delete;
    McHandle& operator=(McHandle&&) = delete;

    /// @brief Append an energy source (plural slot).
    /// @param p Lennard-Jones parameters, as the shared task mirror.
    void set_evaluator(const cpu::task::PairLJCutParams& p) {
        require_configurable_("set_evaluator");
        driver.set_evaluator(cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff});
    }

    /// @brief Install the neighbor list in its MC-correct form.
    ///
    /// delay = 0, NOT the MD params default of 10: a deferred displacement
    /// check is a correctness bug here. An accepted trial move can push an
    /// atom past the skin at any trial, and the very next trial's energy would
    /// then be scored against a stale list — the sampled Hamiltonian would
    /// silently stop being the one the user configured. MD tolerates
    /// `delay 10` because its per-step displacement is bounded by dt; a Monte
    /// Carlo proposal is not.
    ///
    /// @param p Cutoff [Å] and skin [Å], as the shared task mirror.
    void set_neighbor(const cpu::task::NeighborListParams& p) {
        require_configurable_("set_neighbor");
        driver.set_neighbor(
            cpu::md::NeighborList<>{p.cutoff, p.skin, 1, 0, true});
    }

    /// @brief Install the trial-move generator (singular slot).
    /// @tparam M A `cpu::Mover` — one of the four bound move types.
    /// @param m The mover to copy in (they are all small value types).
    template <typename M>
    void set_mover(const M& m) {
        require_configurable_("set_mover");
        driver.set_mover(M{m});
    }

    /// @brief Advance the chain by `n_trials` trial moves.
    ///
    /// The FIRST call materializes the working frame from `frame_obj` and
    /// compiles the plan; every later call continues the same chain, so edits
    /// made to the molrs Frame between calls are ignored.
    ///
    /// @param frame_obj A `molpy.Frame`; results are written back into it.
    /// @param n_trials Number of trial moves to attempt in this call.
    /// @throws std::runtime_error under the MPI launcher, or from `build()`
    ///         when a required component (Mover / energy source / neighbor
    ///         list) is missing.
    void run(nb::object frame_obj, int n_trials) {
        // Probed FIRST, before any frame work: there is no MC task in the
        // launcher protocol, so under a multi-rank host every rank would
        // silently sample its own independent chain and the gathered answer
        // would be one of them, not their average.
        const cpu::task::LauncherGate gate;
        if (gate.active())
            throw std::runtime_error(
                "Monte Carlo is single-rank by design (no MC task exists in "
                "the launcher protocol); run this script outside "
                "atomiverse-launcher / on one rank.");
        if (!built_) {
            driver.build();
            built_ = true;
        }
        auto fref = borrow_molrs_frame(frame_obj);
        if (!materialized_) {
            work_ = cpu::from_molrs(*fref);
            materialized_ = true;
        }
        {
            nb::gil_scoped_release release;
            driver.run(work_, n_trials);
        }
        write_back_(frame_obj, *fref);
    }

    /// @brief Trials of one verb, or of every verb when `verb` is omitted.
    /// @param verb Trial verb name, or `std::nullopt` for the aggregate.
    /// @return Attempt count across every `run()` call so far.
    long long attempts(const std::optional<std::string>& verb) const {
        if (verb)
            return base_().attempts(verb_from_name(*verb));
        return sum_verbs_(&cpu::mc::McBase::attempts);
    }

    /// @brief Accepted trials of one verb, or of every verb.
    /// @param verb Trial verb name, or `std::nullopt` for the aggregate.
    /// @return Accept count across every `run()` call so far.
    long long accepts(const std::optional<std::string>& verb) const {
        if (verb)
            return base_().accepts(verb_from_name(*verb));
        return sum_verbs_(&cpu::mc::McBase::accepts);
    }

    /// @brief Accepted fraction, in [0, 1].
    /// @param verb Trial verb name, or `std::nullopt` for the aggregate.
    /// @return accepts / attempts, or 0.0 when nothing was attempted.
    double acceptance_ratio(const std::optional<std::string>& verb) const {
        const long long a = attempts(verb);
        return a > 0
                   ? static_cast<double>(accepts(verb)) / static_cast<double>(a)
                   : 0.0;
    }

    /// @brief Number of atoms in the chain's current configuration.
    /// @return `nlocal` of the working frame, or 0 before the first `run()`
    ///         (no chain exists yet, so there is no configuration to count).
    int n_atoms() const { return materialized_ ? work_.nlocal() : 0; }

  private:
    /// @brief Refuse a configuration change once the chain has started.
    ///
    /// The execution plan is compiled exactly once, on the first `run()`
    /// (`Driver::build` consumes its pending list, so it is not a repeatable
    /// operation). A setter called afterwards would therefore have no effect
    /// on the running chain — silently, which is the worst outcome — so it
    /// fails loud instead.
    ///
    /// @param setter Name of the setter being refused, for the message.
    /// @throws std::runtime_error when the chain is already running.
    void require_configurable_(const char* setter) const {
        if (built_)
            throw std::runtime_error(
                std::string(setter)
                + ": the sampler has already run and the execution plan is "
                  "compiled once, so this would silently do nothing. "
                  "Configure the sampler fully before the first run(); a "
                  "different configuration means a new sampler object, which "
                  "is also what starts a fresh chain.");
    }

    /// @brief The shared MC loop, for the per-verb tallies.
    /// @return This preset as its `McBase` base.
    const cpu::mc::McBase& base_() const { return driver; }

    /// @brief One of `McBase`'s per-verb tally accessors.
    using VerbTally = long long (cpu::mc::McBase::*)(Verb) const;

    /// @brief Aggregate one per-verb tally over every verb, `Verb::None`
    ///        included — which is what makes the verb-less `attempts()` exactly
    ///        the number of trials the chain decided.
    /// @param tally `&McBase::attempts` or `&McBase::accepts`.
    /// @return The sum of that tally across all five verbs.
    long long sum_verbs_(VerbTally tally) const {
        long long total = 0;
        for (int v = 0; v < kVerbCount; ++v)
            total += (base_().*tally)(static_cast<Verb>(v));
        return total;
    }

    /// @brief Export the working frame into the molrs Frame it came from.
    ///
    /// `cpu::to_molrs` writes into the EXISTING `atoms` block, so it is valid
    /// only while the row count still matches; μVT changes it, and molrs
    /// rejects a block whose columns disagree in length (`RaggedAxis0`). The
    /// rebuild below re-establishes the block at the new N first; `to_molrs`
    /// then fills x/v/f, charge, image, tag, the box and the typed metadata as
    /// on every other driver path.
    ///
    /// @param frame_obj The `molpy.Frame` (needed for the Python-API rebuild).
    /// @param fref The same frame borrowed as a CXX handle.
    void write_back_(nb::object frame_obj, molrs::FrameRef& fref) {
        const auto rows = molrs::frame_block_nrows(fref, "atoms");
        if (rows != static_cast<std::int64_t>(work_.nlocal()))
            rebuild_atoms_block_(frame_obj);
        cpu::to_molrs(work_, fref);
    }

    /// @brief Replace the molrs `atoms` block with one at the chain's current
    ///        population, through the molrs Python API (GIL held).
    ///
    /// Every column is rebuilt so the new block is internally consistent at
    /// the new N — a partial rebuild would leave the untouched columns at the
    /// old length and molrs would reject the block. `mass` and
    /// `atomic_number` are here because `to_molrs` deliberately does NOT write
    /// them (per-run immutable inputs on the fixed-N path); after an insertion
    /// they are outputs, stamped on the new row by `InsertDeleteMove`.
    ///
    /// @param frame_obj The `molpy.Frame` whose `atoms` block is replaced.
    void rebuild_atoms_block_(nb::object frame_obj) {
        const auto n = static_cast<std::size_t>(work_.nlocal());
        nb::object block = nb::module_::import_("molrs").attr("Block")();
        nb::object insert = block.attr("insert");
        insert("x", numpy_column<double>(work_.span(kAtom_x).data(), n));
        insert("y", numpy_column<double>(work_.span(kAtom_y).data(), n));
        insert("z", numpy_column<double>(work_.span(kAtom_z).data(), n));
        insert("vx", numpy_column<double>(work_.span(kAtom_vx).data(), n));
        insert("vy", numpy_column<double>(work_.span(kAtom_vy).data(), n));
        insert("vz", numpy_column<double>(work_.span(kAtom_vz).data(), n));
        insert("fx", numpy_column<double>(work_.span(kAtom_fx).data(), n));
        insert("fy", numpy_column<double>(work_.span(kAtom_fy).data(), n));
        insert("fz", numpy_column<double>(work_.span(kAtom_fz).data(), n));
        insert("charge",
               numpy_column<double>(work_.span(kAtom_charge).data(), n));
        insert("mass", numpy_column<double>(work_.span(kAtom_mass).data(), n));
        // Z is the species key on both sides of the boundary: molrs stores it
        // as an unsigned column, `from_molrs` lands it in `kAtom_type`, and
        // `InsertDeleteMove(z, mass)` stamps that same int on every insertion.
        insert("atomic_number",
               numpy_column<std::uint32_t>(work_.span(kAtom_type).data(), n));
        frame_obj["atoms"] = block;
    }

    bool built_ = false;        ///< Has the execution plan been compiled?
    bool materialized_ = false; ///< Does `work_` hold a live chain?
};

/// @brief Register the methods every MC preset shares.
///
/// The three presets differ only in their acceptance policy, so the setters,
/// the chain entry point and the statistics are identical; `NptMC` adds
/// `current_volume()` on top of this at its own registration site.
///
/// @tparam Preset The C++ preset behind the handle.
/// @param cls The nanobind class being defined.
template <typename Preset>
void bind_mc_common(nb::class_<McHandle<Preset>>& cls) {
    using Handle = McHandle<Preset>;
    cls.def(
           "set_evaluator",
           [](Handle& self, const cpu::task::PairLJCutParams& p) {
               self.set_evaluator(p);
           },
           nb::arg("comp"),
           "Append an energy source (plural slot: each call adds another term "
           "and the trial energies are summed).\n\n"
           "Monte Carlo scores energies only — the `energy` phase runs with "
           "forces switched off — so a pair style here costs a full-frame "
           "energy per trial.\n\n"
           "Two sizing rules the engine does NOT check for you. PairLJCut "
           "needs a NeighborList whose cutoff is >= the LJ cutoff, or pairs "
           "inside the potential's range are never enumerated. And pair "
           "distances are taken under the minimum-image convention (only the "
           "nearest periodic copy of each neighbour is seen), so "
           "cutoff + skin must stay below half the shortest box edge — under "
           "NptMC check that at the SMALLEST volume the chain visits, not "
           "only at the starting cell.")
        .def(
            "set_neighbor",
            [](Handle& self, const cpu::task::NeighborListParams& p) {
                self.set_neighbor(p);
            },
            nb::arg("comp"),
            "Install the neighbor list — the cached list of pairs closer than "
            "cutoff + skin, which is what makes a trial energy cost O(N) "
            "instead of O(N^2). The SKIN [A] is the extra margin the list is "
            "built with beyond the interaction cutoff, so the same list stays "
            "valid while atoms move a little.\n\n"
            "The MC-correct form is built here: every=1, delay=0, check=yes — "
            "the rebuild check runs on every trial and is never deferred, "
            "because a single accepted trial move can push an atom past the "
            "skin at any point (MD can defer it because one timestep moves an "
            "atom by a bounded amount; a Monte Carlo proposal is not "
            "bounded).")
        .def(
            "set_mover",
            [](Handle& self, const cpu::mc::TranslateMove& m) {
                self.set_mover(m);
            },
            nb::arg("mover"),
            "Install the trial-move generator (singular slot: one mover, "
            "replaced rather than added to). Pass a MoveSelector to mix move "
            "types. A displacement move is valid in all three ensembles.")
        .def(
            "set_mover",
            [](Handle& self, const cpu::mc::VolumeMove& m) {
                self.set_mover(m);
            },
            nb::arg("mover"),
            "Install a volume move as the only trial move (singular). It "
            "belongs to NptMC, whose acceptance rule carries the pressure "
            "and volume-entropy terms. On MC (NVT) the canonical rule would "
            "judge it by exp(-dE/(k_B T)) alone — accepted, and sampling no "
            "recognised ensemble — and on GCMC it raises RuntimeError during "
            "run(). A volume-only chain never moves the atoms relative to "
            "each other, so NPT normally goes through a MoveSelector.")
        .def(
            "set_mover",
            [](Handle& self, const cpu::mc::InsertDeleteMove& m) {
                self.set_mover(m);
            },
            nb::arg("mover"),
            "Install an insertion/deletion move as the only trial move "
            "(singular). It belongs to GCMC, whose acceptance rule carries "
            "the chemical potential. On NptMC it raises RuntimeError during "
            "run(); on MC (NVT) the canonical rule would accept it by "
            "energy alone, silently sampling neither NVT nor muVT. muVT "
            "normally goes through a MoveSelector so the atoms also move.")
        .def(
            "set_mover",
            [](Handle& self, const cpu::mc::MoveSelector& m) {
                self.set_mover(m);
            },
            nb::arg("mover"),
            "Install a weighted mix of trial-move types (singular). The "
            "selector fires exactly ONE child per trial — that is the Monte "
            "Carlo contract, and it is why a mixed ensemble goes through a "
            "MoveSelector instead of two set_mover calls.")
        .def("run", &Handle::run, nb::arg("frame"), nb::arg("n_trials"),
             "Attempt n_trials trial moves on a molpy.Frame. GIL released "
             "during sampling.\n\n"
             "n_trials counts SINGLE TRIAL MOVES, not sweeps: each count is "
             "one proposal (one atom displaced, or one volume change, or one "
             "insertion/deletion) followed by one accept/reject decision. It "
             "is NOT the 'one attempted move per particle' sweep some texts "
             "count in, so a chain of N atoms needs roughly N times more "
             "trials here than it would sweeps there.\n\n"
             "Consecutive calls CONTINUE one Markov chain — the standard "
             "run(equilibrate); run(sample) pattern — so the tallies "
             "accumulate and the configuration carries over. The frame is "
             "materialized once, on the first call: edits made to the "
             "molpy.Frame between calls are ignored, and a fresh chain means "
             "a fresh sampler object.\n\n"
             "Results are written back into the SAME molpy.Frame after every "
             "call: positions [A], the box, and the energy of the last "
             "accepted configuration in frame.meta['energy'] [eV]. When the "
             "trial moves changed the particle number the 'atoms' block is "
             "rebuilt at the new N.\n\n"
             "Monte Carlo scores energies only — no gradient is ever "
             "assembled — so the frame's fx/fy/fz come back ZEROED, not as "
             "physical forces; use an MD or Evaluate driver if you want "
             "forces. Velocities are left as they were (an inserted atom "
             "gets zero velocity), because nothing in this sampler moves in "
             "time.\n\n"
             "Raises RuntimeError under the MPI launcher (Monte Carlo is "
             "single-rank by design), and before any sampling when no Mover, "
             "no energy source, or no neighbor list for a pair style was "
             "registered.")
        .def(
            "current_energy",
            [](const Handle& self) {
                return static_cast<double>(self.driver.current_energy());
            },
            "Energy [eV] of the current (last-accepted) configuration. A "
            "rejected trial's energy never survives — this is the same value "
            "written to frame.meta['energy'].")
        .def("attempts", &Handle::attempts,
             nb::arg("verb") = std::optional<std::string>{},
             "Trial moves attempted across every run() call.\n\n"
             "A VERB is what kind of move a trial turned out to be: "
             "'displace' (one atom moved), 'box' (the cell rescaled), "
             "'insert' / 'delete' (one atom added or removed). Pass one of "
             "those names for that verb alone; an unknown name raises "
             "ValueError.\n\n"
             "With no argument you get the aggregate, which also counts the "
             "trials that ended up with NO verb — a deletion drawn on an "
             "empty cell, or a volume draw that would have taken V <= 0. "
             "Those are real, decided trials (always rejected), so the "
             "aggregate is exactly the number of trials run so far, while "
             "the four named verbs can sum to less.")
        .def("accepts", &Handle::accepts,
             nb::arg("verb") = std::optional<std::string>{},
             "Trial moves accepted across every run() call; same verb "
             "argument as attempts().")
        .def("acceptance_ratio", &Handle::acceptance_ratio,
             nb::arg("verb") = std::optional<std::string>{},
             "Accepted fraction in [0, 1] (0 when nothing was attempted); "
             "same verb argument as attempts().\n\n"
             "A healthy chain sits strictly inside (0, 1): 0 means every "
             "trial was rejected (a frozen chain — often a NaN from an "
             "overlapping start), 1 means the exp(-dE/(k_B T)) factor never "
             "bit, i.e. the proposals are too timid to cost any energy.")
        .def("n_atoms", &Handle::n_atoms,
             "Atoms in the current configuration — the fluctuating observable "
             "of the muVT ensemble. 0 before the first run(), when no chain "
             "exists yet.");
}

} // namespace

/// @brief Register the Monte Carlo surface on the `_atomiverse` module.
/// @param m The extension module under construction.
void bind_mc(nb::module_& m) {
    // ── Movers ──
    //
    // Bound as the real C++ components, not as task-spec params mirrors: MC
    // has no launcher task to stay wire-compatible with, so there is nothing
    // for a mirror to buy.

    nb::class_<cpu::mc::TranslateMove>(
        m, "TranslateMove",
        "Single-particle displacement: pick one atom uniformly at random and "
        "move it by an independent uniform draw in [-delta, delta] on each "
        "axis. The displaced atom is wrapped back into the primary cell, so "
        "coordinates stay bounded over a long chain.\n\n"
        "The proposal is SYMMETRIC — proposing the reverse displacement is "
        "exactly as likely as proposing the forward one — so the two "
        "proposal probabilities cancel in the acceptance ratio and nothing "
        "but the energy is left, which is what makes the plain "
        "min(1, exp(-dE/(k_B T))) Metropolis rule correct here.\n\n"
        "delta [A] is the knob that sets the acceptance ratio: too large and "
        "almost every trial overlaps a neighbour and is rejected, too small "
        "and the chain crawls over the configuration space. The usual target "
        "is an acceptance ratio of roughly 0.3-0.5.")
        .def(nb::init<double>(), nb::arg("delta"),
             "Construct with the displacement half-width delta [A] (> 0).");

    nb::class_<cpu::mc::VolumeMove>(
        m, "VolumeMove",
        "Isotropic volume change for NPT-MC: draw dV uniformly in [-dv_max, "
        "dv_max], then scale the cell and every atom position by the same "
        "factor (V'/V)^(1/3) — isotropic meaning the three cell edges are "
        "stretched by that one factor, so the cell keeps its shape.\n\n"
        "Drawing the new volume uniformly IN V (rather than in ln V) is what "
        "pairs with the N k_B T ln(V'/V) term of the NptMC acceptance rule; "
        "the two belong together and neither is a free choice. A draw that "
        "would make the volume zero or negative is rejected outright, with "
        "no acceptance test at all.\n\n"
        "Every atom moves, so a volume trial costs one full-frame energy "
        "just like any other trial — but it is the only move that samples "
        "the volume, so do not starve it in the MoveSelector weights.")
        .def(nb::init<double>(), nb::arg("dv_max"),
             "Construct with the volume half-width dv_max [A^3] (> 0).");

    nb::class_<cpu::mc::InsertDeleteMove>(
        m, "InsertDeleteMove",
        "Grand-canonical insertion/deletion of a single species: with equal "
        "probability either insert one atom at a uniform position in the cell "
        "or delete one uniformly chosen atom.\n\n"
        "A deletion drawn on an empty cell is simply rejected. An inserted "
        "atom carries the configured species and mass and nothing else: zero "
        "velocity, zero charge, no bonds, and a freshly allocated atom tag "
        "(never a recycled one). Its mass MUST be the mass GCMC built its "
        "thermal wavelength Lambda from — mixing the two is a silently wrong "
        "chemical potential, and nothing checks it.")
        .def(nb::init<int, double>(), nb::arg("z"), nb::arg("mass"),
             "Construct with the atomic number z (dimensionless, > 0) "
             "stamped on inserted atoms and their mass [amu] (> 0).\n\n"
             "z is the species key on both sides of the molrs boundary: the "
             "frame's 'atomic_number' column becomes the engine's atom type, "
             "so an inserted Ar atom is z = 18 and comes back out in the "
             "rebuilt 'atomic_number' column.");

    nb::class_<cpu::mc::MoveSelector>(
        m, "MoveSelector",
        "A Mover of Movers: each trial it draws ONE child move type with the "
        "configured weights and fires exactly that one.\n\n"
        "This is how a mixed ensemble is built — NPT as displacements + "
        "volume moves, muVT as displacements + insert/delete. Registering two "
        "movers side by side would instead fire both every trial, which is "
        "not a Monte Carlo trial at all.")
        .def(nb::init<>(),
             "Construct an empty selector; populate with add(). A selector "
             "that reaches run() with no child raises RuntimeError.")
        .def(
            "add",
            [](cpu::mc::MoveSelector& self, const cpu::mc::TranslateMove& m,
               double weight) -> cpu::mc::MoveSelector& {
                self.add(cpu::mc::TranslateMove{m}, weight);
                return self;
            },
            nb::arg("mover"), nb::arg("weight"), nb::rv_policy::reference,
            "Register a child move type with a relative draw weight (> 0); "
            "returns this selector so the calls chain.")
        .def(
            "add",
            [](cpu::mc::MoveSelector& self, const cpu::mc::VolumeMove& m,
               double weight) -> cpu::mc::MoveSelector& {
                self.add(cpu::mc::VolumeMove{m}, weight);
                return self;
            },
            nb::arg("mover"), nb::arg("weight"), nb::rv_policy::reference,
            "Register a child move type with a relative draw weight (> 0).")
        .def(
            "add",
            [](cpu::mc::MoveSelector& self, const cpu::mc::InsertDeleteMove& m,
               double weight) -> cpu::mc::MoveSelector& {
                self.add(cpu::mc::InsertDeleteMove{m}, weight);
                return self;
            },
            nb::arg("mover"), nb::arg("weight"), nb::rv_policy::reference,
            "Register a child move type with a relative draw weight (> 0).");

    // ── Presets ──

    nb::class_<McHandle<cpu::mc::MC>> mc(
        m, "MC",
        "Canonical (NVT) Metropolis Monte Carlo — the particle number N, the "
        "cell volume V and the temperature T are all held fixed.\n\n"
        "Monte Carlo integrates no equations of motion. It walks a MARKOV "
        "CHAIN of configurations: each one is proposed from the previous one "
        "and from nothing else. One TRIAL is a single pass of that walk — the "
        "registered Mover proposes a small random change to the "
        "configuration, the registered energy sources score the changed "
        "configuration, and the proposal is then either kept (accepted) or "
        "undone (rejected).\n\n"
        "Keeping it with probability min(1, exp(-dE / (k_B T))) — the "
        "METROPOLIS rule, in which dE [eV] is the trial energy minus the "
        "energy of the last accepted configuration, k_B is the Boltzmann "
        "constant (8.617333e-5 eV/K) and T [K] the temperature — is what "
        "makes the visited configurations appear with the Boltzmann "
        "probability exp(-E/(k_B T)). A plain average over the chain is then "
        "the NVT ensemble average, with no dynamics and no velocities "
        "involved.\n\n"
        "Configure the sampler completely (energy source, neighbor list, "
        "mover) before the first run(); help(atomiverse.montecarlo) states "
        "the contract the three presets share.");
    mc.def(
        "__init__",
        [](McHandle<cpu::mc::MC>* self, double temperature,
           std::uint64_t seed) {
            new (self)
                McHandle<cpu::mc::MC>(static_cast<AccReal>(temperature), seed);
        },
        nb::arg("temperature"), nb::arg("seed"),
        "Construct with the ensemble temperature [K] (> 0) and the seed of "
        "the single RNG stream shared by the movers and the acceptance test "
        "(one seed, one reproducible chain). A temperature <= 0 raises "
        "ValueError: the Boltzmann weight exp(-dE/(k_B T)) is undefined "
        "there.");
    bind_mc_common(mc);

    nb::class_<McHandle<cpu::mc::NptMc>> npt(
        m, "NptMC",
        "Isothermal-isobaric (NPT) Monte Carlo — the particle number N, the "
        "external pressure P and the temperature T are held fixed while the "
        "cell volume is sampled, so the cell finds the size at which the "
        "system's own pressure matches P.\n\n"
        "Pair it with a MoveSelector mixing TranslateMove and VolumeMove. A "
        "displacement uses the canonical rule of MC. A volume move from V to "
        "V' is accepted with probability "
        "min(1, exp(-beta[dE + P (V' - V) - N k_B T ln(V'/V)])), where beta = "
        "1/(k_B T) and dE [eV] is the trial energy minus the last accepted "
        "energy. The P (V' - V) term is the work done against the external "
        "pressure; the N k_B T ln(V'/V) term is entropic — a volume move "
        "rescales every coordinate together with the cell, i.e. it holds the "
        "atoms' FRACTIONAL positions fixed, so a larger cell hands each of "
        "the N atoms proportionally more room, and that free volume has to be "
        "credited or the box would drift to whatever size lowers the energy "
        "alone.\n\n"
        "Insertions and deletions belong to muVT: this policy has no rule for "
        "them and raises RuntimeError when one is proposed — during run(), "
        "when the trial is judged, not at set_mover().");
    npt.def(
        "__init__",
        [](McHandle<cpu::mc::NptMc>* self, double temperature, double pressure,
           std::uint64_t seed) {
            new (self)
                McHandle<cpu::mc::NptMc>(static_cast<AccReal>(temperature),
                                         static_cast<AccReal>(pressure), seed);
        },
        nb::arg("temperature"), nb::arg("pressure"), nb::arg("seed"),
        "Construct with the ensemble temperature [K] (> 0), the external "
        "pressure [eV/A^3] (LAMMPS sign, positive = compressive; 1 bar ~ "
        "6.2415e-7 eV/A^3) and the RNG seed.");
    bind_mc_common(npt);
    npt.def(
        "current_volume",
        [](const McHandle<cpu::mc::NptMc>& self) {
            return self.driver.current_volume();
        },
        "Volume [A^3] of the current (last-accepted) configuration — the "
        "sampled degree of freedom of this ensemble, and the same cell "
        "written back to the molrs Frame box.");

    nb::class_<McHandle<cpu::mc::Gcmc>> gcmc(
        m, "GCMC",
        "Grand-canonical (muVT) Monte Carlo — the chemical potential mu, the "
        "cell volume V and the temperature T are held fixed while the "
        "particle number N is sampled.\n\n"
        "The cell is treated as being in contact with an infinite reservoir "
        "of one species that it may exchange atoms with. The CHEMICAL "
        "POTENTIAL mu [eV] is the free-energy change of adding one atom to a "
        "system, and the reservoir is held at exactly this value, which makes "
        "it the control knob of the ensemble: raising mu pushes atoms "
        "in, lowering it pulls them out, and the cell settles at whatever <N> "
        "puts it in equilibrium with the reservoir. That is the natural "
        "setting for adsorption and vapour-liquid problems, where N — not the "
        "energy — is the answer you want.\n\n"
        "Pair it with a MoveSelector mixing TranslateMove and "
        "InsertDeleteMove. Because N fluctuates, run() rebuilds the molrs "
        "'atoms' block whenever the population changed, and n_atoms() is the "
        "observable to accumulate for <N>.");
    gcmc.def(
        "__init__",
        [](McHandle<cpu::mc::Gcmc>* self, double temperature, double mu,
           double mass, std::uint64_t seed) {
            new (self)
                McHandle<cpu::mc::Gcmc>(static_cast<AccReal>(temperature),
                                        static_cast<AccReal>(mu), mass, seed);
        },
        nb::arg("temperature"), nb::arg("mu"), nb::arg("mass"), nb::arg("seed"),
        "Construct with the ensemble temperature [K] (> 0), the chemical "
        "potential mu [eV] of the single insertable species, its mass [amu] "
        "(> 0) and the RNG seed.\n\n"
        "mu is the TOTAL chemical potential — the ideal-gas part included, "
        "not the excess part alone. Concretely, the acceptance rule uses mu "
        "only through the ACTIVITY z = exp(beta mu)/Lambda^3 [1/A^3], where "
        "beta = 1/(k_B T) and Lambda = h/sqrt(2 pi m k_B T) [A] is the "
        "thermal de Broglie wavelength — the quantum length scale of a free "
        "particle of mass m at temperature T, which is what carries the "
        "'per unit volume' in z. z is the ideal-gas number density the "
        "reservoir would have at this mu, so mu = k_B T ln(z Lambda^3) is how "
        "you convert a target density into the argument passed here. Same "
        "convention as Frenkel & Smit eq. (5.6.9); transcribed with the two "
        "acceptance rules in include/atomiverse/cpu/montecarlo/gcmc.h and "
        "pinned in regressions/mc/gcmc_lj/REFERENCE.md.\n\n"
        "Because Lambda is built here from `mass`, the InsertDeleteMove that "
        "stamps atoms into the cell MUST carry that same mass — two different "
        "masses mean the sampled chemical potential is not the one requested, "
        "and nothing checks it for you.");
    bind_mc_common(gcmc);
}

#endif // ATV_HAS_MC
