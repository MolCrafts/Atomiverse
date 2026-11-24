/// @file python/src/bindings/bind_mc.cpp
/// @brief Python bindings for the MC and KMC drivers.

#ifdef ATV_HAS_MC

#include "evaluator_setters.h"
#include "molpy_frame_bridge.h"
#include "owned_buffer.h"
#include "units_binding.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "atomiverse/cpu/montecarlo/insert_delete_move.h"
#include "atomiverse/cpu/montecarlo/kmc.h"
#include "atomiverse/cpu/montecarlo/mc.h"
#include "atomiverse/cpu/montecarlo/move_selector.h"
#include "atomiverse/cpu/montecarlo/polymer_moves.h"
#include "atomiverse/cpu/montecarlo/transaction.h"
#include "atomiverse/cpu/montecarlo/translate_move.h"
#include "atomiverse/cpu/montecarlo/volume_move.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/task/internal/potential_builders.h"
#include "atomiverse/cpu/task/launcher_gate.h"
#include "atomiverse/cpu/task/task_spec.h"

using namespace ATV;

namespace {

using Verb = ATV::cpu::mc::Transaction::Verb;

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
/// (`owned_buffer.h`): molpy copies the data into its own column storage on
/// `Block.insert`, and the engine's working frame keeps evolving afterwards,
/// so nothing may alias either side. This wrapper adds the engine-column
/// spelling (pointer + row count) and the dtype conversion the molpy schema
/// asks for — the one call site that needs a non-`double` element type, which
/// is why it cannot go through `result_array.h`.
///
/// @tparam Out numpy element type of the produced array (`double` /
///         `std::uint32_t`) — the dtype the molpy schema declares for that
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

/// Owns the working Frame before the driver so it outlives teardown.
struct McHandle {
    cpu::Frame work_{0};
    std::shared_ptr<cpu::Mover> mover_;
    std::unique_ptr<cpu::mc::MC> driver;

    McHandle(std::uint64_t seed, const std::optional<std::string>& units)
        : McHandle(std::make_unique<cpu::mc::MC>(seed), units) {}
    McHandle(double temperature, std::uint64_t seed,
             const std::optional<std::string>& units)
        : McHandle(std::make_unique<cpu::mc::MC>(temperature, seed), units) {}

    // A live chain owns a frame the driver points at and components that were
    // built against it; neither copies nor moves meaningfully, and nanobind
    // probes both traits. Deleting them keeps the probes honest.
    McHandle(const McHandle&) = delete;
    McHandle& operator=(const McHandle&) = delete;
    McHandle(McHandle&&) = delete;
    McHandle& operator=(McHandle&&) = delete;

    template <typename Params, typename Evaluator>
    void set_categorical(const char* setter, const Params&, Evaluator evaluator) {
        require_configurable_(setter);
        driver->set_evaluator(std::move(evaluator));
    }

    template <typename Evaluator>
    void set_local_evaluator(Evaluator& evaluator, const char*) {
        require_configurable_("set_evaluator");
        driver->set_evaluator_ref(evaluator);
    }

    template <typename Evaluator>
    void set_unmirrored_evaluator(Evaluator evaluator, const char*, bool) {
        require_configurable_("set_evaluator");
        driver->set_evaluator(std::move(evaluator));
    }

    /// @brief Install the neighbor list in its MC-correct form.
    ///
    /// delay = 0 (the NeighborList default, spelled out because a deferred
    /// displacement check is a correctness bug here): an accepted trial move
    /// can push an atom past the skin at any trial, and the very next trial's
    /// energy would then be scored against a stale list — the sampled
    /// Hamiltonian would silently stop being the one the user configured.
    ///
    /// @param p Cutoff [Å] and skin [Å], as the shared task mirror.
    void set_neighbor(const cpu::task::NeighborListParams& p) {
        require_configurable_("set_neighbor");
        driver->set_neighbor(
            cpu::md::NeighborList<>{p.cutoff, p.skin, 1, 0, true});
    }

    void set_mover(std::shared_ptr<cpu::Mover> mover) {
        require_configurable_("set_mover");
        if (!mover)
            throw std::invalid_argument("set_mover requires a Mover");
        driver->set_mover_ref(*mover);
        mover_ = std::move(mover);
    }

    /// @brief Advance the chain by `n_trials` trial moves.
    ///
    /// The FIRST call materializes the working frame from `frame_obj` and
    /// compiles the plan; every later call continues the same chain, so edits
    /// made to the molpy Frame between calls are ignored.
    ///
    /// @param frame_obj A `molpy.Frame`; results are written back into it.
    /// @param n_trials Number of trial moves to attempt in this call.
    /// @throws std::runtime_error under the MPI launcher, or from `build()`
    ///         when a required component (Mover / energy source / neighbor
    ///         list) is missing.
    BigInt run(nb::object frame_obj, BigInt n_trials) {
        if (n_trials < 0)
            throw std::invalid_argument("MC step budget must be nonnegative");
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
            driver->build();
            built_ = true;
        }
        auto fref = borrow_molpy_frame(frame_obj);
        if (!materialized_) {
            if (mover_ && mover_->changes_connectivity())
                validate_bond_annotations_(frame_obj);
            work_ = cpu::from_molpy(*fref);
            materialized_ = true;
        }
        BigInt completed;
        {
            nb::gil_scoped_release release;
            completed = driver->run(work_, n_trials);
        }
        write_back_(frame_obj, *fref);
        return completed;
    }

    /// @brief Trials of one verb, or of every verb when `verb` is omitted.
    /// @param verb Trial verb name, or `std::nullopt` for the aggregate.
    /// @return Attempt count across every `run()` call so far.
    long long attempts(const std::optional<std::string>& verb) const {
        return verb ? driver->attempts(verb_from_name(*verb)) : driver->attempts();
    }

    /// @brief Accepted trials of one verb, or of every verb.
    /// @param verb Trial verb name, or `std::nullopt` for the aggregate.
    /// @return Accept count across every `run()` call so far.
    long long accepts(const std::optional<std::string>& verb) const {
        return verb ? driver->accepts(verb_from_name(*verb)) : driver->accepts();
    }

    /// @brief Accepted fraction, in [0, 1].
    /// @param verb Trial verb name, or `std::nullopt` for the aggregate.
    /// @return accepts / attempts, or 0.0 when nothing was attempted.
    double acceptance_ratio(const std::optional<std::string>& verb) const {
        return verb ? driver->acceptance_ratio(verb_from_name(*verb))
                    : driver->acceptance_ratio();
    }

    /// @brief Number of atoms in the chain's current configuration.
    /// @return `nlocal` of the working frame, or 0 before the first `run()`
    ///         (no chain exists yet, so there is no configuration to count).
    int n_atoms() const { return materialized_ ? work_.nlocal() : 0; }

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

  protected:
    McHandle(std::unique_ptr<cpu::mc::MC> sampler,
             const std::optional<std::string>& units)
        : driver(std::move(sampler)) {
        driver->set_units(ATV::python::resolve_units(units));
    }

  private:
    /// @brief Export the working frame into the molpy Frame it came from.
    ///
    /// `cpu::to_molpy` writes into the EXISTING `atoms` block, so it is valid
    /// only while the row count still matches; μVT changes it, and molpy
    /// rejects a block whose columns disagree in length (`RaggedAxis0`). The
    /// rebuild below re-establishes the block at the new N first; `to_molpy`
    /// then fills x/v/f, charge, image, tag, the box and the typed metadata as
    /// on every other driver path.
    ///
    /// @param frame_obj The `molpy.Frame` (needed for the Python-API rebuild).
    /// @param fref The same frame borrowed as a CXX handle.
    void write_back_(nb::object frame_obj, molrs::FrameRef& fref) {
        const auto rows = molrs::frame_block_n_rows(fref, "atoms");
        if (rows != static_cast<std::int64_t>(work_.nlocal()))
            rebuild_atoms_block_(frame_obj);
        cpu::to_molpy(work_, fref);
        if (driver->changed_connectivity())
            write_bonds_(frame_obj);
    }

    static void validate_bond_annotations_(const nb::object& frame_obj) {
        if (!nb::cast<bool>(frame_obj.attr("keys")().attr("__contains__")("bonds")))
            throw std::invalid_argument("Connectivity-changing MC requires a bonds block");
        const nb::object bonds = frame_obj["bonds"];
        const auto np = nb::module_::import_("numpy");
        for (const auto& key : bonds.attr("keys")()) {
            const auto name = nb::cast<std::string>(key);
            if (name == "atomi" || name == "atomj")
                continue;
            const nb::object values = np.attr("asarray")(bonds[key]);
            if (nb::len(values) > 1 &&
                !nb::cast<bool>(np.attr("all")(np.attr("equal")(values, values[0]))
                                    .attr("item")()))
                throw std::invalid_argument(
                    "Connectivity-changing MC requires uniform bond annotations");
        }
    }

    void write_bonds_(nb::object& frame_obj) {
        const auto tags = work_.span(kAtom_tag);
        const auto count = work_.span(kAtom_num_bond);
        const auto target = work_.span(kAtom_bond_atom);
        std::vector<std::uint32_t> from;
        std::vector<std::uint32_t> to;
        for (int row = 0; row < work_.nlocal(); ++row) {
            for (int col = 0; col < count[row]; ++col) {
                from.push_back(static_cast<std::uint32_t>(tags[row]));
                to.push_back(static_cast<std::uint32_t>(target(row, col)));
            }
        }
        auto bonds = frame_obj["bonds"].attr("copy")();
        bonds.attr("insert")("atomi", numpy_column<std::uint32_t>(from.data(), from.size()));
        bonds.attr("insert")("atomj", numpy_column<std::uint32_t>(to.data(), to.size()));
        frame_obj["bonds"] = std::move(bonds);
    }

    /// @brief Replace the molpy `atoms` block with one at the chain's current
    ///        population, through the molpy Python API (GIL held).
    ///
    /// Every column is rebuilt so the new block is internally consistent at
    /// the new N — a partial rebuild would leave the untouched columns at the
    /// old length and molpy would reject the block. `mass` and the identity
    /// columns (`atomic_number`, `type_id`) are here because `to_molpy`
    /// deliberately does NOT write them (per-run immutable inputs on the
    /// fixed-N path); after an insertion they are outputs, stamped on the new
    /// row by `InsertDeleteMove`. An identity column the input frame did not
    /// carry is not invented on the way back.
    ///
    /// @param frame_obj The `molpy.Frame` whose `atoms` block is replaced.
    void rebuild_atoms_block_(nb::object frame_obj) {
        const auto n = static_cast<std::size_t>(work_.nlocal());
        nb::object block = nb::module_::import_("molpy").attr("Block")();
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
        // One molpy column per Frame column: species Z <-> atomic_number,
        // force-field type <-> type_id, each only when the Frame carries it.
        if (work_.has("atoms", kAtom_atomic_number.key))
            insert("atomic_number",
                   numpy_column<std::uint32_t>(
                       work_.span(kAtom_atomic_number).data(), n));
        if (work_.has("atoms", kAtom_type.key))
            insert("type_id", numpy_column<std::uint32_t>(
                                  work_.span(kAtom_type).data(), n));
        frame_obj["atoms"] = block;
    }

    bool built_ = false;        ///< Has the execution plan been compiled?
    bool materialized_ = false; ///< Does `work_` hold a live chain?
};

struct KmcHandle : McHandle {
    KmcHandle(std::uint64_t seed, const std::optional<std::string>& units)
        : McHandle(std::make_unique<cpu::mc::KMC>(seed), units) {}

    void set_event(std::shared_ptr<cpu::mc::Event> event) {
        require_configurable_("set_event");
        static_cast<cpu::mc::KMC&>(*driver).set_event(std::move(event));
    }

    void set_solver(std::shared_ptr<cpu::mc::KineticSolver> solver) {
        require_configurable_("set_solver");
        static_cast<cpu::mc::KMC&>(*driver).set_solver(std::move(solver));
    }
};

void bind_mc_methods(nb::class_<McHandle>& cls) {
    using Handle = McHandle;
    ATV::python::bind_units(
        cls, [](Handle& h) -> cpu::mc::MC& { return *h.driver; },
        [](const Handle& h, const char* s) { h.require_configurable_(s); });
    cls.def(
            "set_neighbor",
            &Handle::set_neighbor, nb::arg("comp"),
            "Install the neighbor list, with a displacement check on every trial.")
        .def("set_mover", &Handle::set_mover, nb::arg("mover"),
             "Install a mover. Use MoveSelector to draw from a weighted mix.")
        .def(
            "run", &Handle::run, nb::arg("frame"), nb::arg("n_trials"),
            "Advance a molpy.Frame by up to n_trials equilibrium trials, "
            "including rejections. Returns the number of trials performed. "
            "Consecutive calls continue the same chain.")
        .def(
            "current_energy",
            [](const Handle& self) {
                return static_cast<double>(self.driver->current_energy());
            },
            "Energy [energy] of the current (last-accepted) configuration. A "
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
             "same verb argument as attempts(). Acceptance alone does not "
             "measure how well a chain explores the ensemble.")
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

    nb::class_<cpu::Mover>(m, "_Mover");

    nb::class_<cpu::mc::TranslateMove, cpu::Mover>(
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
        "delta [length] is the knob that sets the acceptance ratio: too large "
        "and "
        "many trials may be rejected. A very small displacement can give "
        "a high acceptance ratio with slow exploration.")
        .def(
            nb::init<double>(), nb::arg("delta"),
            "Construct with the displacement half-width delta [length] (> 0).");

    nb::class_<cpu::mc::VolumeMove, cpu::Mover>(
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
             "Construct with the volume half-width dv_max [volume] (> 0).");

    nb::class_<cpu::mc::InsertDeleteMove, cpu::Mover>(
        m, "InsertDeleteMove",
        "Grand-canonical insertion/deletion of a single species: with equal "
        "probability either insert one atom at a uniform position in the cell "
        "or delete one uniformly chosen atom.\n\n"
        "A deletion drawn on an empty cell is simply rejected. An inserted "
        "atom carries the configured identity and mass and nothing else: zero "
        "velocity, zero charge, no bonds, and a freshly allocated atom tag "
        "(never a recycled one). The Frame must contain only this neutral, "
        "unbonded species. The mover and GCMC policy must use the same mass.")
        .def(nb::init<int, int, double>(), nb::arg("type"),
             nb::arg("atomic_number"), nb::arg("mass"),
             "Construct with the force-field type (> 0), the species "
             "atomic_number (Z, 0..118) and the mass [mass] (> 0) stamped on "
             "inserted atoms.\n\n"
             "Each identity is written only where the frame carries it: "
             "type goes to the 'type_id' column and atomic_number to the "
             "'atomic_number' column, so an inserted Ar atom of LJ type 1 is "
             "InsertDeleteMove(type=1, atomic_number=18, mass=39.948) and "
             "comes back out in both rebuilt columns.");

    nb::class_<cpu::mc::MoveSelector, cpu::Mover>(
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
        .def("add", &cpu::mc::MoveSelector::add_child,
             nb::arg("mover"), nb::arg("weight"), nb::rv_policy::reference,
             "Add a mover with a positive relative trial weight.");

    using TagPaths = std::vector<std::vector<TagInt>>;
    nb::class_<cpu::mc::PolymerRotationMove, cpu::Mover>(m, "_PolymerRotationMove");
    nb::class_<cpu::mc::TorsionMove, cpu::mc::PolymerRotationMove>(
        m, "TorsionMove", "Rotate the moving fragment of a directed bond.")
        .def(nb::init<const std::vector<std::array<TagInt, 2>>&, double>(),
             nb::arg("paths"), nb::arg("max_angle"));
    nb::class_<cpu::mc::EndMerRotationMove, cpu::mc::PolymerRotationMove>(
        m, "EndMerRotationMove", "Rotate the last bead of each candidate terminal path.")
        .def(nb::init<const std::vector<std::array<TagInt, 3>>&, double>(),
             nb::arg("paths"), nb::arg("max_angle"));
    nb::class_<cpu::mc::LibrationMove, cpu::mc::PolymerRotationMove>(
        m, "LibrationMove", "Rotate the interior bead and its branches between fixed anchors.")
        .def(nb::init<const std::vector<std::array<TagInt, 3>>&, double>(),
             nb::arg("paths"), nb::arg("max_angle"));
    nb::class_<cpu::mc::DimerFlipMove, cpu::mc::PolymerRotationMove>(
        m, "DimerFlipMove", "Rotate two interior beads and their branches between fixed anchors.")
        .def(nb::init<const std::vector<std::array<TagInt, 4>>&, double>(),
             nb::arg("paths"), nb::arg("max_angle"));
    nb::class_<cpu::mc::PivotMove, cpu::mc::PolymerRotationMove>(
        m, "PivotMove", "Rotate one side of a directed bond about an isotropic axis through the pivot.")
        .def(nb::init<const std::vector<std::array<TagInt, 2>>&, double>(),
             nb::arg("paths"), nb::arg("max_angle"));
    nb::class_<cpu::mc::PolymerClosureMove, cpu::Mover>(m, "_PolymerClosureMove");
    nb::class_<cpu::mc::ConRotMove, cpu::mc::PolymerClosureMove>(
        m, "ConRotMove", "Local concerted rotation with fixed end positions and orientations.")
        .def(nb::init<const std::vector<std::array<TagInt, 8>>&, double>(),
             nb::arg("paths"), nb::arg("max_angle"));
    nb::class_<cpu::mc::ParallelRotationMove, cpu::mc::PolymerClosureMove>(
        m, "ParallelRotationMove", "Parallel rotation with a translating tail.")
        .def(nb::init<const std::vector<std::array<TagInt, 6>>&, double>(),
             nb::arg("paths"), nb::arg("max_angle"));
    nb::class_<cpu::mc::AtomIdentityExchangeMove, cpu::Mover>(
        m, "AtomIdentityExchangeMove", "Exchange the locations of two unbonded atoms.")
        .def(nb::init<std::vector<std::array<TagInt, 2>>>(), nb::arg("pairs"));
    nb::class_<cpu::mc::ReptationMove, cpu::Mover>(
        m, "ReptationMove", "Slithering-snake proposal on fully flexible linear homopolymers.")
        .def(nb::init<TagPaths, double>(), nb::arg("chains"), nb::arg("bond_sigma"));
    nb::class_<cpu::mc::GeneralizedReptationMove, cpu::Mover>(
        m, "GeneralizedReptationMove",
        "Transfer a terminal bead between equivalent linear chains in a variable-length ensemble.")
        .def(nb::init<double>(), nb::arg("bond_sigma"));
    nb::class_<cpu::mc::CBMCMove, cpu::Mover>(
        m, "CBMCMove", "Terminal configurational-bias regrowth on a flexible path.")
        .def(nb::init<TagPaths, int, double, double>(), nb::arg("paths"),
             nb::arg("trials"), nb::arg("bond_sigma"), nb::arg("bias_beta"));
    nb::class_<cpu::mc::ExtendedCBMCMove, cpu::mc::CBMCMove>(
        m, "ExtendedCBMCMove", "Fixed-endpoint configurational-bias regrowth on a flexible path.")
        .def(nb::init<TagPaths, int, double, double>(), nb::arg("paths"),
             nb::arg("trials"), nb::arg("bond_sigma"), nb::arg("bias_beta"));

    nb::class_<cpu::mc::Event>(m, "_Event");
    nb::class_<cpu::mc::KineticSolver>(m, "_KineticSolver");
    nb::class_<cpu::mc::LinearKineticSolver, cpu::mc::KineticSolver>(
        m, "LinearKineticSolver",
        "Gillespie direct solver with linear event selection.")
        .def(nb::init<>());
    nb::class_<cpu::mc::StateTransitions, cpu::mc::Event>(
        m, "StateTransitions",
        "Sparse transition model on integer Frame.meta['kmc_state'] labels.")
        .def(nb::init<const std::vector<cpu::mc::StateTransitions::Edge>&,
                      const std::vector<std::int64_t>&>(),
             nb::arg("transitions"), nb::arg("states") = std::vector<std::int64_t>{},
             "Transitions as (source, destination, rate in 1/ps) tuples. "
             "List isolated state labels in states.");

    nb::class_<McHandle> mc(m, "MC",
        "Monte Carlo driver for equilibrium trials.");
    mc.def(nb::init<std::uint64_t, const std::optional<std::string>&>(),
           nb::arg("seed") = 0, nb::arg("units") = nb::none())
        .def(nb::init<double, std::uint64_t, const std::optional<std::string>&>(),
             nb::arg("temperature"), nb::arg("seed"),
             nb::arg("units") = nb::none(), "Construct a driver with the NVT policy.")
        .def("set_nvt", [](McHandle& self, double temperature) {
            self.require_configurable_("set_nvt");
            self.driver->set_nvt(temperature);
        }, nb::arg("temperature"))
        .def("set_npt", [](McHandle& self, double temperature, double pressure) {
            self.require_configurable_("set_npt");
            self.driver->set_npt(temperature, pressure);
        }, nb::arg("temperature"), nb::arg("pressure"))
        .def("set_gcmc", [](McHandle& self, double temperature, double mu,
                            double mass) {
            self.require_configurable_("set_gcmc");
            self.driver->set_gcmc(temperature, mu, mass);
        }, nb::arg("temperature"), nb::arg("mu"), nb::arg("mass"))
        .def("current_volume", [](const McHandle& self) {
            return static_cast<double>(self.driver->current_volume());
        }, "Volume of the current configuration.");
    bind_mc_methods(mc);
    bind_evaluator_setters<true>(mc,
        "Append a plane-wave evaluator for equilibrium trial energies.");

    nb::class_<KmcHandle, McHandle>(m, "KMC", "Continuous-time kinetic Monte Carlo driver.")
        .def(nb::init<std::uint64_t, const std::optional<std::string>&>(),
             nb::arg("seed") = 0, nb::arg("units") = nb::none())
        .def("run",
             [](KmcHandle& self, nb::object frame, BigInt n_events) {
                 return self.run(std::move(frame), n_events);
             },
             nb::arg("frame"), nb::arg("n_events"),
             "Execute up to n_events kinetic events and return the number executed. "
             "The clock is in Frame.meta['kmc_time_ps'].")
        .def("set_event", &KmcHandle::set_event, nb::arg("event"))
        .def("set_solver", &KmcHandle::set_solver, nb::arg("solver"));
}

#endif // ATV_HAS_MC
