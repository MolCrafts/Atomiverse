/// @file python/src/bindings/bind_md.cpp
/// @brief MD driver + component bindings aligned with the C++ categorical API.
///
/// Public surface mirrors ``examples/cpp/08_md_lj.cpp`` and ``cpu::md::MD``:
///
/// @code
///   sim.set_evaluator(PairLJCut{…});
///   sim.set_neighbor(NeighborList{…});
///   sim.set_integrator(VelocityVerlet{…});
///   sim.set_thermostat(NoseHoover{…});
///   sim.set_observer(InitVelocity{…});
///   sim.set_thermo(N);
///   sim.build(); sim.run(frame, n_steps);
/// @endcode
///
/// Component classes are bound as ``cpu::task::*Params`` mirrors so the MPI
/// launcher TaskSpec path stays byte-compatible. ``MdHandle`` materializes the
/// real C++ components on each setter and records the mirror entry.

#include "capture_stats.h"
#include "evaluator_setters.h"
#include "frame_guards.h"
#include "molpy_frame_bridge.h"

#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/core/unit_system.h"
#include "atomiverse/cpu/dynamics/barostat.h"
#include "atomiverse/cpu/dynamics/configurational_temperature.h"
#include "atomiverse/cpu/dynamics/fix_csvr.h"
#include "atomiverse/cpu/dynamics/fix_langevin.h"
#include "atomiverse/cpu/dynamics/fix_lincs.h"
#include "atomiverse/cpu/dynamics/fix_rattle.h"
#include "atomiverse/cpu/dynamics/fix_shake.h"
#include "atomiverse/cpu/dynamics/gear_pc.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/dynamics/nvt_sllod.h"
#include "atomiverse/cpu/dynamics/parrinello_rahman_npt.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/input_snapshot.h"
#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "units_binding.h"
// The eighteen force sources this file used to spell out are registered from
// `evaluator_setters.h`, which owns their build gates. The four headers below
// are still included HERE because the `PhaseBuilder` escape hatch at the
// bottom of this file constructs those same four components directly.
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/record/thermo.h"
#ifdef ATV_HAS_IO
#include "atomiverse/cpu/record/xyz_recorder.h"
#endif
#ifdef ATV_HAS_IO
#include "atomiverse/cpu/record/mrec_recorder.h"
#endif
#include "atomiverse/cpu/task/launcher_gate.h"
#include "atomiverse/cpu/task/task_spec.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>

#include "atomiverse/cpu/task/task_protocol.h"
#endif

using namespace ATV;
using cpu::task::MDBinding;
using cpu::task::MDComponentEntry;

namespace {

/// Bound MD driver + TaskSpec mirror of its registrations.
struct MdHandle {
    // Ramping components the handle owns so `run()` can hand them their run
    // span (LAMMPS `update->beginstep`/`endstep`), exactly as the launcher
    // path gets it from the task materializer. Declared BEFORE `driver`:
    // members are destroyed in reverse declaration order, so a borrowed
    // component outlives the driver that references it.
    std::optional<cpu::md::NoseHooverNPT> npt_owned;
    std::vector<std::unique_ptr<cpu::md::FixLangevin>> langevin_owned;
#ifdef ATV_HAS_IO
    // Borrowed by the driver so `run()` can commit them: after a run returns
    // the store holds every frame so far, although the run stays live.
    std::vector<std::unique_ptr<cpu::MrecRecorder>> mrec_owned;
#endif
    // The trajectory: the molpy.Frame it runs on and the engine frame that
    // carries it across `run()` calls (engine row order). Declared before
    // the driver, which keeps the engine frame's address.
    nb::object frame_owner;
    std::unique_ptr<cpu::Frame> work;
    std::vector<TagInt> import_tags;      ///< molpy row order, by tag
    cpu::FrameInputSnapshot written_back; ///< the molpy.Frame as run() left it

    cpu::md::MD driver;
    std::vector<MDComponentEntry> mirror;

    /// @brief Refuse a configuration change once the driver has run.
    ///
    /// The execution plan is compiled exactly once (`Driver::build()` consumes
    /// its pending list, so a re-build is not a repeatable operation), which
    /// means a setter called after the first `run()` cannot reach the compiled
    /// plan. Left unchecked it is either silently dropped or detonates the
    /// next `run()` with a component-validation error naming the wrong
    /// culprit, so it fails loud here instead. The same refusal applies on the
    /// MPI launcher path, where the mirror IS re-sent per run: one script must
    /// mean one configuration whether or not it runs under the launcher.
    ///
    /// @param setter Name of the setter being refused, for the message.
    /// @throws std::runtime_error once `run()` has completed.
    void require_configurable_(const char* setter) const {
        if (ran_)
            throw std::runtime_error(
                std::string(setter)
                + ": MD has already run; construct a new MD to change the "
                  "configuration.");
    }

    // Borrow a live component/node; nanobind keeps its Python owner alive.
    template <typename Evaluator>
    void set_local_evaluator(Evaluator& evaluator, const char* name) {
        require_configurable_("set_evaluator");
        driver.set_evaluator_ref(evaluator);
        unmirrored_.push_back(name);
    }

    /// @brief Register a force source the task wire cannot carry.
    ///
    /// `KSpacePPPM` is move-only and is not an `MDComponentParams`
    /// alternative. `set_categorical` would either fail its variant
    /// static_assert or append a mirror entry and then drop the solver on
    /// a launcher replay. `coulomb` feeds `require_charge_column`, because
    /// this registration never lands in `mirror`.
    template <typename Component>
    void set_unmirrored_evaluator(Component comp, const char* name,
                                  bool coulomb) {
        require_configurable_("set_evaluator");
        if (coulomb)
            unmirrored_coulomb_ = true;
        unmirrored_.push_back(name);
        driver.set_evaluator(std::move(comp));
    }

    void set_nvt_sllod_(cpu::md::NvtSllod integrator) {
        require_configurable_("set_integrator");
        unmirrored_.push_back("nvt_sllod");
        driver.set_integrator(std::move(integrator));
    }

    /// @brief Borrow a live `GearPC`: its diagnostics (constraint drift,
    ///        ξ, T_C1) stay readable on the Python object.
    void set_gear_pc_(cpu::md::GearPC& integrator) {
        require_configurable_("set_integrator");
        unmirrored_.push_back("gear_pc");
        driver.set_integrator_ref(integrator);
    }

    /// @brief Borrow a live `ConfigurationalTemperature` observer.
    void set_configurational_temperature_(
        cpu::ConfigurationalTemperature& observer) {
        require_configurable_("set_observer");
        unmirrored_.push_back("configurational_temperature");
        driver.set_observer_ref(observer);
    }

    template <typename Params, typename Component>
    void set_categorical(const char* setter, const Params& p, Component comp) {
        require_configurable_(setter);
        using P = std::decay_t<Params>;
        if constexpr (std::is_same_v<P, cpu::task::PairLJCutParams>
                      || std::is_same_v<P, cpu::task::BondHarmonicParams>
                      || std::is_same_v<P, cpu::task::AngleHarmonicParams>
                      || std::is_same_v<P, cpu::task::PairCoulCutParams>
                      || std::is_same_v<P, cpu::task::CoulLongParams>
                      || std::is_same_v<P, cpu::task::PairLJCutCoulLongParams>
                      || std::is_same_v<P,
                                        cpu::task::DihedralMultiHarmonicParams>
                      // The widened real-space catalogue: six pair styles and
                      // three bonded ones, all on the same plural evaluator
                      // slot — they JOIN the styles above, never replace them.
                      || std::is_same_v<P, cpu::task::PairMorseParams>
                      || std::is_same_v<P, cpu::task::PairBuckParams>
                      || std::is_same_v<P, cpu::task::PairBornParams>
                      || std::is_same_v<P, cpu::task::PairYukawaParams>
                      || std::is_same_v<P, cpu::task::PairWcaParams>
                      || std::is_same_v<P, cpu::task::PairDpdParams>
                      || std::is_same_v<P, cpu::task::BondFeneParams>
                      || std::is_same_v<P, cpu::task::DihedralCharmmParams>
                      || std::is_same_v<P, cpu::task::ImproperHarmonicParams>
                      || std::is_same_v<P, cpu::task::AngleCosineParams>
                      // A uniform field is a force source on the same slot
                      // (force phase), as cpu::md::FixEfield documents.
                      || std::is_same_v<P, cpu::task::FixEfieldParams>)
            driver.set_evaluator(std::move(comp));
#ifdef ATV_HAS_KSPACE
        else if constexpr (std::is_same_v<P, cpu::task::KSpacePmeParams>)
            // Move-only (it owns FFTW plans), so the component arrives here
            // already moved from the setter's prvalue.
            driver.set_evaluator(std::move(comp));
#endif
#ifdef ATV_HAS_SEMIEMPIRICAL
        else if constexpr (std::is_same_v<P, cpu::task::AM1Params>)
            // A semiempirical force source rides the same plural evaluator
            // slot as the pair styles — that is what makes an MD run BOMD.
            driver.set_evaluator(std::move(comp));
#endif
#ifdef ATV_HAS_PW
        else if constexpr (std::is_same_v<P, cpu::task::PWParams>)
            // Plane-wave DFT is the other force source that makes the loop
            // Born-Oppenheimer; move-only (it owns FFTW plans), so the
            // component arrives here already moved from the setter.
            driver.set_evaluator(std::move(comp));
#endif
        else if constexpr (std::is_same_v<P, cpu::task::NeighborListParams>)
            driver.set_neighbor(std::move(comp));
        else if constexpr (std::is_same_v<P, cpu::task::VelocityVerletParams>
                           || std::is_same_v<
                               P, cpu::task::ParrinelloRahmanNptParams>)
            // Neither carries a temperature ramp — no span to install.
            driver.set_integrator(std::move(comp));
        else if constexpr (std::is_same_v<P, cpu::task::NoseHooverNptParams>) {
            // Borrowed, not transferred: `run()` re-installs the `[0,
            // n_steps]` span before every launch. A second NPT replaces the
            // first in place (setters are refused once the driver has run,
            // so nothing is built from the old one yet); the slot then
            // already holds this address.
            npt_owned.emplace(std::move(comp));
            driver.set_integrator_ref(*npt_owned);
        } else if constexpr (std::is_same_v<P, cpu::task::NoseHooverParams>
                             || std::is_same_v<P, cpu::task::FixCsvrParams>)
            driver.set_thermostat(std::move(comp));
        else if constexpr (std::is_same_v<P, cpu::task::FixLangevinParams>) {
            // Same ramp ownership as NPT; the thermostat slot is plural, so
            // the unique_ptr vector keeps every element's address stable.
            langevin_owned.push_back(
                std::make_unique<cpu::md::FixLangevin>(std::move(comp)));
            driver.set_thermostat_ref(*langevin_owned.back());
        } else if constexpr (std::is_same_v<P,
                                            cpu::task::BerendsenBarostatParams>)
            driver.set_barostat(std::move(comp));
        else if constexpr (std::is_same_v<P, cpu::task::InitVelocityParams>
                           || std::is_same_v<P, cpu::task::FixShakeParams>
                           || std::is_same_v<P, cpu::task::FixRattleParams>
                           || std::is_same_v<P, cpu::task::FixLincsParams>)
            driver.set_observer(std::move(comp));
        else if constexpr (std::is_same_v<P, cpu::task::ThermoParams>)
            driver.set_thermo(
                static_cast<const cpu::task::ThermoParams&>(p).freq);
#ifdef ATV_HAS_IO
        else if constexpr (std::is_same_v<P, cpu::task::MrecRecorderParams>) {
            // Borrowed: `run()` commits it before returning.
            mrec_owned.push_back(
                std::make_unique<cpu::MrecRecorder>(std::move(comp)));
            driver.set_record_ref(*mrec_owned.back());
        }
#endif
        else if constexpr (std::is_same_v<P, cpu::task::XyzRecorderParams>)
            // The trajectory sinks own their file handle and their own
            // stride gate, so they go in whole through the recorder slot
            // (plural) rather than through a driver-level cadence like
            // Thermo's. Both are mirrored: under the MPI launcher every rank
            // gathers its owned rows to rank 0, which alone writes.
            driver.set_record(std::move(comp));
        else
            static_assert(cpu::task::detail::unsupported_component_v<P>);
        mirror.push_back(MDComponentEntry{p, std::nullopt});
    }

    /// Escape hatch: explicit phase + order (maps to Driver::on(phase).add).
    template <typename Params, typename Component>
    void add_on_phase(const Params& p, Component comp, MDBinding binding) {
        require_configurable_("on(phase).add");
        driver.on(binding.phase).add(std::move(comp), binding.order);
        mirror.push_back(MDComponentEntry{p, binding});
    }

    /// @brief Reject a chargeless frame before an electrostatic run starts.
    ///
    /// The two boundaries deliberately differ:
    ///
    /// - At the **C++ boundary** `charge` is an *optional* per-atom column.
    ///   `cpu::from_molpy` allocates it either way and zero-fills it when the
    ///   molpy Frame did not carry it, because a C++ caller that hands over a
    ///   frame with no charges is stating a neutral system on purpose.
    /// - At the **Python boundary** that same silence is the common mistake:
    ///   an all-zero charge column and a forgotten charge column produce
    ///   identical, plausible-looking output (energy finite, forces finite,
    ///   electrostatics simply absent), and by the time `from_molpy` has run
    ///   the difference is gone. So the binding checks the molpy Frame — the
    ///   last place where absence is still visible — and throws with a how-to
    ///   instead of running a Coulomb model on charges nobody supplied.
    ///
    /// Intentional zeros stay expressible: insert an explicit zero `charge`
    /// column and the check passes.
    ///
    /// @param frame Borrowed molpy Frame about to be advanced.
    /// @throws std::runtime_error when a Coulomb evaluator is registered and
    ///         the frame's `atoms` block has no `charge` column.
    void require_charge_column(const molrs::FrameRef& frame) const {
        const bool coulomb = unmirrored_coulomb_
                             || std::any_of(mirror.begin(), mirror.end(),
                                            is_coulomb_entry<MDComponentEntry>);
        if (!coulomb || has_molpy_column(frame, "atoms", "charge"))
            return;
        throw std::runtime_error(
            "this MD run registers a Coulomb evaluator but the molpy.Frame "
            "has no per-atom 'charge' column, so every charge would be taken "
            "as 0 e. Add the charges [e] before run(), e.g. "
            "frame[\"atoms\"].insert(\"charge\", q).");
    }

    void run(nb::object frame_obj, std::int64_t n_steps) {
        const cpu::task::LauncherGate gate;
        if (!gate.active()) {
            run_local_(frame_obj, n_steps);
            // Armed only on a COMPLETED run: a run that threw (missing
            // integrator, chargeless frame, ...) left no compiled plan behind,
            // so the caller must still be able to finish configuring.
            ran_ = true;
            return;
        }
#ifdef ATV_HAS_MPI
        if (!unmirrored_.empty())
            throw std::runtime_error(
                "MD.run: the MPI launcher task payload cannot carry "
                + unmirrored_.front() + "; run without the launcher");
        cpu::task::TaskSpec spec(cpu::task::MDTask{
            mirror, static_cast<std::int64_t>(n_steps),
            driver.capture_buffer_bytes(), driver.units().style});
        auto fref = borrow_molpy_frame(frame_obj);
        require_charge_column(*fref);
        cpu::Frame work = cpu::from_molpy_for_run(*fref, driver.units().name());
        const int n_import = work.nlocal();
        std::vector<TagInt> import_tags(
            static_cast<std::size_t>(std::max(n_import, 0)));
        if (n_import > 0)
            std::copy_n(work.span(kAtom_tag).data(), n_import,
                        import_tags.begin());
        cpu::Frame result(0);
        {
            nb::gil_scoped_release release;
            cpu::task::TaskDispatcher dispatcher(MPI_COMM_WORLD);
            result = dispatcher.dispatch(spec, std::move(work)).frame;
        }
        result.restore_import_order(import_tags.data(), n_import);
        cpu::to_molpy(result, *fref);
        ran_ = true;
#endif
    }

    /// @brief Fix the ramp span of the Langevin thermostats and the NPT
    ///        `T_stop` ramp to `[nfirst, nlast]` for every trajectory.
    void set_run_span(std::int64_t nfirst, std::int64_t nlast) {
        require_configurable_("set_run_span");
        if (!npt_owned && langevin_owned.empty())
            throw std::runtime_error("set_run_span requires a Langevin "
                                     "thermostat or a NoseHooverNPT "
                                     "integrator");
        if (nlast <= nfirst)
            throw std::invalid_argument(
                "run span nlast must be greater than nfirst");
        if (npt_owned)
            npt_owned->set_run_span(nfirst, nlast);
        for (auto& langevin : langevin_owned)
            langevin->set_run_span(nfirst, nlast);
        span_fixed = true;
    }

  private:
    bool span_fixed = false; ///< `set_run_span` chose the ramp span

    /// @brief One `run()` without the launcher.
    ///
    /// Consecutive calls on the molpy.Frame the previous one returned,
    /// unedited, extend ONE trajectory (`cpu::md::MD::run` continuation,
    /// LAMMPS `run N pre no`): the integrator history and thermostat
    /// variables, the step count and the recorders' step axis carry over,
    /// as on CUDA. An edit to that frame (positions, velocities, charges,
    /// cell, population) reaches the engine frame as an input change, and
    /// the components that depend on it restart (`invalidate_inputs`).
    /// Another molpy.Frame is an independent simulation. A ramp span is
    /// `set_run_span`'s, else that of the run that starts a trajectory; a
    /// later chunk continues it (the CUDA handle's rule).
    void run_local_(nb::object frame_obj, std::int64_t n_steps) {
        auto frame = borrow_molpy_frame(frame_obj);
        require_charge_column(*frame);
        cpu::Frame incoming =
            cpu::from_molpy_for_run(*frame, driver.units().name());
        const bool continued = work && frame_owner.is(frame_obj);
        std::unique_ptr<cpu::Frame> fresh;
        if (!continued) {
            // A new address: the driver tears the previous run down (it
            // compares the frame it was given), so the old engine frame
            // stays alive until this run has started.
            fresh = std::make_unique<cpu::Frame>(std::move(incoming));
            if (!span_fixed) {
                if (npt_owned)
                    npt_owned->set_run_span(0, n_steps);
                for (auto& langevin : langevin_owned)
                    langevin->set_run_span(0, n_steps);
            }
        } else if (written_back.changes(incoming,
                                        cpu::FrameInputSnapshot::state_input)
                   != InputChange::None) {
            // The engine's row order, so the edit reads as the edit (a
            // velocity edit is a velocity change, not a new population).
            const std::vector<TagInt> engine(work->span(kAtom_tag).data(),
                                             work->span(kAtom_tag).data()
                                                 + work->nlocal());
            const bool same_atoms = incoming.restore_import_order(
                engine.data(), static_cast<int>(engine.size()));
            *work = std::move(incoming);
            if (!same_atoms)
                remember_import_order_(*work);
        }
        cpu::Frame& target = fresh ? *fresh : *work;
        if (fresh)
            remember_import_order_(target);
        {
            nb::gil_scoped_release release;
            driver.run(target, n_steps);
        }
        if (fresh) {
            work = std::move(fresh);
            frame_owner = frame_obj;
        }
        // Owned rows may have been sorted spatially: write each atom back
        // to the row it came from, from a copy, so the engine keeps its
        // order (and its neighbor list) for the next chunk.
        cpu::Frame out(*work);
        out.restore_import_order(import_tags.data(),
                                 static_cast<int>(import_tags.size()));
        cpu::to_molpy(out, *frame);
        // As the next call will read it (converted to the run units).
        written_back.capture(
            cpu::from_molpy_for_run(*frame, driver.units().name()),
            cpu::FrameInputSnapshot::state_input);
#ifdef ATV_HAS_IO
        for (auto& rec : mrec_owned)
            rec->flush();
#endif
    }

    void remember_import_order_(const cpu::Frame& f) {
        import_tags.assign(f.span(kAtom_tag).data(),
                           f.span(kAtom_tag).data() + f.nlocal());
    }

    /// True once a `run()` has completed; see `require_configurable_`.
    bool ran_ = false;
    std::vector<std::string> unmirrored_;
    /// True once `set_unmirrored_evaluator` registered an electrostatic
    /// source. `require_charge_column` ORs this with the mirror scan.
    bool unmirrored_coulomb_ = false;
};

struct MdPhaseBuilder {
    MdHandle* handle;
    ATV::md::Phase phase;
};

} // namespace

void bind_md(nb::module_& m) {
    double constraint_tolerance = cpu::md::FixRattle::default_tolerance();
#ifdef ATV_CUDA_PRECISION_SINGLE
    // One parameter object serves both backends; its default must be
    // representable by either integration precision.
    constraint_tolerance = std::max(
        constraint_tolerance, 64.0 * std::numeric_limits<float>::epsilon());
#endif
    nb::class_<cpu::task::FixShakeParams>(
        m, "FixShake",
        "Force-level SHAKE distance constraints with constraint virial.")
        .def(nb::init<double, double, int>(), nb::arg("dt"),
             nb::arg("tolerance") = constraint_tolerance,
             nb::arg("max_iter") = 500)
        .def("constrain_bond", &cpu::task::FixShakeParams::constrain_bond,
             nb::arg("bond_type"), nb::arg("length"))
        .def("constrain_angle", &cpu::task::FixShakeParams::constrain_angle,
             nb::arg("angle_type"), nb::arg("distance_13"));
    nb::class_<cpu::task::FixRattleParams>(
        m, "FixRattle",
        "RATTLE distance constraints: positions, tangent velocities and "
        "constraint virial.")
        .def(nb::init<double, double, int>(), nb::arg("dt"),
             nb::arg("tolerance") = constraint_tolerance,
             nb::arg("max_iter") = 500)
        .def("constrain_bond", &cpu::task::FixRattleParams::constrain_bond,
             nb::arg("bond_type"), nb::arg("length"))
        .def("constrain_angle", &cpu::task::FixRattleParams::constrain_angle,
             nb::arg("angle_type"), nb::arg("distance_13"));
    nb::class_<cpu::task::FixLincsParams>(
        m, "FixLincs",
        "LINCS distance constraints (CPU): RATTLE's position and velocity "
        "stages solved by a matrix expansion of the given order, iterated "
        "to the tolerance.")
        .def(nb::init<double, double, int, int>(), nb::arg("dt"),
             nb::arg("tolerance") = constraint_tolerance,
             nb::arg("max_iter") = 500, nb::arg("order") = 4)
        .def(
            "constrain_bond",
            [](cpu::task::FixLincsParams& self, int bond_type, double length) {
                self.constrain_bond(bond_type, length);
            },
            nb::arg("bond_type"), nb::arg("length"))
        .def(
            "constrain_angle",
            [](cpu::task::FixLincsParams& self, int angle_type,
               double distance_13) {
                self.constrain_angle(angle_type, distance_13);
            },
            nb::arg("angle_type"), nb::arg("distance_13"));

    nb::enum_<md::Phase>(m, "Phase", "MD phase sequence (LAMMPS-style).")
        .value("initial_integrate", md::Phase::initial_integrate)
        .value("pre_force", md::Phase::pre_force)
        .value("force", md::Phase::force)
        .value("post_force", md::Phase::post_force)
        .value("final_integrate", md::Phase::final_integrate)
        .value("end_of_step", md::Phase::end_of_step);

    // ── Component params (TaskSpec allow-list = binding surface) ──

    nb::class_<cpu::task::VelocityVerletParams>(
        m, "VelocityVerlet", "Symplectic velocity-Verlet integrator.")
        .def(
            "__init__",
            [](cpu::task::VelocityVerletParams* p, double dt) {
                new (p) cpu::task::VelocityVerletParams{dt};
            },
            nb::arg("dt"), "Construct with timestep dt [time].");

    nb::class_<ATV::md::NvtSllodSpec>(
        m, "NvtSllod",
        "NVT SLLOD integrator (LAMMPS `fix nvt/sllod` + `fix deform` ERATE, "
        "`remap v`).\n\n"
        "Replaces VelocityVerlet and thermostats peculiar velocities itself. "
        "Do not add NoseHoover or Langevin beside it. `direction` is the "
        "Voigt index 0..5: x, y, z, yz, xz, xy. `rate` is the engineering "
        "strain rate [1/time]. The strain clock is internal and restarts when "
        "the driver is built again. `tchain` is the Nose-Hoover chain "
        "length (LAMMPS `tchain`; `fix nvt/sllod` defaults it to 1). A tilt "
        "beyond half its box length flips the cell (LAMMPS `flip yes`).\n\n"
        "This record is not part of the MPI task payload. `MD.run` under the "
        "launcher rejects it.")
        .def(
            "__init__",
            [](ATV::md::NvtSllodSpec* p, double temperature, double tau,
               double dt, double rate, int direction, int tchain) {
                new (p) ATV::md::NvtSllodSpec{temperature, tau,       dt,
                                              rate,        direction, tchain};
            },
            nb::arg("temperature"), nb::arg("tau"), nb::arg("dt"),
            nb::arg("rate"), nb::arg("direction"), nb::arg("tchain") = 1,
            "temperature [temperature], tau [time] (LAMMPS Tdamp), dt [time], "
            "engineering rate [1/time], direction 0..5, chain length tchain.");

    nb::class_<cpu::task::FixEfieldParams>(
        m, "FixEfield",
        "Constant uniform electric field (LAMMPS `fix efield Ex Ey Ez`, "
        "run units).\n\n"
        "A force source for MD.set_evaluator on CPU and CUDA: F_i += q_i E, "
        "with E in the run's electric-field unit (V/A for units real and "
        "metal). "
        "The energy U = -sum_i q_i E . r_i^u uses unwrapped coordinates "
        "r^u = r + H n through the frame's image flags (LAMMPS fsum[0]); it "
        "drifts by the work the field does on migrating ions and is not a "
        "bounded potential under PBC. No virial is added (LAMMPS "
        "`fix_modify virial` is off by default). Needs a per-atom 'charge' "
        "column [charge].")
        .def(
            "__init__",
            [](cpu::task::FixEfieldParams* p, double Ex, double Ey, double Ez) {
                new (p) cpu::task::FixEfieldParams{Ex, Ey, Ez};
            },
            nb::arg("Ex"), nb::arg("Ey"), nb::arg("Ez"),
            "Construct with the field components Ex, Ey, Ez [run electric "
            "field].")
        .def_ro("Ex", &cpu::task::FixEfieldParams::Ex,
                "Field x [run electric field].")
        .def_ro("Ey", &cpu::task::FixEfieldParams::Ey,
                "Field y [run electric field].")
        .def_ro("Ez", &cpu::task::FixEfieldParams::Ez,
                "Field z [run electric field].");

    nb::class_<cpu::md::GaussThermostat>(
        m, "GaussThermostat",
        "Gauss least-constraint kinetic thermostat for GearPC (Petravic & "
        "Delhommelle, J. Chem. Phys. 118, 7477 (2003)): v' = F/m - alpha w "
        "with alpha = sum m w.a / sum m w^2, w the peculiar velocity of the "
        "constrained components and a the total acceleration (an applied "
        "field included). The constrained kinetic energy stays exactly at "
        "dof k_B T / 2.\n\n"
        "  kind 'kinetic'    eq. (9): every axis relative to the centre of "
        "mass, 3N-3 dof.\n"
        "  kind 'flow'       eqs. (3), (4), (10): along `axis` relative to "
        "each species' mean velocity (the frame's type column; the species "
        "flows are not thermostatted), barycentric on the other axes, "
        "3N-2-n_species dof.\n"
        "  kind 'transverse' eq. (11): `axis` free, the other two axes "
        "barycentric, 2N-2 dof.\n\n"
        "The paper normalises temperatures by 3N instead of these dof.")
        .def(
            "__init__",
            [](cpu::md::GaussThermostat* p, const std::string& kind,
               double temperature, int axis) {
                cpu::md::GaussKind k;
                if (kind == "kinetic")
                    k = cpu::md::GaussKind::Kinetic;
                else if (kind == "flow")
                    k = cpu::md::GaussKind::Flow;
                else if (kind == "transverse")
                    k = cpu::md::GaussKind::Transverse;
                else
                    throw nb::value_error(
                        ("GaussThermostat: kind must be 'kinetic', 'flow' or "
                         "'transverse', got '"
                         + kind + "'")
                            .c_str());
                if (axis < 0 || axis > 2)
                    throw nb::value_error(
                        "GaussThermostat: axis must be 0, 1 or 2");
                new (p) cpu::md::GaussThermostat{k, temperature, axis};
            },
            nb::arg("kind"), nb::arg("temperature"), nb::arg("axis") = 0,
            "kind 'kinetic' | 'flow' | 'transverse', target temperature "
            "[temperature], field axis 0..2 (flow / transverse).")
        .def_ro("temperature", &cpu::md::GaussThermostat::temperature,
                "Target temperature [temperature].")
        .def_ro("axis", &cpu::md::GaussThermostat::axis, "Field axis.");

    nb::class_<cpu::md::ConfigurationalThermostat>(
        m, "ConfigurationalThermostat",
        "Configurational Nose-Hoover thermostat for GearPC, Petravic & "
        "Delhommelle eqs. (12a-c): r' = v + (xi/T0) dT_C1/dr, "
        "xi' = -Q (T_C1 - T0)/T0, with the first-order configurational "
        "temperature k_B T_C1 = sum F^2 / sum lap U (eq. 7) of the "
        "interaction potential (FixEfield excluded). Every force term must "
        "supply its curvature: PairBorn, CoulLong and KSpaceEwald do; "
        "KSpacePME and KSpacePPPM do not and are refused.")
        .def(
            "__init__",
            [](cpu::md::ConfigurationalThermostat* p, double temperature,
               double Q) {
                new (p) cpu::md::ConfigurationalThermostat{temperature, Q};
            },
            nb::arg("temperature"), nb::arg("Q"),
            "Target T0 [temperature] and feedback constant Q "
            "[length^2/time^2]; xi carries length^2/time.")
        .def_ro("temperature", &cpu::md::ConfigurationalThermostat::temperature,
                "Target temperature [temperature].")
        .def_ro("Q", &cpu::md::ConfigurationalThermostat::Q,
                "Feedback constant [length^2/time^2].");

    nb::class_<cpu::md::GearPC, cpu::Component>(
        m, "GearPC",
        "Fifth-order Gear predictor-corrector integrator on the first-order "
        "equations of motion (r, v), Gear's first-order corrector "
        "(95/288, 1, 25/24, 35/72, 5/48, 1/120). Replaces VelocityVerlet; "
        "without a thermostat the motion is adiabatic, with GaussThermostat "
        "or ConfigurationalThermostat the thermostat is inside the step, so "
        "do not add another thermostat. One force evaluation per step, at "
        "the predicted positions. The higher derivatives start at zero (a "
        "one-time start-up transient). Stable only for omega*dt < 0.114 on a "
        "mode of angular frequency omega: 1 fs for molten NaCl, not 2 fs. "
        "CPU only; spatial decomposition and constraints are refused. The "
        "MD driver borrows this object, so its "
        "diagnostics stay readable here. Not carried by the MPI task "
        "payload.")
        .def(nb::init<double>(), nb::arg("dt"),
             "Adiabatic, timestep dt [time].")
        .def(nb::init<double, cpu::md::GaussThermostat>(), nb::arg("dt"),
             nb::arg("thermostat"),
             "Timestep dt [time] with a Gauss thermostat.")
        .def(nb::init<double, cpu::md::ConfigurationalThermostat>(),
             nb::arg("dt"), nb::arg("thermostat"),
             "Timestep dt [time] with the configurational thermostat.")
        .def_prop_ro("dt", &cpu::md::GearPC::timestep, "Timestep [time].")
        .def_prop_ro("constraint_drift", &cpu::md::GearPC::constraint_drift,
                     "Relative constrained kinetic energy |K/K0 - 1| the last "
                     "Gauss projection removed (truncation diagnostic).")
        .def_prop_ro("gauss_multiplier", &cpu::md::GearPC::gauss_multiplier,
                     "Gauss multiplier alpha [1/time] at the last step.")
        .def_prop_ro("xi", &cpu::md::GearPC::xi,
                     "Configurational thermostat variable xi "
                     "[length^2/time].")
        .def_prop_ro("configurational_temperature",
                     &cpu::md::GearPC::configurational_temperature,
                     "T_C1 [temperature] at the last predicted state "
                     "(configurational thermostat only).");

    nb::class_<cpu::ConfigurationalTemperature, cpu::Component>(
        m, "ConfigurationalTemperature",
        "Observer of the first-order configurational temperature "
        "k_B T_C1 = N / D, N = sum_i |F_i|^2 of the interaction force (an "
        "applied FixEfield excluded), D = sum_i lap_i U (Petravic & "
        "Delhommelle eq. 7), evaluated at post_force every `every` steps. "
        "The run averages <N> and <D> give T_CONF = <N>/(k_B <D>) of eq. (5) "
        "(mean_temperature); each run() is a new run and starts a new "
        "average. Every force "
        "term must supply its curvature (PairBorn, CoulLong, KSpaceEwald). "
        "Under Ewald the Coulomb Laplacian is the background constant "
        "-4 pi k_e sum q^2 / V, which this observer includes. CPU only.")
        .def(nb::init<int>(), nb::arg("every") = 1,
             "Evaluate every `every` steps (>= 1).")
        .def_prop_ro(
            "temperature",
            [](const cpu::ConfigurationalTemperature& o) {
                return o.last().temperature;
            },
            "T_C1 [temperature] of the last evaluation.")
        .def_prop_ro(
            "force_sq",
            [](const cpu::ConfigurationalTemperature& o) {
                return o.last().force_sq;
            },
            "N = sum |F|^2 [force^2] of the last evaluation.")
        .def_prop_ro(
            "laplacian",
            [](const cpu::ConfigurationalTemperature& o) {
                return o.last().laplacian;
            },
            "D = sum lap U [energy/length^2] of the last evaluation.")
        .def_prop_ro(
            "samples",
            [](const cpu::ConfigurationalTemperature& o) {
                return o.samples();
            },
            "Evaluations averaged since the last run() began.")
        .def_prop_ro(
            "mean_temperature",
            [](const cpu::ConfigurationalTemperature& o) {
                return o.mean().temperature;
            },
            "T_CONF = <N>/(k_B <D>) [temperature] over the last run() "
            "(eq. 5; not the mean of T_C1).")
        .def_prop_ro(
            "mean_force_sq",
            [](const cpu::ConfigurationalTemperature& o) {
                return o.mean().force_sq;
            },
            "<N> [force^2] over the last run().")
        .def_prop_ro(
            "mean_laplacian",
            [](const cpu::ConfigurationalTemperature& o) {
                return o.mean().laplacian;
            },
            "<D> [energy/length^2] over the last run().");

    nb::class_<cpu::task::InitVelocityParams>(
        m, "InitVelocity", "Maxwell-Boltzmann velocity initialization.")
        .def(
            "__init__",
            [](cpu::task::InitVelocityParams* p, double temperature,
               unsigned seed) {
                new (p) cpu::task::InitVelocityParams{temperature, seed};
            },
            nb::arg("temperature"), nb::arg("seed"),
            "Construct with target temperature [temperature] and RNG seed.");

    nb::class_<cpu::task::NoseHooverParams>(
        m, "NoseHoover", "Nose-Hoover thermostat (LAMMPS `fix nvt`).")
        .def(
            "__init__",
            [](cpu::task::NoseHooverParams* p, double T_target, double tau_T,
               double dt) {
                new (p) cpu::task::NoseHooverParams{T_target, tau_T, dt};
            },
            nb::arg("T_target"), nb::arg("tau_T"), nb::arg("dt"),
            "Construct with target temperature [temperature], coupling time "
            "tau_T [time] (LAMMPS `Tdamp`), and timestep [time].");

    nb::class_<cpu::task::FixLangevinParams>(
        m, "Langevin", "Langevin thermostat (LAMMPS `fix langevin`).")
        .def(
            "__init__",
            [](cpu::task::FixLangevinParams* p, double T_start, double T_stop,
               double damp, double dt, unsigned seed) {
                new (p) cpu::task::FixLangevinParams{T_start, T_stop, damp, dt,
                                                     seed};
            },
            nb::arg("T_start"), nb::arg("T_stop"), nb::arg("damp"),
            nb::arg("dt"), nb::arg("seed"),
            "Construct with ramp temperatures T_start/T_stop [temperature], "
            "damping "
            "time damp [time] (LAMMPS `damp`), timestep dt [time], and RNG "
            "seed.");

    nb::class_<cpu::task::FixCsvrParams>(
        m, "CSVR",
        "Bussi canonical-sampling velocity-rescale thermostat (CP2K CSVR).")
        .def(
            "__init__",
            [](cpu::task::FixCsvrParams* p, double T_start, double tau,
               double dt, unsigned seed, std::vector<int> types,
               const std::string& bias) {
                std::uint8_t b = 0;
                if (bias == "com")
                    b = static_cast<std::uint8_t>(
                        ATV::md::TemperatureBias::Com);
                else if (bias != "none")
                    throw std::invalid_argument(
                        "CSVR: bias must be 'none' or 'com', got '" + bias
                        + "'");
                for (int t : types)
                    if (t < 0)
                        throw std::invalid_argument(
                            "CSVR: atom types must be >= 0, got "
                            + std::to_string(t));
                new (p) cpu::task::FixCsvrParams{
                    T_start, tau, dt, seed, std::move(types), b};
            },
            nb::arg("T_start"), nb::arg("tau"), nb::arg("dt"), nb::arg("seed"),
            nb::arg("types") = std::vector<int>{}, nb::arg("bias") = "none",
            "Construct with target temperature [temperature], coupling time "
            "tau [time] (CP2K `TIMECON`, LAMMPS `Tdamp`), timestep dt [time] "
            "and RNG seed. `types` restricts the thermostat to atoms whose "
            "force-field type (molpy atoms.type_id) is listed (empty: every "
            "atom); bias='com' measures and rescales velocities relative to "
            "the group's centre-of-mass velocity (LAMMPS `fix_modify temp` "
            "with `compute temp/com`). Several CSVR thermostats may be set on "
            "one driver, each with its own group and seed.")
        .def_ro("types", &cpu::task::FixCsvrParams::types)
        .def_prop_ro("bias", [](const cpu::task::FixCsvrParams& p) {
            return std::string(p.bias == 0 ? "none" : "com");
        });

    nb::class_<cpu::task::NoseHooverNptParams>(
        m, "NoseHooverNPT",
        "Martyna-Tobias-Klein (MTK) Nose-Hoover NPT integrator (LAMMPS "
        "`fix npt`).\n\n"
        "NPT is the constant particle-number / pressure / temperature "
        "ensemble: the cell volume is a dynamical variable driven towards "
        "P_target while a Nose-Hoover chain (a chain of thermostat "
        "variables, each thermostatting the previous one) holds the "
        "temperature at T_target. One Trotter-split step therefore does the "
        "work of integrator + thermostat + barostat, so this REPLACES the "
        "time integrator: install it via set_integrator, never alongside "
        "VelocityVerlet, NoseHoover or BerendsenBarostat.\n\n"
        "The lockstep-xy and constant-surface-tension knobs of the C++ "
        "class (`set_couple_xy`, `set_surface_tension`) are not bound; they "
        "stay at their defaults here.")
        .def(
            "__init__",
            [](cpu::task::NoseHooverNptParams* p, double T_target, double tau_T,
               double P_target, double tau_P, double dt, double drag,
               int mtchain, int mpchain, std::array<bool, 3> couple_axis,
               double T_stop) {
                new (p) cpu::task::NoseHooverNptParams{
                    T_target, tau_T,   P_target, tau_P,       dt,
                    drag,     mtchain, mpchain,  couple_axis, T_stop};
            },
            nb::arg("T_target"), nb::arg("tau_T"), nb::arg("P_target"),
            nb::arg("tau_P"), nb::arg("dt"), nb::arg("drag") = double(0),
            nb::arg("mtchain") = 3, nb::arg("mpchain") = 3,
            nb::arg("couple_axis") = std::array<bool, 3>{true, true, true},
            nb::arg("T_stop") = double(-1),
            "Construct with target temperature T_target [temperature], "
            "thermostat "
            "coupling time tau_T [time] (LAMMPS `Tdamp`), target pressure "
            "P_target [pressure] (LAMMPS sign, positive compressive; in the "
            "run's "
            "pressure unit: metal bar, real atm), barostat coupling time tau_P "
            "[time] (LAMMPS "
            "`Pdamp`), and timestep dt [time]. A coupling time is the "
            "timescale on which that setpoint is enforced; both are usually "
            "tens (tau_T) to hundreds (tau_P) of timesteps.\n\n"
            "Optional LAMMPS `fix npt` knobs, all defaulted to the C++ "
            "values:\n"
            "  drag        extra empirical damping of the Nose-Hoover chain "
            "momenta; dimensionless, 0 disables.\n"
            "  mtchain     atom thermostat chain length (default 3).\n"
            "  mpchain     barostat thermostat chain length (default 3).\n"
            "  couple_axis per-axis barostat enable {x, y, z}; a False axis "
            "is frozen (its cell edge never rescales, so that direction "
            "stays NVT).\n"
            "  T_stop      end temperature [temperature] of a linear setpoint "
            "ramp "
            "across the run span, as in LAMMPS. Any value >= 0 is taken as "
            "the ramp end (equal to T_target means a constant setpoint); "
            "< 0 folds to T_target. The run span is MD.set_run_span's "
            "window, else the first run()'s [0, n_steps]; continued run() "
            "calls continue it.");

    nb::class_<cpu::task::ParrinelloRahmanNptParams>(
        m, "ParrinelloRahmanNPT",
        "Martyna-Tobias-Klein NPT integrator with full cell-matrix degrees "
        "of freedom (Parrinello-Rahman dynamics).\n\n"
        "Same NPT ensemble as NoseHooverNPT, but the cell is driven by a "
        "symmetric strain-rate tensor with six independent components "
        "(xx, yy, zz, xy, xz, yz) instead of three diagonal ones: the tilt "
        "factors evolve, so the cell can shear and not just breathe. That "
        "is what you want for a solid under a structural transition; for a "
        "liquid or an isotropic solid the diagonal NoseHooverNPT is "
        "cheaper and equivalent on average. It likewise REPLACES the time "
        "integrator — install via set_integrator, not alongside "
        "VelocityVerlet.\n\n"
        "The C++ class also exposes a barostat-mass override and chain-"
        "length setters (`W_cell`, `set_drag`, `set_mtchain`, "
        "`set_mpchain`); those are not bound and stay at their defaults.")
        .def(
            "__init__",
            [](cpu::task::ParrinelloRahmanNptParams* p, double T_target,
               double P_target, double tau_T, double tau_P, double dt) {
                new (p) cpu::task::ParrinelloRahmanNptParams{T_target, P_target,
                                                             tau_T, tau_P, dt};
            },
            nb::arg("T_target"), nb::arg("P_target"), nb::arg("tau_T"),
            nb::arg("tau_P"), nb::arg("dt"),
            "Construct with target temperature T_target [temperature], target "
            "hydrostatic pressure P_target [pressure] (LAMMPS sign, positive "
            "compressive; metal bar, real atm), thermostat coupling "
            "time tau_T [time] (LAMMPS `Tdamp`), barostat coupling time "
            "tau_P [time] (LAMMPS `Pdamp`), and timestep dt [time].\n\n"
            "Mind the order: it is (T_target, P_target, tau_T, tau_P, dt) — "
            "both setpoints first, then both coupling times — whereas "
            "NoseHooverNPT pairs each setpoint with its own coupling time "
            "(T_target, tau_T, P_target, tau_P, dt). Passing NoseHooverNPT's "
            "order here silently swaps P_target with tau_T.");

    nb::class_<cpu::task::BerendsenBarostatParams>(
        m, "BerendsenBarostat",
        "Berendsen isotropic barostat (LAMMPS `fix press/berendsen iso`).\n\n"
        "Each end_of_step it reads the instantaneous pressure from the "
        "stress tensor and rescales the box edges and every atom position "
        "by one common factor, nudging the pressure towards P_target. "
        "Being a plain relaxation it is robust and cheap, but it does NOT "
        "sample a rigorous NPT distribution — use it to equilibrate a box, "
        "and NoseHooverNPT (or ParrinelloRahmanNPT) to produce production "
        "NPT statistics.\n\n"
        "It rides ALONGSIDE the integrator: install it with set_barostat "
        "together with VelocityVerlet and a thermostat, unlike the NPT "
        "integrators, which replace the integrator.")
        .def(
            "__init__",
            [](cpu::task::BerendsenBarostatParams* p, double P_target,
               double tau_P, double dt, double bulk_modulus) {
                new (p) cpu::task::BerendsenBarostatParams{P_target, tau_P, dt,
                                                           bulk_modulus};
            },
            nb::arg("P_target"), nb::arg("tau_P"), nb::arg("dt"),
            nb::arg("bulk_modulus") = double(10),
            "Construct with target pressure P_target [pressure] (LAMMPS sign, "
            "positive compressive; metal bar, real atm), barostat "
            "coupling time tau_P [time] (LAMMPS `Pdamp`), and timestep dt "
            "[time].\n\n"
            "bulk_modulus is the tunable modulus beta [pressure] (LAMMPS "
            "`modulus`). It divides the per-step rescale, so a larger value "
            "means a gentler box change and slower pressure relaxation. The "
            "default 10.0 is LAMMPS's, in the run's pressure unit.");

    nb::class_<cpu::task::ThermoParams>(
        m, "Thermo",
        "Thermodynamic output at end_of_step.\n\n"
        "LAMMPS `thermo` step convention: the `step` column is the number of "
        "completed MD steps and the line is the state after that many steps. "
        "Each run() first prints the header and the initial state (step 0), "
        "then the state after every `freq`-th step and after the last step: "
        "run(frame, 1000) at freq=100 prints steps 0, 100, ..., 1000. "
        "freq=0 prints the first and last step of each run only (LAMMPS "
        "`thermo 0`).")
        .def(
            "__init__",
            [](cpu::task::ThermoParams* p, int freq) {
                new (p) cpu::task::ThermoParams{freq};
            },
            nb::arg("freq"),
            "Construct with output frequency (every N steps).");

    // Unconditional: the wheel forces ATV_BUILD_IO=ON through the CMake
    // module DAG (ATV_BUILD_PYTHON -> ATV_BUILD_IO), so the trajectory
    // writer is present on every extension and needs no capability probe.
    nb::class_<cpu::task::XyzRecorderParams>(
        m, "XYZ",
        "XYZ trajectory output at end_of_step.\n\n"
        "Writes one frame — element symbol plus x y z [length] per atom, under "
        "a "
        "comment line carrying the cell and the typed frame metadata — every "
        "`freq` steps. The symbol is derived from the frame's per-atom "
        "`atomic_number` column (Z), so Z=18 is written as Ar.\n\n"
        "The first write of a given recorder truncates the file and every "
        "later write of that same recorder appends, so handing a NEW "
        "XYZ(path, freq) to a second run starts the trajectory over instead "
        "of extending it.\n\n"
        "LAMMPS `dump` step convention: a frame's step is the number of "
        "completed MD steps and the frame is the state after exactly that "
        "many steps. A run() writes the initial state (step 0) and then the "
        "state after every `freq`-th step, up to and including the last: a "
        "10-step run at freq=5 writes 3 frames (steps 0, 5 and 10). "
        "Each run() call advances a freshly materialized working frame, so "
        "chunked run() calls are not one continued trajectory — do not rely "
        "on the step index carrying over between them.")
        .def(
            "__init__",
            [](cpu::task::XyzRecorderParams* p, std::string filename,
               int freq) {
                new (p) cpu::task::XyzRecorderParams{std::move(filename), freq};
            },
            nb::arg("filename"), nb::arg("freq") = 1,
            "Construct with the output path and the write cadence.\n\n"
            "  filename  output XYZ path (truncated on the first write, "
            "appended to afterwards).\n"
            "  freq      write cadence [steps, >= 1; default 1] — one frame "
            "whenever the completed-step count is a multiple of freq "
            "(including the initial state, step 0).\n\n"
            "This object is a plain parameter record; the values are checked "
            "when the recorder is built, so an empty path or freq < 1 raises "
            "ValueError from MD.set_record(...), not from here.");

#ifdef ATV_HAS_IO
    // Bound once here as the task-wire mirror: the CPU handle mirrors it to
    // the MPI launcher, and the CUDA MD handle (bind_cuda.cu) accepts the
    // same record and builds cuda::MrecRecorder from it.
    nb::class_<cpu::task::MrecRecorderParams>(
        m, "MrecRecorder",
        "molrec (`*.mrec`) trajectory output at end_of_step, on the CPU and "
        "the CUDA MD driver alike, and under the MPI launcher.\n\n"
        "Every `every` steps one frame is appended to the store, every value "
        "in the run's LAMMPS units and never converted (real: A, A/fs, "
        "kcal/mol/A, kcal/mol; metal: A, A/ps, eV/A, eV): positions "
        "[length], velocities [velocity], forces [force], charges [charge], "
        "images "
        "(int32 "
        "ix/iy/iz, so unwrapped = x + ix*a + iy*b + iz*c), tags, mass "
        "[mass], the identity columns the frame carries (force-field type as "
        "`type_id`, species as `atomic_number`; an absent one stays absent), "
        "the cell, the energies/virial in the frame metadata, the step and "
        "— with an integrator — time = step*dt [time]. The step is the number "
        "of completed MD steps and the frame is the state after exactly that "
        "many steps (LAMMPS `dump`): a run() of n steps at every=k records "
        "steps 0, k, 2k, ..., up to and including n. Rows are in ascending "
        "tag order. Read it back with `molpy.io.mrec.MrecReader(path)`; the "
        "store's `status` section records `state` (with the `error` when "
        "the run failed), `frames_written` and, on CUDA, the `buffer` "
        "accounting of the driver's capture buffer (`budget_bytes`, "
        "`frame_bytes`, `peak_bytes`, `blocked_episodes`, "
        "`blocked_seconds`).\n\n"
        "The first frame replaces any store at the path. Continued run() "
        "calls on the same frame extend the same trajectory on both "
        "devices, the step axis continuing and the boundary step written "
        "once. Under the MPI launcher every rank "
        "gathers its atoms to rank 0, which alone writes.\n\n"
        "CUDA recording is asynchronous but lossless: frames travel through "
        "the driver's one shared capture buffer (MD(capture_buffer_bytes=…)) "
        "and when it is full the run waits until the consumers free space — "
        "no frame is ever dropped. Waits are logged and recorded in the "
        "`status` section. If writing fails (disk full, I/O error) the run "
        "stops with RuntimeError naming the store and the cause, and the "
        "status records state = \"failed\".")
        .def(
            "__init__",
            [](cpu::task::MrecRecorderParams* p, std::string path, int every,
               std::uint64_t flush_every, bool durable) {
                new (p) cpu::task::MrecRecorderParams{std::move(path), every,
                                                      flush_every, durable};
            },
            nb::arg("path"), nb::arg("every") = 1, nb::arg("flush_every") = 0,
            nb::arg("durable") = true,
            "  path         output store, must end in .mrec\n"
            "  every        write cadence [steps, >= 1; default 1] — one "
            "frame whenever the completed-step count is a multiple of every "
            "(including the initial state, step 0)\n"
            "  flush_every  automatic commit cadence [frames]; 0 lets molpy "
            "commit whole chunks\n"
            "  durable      sync the files on each commit\n\n"
            "The frame buffer is the driver's, not the recorder's: "
            "MD(capture_buffer_bytes=…). Checked by MD.set_record (ValueError "
            "on a bad path or every < 1).")
        .def_ro("path", &cpu::task::MrecRecorderParams::path)
        .def_ro("every", &cpu::task::MrecRecorderParams::every)
        .def_ro("flush_every", &cpu::task::MrecRecorderParams::flush_every)
        .def_ro("durable", &cpu::task::MrecRecorderParams::durable);
#endif

    // ── MD driver (categorical setters) ──
    nb::class_<MdHandle> md(
        m, "MD",
        "Molecular dynamics driver (CPU backend).\n\n"
        "Configure it fully before the first run(): the execution plan is "
        "compiled exactly once, so every setter — and on(phase) — raises "
        "RuntimeError afterwards. A different configuration means a new MD "
        "object. Running again with the same configuration stays legal.");
    ATV::python::bind_units(
        md, [](MdHandle& h) -> cpu::md::MD& { return h.driver; },
        [](const MdHandle& h, const char* s) { h.require_configurable_(s); });
    md.def(nb::init<>())
        .def(
            "set_capture_buffer_bytes",
            [](MdHandle& self, std::uint64_t bytes) {
                self.require_configurable_("set_capture_buffer_bytes");
                self.driver.set_capture_buffer_bytes(bytes);
            },
            nb::arg("bytes"),
            "Byte budget of the driver's shared asynchronous capture buffer; "
            ">= 1. Carried by the MPI launcher.")
        .def(
            "capture_stats",
            [](const MdHandle& self) {
                return capture_stats_dict(self.driver.capture_stats());
            },
            "Capture-buffer accounting of the driver's last run.")
        .def(
            "set_integrator",
            [](MdHandle& self, const cpu::task::VelocityVerletParams& p) {
                self.set_categorical("set_integrator", p,
                                     cpu::md::VelocityVerlet{p.dt});
            },
            nb::arg("comp"), "LAMMPS-style set_integrator (VelocityVerlet).")
        .def(
            "set_integrator",
            [](MdHandle& self, const cpu::task::NoseHooverNptParams& p) {
                self.set_categorical(
                    "set_integrator", p,
                    cpu::md::NoseHooverNPT{p.T_target, p.tau_T, p.P_target,
                                           p.tau_P, p.dt, p.drag, p.mtchain,
                                           p.mpchain, p.couple_axis, p.T_stop});
            },
            nb::arg("comp"),
            "LAMMPS-style set_integrator. NoseHooverNPT (LAMMPS `fix npt`) "
            "is integrator + thermostat + barostat in one Trotter step, so "
            "it takes the integrator slot and no separate thermostat or "
            "barostat is added with it.")
        .def(
            "set_integrator",
            [](MdHandle& self, const cpu::task::ParrinelloRahmanNptParams& p) {
                self.set_categorical(
                    "set_integrator", p,
                    cpu::md::ParrinelloRahmanNPT{p.T_target, p.P_target,
                                                 p.tau_T, p.tau_P, p.dt});
            },
            nb::arg("comp"),
            "LAMMPS-style set_integrator. ParrinelloRahmanNPT is the "
            "full-cell (shearable) variant of the same MTK NPT scheme and "
            "likewise takes the integrator slot.")
        .def(
            "set_integrator",
            [](MdHandle& self, const ATV::md::NvtSllodSpec& p) {
                self.set_nvt_sllod_(cpu::md::NvtSllod{p.temperature_K, p.tau_fs,
                                                      p.dt_fs, p.rate_per_fs,
                                                      p.direction, p.tchain});
            },
            nb::arg("comp"),
            "NVT SLLOD. Not mirrored into the MPI task payload.")
        .def(
            "set_barostat",
            [](MdHandle& self, const cpu::task::BerendsenBarostatParams& p) {
                self.set_categorical("set_barostat", p,
                                     cpu::md::Barostat{p.P_target, p.tau_P,
                                                       p.dt, p.bulk_modulus});
            },
            nb::arg("comp"),
            "LAMMPS-style set_barostat (BerendsenBarostat — `fix "
            "press/berendsen`). Rides alongside the integrator and a "
            "thermostat rather than replacing them, so a full NPT run is "
            "set_integrator(VelocityVerlet) + set_thermostat(...) + "
            "set_barostat(BerendsenBarostat(...)). The slot is plural: "
            "each call appends another barostat.")
        .def(
            "set_thermostat",
            [](MdHandle& self, const cpu::task::NoseHooverParams& p) {
                self.set_categorical(
                    "set_thermostat", p,
                    cpu::md::NoseHoover{p.T_start, p.tau_T, p.dt});
            },
            nb::arg("comp"), "LAMMPS-style set_thermostat (NoseHoover).")
        .def(
            "set_thermostat",
            [](MdHandle& self, const cpu::task::FixLangevinParams& p) {
                self.set_categorical("set_thermostat", p,
                                     cpu::md::FixLangevin{p.T_start, p.T_stop,
                                                          p.damp, p.dt,
                                                          p.seed});
            },
            nb::arg("comp"), "LAMMPS-style set_thermostat (Langevin).")
        .def(
            "set_thermostat",
            [](MdHandle& self, const cpu::task::FixCsvrParams& p) {
                self.set_categorical(
                    "set_thermostat", p,
                    cpu::md::FixCSVR{p.T_start, p.tau, p.dt, p.seed, p.types,
                                     cpu::task::csvr_bias(p.bias)});
            },
            nb::arg("comp"), "LAMMPS-style set_thermostat (CSVR).")
        .def(
            "set_integrator",
            [](MdHandle& self, cpu::md::GearPC& integrator) {
                self.set_gear_pc_(integrator);
            },
            nb::arg("comp"), nb::keep_alive<1, 2>(),
            "Fifth-order Gear predictor-corrector, borrowed. Not mirrored "
            "into the MPI task payload.")
        .def(
            "set_observer",
            [](MdHandle& self, cpu::ConfigurationalTemperature& observer) {
                self.set_configurational_temperature_(observer);
            },
            nb::arg("comp"), nb::keep_alive<1, 2>(),
            "ConfigurationalTemperature observer, borrowed. Not mirrored "
            "into the MPI task payload.")
        .def(
            "set_observer",
            [](MdHandle& self, const cpu::task::InitVelocityParams& p) {
                self.set_categorical(
                    "set_observer", p,
                    cpu::md::InitVelocity{p.temperature, p.seed});
            },
            nb::arg("comp"), "LAMMPS-style set_observer (e.g. InitVelocity).");

    md.def(
        "set_observer",
        [](MdHandle& self, const cpu::task::FixShakeParams& p) {
            self.set_categorical("set_observer", p,
                                 p.component<cpu::md::FixShake>());
        },
        nb::arg("comp"));
    md.def(
        "set_observer",
        [](MdHandle& self, const cpu::task::FixRattleParams& p) {
            self.set_categorical("set_observer", p,
                                 p.component<cpu::md::FixRattle>());
        },
        nb::arg("comp"));
    md.def(
        "set_observer",
        [](MdHandle& self, const cpu::task::FixLincsParams& p) {
            self.set_categorical("set_observer", p,
                                 p.component<cpu::md::FixLincs>());
        },
        nb::arg("comp"));

    // The set_evaluator slot is plural: every call appends another
    // energy/force source and they are summed. The typed spellings are what
    // Python's single `MD.set_evaluator` dispatches to by component type —
    // registered from `evaluator_setters.h`, the one place that knows the
    // eighteen force sources, because `Minimize` exposes the very same names.
    bind_evaluator_setters(
        md,
        // set_evaluator_pw — the MD reading of the same component.
        "Append the plane-wave DFT evaluator as an MD force source — "
        "Born-Oppenheimer dynamics on a plane-wave Hamiltonian, meaning "
        "the electrons are re-converged at every geometry the nuclei "
        "reach, with analytic ionic forces (local Hellmann-Feynman + "
        "Ewald + Kleinman-Bylander) taken at the converged density. "
        "Takes no NeighborList (a plane-wave Hamiltonian is not a pair "
        "style) and reads the frame's 'atomic_number' column [Z]. Like "
        "any MD run it needs a per-atom 'mass' column [mass], and like "
        "any plane-wave run it needs a box.\n\n"
        "A force-producing step uses one electronic solve and contracts its "
        "accepted state for forces. The driver borrows this same method and "
        "keeps its Python owner alive; last_result() is populated on it. "
        "This local provider is not serialized to a launcher task.");

    md.def(
          "set_neighbor",
          [](MdHandle& self, const cpu::task::NeighborListParams& p) {
              self.set_categorical("set_neighbor", p,
                                   cpu::md::NeighborList<>{p.cutoff, p.skin});
          },
          nb::arg("comp"), "LAMMPS-style set_neighbor.")
        .def(
            "set_thermo",
            [](MdHandle& self, int every) {
                self.set_categorical("set_thermo",
                                     cpu::task::ThermoParams{every},
                                     cpu::Thermo{every});
            },
            nb::arg("every"), "LAMMPS-style set_thermo(N) print cadence.")
        .def(
            "set_record",
            [](MdHandle& self, const cpu::task::ThermoParams& p) {
                // Thermo as a record/observer-style component via set_thermo.
                self.set_categorical("set_record", p, cpu::Thermo{p.freq});
            },
            nb::arg("comp"),
            "Register a recorder; Thermo is installed via set_thermo cadence.")
#ifdef ATV_HAS_IO
        .def(
            "set_record",
            [](MdHandle& self, const cpu::task::XyzRecorderParams& p) {
                self.set_categorical("set_record", p,
                                     cpu::XyzRecorder{p.filename, p.freq});
            },
            nb::arg("comp"),
            "Register the XYZ trajectory recorder (LAMMPS `dump`). Binds at "
            "end_of_step and writes one frame every freq steps; raises "
            "ValueError on an empty path or freq < 1.")
#endif
#ifdef ATV_HAS_IO
        .def(
            "set_record",
            [](MdHandle& self, const cpu::task::MrecRecorderParams& p) {
                self.set_categorical("set_record", p,
                                     cpu::MrecRecorder{p.path, p.every,
                                                       p.flush_every,
                                                       p.durable});
            },
            nb::arg("comp"),
            "Register the `*.mrec` trajectory recorder (cpu::MrecRecorder). "
            "Binds at end_of_step; raises ValueError on a path without the "
            ".mrec suffix or every < 1. Carried by the MPI launcher (rank-0 "
            "gather and write).")
#endif
        .def(
            "on",
            [](MdHandle& self, md::Phase phase) {
                // Refused HERE rather than at the builder's add(): on() is the
                // registration entry point, so a caller who has already run
                // gets the contract before constructing a component.
                self.require_configurable_("on");
                return MdPhaseBuilder{&self, phase};
            },
            nb::arg("phase"),
            "Escape hatch: explicit phase binding (prefer categorical "
            "setters). The returned builder carries add_* methods for six "
            "components only — PairLJCut, NeighborList, BondHarmonic, "
            "AngleHarmonic, DihedralMultiHarmonic and Thermo. Every other "
            "force source, including the whole widened real-space catalogue, "
            "goes through set_evaluator, where phase and order are the "
            "driver's business.")
        .def(
            "build", [](MdHandle& self) { self.driver.build(); },
            "Compile the execution plan.")
        .def("set_run_span", &MdHandle::set_run_span, nb::arg("nfirst"),
             nb::arg("nlast"),
             "Fix the Langevin / NoseHooverNPT T_stop ramp to [nfirst, "
             "nlast] completed steps for a trajectory of several run() "
             "calls.")
        .def("run", &MdHandle::run, nb::arg("frame"), nb::arg("n_steps"),
             "Run MD on a molpy.Frame for n_steps. GIL released during "
             "compute; under the MPI launcher this dispatches ONE task to "
             "every rank and writes the gathered result back.\n\n"
             "A run() on the molpy.Frame the previous one returned, "
             "unedited, continues that trajectory (LAMMPS `run N pre no`): "
             "integrator history, thermostat variables, the step count and "
             "the recorders' step axis carry over, as on CUDA. An edit to "
             "that frame restarts what depends on the edited input (e.g. the "
             "Gear history after a velocity edit). Another molpy.Frame starts "
             "an independent simulation. A temperature ramp spans "
             "set_run_span's window, else the first run's [0, n_steps]; "
             "later chunks continue it. Without the launcher only: under it "
             "each call is one task.\n\n"
             "A completed run() freezes the configuration: the plan is "
             "compiled once, so every setter and on(phase) raises "
             "RuntimeError from here on. Running again is still allowed and "
             "re-runs that same plan.\n\n"
             "Raises RuntimeError before any compute if a Coulomb evaluator "
             "is registered and the frame's 'atoms' block carries no "
             "'charge' column [e] — a forgotten column is indistinguishable "
             "from deliberate zeros once the frame reaches C++, so it is "
             "caught here. Insert an explicit zero column to mean zero.")
        .def("enable_timing",
             [](MdHandle& self) { self.driver.enable_timing(); })
        .def("summary", [](MdHandle& self) { return self.driver.summary(); });

    // ── PhaseBuilder escape hatch ──
    nb::class_<MdPhaseBuilder>(m, "PhaseBuilder",
                               "Fluent builder for explicit phase binding.")
        .def(
            "add_pair_lj_cut",
            [](MdPhaseBuilder& self, const cpu::task::PairLJCutParams& p,
               int order) -> MdPhaseBuilder& {
                self.handle->add_on_phase(
                    p, cpu::task::internal::make_pair_lj_cut(p),
                    MDBinding{self.phase, order});
                return self;
            },
            nb::arg("comp"), nb::arg("order") = 0, nb::rv_policy::reference)
        .def(
            "add_neighbor_list",
            [](MdPhaseBuilder& self, const cpu::task::NeighborListParams& p,
               int order) -> MdPhaseBuilder& {
                self.handle->add_on_phase(
                    p, cpu::md::NeighborList<>{p.cutoff, p.skin},
                    MDBinding{self.phase, order});
                return self;
            },
            nb::arg("comp"), nb::arg("order") = 0, nb::rv_policy::reference)
        .def(
            "add_bond_harmonic",
            [](MdPhaseBuilder& self, const cpu::task::BondHarmonicParams& p,
               int order) -> MdPhaseBuilder& {
                self.handle->add_on_phase(p, cpu::BondHarmonic{p.K, p.r0},
                                          MDBinding{self.phase, order});
                return self;
            },
            nb::arg("comp"), nb::arg("order") = 0, nb::rv_policy::reference)
        .def(
            "add_angle_harmonic",
            [](MdPhaseBuilder& self, const cpu::task::AngleHarmonicParams& p,
               int order) -> MdPhaseBuilder& {
                self.handle->add_on_phase(p, cpu::AngleHarmonic{p.K, p.theta0},
                                          MDBinding{self.phase, order});
                return self;
            },
            nb::arg("comp"), nb::arg("order") = 0, nb::rv_policy::reference)
        .def(
            "add_dihedral_multi_harmonic",
            [](MdPhaseBuilder& self,
               const cpu::task::DihedralMultiHarmonicParams& p,
               int order) -> MdPhaseBuilder& {
                self.handle->add_on_phase(p, cpu::DihedralMultiHarmonic{p.Cn},
                                          MDBinding{self.phase, order});
                return self;
            },
            nb::arg("comp"), nb::arg("order") = 0, nb::rv_policy::reference)
        .def(
            "add_thermo",
            [](MdPhaseBuilder& self, const cpu::task::ThermoParams& p,
               int order) -> MdPhaseBuilder& {
                self.handle->add_on_phase(p, cpu::Thermo{p.freq},
                                          MDBinding{self.phase, order});
                return self;
            },
            nb::arg("comp"), nb::arg("order") = 0, nb::rv_policy::reference);
}
