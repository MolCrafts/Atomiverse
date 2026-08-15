/// @file python/src/bindings/bind_md.cpp
/// @brief MD driver + component bindings aligned with the C++ categorical API.
///
/// Public surface mirrors ``examples/07_md_lj.cpp`` and ``cpu::md::MD``:
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

#include "evaluator_setters.h"
#include "molrs_frame_bridge.h"

#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/barostat.h"
#include "atomiverse/cpu/dynamics/fix_csvr.h"
#include "atomiverse/cpu/dynamics/fix_langevin.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/dynamics/parrinello_rahman_npt.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/molrs_exchange.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
// The nine force sources this file used to spell out are registered from
// `evaluator_setters.h`, which owns their headers and their build gates.
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/record/thermo.h"
#include "atomiverse/cpu/record/xyz_recorder.h"
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

/// @brief Does `block` of a molrs Frame carry the column `key`?
///
/// The molrs boundary is the only place where a column's *absence* is still
/// visible: `cpu::from_molrs` allocates every optional per-atom column and
/// leaves an absent one zero-filled, so a Coulomb evaluator downstream cannot
/// tell "no charges were given" from "every charge is 0 e".
///
/// @param frame Borrowed molrs Frame handle.
/// @param block Block name, e.g. `"atoms"`.
/// @param key   Column name, e.g. `"charge"`.
/// @return True when that block carries that column.
bool has_molrs_column(const molrs::FrameRef& frame, const char* block,
                      std::string_view key) {
    for (const ::rust::String& raw : molrs::frame_block_columns(frame, block))
        if (std::string_view{raw.data(), raw.size()} == key)
            return true;
    return false;
}

/// @brief Is this mirror entry an electrostatic evaluator?
///
/// Every Coulomb style reads per-atom charge: the truncated pair style, the
/// real-space Ewald half and the reciprocal PME half alike. The PME mirror is
/// always compiled (the wire layer does not need `ATV_HAS_KSPACE`), so no
/// build gate belongs here.
///
/// @param entry One recorded MD registration.
/// @return True for `PairCoulCut` / `CoulLong` / `KSpacePME` entries.
bool is_coulomb_entry(const MDComponentEntry& entry) {
    return std::visit(
        [](const auto& p) {
            using P = std::decay_t<decltype(p)>;
            return std::is_same_v<P, cpu::task::PairCoulCutParams>
                   || std::is_same_v<P, cpu::task::CoulLongParams>
                   || std::is_same_v<P, cpu::task::KSpacePmeParams>;
        },
        entry.params);
}

/// Bound MD driver + TaskSpec mirror of its registrations.
struct MdHandle {
    // Ramping components the handle owns so `run()` can hand them their run
    // span (LAMMPS `update->beginstep`/`endstep`), exactly as the launcher
    // path gets it from the task materializer. Declared BEFORE `driver`:
    // members are destroyed in reverse declaration order, so a borrowed
    // component outlives the driver that references it.
    std::optional<cpu::md::NoseHooverNPT> npt_owned;
    std::vector<std::unique_ptr<cpu::md::FixLangevin>> langevin_owned;

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

    template <typename Params, typename Component>
    void set_categorical(const char* setter, const Params& p, Component comp) {
        require_configurable_(setter);
        using P = std::decay_t<Params>;
        if constexpr (std::is_same_v<P, cpu::task::PairLJCutParams>
                      || std::is_same_v<P, cpu::task::BondHarmonicParams>
                      || std::is_same_v<P, cpu::task::AngleHarmonicParams>
                      || std::is_same_v<P, cpu::task::PairCoulCutParams>
                      || std::is_same_v<P, cpu::task::CoulLongParams>
                      || std::is_same_v<P,
                                        cpu::task::DihedralMultiHarmonicParams>)
            driver.set_evaluator(std::move(comp));
#ifdef ATV_HAS_KSPACE
        else if constexpr (std::is_same_v<P, cpu::task::KSpacePmeParams>)
            // Move-only (it owns FFTW plans), so the component arrives here
            // already moved from the setter's prvalue.
            driver.set_evaluator(std::move(comp));
#endif
#ifdef ATV_HAS_SEMIEMPIRICAL
        else if constexpr (std::is_same_v<P, cpu::task::AM1EvaluatorParams>)
            // A semiempirical force source rides the same plural evaluator
            // slot as the pair styles — that is what makes an MD run BOMD.
            driver.set_evaluator(std::move(comp));
#endif
#ifdef ATV_HAS_PW
        else if constexpr (std::is_same_v<P, cpu::task::PwEvaluatorParams>)
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
            // n_steps]` span before every launch. A second NPT keeps the first
            // (the singular slot is idempotent on the same component name),
            // so the borrowed storage is never rebuilt underneath the driver.
            if (!npt_owned)
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
        else if constexpr (std::is_same_v<P, cpu::task::InitVelocityParams>)
            driver.set_observer(std::move(comp));
        else if constexpr (std::is_same_v<P, cpu::task::ThermoParams>)
            driver.set_thermo(
                static_cast<const cpu::task::ThermoParams&>(p).freq);
        else if constexpr (std::is_same_v<P, cpu::task::XyzRecorderParams>)
            // The trajectory sink owns its file handle and its own stride
            // gate, so it goes in whole through the recorder slot (plural)
            // rather than through a driver-level cadence like Thermo's.
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
    ///   `cpu::from_molrs` allocates it either way and zero-fills it when the
    ///   molrs Frame did not carry it, because a C++ caller that hands over a
    ///   frame with no charges is stating a neutral system on purpose.
    /// - At the **Python boundary** that same silence is the common mistake:
    ///   an all-zero charge column and a forgotten charge column produce
    ///   identical, plausible-looking output (energy finite, forces finite,
    ///   electrostatics simply absent), and by the time `from_molrs` has run
    ///   the difference is gone. So the binding checks the molrs Frame — the
    ///   last place where absence is still visible — and throws with a how-to
    ///   instead of running a Coulomb model on charges nobody supplied.
    ///
    /// Intentional zeros stay expressible: insert an explicit zero `charge`
    /// column and the check passes.
    ///
    /// @param frame Borrowed molrs Frame about to be advanced.
    /// @throws std::runtime_error when a Coulomb evaluator is registered and
    ///         the frame's `atoms` block has no `charge` column.
    void require_charge_column(const molrs::FrameRef& frame) const {
        const bool coulomb =
            std::any_of(mirror.begin(), mirror.end(), is_coulomb_entry);
        if (!coulomb || has_molrs_column(frame, "atoms", "charge"))
            return;
        throw std::runtime_error(
            "this MD run registers a Coulomb evaluator but the molrs.Frame "
            "has no per-atom 'charge' column, so every charge would be taken "
            "as 0 e. Add the charges [e] before run(), e.g. "
            "frame[\"atoms\"].insert(\"charge\", q).");
    }

    void run(nb::object frame_obj, int n_steps) {
        const cpu::task::LauncherGate gate;
        if (!gate.active()) {
            // Every ramping component gets THIS run's span `[0, n_steps]`,
            // identical to what `TaskMaterializer` installs on the launcher
            // path. Each `run()` is an independent simulation — the molrs
            // overload of `cpu::md::MD::run` tears its run down before
            // returning, so the next call's `LaunchContext::step` restarts at
            // 0 and a span carried over from an earlier call would put the
            // ramp fraction permanently past 1. One continuous temperature
            // ramp is therefore ONE `run()` with the full step budget, not a
            // chunked sequence of them.
            const std::int64_t nlast = n_steps;
            if (npt_owned)
                npt_owned->set_run_span(0, nlast);
            for (auto& langevin : langevin_owned)
                langevin->set_run_span(0, nlast);
            auto frame = borrow_molrs_frame(frame_obj);
            require_charge_column(*frame);
            {
                nb::gil_scoped_release release;
                driver.run(*frame, n_steps);
            }
            // Armed only on a COMPLETED run: a run that threw (missing
            // integrator, chargeless frame, ...) left no compiled plan behind,
            // so the caller must still be able to finish configuring.
            ran_ = true;
            return;
        }
#ifdef ATV_HAS_MPI
        cpu::task::TaskSpec spec(
            cpu::task::MDTask{mirror, static_cast<std::int64_t>(n_steps)});
        auto fref = borrow_molrs_frame(frame_obj);
        require_charge_column(*fref);
        cpu::Frame work = cpu::from_molrs(*fref);
        cpu::Frame result(0);
        {
            nb::gil_scoped_release release;
            cpu::task::TaskDispatcher dispatcher(MPI_COMM_WORLD,
                                                 cpu::task::kProtocolVersion,
                                                 /*handshake=*/false);
            result = dispatcher.dispatch(spec, std::move(work)).frame;
        }
        cpu::to_molrs(result, *fref);
        ran_ = true;
#endif
    }

  private:
    /// True once a `run()` has completed; see `require_configurable_`.
    bool ran_ = false;
};

struct MdPhaseBuilder {
    MdHandle* handle;
    ATV::md::Phase phase;
};

} // namespace

void bind_md(nb::module_& m) {
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
            nb::arg("dt"), "Construct with timestep dt [fs].");

    nb::class_<cpu::task::InitVelocityParams>(
        m, "InitVelocity", "Maxwell-Boltzmann velocity initialization.")
        .def(
            "__init__",
            [](cpu::task::InitVelocityParams* p, double temperature,
               unsigned seed) {
                new (p) cpu::task::InitVelocityParams{temperature, seed};
            },
            nb::arg("temperature"), nb::arg("seed"),
            "Construct with target temperature [K] and RNG seed.");

    nb::class_<cpu::task::NoseHooverParams>(
        m, "NoseHoover", "Nose-Hoover thermostat (LAMMPS `fix nvt`).")
        .def(
            "__init__",
            [](cpu::task::NoseHooverParams* p, double T_target, double tau_T,
               double dt) {
                new (p) cpu::task::NoseHooverParams{T_target, tau_T, dt};
            },
            nb::arg("T_target"), nb::arg("tau_T"), nb::arg("dt"),
            "Construct with target temperature [K], coupling time "
            "tau_T [fs] (LAMMPS `Tdamp`), and timestep [fs].");

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
            "Construct with ramp temperatures T_start/T_stop [K], damping "
            "time damp [fs] (LAMMPS `damp`), timestep dt [fs], and RNG seed.");

    nb::class_<cpu::task::FixCsvrParams>(
        m, "CSVR",
        "Bussi canonical-sampling velocity-rescale thermostat (CP2K CSVR).")
        .def(
            "__init__",
            [](cpu::task::FixCsvrParams* p, double T_start, double tau,
               double dt, unsigned seed) {
                new (p) cpu::task::FixCsvrParams{T_start, tau, dt, seed};
            },
            nb::arg("T_start"), nb::arg("tau"), nb::arg("dt"), nb::arg("seed"),
            "Construct with target temperature [K], coupling time tau [fs] "
            "(CP2K `TIMECON`), timestep dt [fs], and RNG seed.");

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
            "Construct with target temperature T_target [K], thermostat "
            "coupling time tau_T [fs] (LAMMPS `Tdamp`), target pressure "
            "P_target [eV/A^3] (LAMMPS sign, positive compressive; 1 bar ~ "
            "6.2415e-7 eV/A^3), barostat coupling time tau_P [fs] (LAMMPS "
            "`Pdamp`), and timestep dt [fs]. A coupling time is the "
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
            "  T_stop      end temperature [K] of a linear setpoint ramp "
            "across the run span, as in LAMMPS. Any value >= 0 is taken as "
            "the ramp end (equal to T_target means a constant setpoint); "
            "< 0 folds to T_target. The run span itself is wired "
            "automatically by run(), which ramps over [0, n_steps]: every "
            "run() is an independent simulation, so one continuous ramp is "
            "one run() with the full step budget.");

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
            "Construct with target temperature T_target [K], target "
            "hydrostatic pressure P_target [eV/A^3] (LAMMPS sign, positive "
            "compressive; 1 bar ~ 6.2415e-7 eV/A^3), thermostat coupling "
            "time tau_T [fs] (LAMMPS `Tdamp`), barostat coupling time "
            "tau_P [fs] (LAMMPS `Pdamp`), and timestep dt [fs].\n\n"
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
            nb::arg("bulk_modulus") = double(1),
            "Construct with target pressure P_target [eV/A^3] (LAMMPS sign, "
            "positive compressive; 1 bar ~ 6.2415e-7 eV/A^3), barostat "
            "coupling time tau_P [fs] (LAMMPS `Pdamp`), and timestep dt "
            "[fs].\n\n"
            "bulk_modulus is the tunable modulus beta [eV/A^3] (LAMMPS "
            "`modulus`). It divides the per-step rescale, so a larger value "
            "means a gentler box change and slower pressure relaxation. The "
            "default is 1.0, deliberately NOT LAMMPS's 10.0 — see "
            "cpu/dynamics/barostat.h; pass 10.0 for LAMMPS-identical "
            "relaxation.");

    nb::class_<cpu::task::PairLJCutParams>(m, "PairLJCut",
                                           "Lennard-Jones 12-6 pair potential.")
        .def(
            "__init__",
            [](cpu::task::PairLJCutParams* p, double epsilon, double sigma,
               double cutoff) {
                new (p) cpu::task::PairLJCutParams{epsilon, sigma, cutoff};
            },
            nb::arg("epsilon"), nb::arg("sigma"), nb::arg("cutoff"),
            "Construct with epsilon [eV], sigma [A], and cutoff [A].");

    nb::class_<cpu::task::PairCoulCutParams>(
        m, "PairCoulCut",
        "Truncated Coulomb pair style (LAMMPS `pair_style coul/cut`).\n\n"
        "The bare q_i q_j / r sum over the neighbor list, hard-cut at "
        "cutoff. It is self-contained: unlike CoulLong it takes no "
        "reciprocal-space partner, so it is the whole electrostatic model of "
        "the run that installs it. Charges come from the per-atom 'charge' "
        "column [e] of the molrs.Frame; a frame without that column is "
        "rejected by run() instead of being read as neutral.")
        .def(
            "__init__",
            [](cpu::task::PairCoulCutParams* p, double cutoff) {
                new (p) cpu::task::PairCoulCutParams{cutoff};
            },
            nb::arg("cutoff"),
            "Construct with the real-space cutoff [A]. The neighbor list "
            "must be built with a cutoff >= this value.");

    nb::class_<cpu::task::CoulLongParams>(
        m, "CoulLong",
        "Real-space half of an Ewald/PME split (LAMMPS `pair_style "
        "coul/long`).\n\n"
        "Sums erfc(alpha r) q_i q_j / r over the neighbor list; the "
        "complementary smooth part is summed in reciprocal space by the "
        "matched KSpacePME evaluator of the same run. alpha is therefore not "
        "a private knob — it must equal the reciprocal partner's alpha, and "
        "build() refuses a run that installs only one of the two halves.")
        .def(
            "__init__",
            [](cpu::task::CoulLongParams* p, double alpha, double cutoff) {
                new (p) cpu::task::CoulLongParams{alpha, cutoff};
            },
            nb::arg("alpha"), nb::arg("cutoff"),
            "Construct with the Ewald splitting parameter alpha [1/A] — "
            "which must equal the KSpacePME alpha — and the real-space "
            "cutoff [A]. The neighbor list must be built with a cutoff >= "
            "this value.\n\n"
            "alpha sets where the split between the two halves falls: "
            "larger alpha screens the real-space term harder and moves work "
            "out of the neighbor-list sum into the reciprocal one. The "
            "LAMMPS rule of thumb is alpha ~ 3.5 / cutoff, at which the "
            "screened pair term has fallen to erfc(3.5) ~ 7e-7 of the bare "
            "one at the cutoff.");

#ifdef ATV_HAS_KSPACE
    nb::class_<cpu::task::KSpacePmeParams>(
        m, "KSpacePME",
        "Reciprocal-space half of the same Ewald/PME split — smooth "
        "Particle-Mesh Ewald (Essmann et al., J. Chem. Phys. 103, 8577 "
        "(1995)).\n\n"
        "Rather than a direct O(N^2) double sum, each charge is spread onto "
        "nearby points of a mesh with B-spline weights, the mesh is "
        "Fourier-transformed (FFT), multiplied point-by-point by an "
        "influence function fixed by the cell, alpha and order, and "
        "transformed back — O(N log N). It is not a Hamiltonian on its own: "
        "install it together with CoulLong at the SAME alpha, or build() "
        "raises.\n\n"
        "This type exists only in an extension built with "
        "ATV_BUILD_KSPACE=ON (the solver needs FFTW3).")
        .def(
            "__init__",
            [](cpu::task::KSpacePmeParams* p, double alpha, double grid_spacing,
               int order) {
                new (p) cpu::task::KSpacePmeParams{alpha, grid_spacing, order};
            },
            nb::arg("alpha"), nb::arg("grid_spacing") = double(1),
            nb::arg("order") = 6,
            "Construct with the Ewald splitting parameter alpha [1/A], which "
            "must equal the CoulLong alpha.\n\n"
            "  grid_spacing  target real-space mesh spacing [A] (default "
            "1.0). The mesh is derived from the box and rounded to "
            "FFT-friendly sizes, so a smaller spacing buys a finer mesh and "
            "a more accurate — costlier — reciprocal sum.\n"
            "  order         B-spline interpolation order [count, >= 3; "
            "default 6] — the width of the charge-spreading stencil per "
            "dimension.");
#endif

    nb::class_<cpu::task::NeighborListParams>(
        m, "NeighborList", "Cell-list based neighbor list component.")
        .def(
            "__init__",
            [](cpu::task::NeighborListParams* p, double cutoff, double skin) {
                new (p) cpu::task::NeighborListParams{cutoff, skin};
            },
            nb::arg("cutoff"), nb::arg("skin") = double(-1),
            "Construct with cutoff [A] and optional skin distance [A].");

    nb::class_<cpu::task::BondHarmonicParams>(
        m, "BondHarmonic", "Harmonic bond potential V = K(r - r0)^2.")
        .def(
            "__init__",
            [](cpu::task::BondHarmonicParams* p, std::vector<double> K,
               std::vector<double> r0) {
                new (p)
                    cpu::task::BondHarmonicParams{std::move(K), std::move(r0)};
            },
            nb::arg("K"), nb::arg("r0"),
            "Construct with force constants K and equilibrium distances r0 "
            "per bond type.");

    nb::class_<cpu::task::AngleHarmonicParams>(
        m, "AngleHarmonic", "Harmonic angle potential V = K(theta - theta0)^2.")
        .def(
            "__init__",
            [](cpu::task::AngleHarmonicParams* p, std::vector<double> K,
               std::vector<double> theta0) {
                new (p) cpu::task::AngleHarmonicParams{std::move(K),
                                                       std::move(theta0)};
            },
            nb::arg("K"), nb::arg("theta0"),
            "Construct with force constants K and equilibrium angles theta0 "
            "per angle type.");

    nb::class_<cpu::task::DihedralMultiHarmonicParams>(
        m, "DihedralMultiHarmonic", "Multi-harmonic dihedral potential.")
        .def(
            "__init__",
            [](cpu::task::DihedralMultiHarmonicParams* p,
               std::vector<std::vector<double>> Cn) {
                new (p) cpu::task::DihedralMultiHarmonicParams{std::move(Cn)};
            },
            nb::arg("Cn"), "Construct with coefficients Cn per dihedral type.");

    nb::class_<cpu::task::ThermoParams>(m, "Thermo",
                                        "Thermodynamic output at end_of_step.")
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
        "Writes one frame — element symbol plus x y z [A] per atom, under a "
        "comment line carrying the cell and the typed frame metadata — every "
        "`freq` steps. The symbol is derived from the frame's per-atom "
        "`atomic_number` column (Z), so Z=18 is written as Ar.\n\n"
        "The first write of a given recorder truncates the file and every "
        "later write of that same recorder appends, so handing a NEW "
        "XYZ(path, freq) to a second run starts the trajectory over instead "
        "of extending it.\n\n"
        "Within one run() call the step index runs 0..n_steps-1, so the "
        "stride gate opens on the steps 0, freq, 2*freq, ... below n_steps: "
        "a 10-step run at freq=5 writes 2 frames (steps 0 and 5), never 3. "
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
            "whenever step % freq == 0.\n\n"
            "This object is a plain parameter record; the values are checked "
            "when the recorder is built, so an empty path or freq < 1 raises "
            "ValueError from MD.set_record(...), not from here.");

    // ── MD driver (categorical setters) ──
    nb::class_<MdHandle> md(
        m, "MD",
        "Molecular dynamics driver (CPU backend).\n\n"
        "Configure it fully before the first run(): the execution plan is "
        "compiled exactly once, so every setter — and on(phase) — raises "
        "RuntimeError afterwards. A different configuration means a new MD "
        "object. Running again with the same configuration stays legal.");
    md.def(nb::init<>())
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
                    cpu::md::FixCSVR{p.T_start, p.tau, p.dt, p.seed});
            },
            nb::arg("comp"), "LAMMPS-style set_thermostat (CSVR).")
        .def(
            "set_observer",
            [](MdHandle& self, const cpu::task::InitVelocityParams& p) {
                self.set_categorical(
                    "set_observer", p,
                    cpu::md::InitVelocity{p.temperature, p.seed});
            },
            nb::arg("comp"), "LAMMPS-style set_observer (e.g. InitVelocity).");

    // The set_evaluator slot is plural: every call appends another
    // energy/force source and they are summed. The typed spellings are what
    // Python's single `MD.set_evaluator` dispatches to by component type —
    // registered from `evaluator_setters.h`, the one place that knows the nine
    // force sources, because `Minimize` exposes the very same nine names.
    bind_evaluator_setters(
        md,
        // set_evaluator_am1 — the MD reading of the same component.
        "Append the AM1 semiempirical evaluator as an MD force source — "
        "Born-Oppenheimer dynamics, meaning the electrons are "
        "re-converged at every geometry the nuclei reach: one "
        "self-consistent-field (SCF) solve per step, with forces from the "
        "fixed-density central finite difference (step 1e-4 A). Takes no "
        "NeighborList (AM1 is molecular, not a pair style) and reads the "
        "frame's 'atomic_number' column [Z]. Like any MD run it also "
        "needs a per-atom 'mass' column [amu] — the integrator divides by "
        "it.\n\n"
        "The driver installs a FRESH gradient-enabled copy of the five "
        "SCF parameters: the object you pass keeps its own state, so "
        "last_result() on it is NOT populated by run() — per-step "
        "results (energy [eV], charges [e], forces [eV/A]) land on the "
        "frame.",
        // set_evaluator_pw — the MD reading of the same component.
        "Append the plane-wave DFT evaluator as an MD force source — "
        "Born-Oppenheimer dynamics on a plane-wave Hamiltonian, meaning "
        "the electrons are re-converged at every geometry the nuclei "
        "reach, with analytic ionic forces (local Hellmann-Feynman + "
        "Ewald + Kleinman-Bylander) taken at the converged density. "
        "Takes no NeighborList (a plane-wave Hamiltonian is not a pair "
        "style) and reads the frame's 'atomic_number' column [Z]. Like "
        "any MD run it needs a per-atom 'mass' column [amu], and like "
        "any plane-wave run it needs a box.\n\n"
        "Budget it deliberately: a step costs TWO self-consistent-field "
        "solves — the energy solve, then the analytic force path's own — "
        "and each is seconds of work at production settings, so a Python "
        "smoke run wants a small cell, a low ecut_ry and a modest "
        "max_iter.\n\n"
        "The driver installs a FRESH gradient-enabled copy of the "
        "fifteen method parameters: the object you pass keeps its own "
        "state, so last_result() on it is NOT populated by run() — "
        "per-step results (energy [eV], forces [eV/A]) land on the "
        "frame.");

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
            "setters).")
        .def(
            "build", [](MdHandle& self) { self.driver.build(); },
            "Compile the execution plan.")
        .def("run", &MdHandle::run, nb::arg("frame"), nb::arg("n_steps"),
             "Run MD on a molrs.Frame for n_steps. GIL released during "
             "compute; under the MPI launcher this dispatches ONE task to "
             "every rank and writes the gathered result back.\n\n"
             "Each call is an INDEPENDENT simulation starting from the "
             "frame's current state: the step index restarts at 0 and any "
             "temperature ramp is re-installed over [0, n_steps]. A single "
             "continuous ramp is one run() with the full step budget.\n\n"
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
                    p, cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff},
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
