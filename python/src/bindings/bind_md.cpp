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

#include "molrs_frame_bridge.h"

#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
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
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/record/thermo.h"
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
    /// Cumulative MD steps already run — `LaunchContext::step` keeps counting
    /// across chunked `run()` calls, while each `run()` re-ramps over its own
    /// span (LAMMPS `run` command semantics).
    std::int64_t steps_done = 0;

    cpu::md::MD driver;
    std::vector<MDComponentEntry> mirror;

    template <typename Params, typename Component>
    void set_categorical(const Params& p, Component comp) {
        using P = std::decay_t<Params>;
        if constexpr (std::is_same_v<P, cpu::task::PairLJCutParams>
                      || std::is_same_v<P, cpu::task::BondHarmonicParams>
                      || std::is_same_v<P, cpu::task::AngleHarmonicParams>
                      || std::is_same_v<P,
                                        cpu::task::DihedralMultiHarmonicParams>)
            driver.set_evaluator(std::move(comp));
        else if constexpr (std::is_same_v<P, cpu::task::NeighborListParams>)
            driver.set_neighbor(std::move(comp));
        else if constexpr (std::is_same_v<P, cpu::task::VelocityVerletParams>
                           || std::is_same_v<
                               P, cpu::task::ParrinelloRahmanNptParams>)
            // Neither carries a temperature ramp — no span to install.
            driver.set_integrator(std::move(comp));
        else if constexpr (std::is_same_v<P, cpu::task::NoseHooverNptParams>) {
            // Borrowed, not transferred: `run()` re-ramps the setpoint over
            // its own span before every launch. A second NPT keeps the first
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
        else
            static_assert(cpu::task::detail::unsupported_component_v<P>);
        mirror.push_back(MDComponentEntry{p, std::nullopt});
    }

    /// Escape hatch: explicit phase + order (maps to Driver::on(phase).add).
    template <typename Params, typename Component>
    void add_on_phase(const Params& p, Component comp, MDBinding binding) {
        driver.on(binding.phase).add(std::move(comp), binding.order);
        mirror.push_back(MDComponentEntry{p, binding});
    }

    void run(nb::object frame_obj, int n_steps) {
        const cpu::task::LauncherGate gate;
        if (!gate.active()) {
            // Every ramping component gets THIS run's span, mirroring the
            // launcher path (`TaskMaterializer` installs `[0, n_steps]`).
            const std::int64_t nfirst = steps_done;
            const std::int64_t nlast = steps_done + n_steps;
            if (npt_owned)
                npt_owned->set_run_span(nfirst, nlast);
            for (auto& langevin : langevin_owned)
                langevin->set_run_span(nfirst, nlast);
            auto frame = borrow_molrs_frame(frame_obj);
            {
                nb::gil_scoped_release release;
                driver.run(*frame, n_steps);
            }
            steps_done = nlast;
            return;
        }
#ifdef ATV_HAS_MPI
        cpu::task::TaskSpec spec(
            cpu::task::MDTask{mirror, static_cast<std::int64_t>(n_steps)});
        auto fref = borrow_molrs_frame(frame_obj);
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
#endif
    }
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
            "automatically by run(), which ramps over each call's own "
            "[steps_done, steps_done + n_steps].");

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

    // ── MD driver (categorical setters) ──
    nb::class_<MdHandle>(m, "MD", "Molecular dynamics driver (CPU backend).")
        .def(nb::init<>())
        .def(
            "set_integrator",
            [](MdHandle& self, const cpu::task::VelocityVerletParams& p) {
                self.set_categorical(p, cpu::md::VelocityVerlet{p.dt});
            },
            nb::arg("comp"), "LAMMPS-style set_integrator (VelocityVerlet).")
        .def(
            "set_integrator",
            [](MdHandle& self, const cpu::task::NoseHooverNptParams& p) {
                self.set_categorical(
                    p,
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
                    p, cpu::md::ParrinelloRahmanNPT{p.T_target, p.P_target,
                                                    p.tau_T, p.tau_P, p.dt});
            },
            nb::arg("comp"),
            "LAMMPS-style set_integrator. ParrinelloRahmanNPT is the "
            "full-cell (shearable) variant of the same MTK NPT scheme and "
            "likewise takes the integrator slot.")
        .def(
            "set_barostat",
            [](MdHandle& self, const cpu::task::BerendsenBarostatParams& p) {
                self.set_categorical(p,
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
                    p, cpu::md::NoseHoover{p.T_start, p.tau_T, p.dt});
            },
            nb::arg("comp"), "LAMMPS-style set_thermostat (NoseHoover).")
        .def(
            "set_thermostat",
            [](MdHandle& self, const cpu::task::FixLangevinParams& p) {
                self.set_categorical(p, cpu::md::FixLangevin{p.T_start,
                                                             p.T_stop, p.damp,
                                                             p.dt, p.seed});
            },
            nb::arg("comp"), "LAMMPS-style set_thermostat (Langevin).")
        .def(
            "set_thermostat",
            [](MdHandle& self, const cpu::task::FixCsvrParams& p) {
                self.set_categorical(
                    p, cpu::md::FixCSVR{p.T_start, p.tau, p.dt, p.seed});
            },
            nb::arg("comp"), "LAMMPS-style set_thermostat (CSVR).")
        .def(
            "set_observer",
            [](MdHandle& self, const cpu::task::InitVelocityParams& p) {
                self.set_categorical(
                    p, cpu::md::InitVelocity{p.temperature, p.seed});
            },
            nb::arg("comp"), "LAMMPS-style set_observer (e.g. InitVelocity).")
        .def(
            "set_evaluator_pair_lj_cut",
            [](MdHandle& self, const cpu::task::PairLJCutParams& p) {
                self.set_categorical(
                    p, cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff});
            },
            nb::arg("comp"))
        .def(
            "set_evaluator_bond_harmonic",
            [](MdHandle& self, const cpu::task::BondHarmonicParams& p) {
                self.set_categorical(p, cpu::BondHarmonic{p.K, p.r0});
            },
            nb::arg("comp"))
        .def(
            "set_evaluator_angle_harmonic",
            [](MdHandle& self, const cpu::task::AngleHarmonicParams& p) {
                self.set_categorical(p, cpu::AngleHarmonic{p.K, p.theta0});
            },
            nb::arg("comp"))
        .def(
            "set_evaluator_dihedral_multi_harmonic",
            [](MdHandle& self,
               const cpu::task::DihedralMultiHarmonicParams& p) {
                self.set_categorical(p, cpu::DihedralMultiHarmonic{p.Cn});
            },
            nb::arg("comp"))
        .def(
            "set_neighbor",
            [](MdHandle& self, const cpu::task::NeighborListParams& p) {
                self.set_categorical(p,
                                     cpu::md::NeighborList<>{p.cutoff, p.skin});
            },
            nb::arg("comp"), "LAMMPS-style set_neighbor.")
        .def(
            "set_thermo",
            [](MdHandle& self, int every) {
                self.set_categorical(cpu::task::ThermoParams{every},
                                     cpu::Thermo{every});
            },
            nb::arg("every"), "LAMMPS-style set_thermo(N) print cadence.")
        .def(
            "set_record",
            [](MdHandle& self, const cpu::task::ThermoParams& p) {
                // Thermo as a record/observer-style component via set_thermo.
                self.set_categorical(p, cpu::Thermo{p.freq});
            },
            nb::arg("comp"),
            "Register a recorder; Thermo is installed via set_thermo cadence.")
        .def(
            "on",
            [](MdHandle& self, md::Phase phase) {
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
             "every rank and writes the gathered result back.")
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
