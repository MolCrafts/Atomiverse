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

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <utility>
#include <vector>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/dynamics/fix_csvr.h"
#include "atomiverse/cpu/dynamics/fix_langevin.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
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
                           || std::is_same_v<P, cpu::task::NoseHooverNptParams>)
            driver.set_integrator(std::move(comp));
        else if constexpr (std::is_same_v<P, cpu::task::NoseHooverParams>
                           || std::is_same_v<P, cpu::task::FixLangevinParams>
                           || std::is_same_v<P, cpu::task::FixCsvrParams>)
            driver.set_thermostat(std::move(comp));
        else if constexpr (std::is_same_v<P, cpu::task::InitVelocityParams>)
            driver.set_observer(std::move(comp));
        else if constexpr (std::is_same_v<P, cpu::task::ThermoParams>)
            driver.set_thermo(
                static_cast<const cpu::task::ThermoParams&>(p).freq);
        else
            driver.set_observer(std::move(comp));
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
            auto frame = borrow_molrs_frame(frame_obj);
            nb::gil_scoped_release release;
            driver.run(*frame, n_steps);
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
        "MTK Nose-Hoover NPT integrator (LAMMPS `fix npt`); replaces the "
        "time integrator — install via set_integrator, not alongside "
        "VelocityVerlet.")
        .def(
            "__init__",
            [](cpu::task::NoseHooverNptParams* p, double T_target, double tau_T,
               double P_target, double tau_P, double dt) {
                new (p) cpu::task::NoseHooverNptParams{T_target, tau_T,
                                                       P_target, tau_P, dt};
            },
            nb::arg("T_target"), nb::arg("tau_T"), nb::arg("P_target"),
            nb::arg("tau_P"), nb::arg("dt"),
            "Construct with target temperature [K], tau_T [fs], target "
            "pressure [eV/A^3], tau_P [fs], and timestep dt [fs].");

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
                    p, cpu::md::NoseHooverNPT{p.T_target, p.tau_T, p.P_target,
                                              p.tau_P, p.dt});
            },
            nb::arg("comp"),
            "LAMMPS-style set_integrator (NoseHooverNPT — `fix npt` "
            "replaces the plain integrator).")
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
