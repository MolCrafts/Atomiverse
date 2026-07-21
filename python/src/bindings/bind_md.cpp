/// @file python/src/bindings/bind_md.cpp
/// @brief MD driver + component bindings, with the launcher dispatch seam.
///
/// Component classes are bound as the `cpu::task::*Params` mirrors (the
/// TaskSpec allow-list IS the binding surface — Python-visible names and
/// constructor signatures are unchanged). `MD` binds an `MdHandle` that
/// materializes the real C++ component at each `add_*` call AND appends the
/// codec entry, so `run()` can either execute the driver directly (serial
/// path, byte-identical to before) or — when `LauncherGate` reports the
/// process is rank 0 of the MPI launcher — pack the mirror into a
/// `TaskSpec` and hand it to the task protocol (one broadcast, all ranks
/// compute, gathered result written back into the SAME molrs.Frame).

#include "molrs_frame_bridge.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <utility>
#include <vector>

#include "atomiverse/cpu/frame/materialize.h"
#include "atomiverse/cpu/md/init.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/md/nose_hoover.h"
#include "atomiverse/cpu/md/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/recorder/thermo.h"
#include "atomiverse/cpu/task/launcher_gate.h"
#include "atomiverse/cpu/task/task_spec.h"
#include "atomiverse/facade/stage.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>

#include "atomiverse/cpu/task/task_protocol.h"
#endif

using namespace ATV;
using cpu::task::MDBinding;
using cpu::task::MDComponentEntry;

namespace {

/// The bound MD driver plus the TaskSpec mirror of its registrations.
/// Python sees this as class `MD` with the same method surface as before.
struct MdHandle {
    cpu::md::MD driver;
    std::vector<MDComponentEntry> mirror;

    /// Materialize the real component from its params mirror, register it
    /// on the driver (default-binding or explicit), and record the entry.
    template <typename Params, typename Component>
    void add(const Params& p, Component comp,
             std::optional<MDBinding> binding) {
        if (binding)
            driver.on(binding->phase).add(std::move(comp), binding->order);
        else
            driver.add(std::move(comp));
        mirror.push_back(MDComponentEntry{p, binding});
    }

    /// Run on a borrowed molrs.Frame: launcher-hosted rank 0 dispatches
    /// through the task protocol; every other shape runs the driver
    /// directly (the pre-seam path, byte-identical).
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
        cpu::Frame work = cpu::materialize(*fref);
        cpu::Frame result(0);
        {
            nb::gil_scoped_release release;
            // main's dispatcher already handshook with the workers; this
            // is the extension module's own copy of the library, so it
            // constructs its dispatcher handshake-free (task_protocol.h).
            cpu::task::TaskDispatcher dispatcher(MPI_COMM_WORLD,
                                                 cpu::task::kProtocolVersion,
                                                 /*handshake=*/false);
            result = dispatcher.dispatch(spec, std::move(work)).frame;
        }
        cpu::write_back(result, *fref);
#endif
    }
};

/// Fluent builder returned by MdHandle::on(phase).
struct MdPhaseBuilder {
    MdHandle* handle;
    ATV::md::Phase phase;
};

} // namespace

void bind_md(nb::module_& m) {
    // ── Phase enum ──
    nb::enum_<md::Phase>(m, "Phase", "MD phase sequence (LAMMPS-style).")
        .value("initial_integrate", md::Phase::initial_integrate)
        .value("pre_force", md::Phase::pre_force)
        .value("force", md::Phase::force)
        .value("post_force", md::Phase::post_force)
        .value("final_integrate", md::Phase::final_integrate)
        .value("end_of_step", md::Phase::end_of_step);

    // ── Component params (the TaskSpec allow-list IS the binding surface;
    //     Python-visible class names + ctor signatures are unchanged) ──

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

    // ── MD Driver handle ──
    nb::class_<MdHandle>(m, "MD", "Molecular dynamics driver (CPU backend).")
        .def(nb::init<>())
        .def(
            "add_velocity_verlet",
            [](MdHandle& self, const cpu::task::VelocityVerletParams& p) {
                self.add(p, cpu::md::VelocityVerlet{p.dt}, std::nullopt);
            },
            nb::arg("comp"))
        .def(
            "add_init_velocity",
            [](MdHandle& self, const cpu::task::InitVelocityParams& p) {
                self.add(p, cpu::md::InitVelocity{p.temperature, p.seed},
                         std::nullopt);
            },
            nb::arg("comp"))
        .def(
            "add_nose_hoover",
            [](MdHandle& self, const cpu::task::NoseHooverParams& p) {
                self.add(p, cpu::md::NoseHoover{p.T_start, p.tau_T, p.dt},
                         std::nullopt);
            },
            nb::arg("comp"))
        .def(
            "on",
            [](MdHandle& self, md::Phase phase) {
                return MdPhaseBuilder{&self, phase};
            },
            nb::arg("phase"))
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

    // ── PhaseBuilder ──
    nb::class_<MdPhaseBuilder>(m, "PhaseBuilder",
                               "Fluent builder for explicit phase binding.")
        .def(
            "add_pair_lj_cut",
            [](MdPhaseBuilder& self, const cpu::task::PairLJCutParams& p,
               int order) -> MdPhaseBuilder& {
                self.handle->add(p,
                                 cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff},
                                 MDBinding{self.phase, order});
                return self;
            },
            nb::arg("comp"), nb::arg("order") = 0, nb::rv_policy::reference)
        .def(
            "add_neighbor_list",
            [](MdPhaseBuilder& self, const cpu::task::NeighborListParams& p,
               int order) -> MdPhaseBuilder& {
                self.handle->add(p, cpu::md::NeighborList<>{p.cutoff, p.skin},
                                 MDBinding{self.phase, order});
                return self;
            },
            nb::arg("comp"), nb::arg("order") = 0, nb::rv_policy::reference)
        .def(
            "add_bond_harmonic",
            [](MdPhaseBuilder& self, const cpu::task::BondHarmonicParams& p,
               int order) -> MdPhaseBuilder& {
                self.handle->add(p, cpu::BondHarmonic{p.K, p.r0},
                                 MDBinding{self.phase, order});
                return self;
            },
            nb::arg("comp"), nb::arg("order") = 0, nb::rv_policy::reference)
        .def(
            "add_angle_harmonic",
            [](MdPhaseBuilder& self, const cpu::task::AngleHarmonicParams& p,
               int order) -> MdPhaseBuilder& {
                self.handle->add(p, cpu::AngleHarmonic{p.K, p.theta0},
                                 MDBinding{self.phase, order});
                return self;
            },
            nb::arg("comp"), nb::arg("order") = 0, nb::rv_policy::reference)
        .def(
            "add_dihedral_multi_harmonic",
            [](MdPhaseBuilder& self,
               const cpu::task::DihedralMultiHarmonicParams& p,
               int order) -> MdPhaseBuilder& {
                self.handle->add(p, cpu::DihedralMultiHarmonic{p.Cn},
                                 MDBinding{self.phase, order});
                return self;
            },
            nb::arg("comp"), nb::arg("order") = 0, nb::rv_policy::reference)
        .def(
            "add_thermo",
            [](MdPhaseBuilder& self, const cpu::task::ThermoParams& p,
               int order) -> MdPhaseBuilder& {
                self.handle->add(p, cpu::Thermo{p.freq},
                                 MDBinding{self.phase, order});
                return self;
            },
            nb::arg("comp"), nb::arg("order") = 0, nb::rv_policy::reference);
}
