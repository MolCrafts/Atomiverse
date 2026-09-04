/// @file python/src/bindings/bind_semiempirical.cpp
/// @brief AM1 semiempirical bindings: AM1, AM1Evaluator, AM1Result, the
///        `single_point` entry and `AM1Evaluator.forces`.
///
/// AM1 — *Austin Model 1* — is a **semiempirical** electronic-structure
/// method: it solves an approximate quantum-mechanical model of the valence
/// electrons in which most of the expensive electron-electron integrals are
/// dropped (the NDDO approximation, *neglect of diatomic differential
/// overlap*) and the survivors are replaced by numbers fitted once per
/// chemical element. That buys a cost between a classical force field and a
/// Gaussian-basis Hartree-Fock solve, at the price of transferability.
///
/// `AM1Evaluator` publishes energy [eV], Mulliken charges [e] and nuclear
/// forces [eV/Å], so it serves two workflows: `cpu::single_point` for a
/// one-shot solve, and the MD driver as a force source — semiempirical
/// Born-Oppenheimer dynamics, wired on the Python side through
/// `MD.set_evaluator` (`bind_md.cpp`, `set_evaluator_am1`).
///
/// Every result channel writes back into the borrowed molpy.Frame per the
/// pinned to_molrs round-trip contract: energy onto typed meta, charges onto
/// the atoms charge column, forces onto fx/fy/fz.

#include "force_array.h"
#include "molrs_frame_bridge.h"

#include <utility>
#include <vector>

#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/molrs_exchange.h"
#include "atomiverse/cpu/semiempirical/am1_evaluator.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_spin_constants.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"
#include "atomiverse/cpu/task/launcher_gate.h"
#include "atomiverse/cpu/task/task_spec.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>

#include "atomiverse/cpu/task/task_protocol.h"
#endif

using namespace ATV;

namespace {

/// @brief Run one AM1 solve with gradients and hand the forces to Python.
///
/// Goes through `cpu::single_point` — one `Evaluate` pass over the borrowed
/// evaluator, the same pass `regressions/semiempirical/am1_water_force` pins
/// against a re-converged finite difference — so the numbers returned here
/// are the C++ numbers and not a second force recipe: the `Evaluate` preset
/// zeroes forces/energy first, the evaluator unwraps the molecule across
/// periodic images, and AM1 publishes energy [eV], Mulliken charges [e] and
/// forces [eV/Å] onto the transient frame, which is then written back into
/// the borrowed molpy.Frame.
///
/// Local to this process: unlike `single_point`, no MPI launcher task
/// carries a force request, so an active launcher gate is irrelevant here
/// and the solve runs on the calling rank (the `bind_scf.cpp` convention).
///
/// @param evaluator Borrowed evaluator; its `last_result()` is populated.
/// @param frame_obj A `molpy.Frame`; energy [eV], charges [e] and fx/fy/fz
///                  [eV/Å] land on it.
/// @return Nuclear forces F = −∂E/∂R [eV/Å], shape (n_atoms, 3).
/// @throws std::runtime_error (Python `RuntimeError`) when the SCF does not
///         converge within `max_iter`, or when the frame carries an element
///         AM1 has no parameters for.
ForceArray am1_forces(cpu::semiempirical::AM1Evaluator& evaluator,
                      nb::object frame_obj) {
    auto fref = borrow_molrs_frame(frame_obj);
    cpu::Frame work = cpu::from_molrs(*fref);
    {
        nb::gil_scoped_release release;
        cpu::single_point(work, evaluator);
    }
    cpu::to_molrs(work, *fref);

    // `forces_ev_a` is the flat (fx0, fy0, fz0, …) buffer the evaluator reuses
    // on the next solve, so the returned array must own a copy.
    return to_force_array(evaluator.last_result().forces_ev_a);
}

/// @brief Run one SCC-DFTB2 solve over a borrowed molpy.Frame.
///
/// One `cpu::Evaluate` pass — the same pass every DFTB2 regression drives —
/// so the numbers returned here are the C++ numbers and not a second recipe.
/// Energy [eV], Mulliken charges [e] and forces [eV/Å] land back on the frame
/// through the pinned `to_molrs` round trip.
///
/// @param dftb  Borrowed method; its `last_*()` accessors are populated.
/// @param frame_obj A `molpy.Frame`.
/// @throws std::runtime_error (Python `RuntimeError`) when the SCC does not
///         converge within `max_scc`, when the frame carries an element
///         outside H/C/N/O, or on any of the boundary/k-mesh conflicts the
///         method rejects.
void dftb2_run(cpu::semiempirical::DFTB2& dftb, nb::object frame_obj) {
    auto fref = borrow_molrs_frame(frame_obj);
    cpu::Frame work = cpu::from_molrs(*fref);
    {
        nb::gil_scoped_release release;
        cpu::single_point(work, dftb);
    }
    cpu::to_molrs(work, *fref);
}

/// @brief Solve and also return the forces as an (n_atoms, 3) array [eV/Å].
ForceArray dftb2_forces(cpu::semiempirical::DFTB2& dftb, nb::object frame_obj) {
    auto fref = borrow_molrs_frame(frame_obj);
    cpu::Frame work = cpu::from_molrs(*fref);
    {
        nb::gil_scoped_release release;
        cpu::single_point(work, dftb);
    }
    const int n = work.nlocal();
    std::vector<double> flat(static_cast<std::size_t>(3 * n));
    for (int i = 0; i < n; ++i) {
        flat[static_cast<std::size_t>(3 * i + 0)] = work.span(kAtom_fx)[i];
        flat[static_cast<std::size_t>(3 * i + 1)] = work.span(kAtom_fy)[i];
        flat[static_cast<std::size_t>(3 * i + 2)] = work.span(kAtom_fz)[i];
    }
    cpu::to_molrs(work, *fref);
    return to_force_array(flat);
}

} // namespace

void bind_semiempirical(nb::module_& m) {
    using cpu::semiempirical::AM1;
    using cpu::semiempirical::AM1Evaluator;
    using cpu::semiempirical::AM1Result;
    using cpu::semiempirical::DFTB2;
    using cpu::semiempirical::DftbBoundary;
    using cpu::semiempirical::DftbSkSet;
    using cpu::semiempirical::basis_data::DftbSpinConstant;

    // ── AM1 concrete method (parameters assigned in the constructor) ──
    nb::class_<AM1>(m, "AM1",
                    "AM1 (Austin Model 1) semiempirical method — an NDDO "
                    "(neglect of diatomic differential overlap) valence "
                    "Hamiltonian with per-element fitted parameters. Its SCF "
                    "settings are constructor arguments and writable "
                    "members. This class carries the settings; the solve is "
                    "driven through AM1Evaluator (single_point / forces / "
                    "MD.set_evaluator), which owns one of these and hands it "
                    "back from method().")
        .def(nb::init<int, int, int, double, double>(), nb::arg("charge") = 0,
             nb::arg("multiplicity") = 1, nb::arg("max_iter") = 200,
             nb::arg("density_tol") = 1.0e-8, nb::arg("energy_tol") = 1.0e-10,
             "Construct with total charge [e], spin multiplicity, and the "
             "three SCF (self-consistent field) loop settings: iteration "
             "cap, density-change tolerance (dimensionless) and energy "
             "tolerance [eV].")
        .def_rw("charge", &AM1::charge, "Total molecular charge [e].")
        .def_rw("multiplicity", &AM1::multiplicity,
                "Spin multiplicity 2S + 1, where S is the total electron "
                "spin: 1 = singlet (all electrons paired), 2 = doublet, "
                "3 = triplet.")
        .def_rw("max_iter", &AM1::max_iter,
                "Iteration cap for the SCF loop; overrunning it raises "
                "RuntimeError rather than returning an unconverged result.")
        .def_rw("density_tol", &AM1::density_tol,
                "Convergence threshold on the largest absolute change of any "
                "density-matrix element between two successive SCF "
                "iterations (dimensionless).")
        .def_rw("energy_tol", &AM1::energy_tol,
                "Convergence threshold on |dE| between two successive SCF "
                "iterations [eV].");

    // ── AM1Result (= NDDOResult) ──
    nb::class_<AM1Result>(m, "AM1Result",
                          "Converged AM1 solve results (read-only).")
        .def_ro("converged", &AM1Result::converged,
                "True when the SCF met EITHER tolerance — density_tol or "
                "energy_tol, not both. A run that meets neither within "
                "max_iter raises RuntimeError instead of returning, so on a "
                "result you can actually read this is always True.")
        .def_ro("iterations", &AM1Result::iterations,
                "SCF iterations performed.")
        .def_ro("electronic_energy_hartree",
                &AM1Result::electronic_energy_hartree,
                "Total energy [Hartree] — electrons plus core-core "
                "repulsion, i.e. the SAME quantity as total_energy_ev in the "
                "atomic unit of energy (1 Hartree = 27.211386 eV). The field "
                "name says 'electronic' for historical reasons; it is not an "
                "electrons-only energy.")
        .def_ro("heat_of_formation_kcal_mol",
                &AM1Result::heat_of_formation_kcal_mol,
                "Heat of formation [kcal/mol]: the energy released or "
                "consumed in forming this molecule from its elements in "
                "their standard states — the quantity AM1 was parameterized "
                "against, built here from the total energy by the MOPAC "
                "recipe (subtract the tabulated isolated-atom energies, add "
                "the experimental atomic heats of formation).")
        .def_ro("total_energy_ev", &AM1Result::total_energy_ev,
                "Total energy [eV] — the number AM1 writes to "
                "frame.meta['energy'].")
        .def_ro("mulliken_charges", &AM1Result::mulliken_charges,
                "Per-atom partial charges [e] by Mulliken population "
                "analysis — each electron's population is split among the "
                "atomic orbitals it occupies and summed per atom, then "
                "subtracted from the atom's valence-electron count. Always "
                "double precision, unlike the frame's charge column.")
        .def_ro("forces_ev_a", &AM1Result::forces_ev_a,
                "Nuclear forces [eV/A], flat (fx0, fy0, fz0, fx1, ...) over "
                "3 * n_atoms. Empty until a force evaluation ran on THIS "
                "object: AM1Evaluator.forces(frame), or single_point(frame, "
                "evaluator) — the Evaluate pass requests gradients, so it "
                "fills this too. An MD run does NOT fill it: MD.set_evaluator "
                "installs the driver's own copy of the evaluator, and that "
                "copy's per-step forces go to the frame's fx/fy/fz.")
        .def_ro("reference", &AM1Result::reference,
                "Literature reference for the parameterization.");

    // ── AM1Evaluator ──
    nb::class_<AM1Evaluator>(
        m, "AM1Evaluator",
        "AM1 Evaluator: energy [eV], Mulliken charges [e] and nuclear forces "
        "[eV/A]. Plugs into single_point(frame, evaluator) for one solve, "
        "into forces(frame) for the gradient, and into MD.set_evaluator as a "
        "force source — Born-Oppenheimer dynamics, meaning the electrons are "
        "re-converged at every geometry the nuclei reach and the nuclei move "
        "on the resulting energy surface.")
        .def(nb::init<>(),
             "Construct with the default AM1 settings: neutral (charge 0), "
             "singlet, max_iter 200, density_tol 1e-8, energy_tol 1e-10 eV.")
        .def(nb::init<int, int, int, double, double>(), nb::arg("charge") = 0,
             nb::arg("multiplicity") = 1, nb::arg("max_iter") = 200,
             nb::arg("density_tol") = 1.0e-8, nb::arg("energy_tol") = 1.0e-10,
             "Construct the owned AM1 method from the same five settings the "
             "AM1 constructor takes.")
        .def(nb::init<AM1>(), nb::arg("method"),
             "Take ownership of an already-configured AM1 (its settings are "
             "copied in; the object you pass is not aliased).")
        .def("last_result", &AM1Evaluator::last_result,
             nb::rv_policy::reference_internal,
             "AM1Result of the most recent solve driven through THIS "
             "evaluator (forces() or single_point()). Borrowed, not copied: "
             "the next solve overwrites it.")
        .def("method", nb::overload_cast<>(&AM1Evaluator::method),
             nb::rv_policy::reference_internal,
             "The owned AM1 method, borrowed — assigning to its members "
             "(charge, max_iter, ...) reconfigures the next solve, since "
             "every single_point / forces call rebuilds the solver from "
             "them.")
        .def("forces", &am1_forces, nb::arg("frame"),
             "Solve the AM1 SCF at the frame's geometry and return the "
             "nuclear forces F = -dE/dR as an (n_atoms, 3) array [eV/A].\n\n"
             "The gradient is a FIXED-DENSITY central finite difference: the "
             "SCF is converged once, then its electron density P is held "
             "fixed while the energy is re-evaluated at geometries displaced "
             "by +/- h and differenced, with h = 1e-4 A per Cartesian "
             "component. Holding P is not the approximation. At convergence "
             "the energy is stationary against any change of the density "
             "that preserves the electron count, so the density-response "
             "term of dE/dR drops out; and NDDO treats its orbital basis as "
             "orthonormal (no overlap matrix, hence no basis-derivative "
             "'Pulay' term), so nothing else is left over either. The "
             "approximation is the difference quotient itself: it reproduces "
             "the exact NDDO gradient to O(h^2) plus floating-point "
             "round-off. Closed-form NDDO derivatives are a separate engine "
             "queue.\n\n"
             "Only 3 * (n_atoms - 1) differences are taken. The force on the "
             "last atom is filled in as minus the sum of all the others, "
             "which is exact for an isolated molecule because a rigid "
             "translation of the whole system cannot change its energy. So "
             "forces.sum(axis=0) is zero BY CONSTRUCTION and is not an "
             "independent check of the gradient.\n\n"
             "The same values are written back to the frame's fx/fy/fz "
             "columns, alongside the energy [eV] on frame.meta['energy'] and "
             "the Mulliken charges [e] on the atoms charge column — one "
             "solve, every channel published. Read "
             "last_result().forces_ev_a for the same numbers in the flat "
             "(fx0, fy0, fz0, fx1, ...) layout. The global interpreter lock "
             "(Python's one-thread-at-a-time execution lock) is released for "
             "the duration of the solve, so other Python threads keep "
             "running. Raises RuntimeError if the SCF does not converge "
             "within max_iter, or for an element AM1 has no parameters "
             "for.");

    // ── single_point: one Evaluate pass over a borrowed molpy.Frame ──
    m.def(
        "single_point",
        [](nb::object frame_obj, AM1Evaluator& evaluator) {
            const cpu::task::LauncherGate gate;
            auto fref = borrow_molrs_frame(frame_obj);
            cpu::Frame work = cpu::from_molrs(*fref);
            if (!gate.active()) {
                {
                    nb::gil_scoped_release release;
                    cpu::single_point(work, evaluator);
                }
                // Unlike the PW path, AM1 writes a per-atom output (the
                // Mulliken charge column, plus fx/fy/fz), so the work frame
                // must return to the SAME molpy.Frame via the pinned
                // to_molrs round-trip contract.
                cpu::to_molrs(work, *fref);
                return;
            }
#ifdef ATV_HAS_MPI
            // Launcher-hosted rank 0: dispatch a SinglePoint task. Results
            // (energy [eV] + charges [e]) come back through the Frame;
            // NOTE the borrowed evaluator's last_result() is NOT populated
            // on this path — the solve ran inside the per-rank
            // materializers.
            cpu::task::AM1EvaluatorParams p{};
            const AM1& method = evaluator.method();
            p.charge = method.charge;
            p.multiplicity = method.multiplicity;
            p.max_iter = method.max_iter;
            p.density_tol = method.density_tol;
            p.energy_tol = method.energy_tol;
            cpu::task::TaskSpec spec{cpu::task::SinglePointTask{p}};
            cpu::Frame result(0);
            {
                nb::gil_scoped_release release;
                cpu::task::TaskDispatcher dispatcher(
                    MPI_COMM_WORLD, cpu::task::kProtocolVersion,
                    /*handshake=*/false); // main's dispatcher handshook
                result = dispatcher.dispatch(spec, std::move(work)).frame;
            }
            cpu::to_molrs(result, *fref);
#endif
        },
        nb::arg("frame"), nb::arg("evaluator"),
        "Run one AM1 single-point — one SCF solve at the geometry the frame "
        "already holds, with nothing moved afterwards. Results land back on "
        "the same molpy.Frame: total energy [eV] on frame.meta['energy'], "
        "Mulliken charges [e] on the atoms charge column, and nuclear forces "
        "[eV/A] on fx/fy/fz (the Evaluate pass requests gradients, so this "
        "entry computes them too — AM1Evaluator.forces(frame) is the one "
        "that also RETURNS them as an array). The global interpreter lock is "
        "released during compute; read evaluator.last_result() for the "
        "Hartree / kcal-mol detail (serial path only — under the MPI "
        "launcher the solve runs on the worker ranks and only the Frame "
        "comes back).");

    // ── SCC-DFTB2 ─────────────────────────────────────────────────────
    nb::enum_<DftbBoundary>(
        m, "DftbBoundary",
        "Cluster or lattice Hamiltonian — the same fork DFTB+ makes through "
        "its geometry block (a `C` geometry is an isolated molecule, an `S` "
        "geometry is a supercell). These are DIFFERENT Hamiltonians, not an "
        "approximation of one another: under `lattice` the second-order term "
        "carries the Ewald-summed 1/R interaction of every periodic image, "
        "which does not vanish for a neutral cell because the charge "
        "distribution still has a dipole. A large empty box is therefore NOT "
        "a way to get an isolated molecule.")
        .value("cluster", DftbBoundary::cluster,
               "Isolated molecule: no lattice images anywhere.")
        .value("lattice", DftbBoundary::lattice,
               "3-D lattice: image-summed H0/S/repulsive and an Ewald-summed "
               "charge kernel.");

    nb::enum_<DftbSkSet>(m, "DftbSkSet",
                         "Which static Slater-Koster parameter set DFTB2 / "
                         "DFTB3 uses. Both mio and 3ob are transcribed into "
                         "the library; there is no runtime .skf reader. "
                         "Hubbard U^d and collinear W are NOT part of the "
                         "SK set — they are explicit members "
                         "(DFTB2.spin_w_ha, DFTB3.hubbard_du_ha).")
        .value("unset", DftbSkSet::Unset,
               "Sentinel: the method has not been told which SK set to use. "
               "DFTB3 defaults here and throws from launch until assigned. "
               "DFTB2 defaults to mio and throws if the caller assigns this.")
        .value("mio", DftbSkSet::MioSketch,
               "DFTB.org mio-1-1 H/C/N/O — onsite energies, Hubbard U, the "
               "two-centre grids and the spline repulsive.")
        .value("threeob", DftbSkSet::ThreeOb,
               "DFTB.org 3ob-3-1 organic CHNO. Molecular clusters only: it "
               "has no lattice sum, so it rejects boundary=lattice and any "
               "k-mesh beyond Gamma.");

    nb::class_<DftbSpinConstant>(
        m, "DftbSpinConstant",
        "One element's shell-resolved collinear spin constants W^A_ll' "
        "[Hartree], matching DFTB+ SpinConstants input order "
        "(w_ss, w_sp, w_ps, w_pp). Assign a table of these onto "
        "DFTB2.spin_w_ha; the published PBE set is "
        "kDftbSpinConstantsPbeHa.")
        .def(nb::init<>())
        .def_rw("z", &DftbSpinConstant::z, "Atomic number Z (dimensionless).")
        .def_rw("n_shell", &DftbSpinConstant::n_shell,
                "Valence shells: 1 (s only) or 2 (s+p).")
        .def_prop_rw(
            "w_ha",
            [](const DftbSpinConstant& s) {
                return std::array<std::array<double, 2>, 2>{
                    {std::array<double, 2>{s.w_ha[0][0], s.w_ha[0][1]},
                     std::array<double, 2>{s.w_ha[1][0], s.w_ha[1][1]}}};
            },
            [](DftbSpinConstant& s,
               const std::array<std::array<double, 2>, 2>& w) {
                s.w_ha[0][0] = w[0][0];
                s.w_ha[0][1] = w[0][1];
                s.w_ha[1][0] = w[1][0];
                s.w_ha[1][1] = w[1][1];
            },
            "Shell-resolved W_ll' [Ha]; row-major [[ss, sp], [ps, pp]].");

    m.attr("kDftbSpinConstantsPbeHa") =
        nb::cast(cpu::semiempirical::basis_data::kDftbSpinConstantsPbeHa);
    m.attr("kDftbSpinConstantsLdaHa") =
        nb::cast(cpu::semiempirical::basis_data::kDftbSpinConstantsLdaHa);

    nb::class_<DFTB2>(
        m, "DFTB2",
        "SCC-DFTB2 (self-consistent-charge density-functional tight binding) "
        "for H/C/N/O — Elstner et al., Phys. Rev. B 58, 7260 (1998). A "
        "tight-binding valence Hamiltonian built from tabulated two-centre "
        "integrals, with the atomic charges relaxed self-consistently. Costs "
        "roughly a thousandth of a Gaussian-basis DFT solve and reproduces "
        "DFTB+ on the same parameter set.\n\n"
        "Settings are plain writable members, not a config object. Drives "
        "single_point(frame, method) for one solve and forces(frame) for the "
        "gradient; the same object is a force source for MD and geometry "
        "optimisation.")
        .def(nb::init<>(),
             "Construct with the defaults: mio-1-1, boundary=lattice, "
             "neutral closed shell, Gamma point, forces on, stress off, "
             "scc_tol 1e-8.")
        .def_rw("boundary", &DFTB2::boundary,
                "Cluster or lattice Hamiltonian (see DftbBoundary). Default "
                "lattice — a declared cell is a declared lattice, so an "
                "isolated molecule must say so.")
        .def_rw("sk_set", &DFTB2::sk_set, "Slater-Koster parameter set.")
        .def_rw("charge", &DFTB2::charge,
                "Net molecular charge [e], subtracted from the valence "
                "electron count.")
        .def_rw("multiplicity", &DFTB2::multiplicity,
                "Spin multiplicity 2S+1. 1 = closed shell (the default); "
                "anything else runs the collinear spin-polarized path and "
                "requires spin_w_ha (typically kDftbSpinConstantsPbeHa).")
        .def_rw("spin_w_ha", &DFTB2::spin_w_ha,
                "Per-element collinear spin constants W^A_ll' [Hartree], as "
                "in DFTB+ SpinConstants — NOT eV. Empty = unset. Required "
                "when multiplicity != 1; ignored on a closed shell. Assign "
                "kDftbSpinConstantsPbeHa (mio / 3ob are PBE-parameterized). "
                "launch() collapses each element's matrix to the HOAO "
                "diagonal (H W_ss, C/N/O W_pp) and converts Ha→eV, which is "
                "DFTB+ ShellResolvedSpin = No.")
        .def_rw("shell_resolved_spin", &DFTB2::shell_resolved_spin,
                "When true, request DFTB+ ShellResolvedSpin = Yes "
                "(unimplemented). The default false path is the HOAO "
                "collapse. true raises RuntimeError naming "
                "dftb-shell-resolved-spin when the spin channel is taken.")
        .def_rw("scc_tol", &DFTB2::scc_tol,
                "Convergence threshold on max|dq_out - dq_in| [e]. DFTB+'s "
                "default SCCTolerance is 1e-5; this defaults tighter, at "
                "1e-8. Exhausting max_scc raises RuntimeError rather than "
                "returning an unconverged result.")
        .def_rw("max_scc", &DFTB2::max_scc, "Cap on SCC iterations.")
        .def_rw("mix_beta", &DFTB2::mix_beta,
                "Anderson/Pulay charge-mixing relaxation in (0, 1]. The 0.3 "
                "default converges every in-tree case; larger is faster but "
                "stops converging for hydrogen-bonded systems around 0.7.")
        .def_rw("with_forces", &DFTB2::with_forces,
                "Compute analytic nuclear forces F = -dE/dR [eV/A].")
        .def_rw("with_stress", &DFTB2::with_stress,
                "Compute the analytic cell stress [eV/A^3] and publish it to "
                "frame.meta['stress']. Needed for constant-pressure MD.")
        .def_rw("k_mesh", &DFTB2::k_mesh,
                "Monkhorst-Pack mesh subdivisions per reciprocal axis, "
                "(1, 1, 1) = Gamma only. Requires boundary=lattice.")
        .def_rw("k_time_reversal", &DFTB2::k_time_reversal,
                "Fold +/-k pairs (double weight). Exact for a "
                "real-Hamiltonian closed-shell density; energy and forces do "
                "not depend on it.")
        .def_rw("d3", &DFTB2::d3,
                "Add Grimme DFT-D3(BJ) dispersion with the DFTB3-D3(BJ) "
                "damping set.")
        .def_rw("d3_periodic", &DFTB2::d3_periodic,
                "Lattice-sum the D3 correction over an orthorhombic box.")
        .def_rw("ewald_cutoff", &DFTB2::ewald_cutoff,
                "Real-space cutoff for the periodic charge kernel [A]; also "
                "fixes the Ewald splitting parameter. The energy does not "
                "depend on it, only the real/reciprocal work split.")
        .def_rw("ewald_accuracy", &DFTB2::ewald_accuracy,
                "Ewald relative accuracy in (0, 1); sets the |k| cutoff.")
        // Ref-qualified overloads (lvalue / rvalue): nb::overload_cast
        // cannot pick between them, so name the lvalue one directly.
        .def(
            "predictor",
            [](DFTB2& self, int order) -> DFTB2& {
                return self.predictor(order);
            },
            nb::arg("order") = 3, nb::rv_policy::reference_internal,
            "Extrapolate the starting charges from previous solves (Kolafa "
            "ASPC). Along a trajectory the converged Mulliken charges are a "
            "smooth function of time, so earlier steps predict the next "
            "one. This changes WHERE THE SCC STARTS, never where it "
            "converges: the residual gate is untouched, so the energy is "
            "the same and only the iteration count moves — about 1.5x fewer "
            "under thermal MD. Off by default; a single point has no "
            "history to extrapolate from. Returns self, so it chains.")
        .def("reset_predictor", &DFTB2::reset_predictor,
             "Drop the extrapolation history — after a geometry "
             "discontinuity or a restart. History also resets on its own "
             "when the atom count changes.")
        .def("run", &dftb2_run, nb::arg("frame"),
             "Solve at the frame's geometry and publish every channel back "
             "onto it: total energy [eV] on frame.meta['energy'], Mulliken "
             "charges [e] on the atoms charge column, forces [eV/A] on "
             "fx/fy/fz when with_forces, and the cell stress on "
             "frame.meta['stress'] when with_stress. The global interpreter "
             "lock is released during compute.")
        .def("forces", &dftb2_forces, nb::arg("frame"),
             "Solve, publish everything run() publishes, and additionally "
             "RETURN the analytic forces as an (n_atoms, 3) array [eV/A]. "
             "These are closed-form derivatives (Hellmann-Feynman + Pulay + "
             "the charge-kernel and repulsive terms), not finite "
             "differences.")
        .def("last_energy", &DFTB2::last_energy,
             "Total energy of the most recent solve [eV].")
        .def("last_e_band", &DFTB2::last_e_band,
             "Band-structure energy sum(n_i eps_i) [eV] — the DFTB+ "
             "'Band energy' line.")
        .def("last_e_scc", &DFTB2::last_e_scc,
             "Second-order charge-fluctuation energy [eV] — the DFTB+ "
             "'Energy SCC' line.")
        .def("last_e_rep", &DFTB2::last_e_rep,
             "Pair repulsive energy [eV]. last_energy() - last_e_rep() is "
             "DFTB+'s 'Total Electronic energy'.")
        .def("last_e_disp", &DFTB2::last_e_disp,
             "D3(BJ) dispersion energy [eV]; 0 when d3 is off.")
        .def("last_scc_iterations", &DFTB2::last_scc_iterations,
             "SCC iterations the last solve took.")
        .def("last_scc_residual", &DFTB2::last_scc_residual,
             "SCC residual max|dq_out - dq_in| at exit [e].")
        .def("last_stress", &DFTB2::last_stress,
             "Intensive cell stress of the last solve, Voigt "
             "(xx, yy, zz, xy, xz, yz) [eV/A^3]. Zero unless with_stress.")
        .def("last_mulliken_dq", &DFTB2::last_mulliken_dq,
             "Per-atom Mulliken charge fluctuation dq [e] after the last "
             "solve. NOTE the sign convention: this is population minus "
             "reference, so it is the NEGATIVE of the 'atomic gross charge' "
             "DFTB+ prints.")
        .def("last_mulliken_spin", &DFTB2::last_mulliken_spin,
             "Per-atom Mulliken spin population m = N_alpha - N_beta [e].");
}
