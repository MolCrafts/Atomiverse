/// @file python/src/bindings/bind_semiempirical.cpp
/// @brief AM1 semiempirical bindings: AM1, AM1Result, the `single_point`
///        entry, `AM1.forces`, and the AM1-BCC / ABCG2 charge model
///        `AM1BCCCharges`.
///
/// AM1 — *Austin Model 1* — is a **semiempirical** electronic-structure
/// method: it solves an approximate quantum-mechanical model of the valence
/// electrons in which most of the expensive electron-electron integrals are
/// dropped (the NDDO approximation, *neglect of diatomic differential
/// overlap*) and the survivors are replaced by numbers fitted once per
/// chemical element. That buys a cost between a classical force field and a
/// Gaussian-basis Hartree-Fock solve, at the price of transferability.
///
/// `AM1` publishes energy [eV], Mulliken charges [e] and nuclear forces
/// [eV/Å], so it serves two workflows: `cpu::single_point` for a one-shot
/// solve, and the MD driver as a nested force source — semiempirical
/// Born-Oppenheimer dynamics, wired on the Python side through
/// `MD.set_evaluator` (`bind_md.cpp`, `set_evaluator_am1`).
///
/// Every result channel writes back into the borrowed molpy.Frame per the
/// pinned to_molpy round-trip contract: energy onto typed meta, charges onto
/// the atoms charge column, forces onto fx/fy/fz.

#include "force_array.h"
#include "molpy_frame_bridge.h"
#include "nddo_bindings.h"
#include "result_array.h"

#include <utility>
#include <vector>

#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "atomiverse/cpu/semiempirical/am1_bcc_charges.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_spin_constants.h"
#include "atomiverse/cpu/semiempirical/dftb0.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"
#include "atomiverse/cpu/semiempirical/mndo_solver.h"
#include "atomiverse/cpu/semiempirical/pm3_solver.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"
#include "atomiverse/cpu/task/launcher_gate.h"
#include "atomiverse/cpu/task/task_spec.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>

#include "atomiverse/cpu/task/task_protocol.h"
#endif

using namespace ATV;

namespace {

/// @brief Evaluate energy and charges over a borrowed molpy.Frame.
///
/// Calls the method's Frame entry and publishes through the existing molrs
/// bridge. Existing force columns are preserved by run().
///
/// @param dftb  Borrowed method; its `last_*()` accessors are populated.
/// @param frame_obj A `molpy.Frame`.
/// @throws std::runtime_error (Python `RuntimeError`) when the SCC does not
///         converge within `max_scc`, when the frame carries an element
///         outside H/C/N/O, or on any of the boundary/k-mesh conflicts the
///         method rejects.
template <class Method>
void dftb_run(Method& dftb, nb::object frame_obj) {
    if (cpu::task::LauncherGate{}.active())
        throw std::runtime_error("DFTB has no MPI single-point task payload; "
                                 "run without the launcher");
    auto fref = borrow_molpy_frame(frame_obj);
    cpu::Frame work = cpu::from_molpy(*fref);
    {
        nb::gil_scoped_release release;
        dftb.run(work);
    }
    cpu::to_molpy(work, *fref);
}

/// @brief Solve and also return the forces as an (n_atoms, 3) array [eV/Å].
template <class Method>
ForceArray dftb_forces(Method& dftb, nb::object frame_obj) {
    if (cpu::task::LauncherGate{}.active())
        throw std::runtime_error("DFTB has no MPI single-point task payload; "
                                 "run without the launcher");
    auto fref = borrow_molpy_frame(frame_obj);
    cpu::Frame work = cpu::from_molpy(*fref);
    {
        nb::gil_scoped_release release;
        dftb.forces(work);
    }
    const int n = work.nlocal();
    std::vector<double> flat(static_cast<std::size_t>(3 * n));
    for (int i = 0; i < n; ++i) {
        flat[static_cast<std::size_t>(3 * i + 0)] = work.span(kAtom_fx)[i];
        flat[static_cast<std::size_t>(3 * i + 1)] = work.span(kAtom_fy)[i];
        flat[static_cast<std::size_t>(3 * i + 2)] = work.span(kAtom_fz)[i];
    }
    cpu::to_molpy(work, *fref);
    return to_force_array(flat);
}

} // namespace

void bind_semiempirical(nb::module_& m) {
    using cpu::semiempirical::AM1;
    using cpu::semiempirical::AM1BCCCharges;
    using cpu::semiempirical::AM1Result;
    using cpu::semiempirical::DFTB0;
    using cpu::semiempirical::DFTB2;
    using cpu::semiempirical::DftbBoundary;
    using cpu::semiempirical::DftbSkSet;
    using cpu::semiempirical::basis_data::DftbSpinConstant;

    bind_nddo_method<AM1>(m, "AM1", "cpu", true);
    bind_nddo_method<cpu::semiempirical::RM1>(m, "RM1", "cpu", true);
    bind_nddo_method<cpu::semiempirical::MNDO>(m, "MNDO", "cpu");
    bind_nddo_method<cpu::semiempirical::PM3>(m, "PM3", "cpu");

    // ── AM1Result (= NDDOResult) ──
    nb::class_<AM1Result>(m, "AM1Result",
                          "Converged AM1 solve results (read-only).")
        .def_ro("converged", &AM1Result::converged,
                "True only when both density_tol and energy_tol are met. "
                "Failed calculations clear convergence and do not publish "
                "Frame results.")
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
                "object: AM1.forces(frame), or single_point(frame, method) — "
                "the Evaluate pass requests gradients, so it fills this too. "
                "Energy-only run() clears this channel. Native borrowed "
                "driver mounts retain results on the original method object.")
        .def_ro("reference", &AM1Result::reference,
                "Literature reference for the parameterization.");

    // Both method aliases refer to the one canonical NDDOResult C++ type.
    m.attr("RM1Result") = m.attr("AM1Result");
    m.attr("MNDOResult") = m.attr("AM1Result");
    m.attr("PM3Result") = m.attr("AM1Result");

    // Molecular charge models accept the chemical graph, not a compute Frame.
    nb::class_<AM1BCCCharges>(
        m, "AM1BCCCharges",
        "AM1-BCC or ABCG2 charges at the molecule's current geometry. "
        "assign(molecule) returns an owned float64 array in graph atom order "
        "without modifying the input molpy.Atomistic. Geometry optimization "
        "is a separate operation; correction is independent of GAFF "
        "generation.")
        .def(nb::init<AM1, std::string_view>(), nb::arg("method") = AM1{},
             nb::arg("correction") = "bcc")
        .def(
            "assign",
            [](AM1BCCCharges& self, nb::object molecule) {
                auto atomistic =
                    nb::module_::import_("molpy").attr("Atomistic");
                if (!nb::isinstance(molecule, atomistic))
                    throw nb::type_error("AM1BCCCharges.assign expects a "
                                         "molpy.Atomistic molecular graph");
                auto frame = molecule.attr("to_frame")();
                // A coordinate an atom lacks is a null cell of its column;
                // the native frame would read it as 0, so refuse it here.
                nb::object atoms = frame.attr("__getitem__")("atoms");
                for (const char* axis : {"x", "y", "z"}) {
                    // An absent column is refused natively, by name.
                    if (PyObject_IsTrue(atoms.attr("keys")()
                                            .attr("__contains__")(axis)
                                            .ptr())
                        != 1)
                        continue;
                    nb::object mask = atoms.attr("validity")(axis);
                    if (mask.is_none())
                        continue;
                    for (nb::handle held : mask)
                        if (PyObject_IsTrue(held.ptr()) != 1)
                            throw nb::value_error(
                                (std::string("AM1BCCCharges: coordinate atoms.")
                                 + axis + " is missing on some atoms")
                                    .c_str());
                }
                auto transport = borrow_molpy_frame(frame);
                std::vector<double> charges;
                {
                    nb::gil_scoped_release release;
                    charges = self.assign(*transport);
                }
                return to_vector_array(charges);
            },
            nb::arg("molecule"),
            "Return partial charges [e] in molecule atom order without "
            "mutation. "
            "Requires a complete molpy.Atomistic with x/y/z coordinates [A].")
        .def("last_am1_result", &AM1BCCCharges::last_am1_result,
             nb::rv_policy::reference_internal,
             "AM1 result before equivalencing and correction; borrowed.")
        .def_prop_ro("correction", &AM1BCCCharges::correction)
        .def("needs_equivalencing", &AM1BCCCharges::needs_equivalencing);

    // ── single_point: one Evaluate pass over a borrowed molpy.Frame ──
    m.def(
        "single_point",
        [](nb::object frame_obj, AM1& method) {
            const cpu::task::LauncherGate gate;
            auto fref = borrow_molpy_frame(frame_obj);
            cpu::Frame work = cpu::from_molpy(*fref);
            if (!gate.active()) {
                {
                    nb::gil_scoped_release release;
                    cpu::single_point(work, method);
                }
                // Unlike the PW path, AM1 writes a per-atom output (the
                // Mulliken charge column, plus fx/fy/fz), so the work frame
                // must return to the SAME molpy.Frame via the pinned
                // to_molpy round-trip contract.
                cpu::to_molpy(work, *fref);
                return;
            }
#ifdef ATV_HAS_MPI
            // Launcher-hosted rank 0: dispatch a SinglePoint task. Results
            // (energy [eV] + charges [e]) come back through the Frame;
            // NOTE the borrowed method's last_result() is NOT populated on
            // this path — the solve ran inside the per-rank materializers.
            cpu::task::AM1Params p{};
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
            cpu::to_molpy(result, *fref);
#endif
        },
        nb::arg("frame"), nb::arg("method"),
        "Run one AM1 single-point — one SCF solve at the geometry the frame "
        "already holds, with nothing moved afterwards. Results land back on "
        "the same molpy.Frame: total energy [eV] on frame.meta['energy'], "
        "Mulliken charges [e] on the atoms charge column, and nuclear forces "
        "[eV/A] on fx/fy/fz (the Evaluate pass requests gradients, so this "
        "entry computes them too — AM1.forces(frame) is the one that also "
        "RETURNS them as an array). The global interpreter lock is released "
        "during compute; read method.last_result() for the "
        "Hartree / kcal-mol detail (serial path only — under the MPI "
        "launcher the solve runs on the worker ranks and only the Frame "
        "comes back).");

    nb::class_<DFTB0, cpu::Component>(m, "DFTB0",
                                      "Non-SCC mio-1-1 tight binding for "
                                      "neutral closed-shell CHNO molecules.")
        .def(nb::init<>())
        .def("run", &dftb_run<DFTB0>, nb::arg("frame"),
             "Overwrite energy [eV] and charges [e]; preserve existing forces.")
        .def("forces", &dftb_forces<DFTB0>, nb::arg("frame"),
             "Overwrite energy, charges and forces; return forces [eV/A].")
        .def("last_energy", &DFTB0::last_energy)
        .def("last_charges", [](const DFTB0& self) {
            const auto q = self.last_charges();
            return std::vector<double>(q.begin(), q.end());
        });

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
               "Sentinel: no SK set chosen. A DFTB2 constructed with it "
               "throws from launch.")
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

    nb::class_<DFTB2, cpu::Component>(
        m, "DFTB2",
        "SCC-DFTB2 (self-consistent-charge density-functional tight binding) "
        "for H/C/N/O — Elstner et al., Phys. Rev. B 58, 7260 (1998). A "
        "tight-binding valence Hamiltonian built from tabulated two-centre "
        "integrals, with the atomic charges relaxed self-consistently. Costs "
        "roughly a thousandth of a Gaussian-basis DFT solve and reproduces "
        "DFTB+ on the same parameter set.\n\n"
        "The SK set, charge and multiplicity are constructor arguments; the "
        "other settings are plain writable members, not a config object. "
        "Drives single_point(frame, method) for one solve and forces(frame) "
        "for the gradient; the same object is a force source for MD and "
        "geometry optimisation.")
        .def(nb::init<DftbSkSet, int, int>(), nb::arg("sk_set"),
             nb::arg("charge") = 0, nb::arg("multiplicity") = 1,
             "Construct with the Slater-Koster parameter set (DftbSkSet.mio "
             "or DftbSkSet.threeob), net molecular charge [e] and spin "
             "multiplicity 2S+1. These three are fixed afterwards; the other "
             "settings are writable members defaulting to boundary=lattice, "
             "Gamma point, forces on, stress off, scc_tol 1e-8.")
        .def_rw("boundary", &DFTB2::boundary,
                "Cluster or lattice Hamiltonian (see DftbBoundary). Default "
                "lattice — a declared cell is a declared lattice, so an "
                "isolated molecule must say so.")
        .def_ro("sk_set", &DFTB2::sk_set,
                "Slater-Koster parameter set fixed at construction "
                "(read-only).")
        .def_ro("charge", &DFTB2::charge,
                "Net molecular charge [e], subtracted from the valence "
                "electron count; fixed at construction (read-only).")
        .def_ro("multiplicity", &DFTB2::multiplicity,
                "Spin multiplicity 2S+1, fixed at construction (read-only). "
                "1 = closed shell (the default); anything else runs the "
                "collinear spin-polarized path and requires spin_w_ha "
                "(typically kDftbSpinConstantsPbeHa).")
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
        .def("run", &dftb_run<DFTB2>, nb::arg("frame"),
             "Overwrite energy [eV] and charges [e]; preserve existing forces.")
        .def("forces", &dftb_forces<DFTB2>, nb::arg("frame"),
             "Overwrite energy, charges and forces; return an (n_atoms, 3) "
             "force array [eV/A].")
        .def(
            "stress",
            [](DFTB2& self, nb::object frame_obj) {
                if (cpu::task::LauncherGate{}.active())
                    throw std::runtime_error(
                        "DFTB has no MPI single-point task payload; "
                        "run without the launcher");
                auto fref = borrow_molpy_frame(frame_obj);
                auto work = cpu::from_molpy(*fref);
                std::array<double, 6> stress;
                {
                    nb::gil_scoped_release release;
                    stress = self.stress(work);
                }
                cpu::to_molpy(work, *fref);
                return stress;
            },
            nb::arg("frame"),
            "Overwrite energy, charges and extensive virial [eV]; return "
            "intensive Voigt stress [eV/A^3].")
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
             "(xx, yy, zz, xy, xz, yz) [eV/A^3]. Zero when stress was not "
             "requested.")
        .def("last_mulliken_dq", &DFTB2::last_mulliken_dq,
             "Per-atom Mulliken charge fluctuation dq [e] after the last "
             "solve. NOTE the sign convention: this is population minus "
             "reference, so it is the NEGATIVE of the 'atomic gross charge' "
             "DFTB+ prints.")
        .def("last_mulliken_spin", &DFTB2::last_mulliken_spin,
             "Per-atom Mulliken spin population m = N_alpha - N_beta [e].");
}
