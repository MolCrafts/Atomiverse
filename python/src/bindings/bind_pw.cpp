/// @file python/bind_pw.cpp
/// @brief Plane-wave DFT bindings backed directly by cpu::pw::PW.
///
/// **Plane-wave DFT** solves the Kohn-Sham equations of density-functional
/// theory — the one-electron eigenvalue problem whose orbitals reproduce the
/// interacting system's electron density — in a basis of plane waves
/// \(e^{i(k+G)r}\), the natural basis for a *periodic* system, since every
/// plane wave already has the periodicity of the cell. That is also the
/// binding's hardest boundary: without a cell there is no basis, so every
/// entry point here refuses a frame with no box instead of converging on
/// nothing and reporting NaN.
///
/// The surface is the C++ one, member for member: construct `PW`, set the
/// members (there is no `PWConfig` bag and no factory), `build(frame)` once,
/// then ask for what you need at a geometry —
///
/// | Call | Returns | Writes back to the frame |
/// | ---- | ------- | ------------------------ |
/// | `run(frame)` | `PWResult` (energies [Ry]/[Ha], smearing channels) | nothing |
/// | `forces(frame)` | `(n_atoms, 3)` [eV/Å] | fx/fy/fz [eV/Å] + energy [eV] |
/// | `stress(frame)` | Voigt 6-vector [eV/Å³] | nothing |
///
/// Units are the solver's on `PWResult` (Rydberg, the plane-wave convention:
/// 1 Ry = ½ Hartree, and 1 Ry = 13.605693 eV exactly as this engine spells it
/// — `cpu::pw::qe_units::rydberg_to_ev`, the Quantum ESPRESSO-rounded value)
/// and the engine's everywhere else (eV, Å) — forces in eV/Å, stress in
/// eV/Å³, `frame.meta['energy']` in eV.
///
/// Every entry point reads the frame's per-atom `atomic_number` column [Z]
/// (that is what selects the pseudopotential and the valence-electron count)
/// plus positions [Å] and the box; a frame missing that column is refused by
/// the molrs → Frame conversion before any solve starts.
///
/// `PWEvaluator` is the same method wearing its Component face: hand it to
/// `MD.set_evaluator` and the run becomes plane-wave Born-Oppenheimer
/// dynamics — the electrons are re-converged at every geometry the nuclei
/// reach. Budget **two** SCF solves per force-producing step, not one: the
/// evaluator first runs the energy SCF (`PW::run`) and then calls
/// `pw_forces_analytic`, which converges its own density before assembling
/// the forces (`cpu::pw::PWEvaluator::launch`).

#include "force_array.h"
#include "molrs_frame_bridge.h"
#include "result_array.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#ifdef ATV_HAS_PW
// The functional-name table is shared with the Gaussian SCF binding.
#include "xc_names.h"

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/molrs_exchange.h"
// `copy_pw_params` is the ONE definition of "the same method, no state" (see
// its comment: two file-local copies once silently dropped `gth_name`), so
// the binding reuses it rather than writing a third member list.
#include "atomiverse/cpu/planewave/internal/scf_loop.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/qe_units.h"

namespace {

using ATV::cpu::pw::PW;
using ATV::cpu::pw::PWEvaluator;
using ATV::cpu::pw::PWResult;
using ATV::cpu::pw::Smearing;

/// @brief Metallic-smearing kinds and the names Python spells them with.
///
/// *Smearing* replaces the sharp occupied/empty step at the Fermi level by a
/// smooth occupation function of a chosen width, which is what makes a metal
/// converge at a finite k-mesh. Names are the enumerators of
/// `cpu::pw::Smearing`, `none` being fixed (insulating) occupations.
constexpr std::pair<const char*, Smearing> kSmearingNames[] = {
    {"none", Smearing::none},
    {"fermi_dirac", Smearing::fermi_dirac},
    {"methfessel_paxton", Smearing::methfessel_paxton},
    {"gaussian", Smearing::gaussian},
};

/// @brief Smearing name → `Smearing`.
/// @param name One of the names in `kSmearingNames`.
/// @return The matching enumerator.
/// @throws std::invalid_argument (Python `ValueError`) on any other name,
///         listing the supported ones — no fallback occupation scheme.
Smearing smearing_from_name(const std::string& name) {
    std::string supported;
    for (const auto& [key, value] : kSmearingNames) {
        if (name == key)
            return value;
        if (!supported.empty())
            supported += ", ";
        supported += key;
    }
    throw std::invalid_argument("unknown smearing '" + name
                                + "'; supported: " + supported);
}

/// @brief `Smearing` → the name `smearing_from_name` accepts for it.
/// @param smearing A smearing enumerator.
/// @return Its published name.
/// @throws std::invalid_argument when the enumerator is missing from the
///         table — a binding left behind by a new smearing kind, which would
///         otherwise make the property round-trip lie.
std::string smearing_name(Smearing smearing) {
    for (const auto& [key, value] : kSmearingNames)
        if (smearing == value)
            return key;
    throw std::invalid_argument(
        "smearing enumerator " + std::to_string(static_cast<int>(smearing))
        + " has no name in the Python binding table (bind_pw.cpp)");
}

/// @brief Materialize a borrowed molrs.Frame and refuse a non-periodic one.
///
/// A plane-wave basis is *defined* by the cell: the basis functions are the
/// plane waves commensurate with it, and the cutoff `ecut_ry` selects them by
/// their reciprocal-lattice length. With no cell there is no reciprocal
/// lattice, so the run has nothing to expand in — today it would grind
/// through its iteration budget and report a NaN energy, which is exactly the
/// failure a user cannot diagnose.
///
/// The check is on the materialized Frame rather than on the molrs handle
/// because that is where "no box at all" and "a box with zero volume" become
/// the same, well-defined question.
///
/// @param src Borrowed molrs.Frame.
/// @return The materialized execution Frame (positions [Å], Z, cell).
/// @throws std::runtime_error (Python `RuntimeError`) when the frame carries
///         no cell, naming the attribute to set.
ATV::cpu::Frame periodic_frame(const molrs::FrameRef& src) {
    ATV::cpu::Frame work = ATV::cpu::from_molrs(src);
    if (!(work.box.volume() > 0.0))
        throw std::runtime_error(
            "plane-wave DFT requires a periodic cell, but this molrs.Frame "
            "has no box (or one of zero volume). Set frame.box before "
            "build()/run()/forces()/stress(), e.g. "
            "frame.box = molrs.Box.cube(10.0).");
    return work;
}

/// @brief Copy per-atom force triples into one flat row-major buffer.
///
/// `to_force_array` copies from a contiguous `(fx0, fy0, fz0, fx1, …)` span;
/// the solver hands back a vector of triples. Flattening explicitly rather
/// than reinterpreting the triples' storage keeps the conversion typed.
///
/// @param forces Per-atom forces [eV/Å].
/// @return 3·n_atoms values in row-major order [eV/Å].
std::vector<double>
flatten_forces(const std::vector<std::array<double, 3>>& forces) {
    std::vector<double> flat;
    flat.reserve(forces.size() * 3u);
    for (const auto& f : forces) {
        flat.push_back(f[0]);
        flat.push_back(f[1]);
        flat.push_back(f[2]);
    }
    return flat;
}

/// @brief Run one plane-wave SCF, take the analytic ionic forces, and write
///        both back to the borrowed molrs.Frame.
///
/// The energy published on the frame is the one belonging to *this* SCF —
/// `pw_forces_analytic` reports it through its out-parameter — so the number
/// and the forces describe the same converged density and no second solve is
/// paid for. `PW::last_result()` is deliberately untouched: the analytic
/// force path runs its own SCF and never claims the method's result slot.
///
/// @param solver    Configured plane-wave method (already built, or built
///                  lazily inside the force path).
/// @param frame_obj A `molrs.Frame`; fx/fy/fz [eV/Å] and energy [eV] land on
///                  it.
/// @return Forces [eV/Å], shape (n_atoms, 3).
/// @throws std::runtime_error (Python `RuntimeError`) on a frame with no
///         cell, and if the solver returns a force count that disagrees with
///         the atom count.
/// @throws std::invalid_argument (Python `ValueError`) from
///         `pw_forces_analytic`: odd valence-electron count on the
///         restricted fixed-occupation path, incompatible multiplicity
///         parity, empty frame.
ForceArray pw_forces(PW& solver, nb::object frame_obj) {
    auto fref = borrow_molrs_frame(frame_obj);
    ATV::cpu::Frame work = periodic_frame(*fref);
    std::vector<double> flat;
    {
        nb::gil_scoped_release release;
        double energy_ry = 0.0;
        const auto forces =
            ATV::cpu::pw::pw_forces_analytic(work, solver, &energy_ry);
        if (forces.size() != static_cast<std::size_t>(work.nlocal()))
            throw std::runtime_error(
                "cpu::pw::pw_forces_analytic returned "
                + std::to_string(forces.size()) + " force triples for "
                + std::to_string(work.nlocal()) + " atoms");
        auto fx = work.span(ATV::kAtom_fx);
        auto fy = work.span(ATV::kAtom_fy);
        auto fz = work.span(ATV::kAtom_fz);
        for (int i = 0; i < work.nlocal(); ++i) {
            const auto& f = forces[static_cast<std::size_t>(i)];
            fx[i] = static_cast<ATV::AccReal>(f[0]);
            fy[i] = static_cast<ATV::AccReal>(f[1]);
            fz[i] = static_cast<ATV::AccReal>(f[2]);
        }
        work.meta.get(ATV::kEnergy) =
            energy_ry * ATV::cpu::pw::qe_units::rydberg_to_ev;
        flat = flatten_forces(forces);
    }
    ATV::cpu::to_molrs(work, *fref);
    return to_force_array(std::span<const double>(flat.data(), flat.size()));
}

/// @brief Run one plane-wave SCF and return the analytic cell stress.
///
/// Stress is the derivative of the energy with respect to a *strain* of the
/// cell, so it reports on the cell, not on the atoms: nothing is written back
/// to the frame. Mechanical (CP2K) sign convention,
/// \(\sigma = -(1/\Omega)\,\partial E/\partial\varepsilon\) — positive under
/// compression.
///
/// @param solver    Configured plane-wave method.
/// @param frame_obj A `molrs.Frame`; read only.
/// @return Voigt stress (xx, yy, zz, xy, xz, yz) [eV/Å³].
/// @throws std::runtime_error (Python `RuntimeError`) on a frame with no
///         cell.
/// @throws std::invalid_argument (Python `ValueError`) from `pw_stress`:
///         odd valence-electron count on the restricted fixed-occupation
///         path, `spin_polarized` with metallic smearing (no analytic stress
///         path), incompatible multiplicity parity, empty frame.
VectorArray pw_cell_stress(PW& solver, nb::object frame_obj) {
    auto fref = borrow_molrs_frame(frame_obj);
    ATV::cpu::Frame work = periodic_frame(*fref);
    std::array<double, 6> stress{};
    {
        nb::gil_scoped_release release;
        stress = ATV::cpu::pw::pw_stress(work, solver);
    }
    return to_vector_array(
        std::span<const double>(stress.data(), stress.size()));
}

} // namespace
#endif // ATV_HAS_PW

/// @brief Register the plane-wave classes on the `_atomiverse` module.
///        Everything below is compiled only with `ATV_BUILD_PW=ON`, so on any
///        other build the module carries no `PW` attribute — which is what
///        `atomiverse.pw` probes for before populating its `__all__`.
/// @param m The nanobind module under construction.
void bind_pw(nb::module_& m) {

#ifdef ATV_HAS_PW
    nb::class_<PWResult>(m, "PWResult",
                         "Result of a plane-wave DFT calculation.")
        .def_prop_ro(
            "energy", [](const PWResult& r) { return r.energy; },
            "Total energy [Ry]. With metallic smearing this is the Mermin "
            "free energy E - TS, the quantity the forces and stress are the "
            "derivatives of.")
        .def_prop_ro(
            "energy_ha", [](const PWResult& r) { return r.energy_ha; },
            "The same total energy in Hartree (1 Ha = 2 Ry).")
        .def_prop_ro(
            "converged", [](const PWResult& r) { return r.converged; },
            "Whether SCF converged within max_iter.")
        .def_prop_ro(
            "iterations", [](const PWResult& r) { return r.iterations; },
            "Number of SCF iterations taken.")
        .def_prop_ro(
            "magnetization", [](const PWResult& r) { return r.magnetization; },
            "Collinear magnetization N_up - N_down [electrons]. Exactly 0 on "
            "a restricted (spin_polarized = False) run. With fixed "
            "occupations it is exactly multiplicity - 1; with metallic "
            "smearing it is self-consistent and need not be an integer.")
        .def_prop_ro(
            "e_fermi", [](const PWResult& r) { return r.e_fermi; },
            "Fermi level [Ry] — the chemical potential the smearing search "
            "converged to. 0 with fixed occupations, which have no such "
            "search.")
        .def_prop_ro(
            "n_elec_occ", [](const PWResult& r) { return r.n_elec_occ; },
            "Electron count sum(w f) recovered from the occupations of the "
            "last iteration — the sanity check on the Fermi-level search. 0 "
            "on the fixed-occupation path, which never runs one.")
        .def_prop_ro(
            "e_entropy", [](const PWResult& r) { return r.e_entropy; },
            "The -TS term [Ry] already included in `energy`. Nonzero only "
            "with metallic smearing: fractional occupations carry an "
            "electronic entropy, fixed ones do not.")
        .def_prop_ro(
            "n_kpts", [](const PWResult& r) { return r.n_kpts; },
            "Total Monkhorst-Pack k-points of this run (the k_mesh**3 set "
            "after symmetry-free generation); k_mesh = 1 is the Gamma point "
            "alone.")
        .def("__repr__", [](const PWResult& r) {
            std::string conv = r.converged ? "converged" : "NOT converged";
            return "PWResult(" + conv + ", E=" + std::to_string(r.energy)
                   + " Ry, iters=" + std::to_string(r.iterations) + ")";
        });

    nb::class_<PW>(
        m, "PW",
        "Plane-wave density-functional theory (DFT) for a periodic cell: the "
        "Kohn-Sham orbitals are expanded in plane waves e^{i(k+G)r}, the "
        "basis a periodic system asks for. Parameters are plain members "
        "(there is no PWConfig bag and no factory) — default-construct, "
        "assign, then build(frame) / run(frame) / forces(frame) / "
        "stress(frame).\n\n"
        "Every entry point reads the frame's per-atom 'atomic_number' column "
        "[Z], the positions [A] and the box, and refuses a frame with no box.")
        .def(nb::init<>())
        .def_rw("upf_path", &PW::upf_path,
                "Path to a NORM-CONSERVING pseudopotential file in Unified "
                "Pseudopotential Format (UPF) — a *pseudopotential* replaces "
                "the nucleus and the chemically inert core electrons by an "
                "effective potential, which is what makes a plane-wave "
                "expansion affordable. Ultrasoft and PAW files are refused on "
                "read: this path adds no augmentation charge, so their "
                "densities would be wrong by an unbounded amount. Mutually "
                "exclusive with gth_name — setting both, or neither, raises "
                "ValueError at build(). There is no default and no "
                "bare-Coulomb fallback: a point charge has no finite "
                "plane-wave cutoff.")
        .def_rw("gth_name", &PW::gth_name,
                "In-tree analytic Goedecker-Teter-Hutter (GTH) "
                "pseudopotential, e.g. 'GTH-PBE-q4' (Si, 4 valence "
                "electrons) or 'GTH-PBE-q1' (H) — an analytic form whose "
                "published parameters ship with the engine, so no external "
                "file is needed. In-tree coverage is H, O, Na, Cl, Ar, Si, "
                "Ni, Zn. ONE potential serves the whole cell, so a frame "
                "mixing atomic numbers raises ValueError; multi-species "
                "plane-wave runs are not implemented. Mutually exclusive with "
                "upf_path.")
        .def_rw("ecut_ry", &PW::ecut_ry,
                "Wavefunction cutoff [Ry], default 30. A plane wave "
                "e^{i(k+G)r} is kept only when its kinetic energy — |k+G|^2 "
                "in Rydberg atomic units — stays below this, so it is the one "
                "knob that sets basis size and cost; converge it for the "
                "system rather than trusting the default.")
        .def_rw("k_mesh", &PW::k_mesh,
                "Monkhorst-Pack mesh size per dimension (dimensionless), "
                "default 4. The Brillouin zone is sampled on a uniform "
                "k_mesh**3 grid of equal weight and no symmetry reduction; "
                "k_mesh = 1 is the Gamma point alone.")
        .def_rw("tol", &PW::tol,
                "SCF convergence threshold on the L2 norm of the "
                "density change between two iterations (dimensionless), "
                "default 1e-6. Self-consistent field (SCF) means the "
                "potential is rebuilt from the density it produced until the "
                "two agree.")
        .def_rw("max_iter", &PW::max_iter,
                "Cap on SCF iterations (dimensionless count), default 50. "
                "Overrunning it does NOT raise: the result comes back with "
                "converged = False, and forces/stress warn and use the last "
                "density — check the flag.")
        .def_rw("mix_beta", &PW::mix_beta,
                "Linear density-mixing fraction (Quantum ESPRESSO "
                "`mixing_beta`, dimensionless), default 0.7. Feeding the raw "
                "output density back would oscillate; a fraction of it is "
                "blended in instead, smaller being slower but steadier.")
        .def_rw("pulay_depth", &PW::pulay_depth,
                "Number of previous densities the Pulay/DIIS mixer keeps "
                "(dimensionless count), default 5. Pulay mixing extrapolates "
                "the next input density from that history instead of from "
                "the last iteration alone.")
        .def_prop_rw(
            "xc", [](const PW& solver) { return xc_name(solver.xc); },
            [](PW& solver, const std::string& name) {
                solver.xc = xc_from_name(name);
            },
            "Exchange-correlation functional by NAME (default 'PBE') — the "
            "recipe supplying the exchange and correlation energy the "
            "Kohn-Sham reference system leaves out. Same table as "
            "RKS(functional=...): 'LSDA', 'PBE', 'PBEsol', 'RevPBE', 'RPBE', "
            "'BLYP', 'OptB88VdW', 'B3LYP', 'PBE0', 'B2PLYP'. An unknown name "
            "raises ValueError listing them — there is no fallback "
            "functional.")
        .def_rw("spin_polarized", &PW::spin_polarized,
                "Collinear spin-polarized Kohn-Sham (nspin = 2): the "
                "spin-up and spin-down densities are solved for separately. "
                "Default False (restricted, every band doubly occupied), "
                "which requires an even valence-electron count — an "
                "open-shell cell such as a single H atom needs this True.")
        .def_rw("multiplicity", &PW::multiplicity,
                "Spin multiplicity M = 2S + 1, read only when "
                "spin_polarized. With fixed occupations it sets the "
                "magnetization exactly: N_up - N_down = M - 1 (1 = singlet, "
                "2 = doublet, ...). With metallic smearing it only seeds the "
                "initial spin split, which the SCF is then free to change.")
        .def_prop_rw(
            "smearing",
            [](const PW& solver) { return smearing_name(solver.smearing); },
            [](PW& solver, const std::string& name) {
                solver.smearing = smearing_from_name(name);
            },
            "Metallic smearing kind by NAME (default 'none'): 'none' keeps "
            "fixed insulating occupations, 'fermi_dirac', "
            "'methfessel_paxton' and 'gaussian' spread the occupation over a "
            "window of width `degauss` around the Fermi level, which is what "
            "lets a metal converge on a finite k-mesh. An unknown name "
            "raises ValueError listing the four.")
        .def_rw("degauss", &PW::degauss,
                "Smearing width [Ry] (Quantum ESPRESSO `degauss`), default "
                "0.02. Must be > 0 whenever smearing is not 'none'.")
        .def_rw("mp_order", &PW::mp_order,
                "Methfessel-Paxton expansion order N (default 1), read only "
                "by the 'methfessel_paxton' smearing.")
        .def_rw("extra_bands", &PW::extra_bands,
                "Empty bands beyond ceil(N_e/2) to solve for when smearing "
                "is on (default 4, per spin channel when collinear). "
                "Fractional occupations need states above the Fermi level to "
                "put electrons in.")
        .def(
            "build",
            [](PW& solver, nb::object frame_obj) {
                auto fref = borrow_molrs_frame(frame_obj);
                ATV::cpu::Frame work = periodic_frame(*fref);
                nb::gil_scoped_release release;
                solver.build(work);
            },
            nb::arg("frame"),
            "Validate the reference frame and perform one-time setup: it "
            "checks that exactly one pseudopotential source is set and that "
            "it resolves (the UPF file parses, or the GTH name exists for "
            "this frame's element). run()/forces()/stress() call it lazily, "
            "so build() is only about failing early.\n\n"
            "Raises RuntimeError when the frame carries no box — a "
            "plane-wave basis is defined by the cell — and ValueError when "
            "neither or both of upf_path / gth_name are set, or the named "
            "potential is unknown or unreadable.")
        .def(
            "run",
            [](PW& solver, nb::object frame_obj) {
                auto fref = borrow_molrs_frame(frame_obj);
                ATV::cpu::Frame work = periodic_frame(*fref);
                PWResult result;
                {
                    nb::gil_scoped_release release;
                    result = solver.run(work);
                }
                return result;
            },
            nb::arg("frame"),
            "Run SCF at the frame's current geometry and return the "
            "PWResult; energies stay on the result (Ry / Hartree) and the "
            "frame is not written to. The GIL is released during the solve, "
            "so other Python threads keep running.\n\n"
            "Failing to converge within max_iter is NOT an error: the result "
            "comes back with converged = False and the last density's "
            "energy, so check the flag.\n\n"
            "Raises RuntimeError on a frame with no box, and ValueError when "
            "a restricted run is asked for an odd valence-electron count "
            "(set spin_polarized, or a smearing), when multiplicity and the "
            "electron count have incompatible parity, when gth_name is used "
            "on a frame that mixes atomic numbers (one potential serves the "
            "whole cell), and when the pseudopotential source is missing or "
            "doubly set.")
        .def("forces", &pw_forces, nb::arg("frame"),
             "Run one SCF and return the analytic ionic forces F = -dE/dR as "
             "an (n_atoms, 3) array [eV/A] — local Hellmann-Feynman + Ewald "
             "ion-ion + Kleinman-Bylander nonlocal, computed from the "
             "converged density rather than by differencing energies. The "
             "same values are written to the frame's fx/fy/fz columns and "
             "that SCF's total energy [eV] to frame.meta['energy'] "
             "(replacing any previous value); last_result is NOT updated, "
             "because this path runs its own SCF.\n\n"
             "Raises RuntimeError on a frame with no box, and ValueError "
             "when the restricted fixed-occupation path is asked for an odd "
             "valence-electron count (set spin_polarized, or a smearing), "
             "or when the frame has no atoms.")
        .def("stress", &pw_cell_stress, nb::arg("frame"),
             "Run one SCF and return the analytic Nielsen-Martin cell stress "
             "as a Voigt 6-vector (xx, yy, zz, xy, xz, yz) [eV/A^3]. "
             "Mechanical (CP2K) sign, sigma = -(1/V) dE/d(strain), so a "
             "compressed cell gives a positive diagonal. Reports on the cell "
             "only: no geometry, force or energy is written back, and "
             "last_result is not updated.\n\n"
             "Cost is one SCF plus twelve non-self-consistent strained "
             "diagonalizations (the Nielsen-Martin form of the cell "
             "derivative), not twelve full SCFs.\n\n"
             "Raises RuntimeError on a frame with no box (or one of zero "
             "volume), and ValueError on an odd valence-electron count "
             "without spin_polarized or smearing, on spin_polarized combined "
             "with a smearing (no analytic stress path yet), and on a frame "
             "with no atoms.")
        .def_prop_ro(
            "last_result",
            [](const PW& solver) { return solver.last_result(); },
            "Copy of the most recent run() result. Neither forces() nor "
            "stress() touches it — those run their own SCF — so before the "
            "first run() this is a default PWResult (energy 0, converged "
            "False, iterations 0).");

    nb::class_<PWEvaluator>(
        m, "PWEvaluator",
        "The plane-wave method as an MD force source: hand it to "
        "MD.set_evaluator and the run is plane-wave Born-Oppenheimer "
        "dynamics — the electrons are re-converged at every geometry the "
        "nuclei reach. Takes no NeighborList (a plane-wave Hamiltonian is "
        "not a pair style) and reads the frame's 'atomic_number' column [Z]; "
        "like any MD run it also needs a per-atom 'mass' column [amu] and a "
        "box.\n\n"
        "Cost per force-producing step is TWO SCF solves, not one: the "
        "energy SCF publishes frame.meta['energy'] [eV], then the analytic "
        "force path converges its own density before assembling F [eV/A]. "
        "The two agree to the SCF tolerance — ~3e-6 Ry apart on the pinned "
        "Si2 fixture of python/tests/test_pw.py — but are not the same "
        "solve.\n\n"
        "The evaluator publishes nothing on itself in Python — there is no "
        "bound result accessor — so read per-step energy and forces off the "
        "frame.")
        .def(
            "__init__",
            [](PWEvaluator* self, const PW& method) {
                // Parameters are copied, not moved: moving would leave the
                // Python-held PW a hollow shell that still looks usable.
                new (self)
                    PWEvaluator(ATV::cpu::pw::internal::copy_pw_params(method));
            },
            nb::arg("method"),
            "Wrap a configured PW. The method's fifteen parameters are "
            "COPIED (a PW owns FFTW plans, which cannot be shared), so the "
            "object you pass stays usable and later edits to it do not reach "
            "this evaluator. MD.set_evaluator copies the same fifteen once "
            "more, with force evaluation switched on, so per-step results "
            "land on the frame and not on this object either.");
#else
    (void)m;
#endif
}
