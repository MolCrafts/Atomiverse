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
/// The surface is the C++ one, member for member: construct `PW` from its
/// identity (pseudopotential, `ecut_ry`, `k_mesh`, `xc`), set the knob
/// members (there is no `PWConfig` bag and no factory), `build(frame)` once,
/// then ask for what you need at a geometry —
///
/// | Call | Returns | Writes back to the frame |
/// | ---- | ------- | ------------------------ |
/// | `run(frame)` | `PWResult` (energies [Ry]/[Ha], smearing channels) | energy [eV] |
/// | `forces(frame)` | `(n_atoms, 3)` [eV/Å] | fx/fy/fz [eV/Å] + energy [eV] |
/// | `stress(frame)` | Voigt 6-vector [eV/Å³] | stress [eV], energy [eV] |
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
/// the molpy → Frame conversion before any solve starts.
///
/// Hand a `PW` to `MD.set_evaluator` and the run becomes plane-wave
/// Born-Oppenheimer dynamics: the driver borrows the same method as a
/// nested node, which re-converges the electrons at every geometry and
/// accumulates energy and analytic forces from that one SCF.

#include "force_array.h"
#include "molpy_frame_bridge.h"
#include "result_array.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
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
#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/qe_units.h"

namespace {

using ATV::cpu::pw::PW;
using ATV::cpu::pw::PWResult;
using ATV::electronic::Smearing;

/// @brief Metallic-smearing kinds and the names Python spells them with.
///
/// *Smearing* replaces the sharp occupied/empty step at the Fermi level by a
/// smooth occupation function of a chosen width, which is what makes a metal
/// converge at a finite k-mesh. Names are the enumerators of
/// `ATV::electronic::Smearing`, `none` being fixed (insulating) occupations.
constexpr std::pair<const char*, ATV::electronic::Smearing> kSmearingNames[] = {
    {"none", ATV::electronic::Smearing::none},
    {"fermi_dirac", ATV::electronic::Smearing::fermi_dirac},
    {"methfessel_paxton", ATV::electronic::Smearing::methfessel_paxton},
    {"gaussian", ATV::electronic::Smearing::gaussian},
};

/// @brief Smearing name → `Smearing`.
/// @param name One of the names in `kSmearingNames`.
/// @return The matching enumerator.
/// @throws std::invalid_argument (Python `ValueError`) on any other name,
///         listing the supported ones — no fallback occupation scheme.
ATV::electronic::Smearing smearing_from_name(const std::string& name) {
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
std::string smearing_name(ATV::electronic::Smearing smearing) {
    for (const auto& [key, value] : kSmearingNames)
        if (smearing == value)
            return key;
    throw std::invalid_argument(
        "smearing enumerator " + std::to_string(static_cast<int>(smearing))
        + " has no name in the Python binding table (bind_pw.cpp)");
}

/// @brief Materialize a borrowed molpy.Frame and refuse a non-periodic one.
///
/// A plane-wave basis is *defined* by the cell: the basis functions are the
/// plane waves commensurate with it, and the cutoff `ecut_ry` selects them by
/// their reciprocal-lattice length. With no cell there is no reciprocal
/// lattice, so the run has nothing to expand in — today it would grind
/// through its iteration budget and report a NaN energy, which is exactly the
/// failure a user cannot diagnose.
///
/// The check is on the materialized Frame rather than on the molpy handle
/// because that is where "no box at all" and "a box with zero volume" become
/// the same, well-defined question.
///
/// @param src Borrowed molpy.Frame.
/// @return The materialized execution Frame (positions [Å], Z, cell).
/// @throws std::runtime_error (Python `RuntimeError`) when the frame carries
///         no cell, naming the attribute to set.
ATV::cpu::Frame periodic_frame(const molrs::FrameRef& src) {
    ATV::cpu::Frame work = ATV::cpu::from_molpy(src);
    if (!(work.box.volume() > 0.0))
        throw std::runtime_error(
            "plane-wave DFT requires a periodic cell, but this molpy.Frame "
            "has no box (or one of zero volume). Set frame.box before "
            "build()/run()/forces()/stress(), e.g. "
            "frame.box = molpy.Box.cube(10.0).");
    return work;
}

/// @brief Run one plane-wave SCF, take the analytic ionic forces, and write
///        both back to the borrowed molpy.Frame.
///
/// `PW::forces` runs one SCF and writes both the forces and the energy of
/// that same SCF onto the frame, so the number and the forces describe the
/// same converged density; `PW::last_result()` holds that SCF afterwards.
///
/// @param solver    Configured plane-wave method (already built, or built
///                  lazily inside the force path).
/// @param frame_obj A `molpy.Frame`; fx/fy/fz [eV/Å] and energy [eV] land on
///                  it.
/// @return Forces [eV/Å], shape (n_atoms, 3).
/// @throws std::runtime_error (Python `RuntimeError`) on a frame with no
///         cell, and if the solver returns a force count that disagrees with
///         the atom count.
/// @throws std::invalid_argument (Python `ValueError`) from
///         `PW::forces`: odd valence-electron count on the
///         restricted fixed-occupation path, incompatible multiplicity
///         parity, empty frame.
ForceArray pw_forces(PW& solver, nb::object frame_obj) {
    auto fref = borrow_molpy_frame(frame_obj);
    ATV::cpu::Frame work = periodic_frame(*fref);
    std::vector<double> flat;
    {
        nb::gil_scoped_release release;
        solver.forces(work);
        const auto fx = work.span(ATV::kAtom_fx);
        const auto fy = work.span(ATV::kAtom_fy);
        const auto fz = work.span(ATV::kAtom_fz);
        flat.reserve(static_cast<std::size_t>(3 * work.nlocal()));
        for (int i = 0; i < work.nlocal(); ++i) {
            flat.push_back(static_cast<double>(fx[i]));
            flat.push_back(static_cast<double>(fy[i]));
            flat.push_back(static_cast<double>(fz[i]));
        }
    }
    ATV::cpu::to_molpy(work, *fref);
    return to_force_array(std::span<const double>(flat.data(), flat.size()));
}

/// @brief Run one plane-wave SCF and return the analytic cell stress.
///
/// Stress is the derivative of the energy with respect to a *strain* of the
/// cell. Writes extensive virial and energy back while preserving geometry. Mechanical (CP2K) sign convention,
/// \(\sigma = -(1/\Omega)\,\partial E/\partial\varepsilon\) — positive under
/// compression.
///
/// @param solver    Configured plane-wave method.
/// @param frame_obj A `molpy.Frame`; energy and extensive virial are overwritten.
/// @return Voigt stress (xx, yy, zz, xy, xz, yz) [eV/Å³].
/// @throws std::runtime_error (Python `RuntimeError`) on a frame with no
///         cell.
/// @throws std::invalid_argument (Python `ValueError`) from `PW::stress`:
///         odd valence-electron count on the restricted fixed-occupation
///         path, `spin_polarized` with metallic smearing (no analytic stress
///         path), incompatible multiplicity parity, empty frame.
VectorArray pw_cell_stress(PW& solver, nb::object frame_obj) {
    auto fref = borrow_molpy_frame(frame_obj);
    ATV::cpu::Frame work = periodic_frame(*fref);
    std::array<double, 6> stress{};
    {
        nb::gil_scoped_release release;
        stress = solver.stress(work);
    }
    ATV::cpu::to_molpy(work, *fref);
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

    using ATV::cpu::pw::GthPseudopotential;
    using ATV::cpu::pw::UpfPseudopotential;
    using ATV::electronic::NormConservingPotential;
    nb::class_<NormConservingPotential>(m, "NormConservingPotential")
        .def_prop_ro("atomic_number", &NormConservingPotential::atomic_number)
        .def_prop_ro("valence", &NormConservingPotential::valence);
    nb::class_<GthPseudopotential, NormConservingPotential>(
        m, "GthPseudopotential")
        .def(
            "__init__",
            [](GthPseudopotential* self, int z, const std::string& name) {
                new (self)
                    GthPseudopotential(static_cast<molrs::Element>(z), name);
            },
            nb::arg("atomic_number"), nb::arg("name"));
    nb::class_<UpfPseudopotential, NormConservingPotential>(
        m, "UpfPseudopotential")
        .def(nb::init<const std::string&>(), nb::arg("path"));

    nb::class_<PW>(
        m, "PW",
        "Plane-wave density-functional theory (DFT) for a periodic cell: the "
        "Kohn-Sham orbitals are expanded in plane waves e^{i(k+G)r}, the "
        "basis a periodic system asks for. The pseudopotential, cutoff, "
        "k-mesh and functional are constructor arguments, fixed afterwards; "
        "the SCF knobs are plain writable members (there is no PWConfig bag "
        "and no factory). Construct, then build(frame) / run(frame) / "
        "forces(frame) / stress(frame).\n\n"
        "Every entry point reads the frame's per-atom 'atomic_number' column "
        "[Z], the positions [A] and the box, and refuses a frame with no box.")
        .def(
            "__init__",
            [](PW* self,
               std::shared_ptr<const NormConservingPotential> potential,
               double ecut_ry, int k_mesh, const std::string& xc) {
                new (self)
                    PW(std::move(potential), ecut_ry, k_mesh, xc_from_name(xc));
            },
            nb::kw_only(), nb::arg("potential"), nb::arg("ecut_ry"),
            nb::arg("k_mesh"), nb::arg("xc") = "PBE",
            "Construct with an immutable NC potential, cutoff [Ry] and k mesh.")
        .def_prop_ro(
            "potential", [](const PW& pw) { return pw.potential; },
            "The immutable NC potential shared by this method.")
        .def_ro("ecut_ry", &PW::ecut_ry,
                "Wavefunction cutoff [Ry] fixed at construction (read-only).")
        .def_ro("k_mesh", &PW::k_mesh,
                "Monkhorst-Pack mesh size per dimension fixed at construction "
                "(read-only).")
        .def_prop_ro(
            "xc", [](const PW& solver) { return xc_name(solver.xc); },
            "Exchange-correlation functional name fixed at construction "
            "(read-only).")
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
                "ATV::electronic::Smearing width [Ry] (Quantum ESPRESSO "
                "`degauss`), default "
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
                auto fref = borrow_molpy_frame(frame_obj);
                ATV::cpu::Frame work = periodic_frame(*fref);
                nb::gil_scoped_release release;
                solver.build(work);
            },
            nb::arg("frame"),
            "Validate the reference frame and perform one-time setup: it "
            "checks that the pseudopotential source resolves (the UPF file "
            "parses, or the GTH name exists for "
            "this frame's element). run()/forces()/stress() call it lazily, "
            "so build() is only about failing early.\n\n"
            "Raises RuntimeError when the frame carries no box — a "
            "plane-wave basis is defined by the cell — and ValueError when "
            "the named potential is unknown or unreadable.")
        .def(
            "run",
            [](PW& solver, nb::object frame_obj) {
                auto fref = borrow_molpy_frame(frame_obj);
                ATV::cpu::Frame work = periodic_frame(*fref);
                PWResult result;
                {
                    nb::gil_scoped_release release;
                    result = solver.run(work);
                }
                ATV::cpu::to_molpy(work, *fref);
                return result;
            },
            nb::arg("frame"),
            "Run SCF at the frame's current geometry and return the "
            "PWResult (Ry / Hartree), and overwrite frame.meta['energy'] "
            "in eV. The GIL is released during the solve, "
            "so other Python threads keep running.\n\n"
            "Failing to converge within max_iter is NOT an error: the result "
            "comes back with converged = False and the last density's "
            "energy, so check the flag.\n\n"
            "Raises RuntimeError on a frame with no box, and ValueError when "
            "a restricted run is asked for an odd valence-electron count "
            "(set spin_polarized, or a smearing), when multiplicity and the "
            "electron count have incompatible parity, or Frame species differ "
            "from the provider atomic number.")
        .def("forces", &pw_forces, nb::arg("frame"),
             "Run one SCF and return the analytic ionic forces F = -dE/dR as "
             "an (n_atoms, 3) array [eV/A] — local Hellmann-Feynman + Ewald "
             "ion-ion + Kleinman-Bylander nonlocal, computed from the "
             "converged density rather than by differencing energies. The "
             "same values are written to the frame's fx/fy/fz columns and "
             "that SCF's total energy [eV] to frame.meta['energy'] "
             "(replacing any previous value); last_result holds that SCF.\n\n"
             "Raises RuntimeError on a frame with no box, and ValueError "
             "when the restricted fixed-occupation path is asked for an odd "
             "valence-electron count (set spin_polarized, or a smearing), "
             "or when the frame has no atoms.")
        .def("stress", &pw_cell_stress, nb::arg("frame"),
             "Run one SCF and return the analytic Nielsen-Martin cell stress "
             "as a Voigt 6-vector (xx, yy, zz, xy, xz, yz) [eV/A^3]. "
             "Mechanical (CP2K) sign, sigma = -(1/V) dE/d(strain), so a "
             "compressed cell gives a positive diagonal. Reports on the cell "
             "without moving atoms: energy [eV] and the extensive virial "
             "[eV] overwrite frame metadata; last_result holds that SCF.\n\n"
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
            "Copy of the most recent run(), forces() or stress() SCF result. "
            "Before the first solve this is a default PWResult (energy 0, "
            "converged "
            "False, iterations 0).");

#else
    (void)m;
#endif
}
