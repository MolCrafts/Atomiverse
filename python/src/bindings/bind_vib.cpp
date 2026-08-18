/// @file python/src/bindings/bind_vib.cpp
/// @brief Harmonic vibrational analysis (spec py-vib): the central
///        finite-difference Hessian driver `cpu::vib::Hessian` and its result.
///
/// Public surface mirrors the C++ preset and the `regressions/semiempirical/
/// se-analytic-hessian` shape:
///
/// @code
///   driver = Hessian()                       # standard conditions
///   driver.set_evaluator(PairLJCut(eps, sigma, cutoff))
///   driver.set_neighbor(NeighborList(cutoff, skin))
///   result = driver.run(frame)               # molpy.Frame in, result out
///   result.frequencies                       # [cm^-1], 3N-6 / 3N-5 entries
/// @endcode
///
/// Three properties of this binding are worth stating up front.
///
///  1. **No file output by default.** The C++ preset defaults to
///     `"frequencies.tsv"` / prefix `"mode"`, which writes into whatever
///     directory the process happens to sit in. The Python defaults are the
///     empty path and the empty prefix — the explicit "disabled" spelling the
///     two Recorders now honour (`frequency_report.cpp`,
///     `normal_mode_writer.cpp`). Pass a path to opt in.
///  2. **No launcher gate.** There is no vibration task in the MPI launcher
///     protocol and no `LauncherGate` probe here: like the Gaussian SCF
///     methods (`bind_scf.cpp`) a Hessian run is simply local to this process.
///     The 6N displacement loop is deterministic, so N ranks each running it
///     would agree rather than silently diverge the way N Markov chains do —
///     which is why MC refuses and this does not.
///  3. **Lazy build.** `Driver::validate_components()` needs the force source
///     to be registered already, so the plan is compiled on the first `run()`,
///     not in the constructor. As in MC that makes the plan a
///     compile-exactly-once affair, so the setters refuse to run afterwards.
///
/// Whole body gated on `ATV_HAS_VIB` (emitted by `AtvProductModules.cmake` for
/// `ATV_BUILD_VIB=ON`), matching the `bind.cpp` dispatch line.

#ifdef ATV_HAS_VIB

#include "molrs_frame_bridge.h"
#include "result_array.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/molrs_exchange.h"
#include "atomiverse/cpu/vibration/hessian.h"
#include "atomiverse/cpu/vibration/hessian_result.h"

#ifdef ATV_HAS_NEIGHBOR
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#endif
#ifdef ATV_HAS_POTENTIALS
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#endif
#ifdef ATV_HAS_SEMIEMPIRICAL
#include "atomiverse/cpu/semiempirical/am1_evaluator.h"
#endif
#include "atomiverse/cpu/task/task_spec.h"

using namespace ATV;

namespace {

/// @brief Whole contiguous storage of an xtensor as a read-only span.
/// @tparam Rank Tensor rank (1 for `frequencies`, 2 for `hessian`).
/// @param tensor Row-major xtensor owned by the result object.
/// @return A span over its elements; empty when the tensor is empty.
template <std::size_t Rank>
std::span<const double> tensor_span(const xt::xtensor<double, Rank>& tensor) {
    return std::span<const double>(tensor.data(), tensor.size());
}

/// @brief One bound Hessian driver: the preset, its working frame, and the
///        molrs write-back.
///
/// Declaration order matters. `work_` comes BEFORE `driver`: members are
/// destroyed in reverse declaration order, and the driver's teardown path
/// dereferences the frame the run was bound to (`DestroyContext{ctx,
/// *frame_}`), so the frame has to outlive it. Same rule as `McHandle`.
struct VibHandle {
    cpu::Frame work_{0};
    cpu::vib::Hessian driver;

    /// @brief Construct the preset with its thermochemistry inputs and the
    ///        two Recorder paths.
    /// @param temperature Temperature [K] of the RRHO thermochemistry.
    /// @param pressure Pressure [Pa] of the translational (Sackur-Tetrode)
    ///        entropy term.
    /// @param multiplicity Spin multiplicity 2S+1 (dimensionless).
    /// @param symmetry_number Rotational symmetry number sigma
    ///        (dimensionless).
    /// @param frequency_tsv Path for the frequency TSV; empty disables it.
    /// @param mode_xyz_prefix Prefix for the per-mode XYZ animations; empty
    ///        disables them.
    VibHandle(double temperature, double pressure, int multiplicity,
              int symmetry_number, std::string frequency_tsv,
              std::string mode_xyz_prefix)
        : driver(temperature, pressure, multiplicity, symmetry_number,
                 std::move(frequency_tsv), std::move(mode_xyz_prefix)) {}

    // A configured handle owns a frame the driver points at and components
    // built against it; neither copies nor moves meaningfully, and nanobind
    // probes both traits. Deleting them keeps the probes honest.
    VibHandle(const VibHandle&) = delete;
    VibHandle& operator=(const VibHandle&) = delete;
    VibHandle(VibHandle&&) = delete;
    VibHandle& operator=(VibHandle&&) = delete;

#ifdef ATV_HAS_POTENTIALS
    /// @brief Append a Lennard-Jones energy/force source (plural slot).
    /// @param p Lennard-Jones parameters, as the shared task mirror.
    void set_evaluator_lj(const cpu::task::PairLJCutParams& p) {
        require_configurable_("set_evaluator");
        driver.set_evaluator(cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff});
    }
#endif

#ifdef ATV_HAS_SEMIEMPIRICAL
    /// @brief Append the AM1 semiempirical method as the force source.
    ///
    /// Handed a LIVE component rather than a params mirror, exactly as
    /// `MdHandle::set_evaluator_am1` is: `AM1Evaluator` is already bound for
    /// its own `single_point` / `forces` surface, so Python has the real
    /// thing. The five SCF settings are read back off the owned method and a
    /// FRESH gradient-enabled copy is registered — the Hessian differences
    /// forces, so gradients are not optional here.
    ///
    /// @param evaluator The evaluator whose settings are copied in.
    void set_evaluator_am1(const cpu::semiempirical::AM1Evaluator& evaluator) {
        require_configurable_("set_evaluator");
        const cpu::semiempirical::AM1& method = evaluator.method();
        driver.set_evaluator(cpu::semiempirical::AM1Evaluator{
            method.charge, method.multiplicity, method.max_iter,
            method.density_tol, method.energy_tol}
                                 .with_gradient(true));
    }
#endif

#ifdef ATV_HAS_NEIGHBOR
    /// @brief Install the neighbor list in its finite-difference form.
    ///
    /// `every=1, delay=0, check=yes` — the shape the phonon regressions build
    /// (`regressions/pw/fd-phonon-supercell`,
    /// `regressions/response/phonon-dispersion-si`). The MD params default of
    /// `delay=10` is wrong here for the same reason it is wrong for Monte
    /// Carlo: the displacement loop is not a trajectory, so "ten steps since
    /// the last rebuild" bounds nothing. With the check ON the half-skin test
    /// still decides, so a +/-1e-3 A displacement pattern pays for exactly one
    /// build rather than 6N of them.
    ///
    /// @param p Cutoff [A] and skin [A], as the shared task mirror.
    void set_neighbor(const cpu::task::NeighborListParams& p) {
        require_configurable_("set_neighbor");
        driver.set_neighbor(
            cpu::md::NeighborList<>{p.cutoff, p.skin, 1, 0, true});
    }
#endif

    /// @brief Assemble the 6N-displacement Hessian on a borrowed molpy.Frame.
    ///
    /// The plan is compiled here on the first call — `validate_components()`
    /// needs the force source, which is only known once the setters have run.
    ///
    /// @param frame_obj A `molpy.Frame`; the equilibrium geometry is restored
    ///        before it is written back, and the force columns come back
    ///        carrying the forces [eV/A] evaluated at that geometry.
    /// @return The Hessian, frequencies, normal modes and RRHO thermochemistry.
    /// @throws std::runtime_error from `build()` when no force source was
    ///         registered.
    /// @throws std::invalid_argument when an atom carries an atomic number the
    ///         IUPAC-2013 mass table does not cover (Z outside 1..18).
    cpu::vib::HessianResult run(nb::object frame_obj) {
        auto fref = borrow_molrs_frame(frame_obj);
        work_ = cpu::from_molrs(*fref);
        cpu::vib::HessianResult result;
        {
            nb::gil_scoped_release release;
            if (!built_) {
                driver.build();
                built_ = true;
            }
            result = driver.run(work_);
        }
        cpu::to_molrs(work_, *fref);
        ran_ = true;
        return result;
    }

    /// @brief The result of the most recent `run()`.
    /// @return A copy of the cached result — bit-identical to what that
    ///         `run()` returned.
    /// @throws std::runtime_error before the first `run()`; there is no
    ///         zero-filled placeholder result to hand out.
    cpu::vib::HessianResult last_result() const {
        if (!ran_)
            throw std::runtime_error(
                "last_result(): no vibrational analysis has run yet on this "
                "Hessian. Call run(frame) first — it returns the same object.");
        return driver.last_result();
    }

  private:
    /// @brief Refuse a configuration change once the analysis has run.
    ///
    /// `Driver::build()` consumes its pending list, so the plan is compiled
    /// exactly once (on the first `run()`); a setter called afterwards could
    /// not reach it. Failing loud beats silently doing nothing.
    ///
    /// @param setter Name of the setter being refused, for the message.
    /// @throws std::runtime_error once `run()` has completed.
    void require_configurable_(const char* setter) const {
        if (built_)
            throw std::runtime_error(
                std::string(setter)
                + ": this Hessian has already run and the execution plan is "
                  "compiled once, so the change would silently do nothing. "
                  "Configure the driver fully before the first run(); a "
                  "different configuration means a new Hessian object.");
    }

    bool built_ = false; ///< Has the execution plan been compiled?
    bool ran_ = false;   ///< Has a run() completed at least once?
};

} // namespace

/// @brief Register the vibrational-analysis surface on the `_atomiverse`
///        module.
/// @param m The extension module under construction.
void bind_vib(nb::module_& m) {
    using cpu::vib::HessianResult;

    // ── Result ──

    nb::class_<HessianResult>(
        m, "HessianResult",
        "Output of a harmonic vibrational analysis: the Cartesian Hessian "
        "(the matrix of second derivatives of the energy with respect to the "
        "atomic coordinates), the normal modes, and the rigid-rotor / "
        "harmonic-oscillator (RRHO) thermochemistry — the finite-temperature "
        "energies and entropies that follow from treating the molecule as a "
        "rigid body translating and rotating freely while its normal modes "
        "oscillate independently.\n\n"
        "TRANSLATIONS AND ROTATIONS ARE PROJECTED OUT, PySCF-style, before "
        "the mass-weighted Hessian is diagonalized. `frequencies` therefore "
        "has exactly 3N-6 entries — 3N-5 for a linear molecule, and NONE at "
        "all (an empty array) for a lone atom, which has no vibration to "
        "report. A diatomic reports ONE frequency. It never contains five or "
        "six near-zero entries for you to filter out.\n\n"
        "An IMAGINARY mode — a direction in which the energy curves DOWN, so "
        "the geometry is a saddle point rather than a minimum — is reported "
        "as a NEGATIVE wavenumber (the universal `imaginary_freq=False` "
        "chemistry convention), and `n_imaginary` counts them.\n\n"
        "Units: frequencies [cm^-1]; Hessian [eV/A^2]; energies [Hartree]; "
        "entropies [Hartree/K]. The energies are Hartree because the "
        "thermochemistry is a faithful port of PySCF's `thermo()`, which "
        "reports in atomic units; 1 Hartree = 27.211386 eV.\n\n"
        "THE ELECTRONIC ENERGY E0 IS CARRIED AS ZERO. Nothing here adds the "
        "potential energy of the equilibrium geometry, so `u_tot`, `h_tot` "
        "and `g_tot` are the THERMAL parts of U, H and G — add your own E0 "
        "(in Hartree) to get an absolute energy. One visible consequence: "
        "`g_tot` and `g_thermal` carry the same number.\n\n"
        "Every array property returns a FRESH numpy array that owns its "
        "buffer, so it is unaffected by a later run() on the same driver.")
        // `nb::rv_policy::move` on every array property: the default for a
        // property getter is `reference_internal`, which nanobind refuses for
        // an ndarray that already carries its own capsule owner — and these
        // deliberately do, since they are copies that outlive the driver.
        .def_prop_ro(
            "frequencies",
            [](const HessianResult& r) {
                return to_vector_array(tensor_span(r.frequencies));
            },
            nb::rv_policy::move,
            "Harmonic wavenumbers [cm^-1], ascending — a 1-D array of the "
            "PROJECTED mode count (3N-6, or 3N-5 for a linear molecule; "
            "empty for a lone atom). A wavenumber is a frequency divided by "
            "the speed of light, the unit vibrational spectra are quoted in. "
            "Negative entries are imaginary modes.")
        .def_prop_ro(
            "hessian",
            [](const HessianResult& r) {
                return to_matrix_array(tensor_span(r.hessian),
                                       r.hessian.shape(0), r.hessian.shape(1));
            },
            nb::rv_policy::move,
            "Cartesian Hessian d^2E / dR_i dR_j as a dense symmetric "
            "[3N, 3N] array [eV/A^2]. Row/column c is atom c//3, axis c%3 "
            "(x, y, z). This is the raw matrix BEFORE mass-weighting and "
            "before the translation/rotation projection — all 3N coordinates "
            "are present.")
        .def_prop_ro(
            "normal_modes",
            [](const HessianResult& r) {
                return to_matrix_array(tensor_span(r.normal_modes),
                                       r.normal_modes.shape(0),
                                       r.normal_modes.shape(1));
            },
            nb::rv_policy::move,
            "Normal-mode eigenvectors as a [3N, n_modes] array "
            "(dimensionless): column k is mode k of `frequencies`, lifted "
            "back to the full 3N Cartesian basis. The vectors are "
            "MASS-WEIGHTED, so the Cartesian displacement of atom i along "
            "mode k is column[3i:3i+3] / sqrt(m_i) with m_i [amu] — the same "
            "un-weighting the XYZ animations apply.")
        .def_prop_ro(
            "n_atoms", [](const HessianResult& r) { return r.n_atoms; },
            "Number of atoms N in the analysed frame (the Hessian is "
            "3N x 3N).")
        .def_prop_ro(
            "n_imaginary", [](const HessianResult& r) { return r.n_imaginary; },
            "Number of imaginary modes — the entries of `frequencies` that "
            "are not positive (an imaginary mode is stored as a NEGATIVE "
            "wavenumber). 0 at a genuine minimum, 1 at a first-order saddle "
            "point such as a transition state.")
        .def_prop_ro(
            "is_linear", [](const HessianResult& r) { return r.is_linear; },
            "True when the molecule was classified as a linear rotor, which "
            "is what makes the mode count 3N-5 instead of 3N-6.")
        .def_prop_ro(
            "zpe", [](const HessianResult& r) { return r.zpe; },
            "Harmonic zero-point energy (ZPE) [Hartree] — the vibrational "
            "energy the molecule still carries at 0 K, because a quantum "
            "harmonic oscillator cannot be brought to rest. It is "
            "0.5 * sum_i h c nu_i over the REAL modes, where h is the Planck "
            "constant, c the speed of light and nu_i the wavenumber [cm^-1] "
            "of mode i. Imaginary modes are excluded from the sum.")
        .def_prop_ro(
            "u_tot", [](const HessianResult& r) { return r.u_tot; },
            "Internal energy U = ZPE + thermal vibrational + translational + "
            "rotational contributions [Hartree], at the configured "
            "temperature. The electronic energy E0 is carried as ZERO, so "
            "this is the thermal part of U and not an absolute internal "
            "energy; add your own E0 [Hartree] for that.")
        .def_prop_ro(
            "h_tot", [](const HessianResult& r) { return r.h_tot; },
            "Enthalpy H = U + PV [Hartree]; the +k_B T PV term of an ideal "
            "gas (k_B = Boltzmann constant, T = configured temperature) sits "
            "in the translational part. Same E0 = 0 convention as `u_tot`.")
        .def_prop_ro(
            "g_tot", [](const HessianResult& r) { return r.g_tot; },
            "Gibbs free energy G = H - T S [Hartree] (S = `s_tot`). Same "
            "E0 = 0 convention as `u_tot`, so this is NOT an absolute free "
            "energy and it equals `g_thermal` exactly.")
        .def_prop_ro(
            "g_thermal", [](const HessianResult& r) { return r.g_thermal; },
            "Thermal correction to the Gibbs free energy, G_tot - E0 "
            "[Hartree]. Because E0 is carried as zero it is numerically "
            "`g_tot`; it is published under its own name because the "
            "correction — not the absolute G — is what compares across codes "
            "without first matching electronic energies.")
        .def_prop_ro(
            "s_tot", [](const HessianResult& r) { return r.s_tot; },
            "Total entropy S = S_elec + S_trans + S_rot + S_vib "
            "[Hartree/K].")
        .def_prop_ro(
            "s_trans", [](const HessianResult& r) { return r.s_trans; },
            "Translational entropy [Hartree/K] of the molecule moving freely "
            "through space, from the Sackur-Tetrode equation (the ideal-gas "
            "expression for that entropy). This is the ONLY term the "
            "configured pressure enters.")
        .def_prop_ro(
            "s_rot", [](const HessianResult& r) { return r.s_rot; },
            "Rotational entropy [Hartree/K] of the molecule treated as a "
            "RIGID ROTOR — a rigid body whose moments of inertia are those "
            "of the equilibrium geometry. Depends on the configured "
            "rotational symmetry number.")
        .def_prop_ro(
            "s_vib", [](const HessianResult& r) { return r.s_vib; },
            "Vibrational entropy [Hartree/K], one independent quantum "
            "harmonic oscillator per real mode.")
        .def_prop_ro(
            "s_elec", [](const HessianResult& r) { return r.s_elec; },
            "Electronic entropy R ln(M) [Hartree/K], where M is the "
            "configured spin multiplicity and R is the gas constant in "
            "Hartree/K — every entropy here is a PER-MOLECULE quantity, so "
            "that constant is numerically the Boltzmann constant in those "
            "units. Zero for a closed-shell singlet (M = 1, ln 1 = 0): this "
            "term only counts the degeneracy of the electronic ground "
            "state.")
        .def_prop_ro(
            "units", [](const HessianResult& r) { return r.units; },
            "Physical unit tag of `hessian` — always 'eV/Angstrom^2'.")
        .def("__repr__", [](const HessianResult& r) {
            return "HessianResult(n_atoms=" + std::to_string(r.n_atoms)
                   + ", n_modes=" + std::to_string(r.frequencies.size())
                   + ", n_imaginary=" + std::to_string(r.n_imaginary)
                   + (r.is_linear ? ", linear)" : ")");
        });

    // ── Driver ──

    nb::class_<VibHandle> hessian(
        m, "Hessian",
        "Harmonic vibrational analysis by central finite differences (CPU).\n\n"
        "HARMONIC means the energy is approximated by its second-order Taylor "
        "expansion about a stationary point — the molecule as a set of "
        "coupled springs. Diagonalizing that quadratic form splits it into "
        "independent oscillations, the NORMAL MODES, each with one "
        "characteristic frequency. That is what this class computes, together "
        "with the thermodynamics those frequencies imply.\n\n"
        "The driver displaces every Cartesian coordinate by +/- h "
        "(h = 1e-3 A, fixed), differences the forces the registered evaluator "
        "produces, symmetrizes the resulting matrix, mass-weights it with the "
        "IUPAC-2013 standard atomic weights, projects out the translations "
        "and rotations and diagonalizes the rest. That is 6N displaced force "
        "evaluations for N atoms (plus one more at the restored geometry on "
        "the final pass) — the cost scales with the force source, so an "
        "electronic-structure evaluator on a large molecule is expensive by "
        "construction.\n\n"
        "The force slot takes ANY bound energy/force source: a pair style "
        "(with a NeighborList), or the AM1 semiempirical evaluator. There is "
        "no default — a run with nothing registered raises RuntimeError, "
        "because a Hessian of no potential is not a physical zero.\n\n"
        "MASSES COME FROM THE ATOMIC NUMBER, not from the frame's `mass` "
        "column: the analysis reads the `atomic_number` column [Z] and looks "
        "the IUPAC-2013 standard atomic weight up per element, so editing "
        "`mass` cannot move a frequency. The table covers Z = 1..18 (H..Ar) "
        "and has no fallback; anything else raises ValueError naming the "
        "atomic number.\n\n"
        "Configure the driver fully before the first run(): the execution "
        "plan is compiled exactly once, so every setter raises RuntimeError "
        "afterwards. A different configuration means a new Hessian object.\n\n"
        "This is molecular vibrational analysis — free-molecule translations "
        "and rotations are projected out. Periodic phonons are a different "
        "calculation and are not bound.");
    hessian.def(
        "__init__",
        [](VibHandle* self, double temperature, double pressure,
           int multiplicity, int symmetry_number,
           const std::string& frequency_tsv,
           const std::string& mode_xyz_prefix) {
            new (self)
                VibHandle(temperature, pressure, multiplicity, symmetry_number,
                          frequency_tsv, mode_xyz_prefix);
        },
        nb::arg("temperature") = 298.15, nb::arg("pressure") = 101325.0,
        nb::arg("multiplicity") = 1, nb::arg("symmetry_number") = 1,
        nb::arg("frequency_tsv") = "", nb::arg("mode_xyz_prefix") = "",
        "Construct the driver. Every argument configures the "
        "rigid-rotor/harmonic-oscillator thermochemistry or the optional file "
        "output; the finite-difference step itself is not tunable.\n\n"
        "  temperature      [K, default 298.15] of the thermal corrections.\n"
        "  pressure         [Pa, default 101325.0 = 1 atm] — enters the "
        "translational entropy (the Sackur-Tetrode ideal-gas expression) and "
        "nothing else.\n"
        "  multiplicity     spin multiplicity M = 2S+1 (dimensionless, "
        "default 1 = closed-shell singlet). It contributes the electronic "
        "entropy R ln(M) and nothing else.\n"
        "  symmetry_number  rotational symmetry number sigma (dimensionless, "
        "default 1) — the number of indistinguishable orientations the "
        "molecule has under rotation (2 for H2O, 3 for NH3, 12 for benzene). "
        "It divides the rotational partition function (the count of thermally "
        "accessible rotational states), so leaving it at 1 for a symmetric "
        "molecule OVERESTIMATES the rotational entropy by R ln(sigma) and "
        "under-estimates the free energy; nothing derives it for you.\n"
        "  frequency_tsv    path of an optional tab-separated frequency "
        "report. EMPTY BY DEFAULT, which writes no file — pass a path to opt "
        "in.\n"
        "  mode_xyz_prefix  prefix of optional per-mode XYZ animations, one "
        "file '<prefix>_mode<k>.xyz' per REAL mode (imaginary modes are not "
        "animated). EMPTY BY DEFAULT, which writes no files.");

#ifdef ATV_HAS_POTENTIALS
    hessian.def("set_evaluator", &VibHandle::set_evaluator_lj, nb::arg("comp"),
                "Append an energy/force source (plural slot: each call adds "
                "another term and the forces are summed).\n\n"
                "PairLJCut needs a NeighborList whose cutoff is at least the "
                "LJ cutoff, or pairs inside the potential's range are never "
                "enumerated. Pair distances are taken under the "
                "minimum-image convention, so cutoff + skin must stay below "
                "half the shortest box edge — and for an isolated molecule "
                "the cell must be larger than twice the cutoff, or the "
                "molecule interacts with its own periodic image and the "
                "frequencies are not the free-molecule ones.");
#endif

#ifdef ATV_HAS_SEMIEMPIRICAL
    hessian.def("set_evaluator", &VibHandle::set_evaluator_am1, nb::arg("comp"),
                "Append the AM1 semiempirical method as the force source — "
                "one self-consistent-field solve per displaced geometry, so "
                "6N solves in total. Takes no NeighborList (AM1 is molecular, "
                "not a pair style) and reads the frame's 'atomic_number' "
                "column [Z].\n\n"
                "The driver installs a FRESH gradient-enabled copy of the "
                "five SCF settings: the object you pass keeps its own state, "
                "so its last_result() is NOT populated by this run.");
#endif

#ifdef ATV_HAS_NEIGHBOR
    hessian.def("set_neighbor", &VibHandle::set_neighbor, nb::arg("comp"),
                "Install the neighbor list — the cached list of pairs closer "
                "than cutoff + skin, which is what makes a force evaluation "
                "cost O(N) instead of O(N^2). The SKIN [A] is the extra "
                "margin beyond the interaction cutoff that keeps the same "
                "list valid while atoms move a little.\n\n"
                "Built here in the finite-difference form (every=1, delay=0, "
                "check=yes): the displacement loop is not a trajectory, so "
                "the MD default of deferring the rebuild check by ten steps "
                "bounds nothing. Needed only by pair styles; an "
                "electronic-structure force source takes none.");
#endif

    hessian
        .def("run", &VibHandle::run, nb::arg("frame"),
             "Assemble the Hessian at the frame's current geometry and return "
             "a HessianResult. GIL released during the 6N force loop.\n\n"
             "The geometry is treated as an EQUILIBRIUM one and is restored "
             "before the frame is written back (the force columns come back "
             "holding the forces [eV/A] at that geometry) — the harmonic "
             "approximation only means anything at a stationary point, so "
             "minimize first. "
             "Running on a non-stationary geometry does not fail; it silently "
             "returns the curvature of a slope, usually with spurious "
             "imaginary modes.\n\n"
             "The execution plan is compiled on this first call, which is why "
             "every setter raises RuntimeError from here on. Running again is "
             "still allowed and repeats the analysis.\n\n"
             "Raises RuntimeError when no force source was registered, and "
             "ValueError naming the atomic number when an atom's Z is outside "
             "the mass table's 1..18 range.")
        .def("last_result", &VibHandle::last_result,
             "The HessianResult of the most recent run() — the same values, "
             "returned as an independent copy. Raises RuntimeError before the "
             "first run(): there is no zero-filled placeholder result.");
}

#endif // ATV_HAS_VIB
