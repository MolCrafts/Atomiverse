/// @file python/src/bindings/bind_potentials.cpp
/// @brief Real-space force-field and neighbor-list parameter types.
///
/// The pair, bonded and Coulomb styles (plus `NeighborList`) are bound as
/// their `cpu::task` wire mirrors: a Python object is a typed parameter set
/// that a driver facade hands to its `set_evaluator_*` / `set_neighbor`
/// setter. Every force-consuming driver (MD, Minimize, MC) shares these
/// types, so they register under `ATV_BUILD_POTENTIALS` rather than inside
/// one driver's binding — a minimize-only extension carries them too.

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <utility>
#include <vector>

#include "atomiverse/cpu/task/task_spec.h"

namespace nb = nanobind;
using namespace ATV;

void bind_potentials(nb::module_& m) {
    nb::class_<cpu::task::PairLJCutParams>(m, "PairLJCut",
                                           "Lennard-Jones 12-6 pair potential.")
        .def(
            "__init__",
            [](cpu::task::PairLJCutParams* p, double epsilon, double sigma,
               double cutoff, double s12, double s13, double s14,
               bool energy_shift) {
                new (p) cpu::task::PairLJCutParams{
                    epsilon, sigma, cutoff, s12, s13,         s14,
                    {},      {},    {},     {},  energy_shift};
            },
            nb::arg("epsilon"), nb::arg("sigma"), nb::arg("cutoff"),
            nb::arg("s12") = 1.0, nb::arg("s13") = 1.0, nb::arg("s14") = 1.0,
            nb::arg("energy_shift") = true,
            "Construct with epsilon [eV], sigma [A], and cutoff [A].\n\n"
            "s12, s13, s14 are the 1-2 / 1-3 / 1-4 pair weights (LAMMPS "
            "special_bonds). 0 drops that shell, a weight in (0, 1] scales "
            "that pair's energy and force. The default is (1, 1, 1). "
            "energy_shift subtracts V(cutoff) per pair (LAMMPS pair_modify "
            "shift); default on.")
        .def_static(
            "table",
            [](std::vector<int> type_i, std::vector<int> type_j,
               std::vector<double> epsilon, std::vector<double> sigma,
               double cutoff, double s12, double s13, double s14,
               bool energy_shift) {
                return cpu::task::PairLJCutParams{0.0,
                                                  0.0,
                                                  cutoff,
                                                  s12,
                                                  s13,
                                                  s14,
                                                  std::move(type_i),
                                                  std::move(type_j),
                                                  std::move(epsilon),
                                                  std::move(sigma),
                                                  energy_shift};
            },
            nb::arg("type_i"), nb::arg("type_j"), nb::arg("epsilon"),
            nb::arg("sigma"), nb::arg("cutoff"), nb::arg("s12") = 1.0,
            nb::arg("s13") = 1.0, nb::arg("s14") = 1.0,
            nb::arg("energy_shift") = true,
            "Multi-type form: one row per type pair — type_i, type_j (frame "
            "atom kinds: atoms.type_id when stated, else Z), epsilon [eV], "
            "sigma [A]. Unlisted cross pairs mix Lorentz-Berthelot from the "
            "diagonal; a type with no diagonal has no LJ site. CPU only.");

    // ── The widened real-space catalogue ──
    //
    // Six pair styles and three bonded ones, all unconditional: a real-space
    // style is part of ATV_BUILD_POTENTIALS with no flag of its own, exactly
    // like PairLJCut / BondHarmonic. None of the six pair styles is a
    // NeighborConsumer, and each as constructed from these mirrors walks all
    // pairs O(N^2) — including PairBorn, whose only neighbor-list-aware arm is
    // the single-type C++ constructor the table mirror never reaches. So a run
    // that installs only these needs no NeighborList at all.

    nb::class_<cpu::task::PairMorseParams>(
        m, "PairMorse",
        "Morse pair potential V = D(1 - exp(-alpha(r - r0)))^2, shifted to "
        "zero at the cutoff.\n\n"
        "The textbook model of a chemical bond: a minimum at r0, harmonic for "
        "small displacements, and — unlike a spring — flattening out to the "
        "finite dissociation energy D as r grows, so the bond can break. "
        "alpha sets how fast that flattening happens; a large alpha is a "
        "narrow, stiff well.\n\n"
        "The tree's demonstration that a new pair style costs two files and "
        "no registry, written for readability rather than speed: it walks "
        "every pair O(N^2) with no neighbor list. Use it at the scale where "
        "that is honest — a dimer, a small cluster, a teaching example.")
        .def(
            "__init__",
            [](cpu::task::PairMorseParams* p, double D, double alpha, double r0,
               double cutoff) {
                new (p) cpu::task::PairMorseParams{D, alpha, r0, cutoff};
            },
            nb::arg("D"), nb::arg("alpha"), nb::arg("r0"), nb::arg("cutoff"),
            "Construct with well depth D [eV], width alpha [1/A], "
            "equilibrium separation r0 [A], and cutoff [A].");

    nb::class_<cpu::task::PairBuckParams>(
        m, "PairBuck",
        "Buckingham pair potential U = A exp(-r/rho) - C/r^6, shifted at the "
        "cutoff (LAMMPS `pair_style buck` with `pair_modify shift yes`).\n\n"
        "The two terms are the two physical effects of a closed-shell pair. "
        "The Born-Mayer exponential is Pauli repulsion — the energy cost of "
        "forcing two filled electron clouds to overlap, falling off over the "
        "softness length rho. The -C/r^6 term is London dispersion, the "
        "attraction between the instantaneous dipoles two polarizable atoms "
        "induce in each other.\n\n"
        "Single-type: one (A, rho, C) applies to every pair. Mind the short-"
        "range limit — the Born-Mayer wall is finite while -C/r^6 is not, so "
        "Buckingham has a spurious attractive collapse as r -> 0 that "
        "PairBorn's +D/r^8 term cures.")
        .def(
            "__init__",
            [](cpu::task::PairBuckParams* p, double A, double rho, double C,
               double cutoff) {
                new (p) cpu::task::PairBuckParams{A, rho, C, cutoff};
            },
            nb::arg("A"), nb::arg("rho"), nb::arg("C"), nb::arg("cutoff"),
            "Construct with Born-Mayer prefactor A [eV], softness length "
            "rho [A], dispersion coefficient C [eV*A^6], and cutoff [A].");

    nb::class_<cpu::task::PairBornParams>(
        m, "PairBorn",
        "Born-Mayer-Huggins pair table U = A exp((sigma - r)/rho) - C/r^6 + "
        "D/r^8, shifted at the cutoff (LAMMPS `pair_style born`) — the "
        "Fumi-Tosi short-range half of a molten-salt model. Add the Coulomb "
        "term as a separate evaluator.\n\n"
        "The table is per TYPE PAIR and the keys are ATOMIC NUMBERS: the "
        "frame's per-atom type column is filled from the molrs "
        "`atomic_number` column, so a sodium-chloride row is type_i=11, "
        "type_j=17 and an argon dimer is 18, 18 — not a 0-based index.\n\n"
        "Every interacting pair must be listed, and the table is symmetric in "
        "(i, j). An unlisted pair does NOT interact: Fumi-Tosi parameters are "
        "not Lorentz-Berthelot, so there is no mixing rule to fall back on.")
        .def(
            "__init__",
            [](cpu::task::PairBornParams* p, std::vector<int> type_i,
               std::vector<int> type_j, std::vector<double> A,
               std::vector<double> rho, std::vector<double> sigma,
               std::vector<double> C, std::vector<double> D, double cutoff) {
                new (p) cpu::task::PairBornParams{
                    std::move(type_i), std::move(type_j),
                    std::move(A),      std::move(rho),
                    std::move(sigma),  std::move(C),
                    std::move(D),      cutoff};
            },
            nb::arg("type_i"), nb::arg("type_j"), nb::arg("A"), nb::arg("rho"),
            nb::arg("sigma"), nb::arg("C"), nb::arg("D"), nb::arg("cutoff"),
            "Construct from PARALLEL sequences, one entry per table row: "
            "type_i / type_j (atomic numbers), A [eV], rho [A], sigma [A], "
            "C [eV*A^6], D [eV*A^8], plus the one shared cutoff [A].\n\n"
            "All seven sequences are one table seen column-wise and must have "
            "the same length. A single-type model is one row, e.g. "
            "type_i=[18], type_j=[18], A=[1000.0], ...\n\n"
            "This object is a plain parameter record and validates nothing; "
            "the table is transposed into coefficient rows when the evaluator "
            "is installed, so a length mismatch raises ValueError from "
            "MD.set_evaluator(...), not from here — the same arrangement as "
            "XYZ and MD.set_record(...).");

    nb::class_<cpu::task::PairYukawaParams>(
        m, "PairYukawa",
        "Screened-Coulomb (Yukawa) pair potential U = A exp(-kappa r)/r, "
        "shifted at the cutoff.\n\n"
        "SCREENING is what a cloud of mobile opposite charges — counter-ions "
        "around a colloid, electrons in a plasma — does to a bare charge: it "
        "neutralizes it beyond a distance 1/kappa, turning the 1/r Coulomb "
        "tail into an exponentially cut one. kappa = 0 recovers unscreened "
        "A/r.\n\n"
        "Despite the name it reads NO per-atom charge column: the amplitude A "
        "is the whole interaction strength, which is what makes this a "
        "colloid / screened-plasma model rather than an electrostatic one. "
        "For real charges use PairCoulCut, or CoulLong + KSpacePME.")
        .def(
            "__init__",
            [](cpu::task::PairYukawaParams* p, double A, double kappa,
               double cutoff) {
                new (p) cpu::task::PairYukawaParams{A, kappa, cutoff};
            },
            nb::arg("A"), nb::arg("kappa"), nb::arg("cutoff"),
            "Construct with amplitude A [eV*A] (positive is repulsive), "
            "screening wavenumber kappa [1/A], and cutoff [A].");

    nb::class_<cpu::task::PairWcaParams>(
        m, "PairWCA",
        "Weeks-Chandler-Andersen pair potential: Lennard-Jones 12-6 cut and "
        "shifted at r_c = 2^(1/6) sigma, so U(r_c) = F(r_c) = 0 and the style "
        "is purely repulsive.\n\n"
        "There is no cutoff argument, deliberately: the cutoff is DERIVED "
        "from sigma (the LJ minimum), not chosen — that is the definition of "
        "the style. Beyond it the contribution is exactly zero, not merely "
        "small.")
        .def(
            "__init__",
            [](cpu::task::PairWcaParams* p, double epsilon, double sigma) {
                new (p) cpu::task::PairWcaParams{epsilon, sigma};
            },
            nb::arg("epsilon"), nb::arg("sigma"),
            "Construct with the parent LJ well depth epsilon [eV] and length "
            "scale sigma [A].");

    nb::class_<cpu::task::PairDpdParams>(
        m, "PairDpd",
        "Groot-Warren dissipative particle dynamics (DPD).\n\n"
        "DPD is a COARSE-GRAINED method: a particle stands for a blob of "
        "fluid rather than an atom, so the pair force is soft (finite at "
        "r = 0) and the degrees of freedom that were coarsened away come back "
        "as a pairwise thermostat. The conservative half is "
        "F^C = a w(r) rhat with w = 1 - r/r_c, rhat the unit vector along "
        "r_i - r_j; the energy reported for it is U^C = a r_c w^2 / 2 [eV]. "
        "The thermostat half is a matched dissipative/random pair with "
        "sigma^2 = 2 gamma k_B T.\n\n"
        "It is a THERMOSTAT as well as a potential — install it instead of, "
        "not beside, NoseHoover / Langevin / CSVR, and give it the same dt "
        "the integrator uses (the random force scales as 1/sqrt(dt)). It does "
        "NOT replace the integrator: VelocityVerlet still advances the "
        "positions.\n\n"
        "Single-amplitude only: the C++ class also has a multi-component "
        "constructor taking a square a_matrix[type_i][type_j], and that one is "
        "NOT bound here (it has no wire mirror either — it is a dense matrix "
        "indexed by atomic number, not a list of keyed rows like PairBorn's "
        "table). One `a` applies to every pair.\n\n"
        "Two consequences worth knowing before reading a number off a run:\n"
        "  * the reported stress carries the CONSERVATIVE virial only. The "
        "dissipative and random forces are thermostat terms with zero mean at "
        "equilibrium, and the random one diverges as 1/sqrt(dt), so tallying "
        "them would put timestep-dependent noise into every pressure without "
        "changing its average.\n"
        "  * the random force is REDRAWN every step from the {seed, step, i, "
        "j} stream, where i and j are atom INDICES. A run is reproducible for "
        "a given seed, step budget and atom ordering; a single point is not "
        "reproducible across different steps unless gamma = 0, which switches "
        "both thermostat terms off and leaves the conservative half alone.")
        .def(
            "__init__",
            [](cpu::task::PairDpdParams* p, double a, double gamma,
               double temperature, double cutoff, double dt,
               std::uint64_t seed) {
                new (p) cpu::task::PairDpdParams{a,      gamma, temperature,
                                                 cutoff, dt,    seed};
            },
            nb::arg("a"), nb::arg("gamma"), nb::arg("temperature"),
            nb::arg("cutoff"), nb::arg("dt"),
            nb::arg("seed") = std::uint64_t(1),
            "Construct with soft repulsion amplitude a [eV/A] — a FORCE, the "
            "value of F^C at contact, not an energy — dissipative friction "
            "gamma [eV*fs/A^2], thermostat temperature [K], cutoff r_c [A], "
            "timestep dt [fs], and the random-stream seed "
            "(dimensionless).\n\n"
            "gamma = 0 switches BOTH the dissipative and the random term off "
            "(sigma = sqrt(2 gamma k_B T)), leaving the conservative soft "
            "repulsion alone — which is the only way to get a reproducible "
            "single point out of this style.");

    nb::class_<cpu::task::BondFeneParams>(
        m, "BondFene",
        "Kremer-Grest FENE bond U = -K R0^2 ln(1 - (r/R0)^2) / 2.\n\n"
        "FENE is Finitely Extensible Nonlinear Elastic: the energy diverges "
        "as the bond length approaches the maximum extension R0, so unlike a "
        "harmonic spring the bond cannot be stretched arbitrarily far. And it "
        "says so — a bond at or beyond R0 raises RuntimeError from run() "
        "rather than reporting an infinite energy.\n\n"
        "The standard coarse-grained polymer bond, and the reason it is "
        "standard: paired with a repulsive WCA pair style, the hard ceiling "
        "on bond length plus the hard floor on non-bonded approach makes it "
        "geometrically impossible for two chains to pass through each other.")
        .def(
            "__init__",
            [](cpu::task::BondFeneParams* p, std::vector<double> K,
               std::vector<double> R0) {
                new (p) cpu::task::BondFeneParams{std::move(K), std::move(R0)};
            },
            nb::arg("K"), nb::arg("R0"),
            "Construct with spring constants K [eV/A^2] and maximum "
            "extensions R0 [A] per bond type.");

    nb::class_<cpu::task::DihedralCharmmParams>(
        m, "DihedralCharmm",
        "CHARMM periodic torsion V = K(1 + cos(n phi - delta)).\n\n"
        "This is LAMMPS `dihedral_style charmm` WITHOUT the optional 1-4 "
        "LJ/Coulomb weight factors — those belong to the pair styles via "
        "special-bonds weights, not to the torsion. The C++ class also "
        "exposes a stress-skipping performance switch (`set_compute_stress`); "
        "it is not bound and stays at its default, so the torsion always "
        "contributes to the virial.")
        .def(
            "__init__",
            [](cpu::task::DihedralCharmmParams* p, std::vector<double> K,
               std::vector<int> n, std::vector<double> delta) {
                new (p) cpu::task::DihedralCharmmParams{
                    std::move(K), std::move(n), std::move(delta)};
            },
            nb::arg("K"), nb::arg("n"), nb::arg("delta"),
            "Construct with force constants K [eV], multiplicities n "
            "(integers >= 1) and phase shifts delta [rad] per dihedral type. "
            "n is the number of minima per full turn, so n = 3 is the "
            "threefold barrier of an sp3-sp3 bond; delta shifts where those "
            "minima sit.");

    nb::class_<cpu::task::ImproperHarmonicParams>(
        m, "ImproperHarmonic",
        "Harmonic improper torsion V = K(chi - chi0)^2 (LAMMPS "
        "`improper_style harmonic`), where chi is the angle between the two "
        "planes of the 4-tuple.\n\n"
        "Reads the frame's IMPROPER topology — a separate block from "
        "`dihedrals`, holding the out-of-plane tuples that keep a group "
        "planar (an sp2 centre, an aromatic ring).")
        .def(
            "__init__",
            [](cpu::task::ImproperHarmonicParams* p, std::vector<double> K,
               std::vector<double> chi0) {
                new (p) cpu::task::ImproperHarmonicParams{std::move(K),
                                                          std::move(chi0)};
            },
            nb::arg("K"), nb::arg("chi0"),
            "Construct with force constants K [eV/rad^2] and equilibrium "
            "angles chi0 [rad] per improper type.");

    nb::class_<cpu::task::PairCoulCutParams>(
        m, "PairCoulCut",
        "Truncated Coulomb pair style (LAMMPS `pair_style coul/cut`).\n\n"
        "The bare q_i q_j / r sum over the neighbor list, hard-cut at "
        "cutoff. It is self-contained: unlike CoulLong it takes no "
        "reciprocal-space partner, so it is the whole electrostatic model of "
        "the run that installs it. Charges come from the per-atom 'charge' "
        "column [e] of the molpy.Frame; a frame without that column is "
        "rejected by run() instead of being read as neutral.")
        .def(
            "__init__",
            [](cpu::task::PairCoulCutParams* p, double cutoff, double s12,
               double s13, double s14, double coulomb) {
                new (p) cpu::task::PairCoulCutParams{cutoff, s12, s13, s14,
                                                     coulomb};
            },
            nb::arg("cutoff"), nb::arg("s12") = 1.0, nb::arg("s13") = 1.0,
            nb::arg("s14") = 1.0, nb::arg("coulomb") = ATV::units::coulomb,
            "Construct with the real-space cutoff [A]. The neighbor list "
            "must be built with a cutoff >= this value. s12, s13, s14 are the "
            "1-2 / 1-3 / 1-4 Coulomb weights (LAMMPS special_bonds coul). "
            "coulomb is the constant k in k q_i q_j / r [eV A / e^2]; the "
            "default 14.399645 is LAMMPS metal qqr2e.");

    nb::class_<cpu::task::PairLJCutCoulLongParams>(
        m, "PairLJCutCoulLong",
        "Fused multi-type Lennard-Jones 12-6 + real-space Ewald Coulomb "
        "(LAMMPS lj/cut/coul/long), the Amber/GAFF nonbonded style.")
        .def(
            "__init__",
            [](cpu::task::PairLJCutCoulLongParams* p, std::vector<int> type_i,
               std::vector<int> type_j, std::vector<double> epsilon,
               std::vector<double> sigma, double alpha, double cutoff,
               std::array<double, 3> special_lj,
               std::array<double, 3> special_coul, bool energy_shift) {
                new (p) cpu::task::PairLJCutCoulLongParams{std::move(type_i),
                                                           std::move(type_j),
                                                           std::move(epsilon),
                                                           std::move(sigma),
                                                           alpha,
                                                           cutoff,
                                                           special_lj,
                                                           special_coul,
                                                           energy_shift};
            },
            nb::arg("type_i"), nb::arg("type_j"), nb::arg("epsilon"),
            nb::arg("sigma"), nb::arg("alpha"), nb::arg("cutoff"),
            nb::arg("special_lj") = std::array<double, 3>{1.0, 1.0, 1.0},
            nb::arg("special_coul") = std::array<double, 3>{1.0, 1.0, 1.0},
            nb::arg("energy_shift") = false,
            "One LJ row per type pair: type_i, type_j (frame atom kinds — "
            "atoms.type_id when stated, else Z), epsilon [eV], sigma [A]. "
            "Unlisted cross pairs mix Lorentz-Berthelot from the diagonal; a "
            "type with no diagonal has no LJ site. alpha [1/A] must equal the "
            "KSpacePME/KSpacePPPM alpha; cutoff [A] is shared by LJ and "
            "Coulomb. special_lj / special_coul are the 1-2/1-3/1-4 weights "
            "(Amber: (0, 0, 0.5) and (0, 0, 5/6)). energy_shift subtracts "
            "V_LJ(rc) (LAMMPS pair_modify shift).");

    nb::class_<cpu::task::ImproperCvffParams>(
        m, "ImproperCvff",
        "Amber/GAFF improper torsion V = K(1 + d cos(n phi)), d = +/-1 "
        "(LAMMPS improper_style cvff).")
        .def(
            "__init__",
            [](cpu::task::ImproperCvffParams* p, std::vector<double> K,
               std::vector<int> d, std::vector<int> n) {
                new (p) cpu::task::ImproperCvffParams{
                    std::move(K), std::move(d), std::move(n)};
            },
            nb::arg("K"), nb::arg("d"), nb::arg("n"),
            "Per improper type: K [eV], d (+1 or -1), n (0..6). phi is the "
            "torsion angle of the improper 4-tuple in its frame order (Amber "
            "puts the central atom third).");

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

    nb::class_<cpu::task::KSpacePppmParams>(
        m, "KSpacePPPM",
        "Reciprocal-space half of an Ewald split evaluated as "
        "Hockney-Eastwood PPPM with ik differentiation, the LAMMPS "
        "`pppm` default (`differentiation_flag = 0`).\n\n"
        "Charge is spread with the same B-splines as smooth PME. The mesh "
        "is multiplied by the ik Green's function G(k) (alias sum, not the "
        "Essmann Euler modulus), the energy is coulomb/(2V) * |rho|^2 * G, "
        "and the force is q times the Cartesian field i*k*V gathered with "
        "the spreading weights. Install it with CoulLong at the SAME alpha. "
        "It is not carried by the MPI task wire.\n\n"
        "This type exists only in an extension built with "
        "ATV_BUILD_KSPACE=ON.")
        .def(
            "__init__",
            [](cpu::task::KSpacePppmParams* p, double alpha,
               double grid_spacing, int order) {
                new (p) cpu::task::KSpacePppmParams{alpha, grid_spacing, order};
            },
            nb::arg("alpha"), nb::arg("grid_spacing") = double(1),
            nb::arg("order") = 6,
            "Construct with alpha [1/A], which must equal the CoulLong "
            "alpha.\n\n"
            "  grid_spacing  target mesh spacing [A] (default 1.0). Rounded "
            "to sizes whose factors are 2, 3 and 5.\n"
            "  order         B-spline order [count, 3..12; default 6].");
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
            "Construct with force constants K [eV/A^2] and equilibrium "
            "distances r0 [A], one entry per bond type. The bond type is the "
            "frame's `bonds` block `type_id` column.");

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
            "Construct with force constants K [eV/rad^2] and equilibrium "
            "angles theta0 [rad], one entry per angle type (the frame's "
            "`angles` block `type_id` column).");

    nb::class_<cpu::task::DihedralMultiHarmonicParams>(
        m, "DihedralMultiHarmonic",
        "Multi-harmonic (Ryckaert-Bellemans) dihedral potential "
        "V = sum_n C_n cos^n(phi), where phi is the torsion angle about the "
        "central bond of the 4-tuple (LAMMPS `dihedral_style "
        "multi/harmonic`).")
        .def(
            "__init__",
            [](cpu::task::DihedralMultiHarmonicParams* p,
               std::vector<std::vector<double>> Cn) {
                new (p) cpu::task::DihedralMultiHarmonicParams{std::move(Cn)};
            },
            nb::arg("Cn"),
            "Construct with the coefficients Cn [eV] — one inner sequence per "
            "dihedral type, its k-th entry multiplying cos^k(phi).");
}
