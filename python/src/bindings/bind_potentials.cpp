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
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <utility>
#include <vector>

#include "atomiverse/cpu/task/task_spec.h"
#ifdef ATV_HAS_KSPACE
#include "atomiverse/core/pppm_accuracy.h"
#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "frame_guards.h"
#include "molpy_frame_bridge.h"
#endif

namespace nb = nanobind;
using namespace ATV;

void bind_potentials(nb::module_& m) {
    nb::class_<LJTypeCoeff>(
        m, "LJTypeCoeff",
        "One Lennard-Jones type-pair row (LAMMPS `pair_coeff i j eps sigma "
        "[cut_lj]`), keyed by the frame's per-atom type value.")
        .def(
            "__init__",
            [](LJTypeCoeff* c, int type_i, int type_j, double epsilon,
               double sigma, std::optional<double> cutoff) {
                new (c) LJTypeCoeff{type_i, type_j, epsilon, sigma, cutoff};
            },
            nb::arg("type_i"), nb::arg("type_j"), nb::arg("epsilon"),
            nb::arg("sigma"), nb::arg("cutoff") = nb::none(),
            "Construct with the two types, epsilon [energy], sigma [length] "
            "and an "
            "optional per-pair LJ cutoff [length]. An unset cutoff on a listed "
            "row "
            "is the style's global cutoff; a pair with no row at all mixes "
            "its cutoff arithmetically from the diagonals.")
        .def_ro("type_i", &LJTypeCoeff::type_i)
        .def_ro("type_j", &LJTypeCoeff::type_j)
        .def_ro("epsilon", &LJTypeCoeff::epsilon)
        .def_ro("sigma", &LJTypeCoeff::sigma)
        .def_ro("cutoff", &LJTypeCoeff::cutoff);

    nb::class_<R4TypeCoeff>(
        m, "R4TypeCoeff",
        "One force-shifted r^-4 attraction row: U(r) = -S [ (sigma/r)^4 - "
        "(sigma/rc)^4 + 4 (sigma/rc)^4 (r/rc - 1) ] for r < rc, else 0, so "
        "U and F vanish at rc.")
        .def(
            "__init__",
            [](R4TypeCoeff* c, int type_i, int type_j, double strength,
               double sigma, double cutoff) {
                new (c) R4TypeCoeff{type_i, type_j, strength, sigma, cutoff};
            },
            nb::arg("type_i"), nb::arg("type_j"), nb::arg("strength"),
            nb::arg("sigma"), nb::arg("cutoff"),
            "Construct with the two types, strength S [energy] (S > 0 "
            "attracts), "
            "sigma [length] and the term's own cutoff rc [length].")
        .def_ro("type_i", &R4TypeCoeff::type_i)
        .def_ro("type_j", &R4TypeCoeff::type_j)
        .def_ro("strength", &R4TypeCoeff::strength)
        .def_ro("sigma", &R4TypeCoeff::sigma)
        .def_ro("cutoff", &R4TypeCoeff::cutoff);

    nb::class_<cpu::task::PairLJCutParams>(m, "PairLJCut",
                                           "Lennard-Jones 12-6 pair potential.")
        .def(
            "__init__",
            [](cpu::task::PairLJCutParams* p, double epsilon, double sigma,
               double cutoff, double s12, double s13, double s14,
               bool energy_shift) {
                new (p) cpu::task::PairLJCutParams{
                    epsilon, sigma, cutoff, s12,   s13,
                    s14,     {},    {},     false, energy_shift};
            },
            nb::arg("epsilon"), nb::arg("sigma"), nb::arg("cutoff"),
            nb::arg("s12") = 1.0, nb::arg("s13") = 1.0, nb::arg("s14") = 1.0,
            nb::arg("energy_shift") = true,
            "Construct with epsilon [energy], sigma [length], and cutoff "
            "[length].\n\n"
            "s12, s13, s14 are the 1-2 / 1-3 / 1-4 pair weights (LAMMPS "
            "special_bonds). 0 drops that shell, a weight in (0, 1] scales "
            "that pair's energy and force. The default is (1, 1, 1). "
            "energy_shift subtracts V(cutoff) per pair (LAMMPS pair_modify "
            "shift); default on.")
        .def(
            "__init__",
            [](cpu::task::PairLJCutParams* p, std::vector<LJTypeCoeff> coeffs,
               double cutoff, double s12, double s13, double s14,
               bool energy_shift) {
                new (p) cpu::task::PairLJCutParams{
                    0.0, 0.0,   cutoff,      s12, s13, s14, std::move(coeffs),
                    {},  false, energy_shift};
            },
            nb::arg("coeffs"), nb::arg("cutoff"), nb::arg("s12") = 1.0,
            nb::arg("s13") = 1.0, nb::arg("s14") = 1.0,
            nb::arg("energy_shift") = true,
            "Per-type form: a list of LJTypeCoeff rows (types are the frame's "
            "force-field type, molpy atoms.type_id) and the global cutoff "
            "[length]. Unlisted pairs mix Lorentz-Berthelot (geometric "
            "epsilon, "
            "arithmetic sigma and cutoff, LAMMPS init_one); a type with no "
            "diagonal row has no LJ site. With energy_shift (default on) "
            "every pair is shifted at its own cutoff.")
        .def(
            "set_r4_attraction",
            [](nb::object self, std::vector<R4TypeCoeff> rows) {
                nb::cast<cpu::task::PairLJCutParams&>(self).r4 =
                    std::move(rows);
                return self;
            },
            nb::arg("rows"),
            "Replace the force-shifted r^-4 attraction rows (R4TypeCoeff) "
            "and return self. Needs the per-type form; the term is scaled by "
            "the same special-bond weight as LJ.")
        .def(
            "set_type_coeff",
            [](nb::object self, const LJTypeCoeff& row) {
                auto& p = nb::cast<cpu::task::PairLJCutParams&>(self);
                if (!p.typed())
                    throw std::logic_error(
                        "PairLJCut.set_type_coeff: this PairLJCut was built "
                        "with a single epsilon/sigma and has no type table; "
                        "construct it from a coefficient list instead");
                p.coeffs.push_back(row);
                return self;
            },
            nb::arg("row"),
            "Add or override one LJTypeCoeff row (a later row wins; a new "
            "type pair extends the table) and return self. Per-type form "
            "only, as cpu/cuda PairLJCut::set_type_coeff.")
        .def(
            "set_tail_correction",
            [](nb::object self, bool on) {
                nb::cast<cpu::task::PairLJCutParams&>(self).tail = on;
                return self;
            },
            nb::arg("on"),
            "LAMMPS `pair_modify tail yes`: add the analytic LJ tail energy "
            "etail/V and pressure beyond each pair's cutoff (default off); "
            "returns self.");

    nb::class_<cpu::task::PairLJCutCoulLongParams>(
        m, "PairLJCutCoulLong",
        "Fused typed Lennard-Jones + Ewald real-space Coulomb (LAMMPS "
        "`pair_style lj/cut/coul/long`), on CPU and CUDA MD.\n\n"
        "The real-space half of an Ewald split: the run must also install "
        "KSpacePME at the SAME alpha. Defaults match the C++ class: Amber "
        "special bonds (LJ 0/0/0.5, Coulomb 0/0/5/6) and LJ shifted at each "
        "pair's own cutoff.")
        .def(
            "__init__",
            [](cpu::task::PairLJCutCoulLongParams* p,
               std::vector<LJTypeCoeff> coeffs, double alpha, double cutoff) {
                new (p) cpu::task::PairLJCutCoulLongParams{std::move(coeffs),
                                                           alpha, cutoff};
            },
            nb::arg("coeffs"), nb::arg("alpha"), nb::arg("cutoff"),
            "Construct from LJTypeCoeff rows, the Ewald alpha [1/length] and "
            "the "
            "Coulomb (and global LJ) cutoff [length]. Unlisted pairs mix "
            "Lorentz-Berthelot with arithmetic cutoffs (LAMMPS init_one).")
        .def(
            "set_special_bonds",
            [](nb::object self, double lj12, double lj13, double lj14,
               double c12, double c13, double c14) {
                auto& p = nb::cast<cpu::task::PairLJCutCoulLongParams&>(self);
                p.lj12 = lj12;
                p.lj13 = lj13;
                p.lj14 = lj14;
                p.c12 = c12;
                p.c13 = c13;
                p.c14 = c14;
                return self;
            },
            nb::arg("lj12"), nb::arg("lj13"), nb::arg("lj14"), nb::arg("c12"),
            nb::arg("c13"), nb::arg("c14"),
            "LAMMPS `special_bonds lj a b c coul d e f`; returns self.")
        .def(
            "set_special_bonds_amber",
            [](nb::object self) {
                auto& p = nb::cast<cpu::task::PairLJCutCoulLongParams&>(self);
                p.lj12 = p.lj13 = 0.0;
                p.lj14 = 0.5;
                p.c12 = p.c13 = 0.0;
                p.c14 = 5.0 / 6.0;
                return self;
            },
            "LAMMPS `special_bonds amber` (the default); returns self.")
        .def(
            "set_energy_shift",
            [](nb::object self, bool on) {
                nb::cast<cpu::task::PairLJCutCoulLongParams&>(self)
                    .energy_shift = on;
                return self;
            },
            nb::arg("on"),
            "LAMMPS `pair_modify shift` (default on); returns self.")
        .def(
            "set_r4_attraction",
            [](nb::object self, std::vector<R4TypeCoeff> rows) {
                nb::cast<cpu::task::PairLJCutCoulLongParams&>(self).r4 =
                    std::move(rows);
                return self;
            },
            nb::arg("rows"),
            "Replace the force-shifted r^-4 attraction rows (R4TypeCoeff) "
            "and return self; scaled by the LJ special weight.")
        .def(
            "set_type_coeff",
            [](nb::object self, const LJTypeCoeff& row) {
                nb::cast<cpu::task::PairLJCutCoulLongParams&>(self)
                    .coeffs.push_back(row);
                return self;
            },
            nb::arg("row"),
            "Add or override one LJTypeCoeff row (a later row wins; a new "
            "type pair extends the table) and return self.")
        .def(
            "set_tail_correction",
            [](nb::object self, bool on) {
                nb::cast<cpu::task::PairLJCutCoulLongParams&>(self).tail = on;
                return self;
            },
            nb::arg("on"),
            "LAMMPS `pair_modify tail yes`: add the analytic LJ tail energy "
            "etail/V and pressure beyond each pair's own LJ cutoff (default "
            "off); returns self.");

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
            "Construct with well depth D [energy], width alpha [1/length], "
            "equilibrium separation r0 [length], and cutoff [length].");

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
            "Construct with Born-Mayer prefactor A [energy], softness length "
            "rho [length], dispersion coefficient C [energy*length^6], and "
            "cutoff [length].");

    nb::class_<cpu::task::PairBornParams>(
        m, "PairBorn",
        "Born-Mayer-Huggins pair table U = A exp((sigma - r)/rho) - C/r^6 + "
        "D/r^8, shifted at the cutoff (LAMMPS `pair_style born`) — the "
        "Fumi-Tosi short-range half of a molten-salt model. Add the Coulomb "
        "term as a separate evaluator.\n\n"
        "The table is per TYPE PAIR and the keys are FORCE-FIELD TYPES: the "
        "frame's per-atom type column is filled from the molpy "
        "`atoms.type_id` column (u64), not from `atomic_number` (the "
        "species), and the keys are used as-is.\n\n"
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
            "type_i / type_j (force-field types, atoms.type_id), A [energy], "
            "rho [length], sigma [length], "
            "C [energy*length^6], D [energy*length^8], plus the one shared "
            "cutoff [length].\n\n"
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
            "Construct with amplitude A [energy*length] (positive is "
            "repulsive), "
            "screening wavenumber kappa [1/length], and cutoff [length].");

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
            "Construct with the parent LJ well depth epsilon [energy] and "
            "length "
            "scale sigma [length].");

    nb::class_<cpu::task::PairDpdParams>(
        m, "PairDpd",
        "Groot-Warren dissipative particle dynamics (DPD).\n\n"
        "DPD is a COARSE-GRAINED method: a particle stands for a blob of "
        "fluid rather than an atom, so the pair force is soft (finite at "
        "r = 0) and the degrees of freedom that were coarsened away come back "
        "as a pairwise thermostat. The conservative half is "
        "F^C = a w(r) rhat with w = 1 - r/r_c, rhat the unit vector along "
        "r_i - r_j; the energy reported for it is U^C = a r_c w^2 / 2 "
        "[energy]. "
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
            "Construct with soft repulsion amplitude a [force] — a FORCE, the "
            "value of F^C at contact, not an energy — dissipative friction "
            "gamma [energy*time/length^2], thermostat temperature "
            "[temperature], cutoff r_c [length], "
            "timestep dt [time], and the random-stream seed "
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
            "Construct with spring constants K [energy/length^2] and maximum "
            "extensions R0 [length] per bond type.");

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
            "Construct with force constants K [energy], multiplicities n "
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
            "Construct with force constants K [energy/rad^2] and equilibrium "
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
               double s13, double s14, std::optional<double> coulomb) {
                if (coulomb && !(*coulomb > 0.0))
                    throw nb::value_error(
                        "PairCoulCut: coulomb must be finite and > 0");
                new (p) cpu::task::PairCoulCutParams{cutoff, s12, s13, s14,
                                                     coulomb.value_or(0.0)};
            },
            nb::arg("cutoff"), nb::arg("s12") = 1.0, nb::arg("s13") = 1.0,
            nb::arg("s14") = 1.0, nb::arg("coulomb") = nb::none(),
            "Construct with the real-space cutoff [length]. The neighbor list "
            "must be built with a cutoff >= this value. s12, s13, s14 are the "
            "1-2 / 1-3 / 1-4 Coulomb weights (LAMMPS special_bonds coul). "
            "coulomb is the constant k in k q_i q_j / r, in the run's units; "
            "None (default) uses the run's qqrd2e, as LAMMPS coul/cut does "
            "(real: 332.06371, metal: 14.399645, lj: 1).");

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
            "Per improper type: K [energy], d (+1 or -1), n (0..6). phi is the "
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
            "Construct with the Ewald splitting parameter alpha [1/length] — "
            "which must equal the KSpacePME alpha — and the real-space "
            "cutoff [length]. The neighbor list must be built with a cutoff >= "
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
            "Construct with the Ewald splitting parameter alpha [1/length], "
            "which "
            "must equal the CoulLong alpha.\n\n"
            "  grid_spacing  target real-space mesh spacing [length] (default "
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
        "Charge is spread with cardinal B-splines on the LAMMPS centred "
        "stencil (for odd orders the nearest grid point is the centre; smooth "
        "PME's one-sided stencil sits half a cell away). The mesh "
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
               double grid_spacing, int order,
               std::optional<std::array<int, 3>> mesh) {
                if (mesh)
                    for (int v : *mesh)
                        if (v <= 0)
                            throw nb::value_error(
                                "KSpacePPPM: every mesh entry must be > 0");
                new (p) cpu::task::KSpacePppmParams{
                    alpha, grid_spacing, order,
                    mesh.value_or(std::array<int, 3>{0, 0, 0})};
            },
            nb::arg("alpha"), nb::arg("grid_spacing") = double(1),
            nb::arg("order") = 6, nb::arg("mesh") = nb::none(),
            "Construct with alpha [1/length], which must equal the CoulLong "
            "alpha.\n\n"
            "  grid_spacing  target mesh spacing [length] (default 1.0). "
            "Rounded "
            "to sizes whose factors are 2, 3 and 5.\n"
            "  order         B-spline order [count, 3..12; default 6].\n"
            "  mesh          explicit (K0, K1, K2) grid points (LAMMPS "
            "kspace_modify mesh); overrides grid_spacing. Raised to 2-3-5 "
            "sizes and to at least order + 1.")
        .def_static(
            "from_accuracy",
            [](nb::object frame_obj, double accuracy, double cutoff, int order,
               std::optional<double> alpha,
               std::optional<std::array<int, 3>> mesh,
               const std::string& units_style) {
                auto frame = borrow_molpy_frame(frame_obj);
                if (!has_molpy_column(*frame, "atoms", "charge"))
                    throw nb::value_error(
                        "KSpacePPPM.from_accuracy: the frame has no 'charge' "
                        "column");
                const cpu::Frame work = cpu::from_molpy(*frame);
                longrange::PppmAccuracyRequest req;
                req.accuracy = accuracy;
                req.cutoff = cutoff;
                req.order = order;
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        req.cell[3 * r + c] = work.box.cell()(r, c);
                req.natoms = work.nlocal();
                const StorageReal* q = work.span(kAtom_charge);
                for (int i = 0; i < work.nlocal(); ++i)
                    req.qsqsum += double(q[i]) * double(q[i]);
                if (alpha)
                    req.alpha = *alpha;
                if (mesh)
                    for (int d = 0; d < 3; ++d)
                        req.mesh[d] = (*mesh)[d];
                // The run's LAMMPS row: two_charge and qqrd2e of PPPM::init.
                try {
                    req.unit_system = units::from_style(units_style);
                } catch (const std::invalid_argument& e) {
                    throw nb::value_error(e.what());
                }
                const auto r = longrange::pppm_from_accuracy(req);
                return cpu::task::KSpacePppmParams{
                    r.alpha, 1.0, r.order, {r.mesh[0], r.mesh[1], r.mesh[2]}};
            },
            nb::arg("frame"), nb::arg("accuracy"), nb::arg("cutoff"),
            nb::arg("order") = 6, nb::arg("alpha") = nb::none(),
            nb::arg("mesh") = nb::none(), nb::arg("units") = "lj",
            "LAMMPS `kspace_style pppm <accuracy>` for this frame: alpha "
            "and mesh from the relative RMS force accuracy, the real-space "
            "Coulomb cutoff, the cell and the charges (PPPM::init: "
            "set_grid_global + adjust_gewald). two_charge and qqrd2e come "
            "from `units`, the LAMMPS style of the run that will use the "
            "result (pass the driver's units_style). "
            "The mesh counts stay fixed after this call. A later cell change "
            "rebuilds the Green's function only (PPPM::setup), and does not "
            "call set_grid_global. `alpha` / `mesh` fix one of them "
            "(kspace_modify gewald / mesh). Install CoulLong(p.alpha, "
            "cutoff) with the result; the same parameters run on CPU and "
            "CUDA.")
        .def_ro("alpha", &cpu::task::KSpacePppmParams::alpha,
                "Ewald alpha [1/length].")
        .def_ro("grid_spacing", &cpu::task::KSpacePppmParams::grid_spacing,
                "Target mesh spacing [length] (unused when mesh is set).")
        .def_ro("order", &cpu::task::KSpacePppmParams::order,
                "Charge-assignment order.")
        .def_prop_ro(
            "mesh",
            [](const cpu::task::KSpacePppmParams& p) -> nb::object {
                if (p.mesh[0] <= 0)
                    return nb::none();
                return nb::cast(p.mesh);
            },
            "Explicit (K0, K1, K2), or None when derived from grid_spacing.");
#endif

    nb::class_<cpu::task::NeighborListParams>(
        m, "NeighborList", "Cell-list based neighbor list component.")
        .def(
            "__init__",
            [](cpu::task::NeighborListParams* p, double cutoff, double skin) {
                new (p) cpu::task::NeighborListParams{cutoff, skin};
            },
            nb::arg("cutoff"), nb::arg("skin") = double(-1),
            "Construct with cutoff and optional skin distance, in the run's "
            "length unit. Without a skin the run's LAMMPS units default "
            "applies (lj 0.3, real/metal/electron 2.0, si 0.001, "
            "cgs/micro/nano 0.1).");

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
            "Construct with force constants K [energy/length^2] and "
            "equilibrium "
            "distances r0 [length], one entry per bond type. The bond type is "
            "the "
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
            "Construct with force constants K [energy/rad^2] and equilibrium "
            "angles theta0 [rad], one entry per angle type (the frame's "
            "`angles` block `type_id` column).");

    nb::class_<cpu::task::AngleCosineParams>(
        m, "AngleCosine",
        "Cosine angle potential V = K(1 + cos theta) (LAMMPS `angle_style "
        "cosine`).\n\n"
        "Minimal at theta = pi: the bending stiffness of a semiflexible "
        "Kremer-Grest chain. There is no equilibrium-angle parameter.")
        .def(
            "__init__",
            [](cpu::task::AngleCosineParams* p, std::vector<double> K) {
                new (p) cpu::task::AngleCosineParams{std::move(K)};
            },
            nb::arg("K"),
            "Construct with the stiffness K [energy], one entry per angle type "
            "(the frame's `angles` block `type_id` column).");

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
            "Construct with the coefficients Cn [energy] — one inner sequence "
            "per "
            "dihedral type, its k-th entry multiplying cos^k(phi).");
}
