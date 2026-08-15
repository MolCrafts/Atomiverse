/// @file python/src/bindings/bind_minimizer.cpp
/// @brief CPU Minimize bindings aligned with the C++ categorical API.
///
/// Public surface mirrors ``cpu::minimizer::Minimize``:
///
/// @code
///   opt.set_optimizer(LBFGS{…});           // or CG / FIRE / a cell relaxer
///   opt.add_criterion(ForceConverged{…});  // optional; default in build
///   opt.set_evaluator(PairLJCut{…});       // any bound force source
///   opt.set_neighbor(NeighborList{…});
///   opt.build(); opt.run(frame, max_iter);
/// @endcode
///
/// The evaluator setters are spelled exactly as `bind_md.cpp` spells them —
/// one typed `set_evaluator_*` per component type — because both facades read
/// ONE dispatch table (`atomiverse/_dispatch.py`), and a table entry is a
/// method NAME. A force source is a force source: whatever drives MD drives a
/// geometry optimization.

#include "evaluator_setters.h"
#include "frame_guards.h"
#include "molrs_frame_bridge.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "atomiverse/cpu/frame/molrs_exchange.h"
// LBFGS and CG are constructed here because their fields ARE their
// constructor arguments; FIRE, the four cell relaxers and the three criteria
// come from `task/internal/minimize_builders.h`, and the nine force sources
// from `evaluator_setters.h` — each header owns the components it builds.
#include "atomiverse/cpu/minimize/cg.h"
#include "atomiverse/cpu/minimize/lbfgs.h"
#include "atomiverse/cpu/minimize/minimize.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/task/internal/minimize_builders.h"
#include "atomiverse/cpu/task/launcher_gate.h"
#include "atomiverse/cpu/task/task_spec.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>

#include "atomiverse/cpu/task/task_protocol.h"
#endif

using namespace ATV;

namespace {

using cpu::task::ForceConvergedParams;
using cpu::task::MinimizeBinding;
using cpu::task::MinimizeComponentEntry;
using cpu::task::MinimizeComponentParams;
using cpu::task::MinimizeCriterionParams;
using cpu::task::TaskResult;

/// @brief Is `P` one of the alternatives of the variant `V`?
///
/// Used against `MinimizeComponentParams`, which IS the Minimize launcher
/// wire's allow-list: a params type outside it cannot be mirrored into a
/// `MinimizeTask`, and the handle has to say so rather than dispatch a payload
/// missing a force source.
template <typename P, typename V>
struct is_variant_alternative : std::false_type {};

template <typename P, typename... Ts>
struct is_variant_alternative<P, std::variant<Ts...>>
    : std::bool_constant<(std::is_same_v<P, Ts> || ...)> {};

template <typename P>
inline constexpr bool wire_carries_v =
    is_variant_alternative<P, MinimizeComponentParams>::value;

/// Bound Minimize driver + TaskSpec mirror of its registrations.
struct MinimizeHandle {
    cpu::minimizer::Minimize driver;
    std::vector<MinimizeComponentEntry> mirror;
    MinimizeCriterionParams convergence{};
    bool user_criterion = false;

    /// @brief Refuse a configuration change once the driver has run.
    ///
    /// `Driver::build()` consumes its pending registrations, so the execution
    /// plan is compiled exactly once and a setter called after the first
    /// `run()` cannot reach it: it is either silently dropped or detonates the
    /// next `run()` with a component-validation error naming the wrong
    /// culprit. Same guard, same wording as `MdHandle::require_configurable_`
    /// and `bind_mc.cpp`'s.
    ///
    /// Re-RUNNING stays legal — `cpu::minimizer::Minimize` is re-runnable by
    /// contract (each `run()` is a complete optimization), and chunked
    /// optimization is the normal way to interleave analysis.
    ///
    /// @param setter Name of the setter being refused, for the message.
    /// @throws std::runtime_error once `run()` has completed.
    void require_configurable_(const char* setter) const {
        if (ran_)
            throw std::runtime_error(
                std::string(setter)
                + ": Minimize has already run; construct a new Minimize to "
                  "change the configuration.");
    }

    template <typename Params, typename Component>
    void set_categorical(const char* setter, const Params& p,
                         Component component) {
        require_configurable_(setter);
        using P = std::decay_t<Params>;
        // Named BEFORE the move, and only for the types the wire cannot
        // carry — `if constexpr` keeps the common path free of the string.
        if constexpr (!wire_carries_v<P>) {
            unmirrored_.push_back(component.name());
            // The mirror is the charge guard's evidence, and an unmirrored
            // registration leaves none behind — so an electrostatic one is
            // remembered here as a typed fact, not re-derived from that
            // display name. See `require_charge_column`.
            if constexpr (is_coulomb_params_v<P>)
                unmirrored_coulomb_ = true;
        }

        if constexpr (std::is_same_v<P, cpu::task::PairLJCutParams>
                      || std::is_same_v<P, cpu::task::PairCoulCutParams>
                      || std::is_same_v<P, cpu::task::CoulLongParams>
                      || std::is_same_v<P, cpu::task::BondHarmonicParams>
                      || std::is_same_v<P, cpu::task::AngleHarmonicParams>
                      || std::is_same_v<P,
                                        cpu::task::DihedralMultiHarmonicParams>)
            driver.set_evaluator(std::move(component));
#ifdef ATV_HAS_KSPACE
        else if constexpr (std::is_same_v<P, cpu::task::KSpacePmeParams>)
            // Move-only (it owns FFTW plans), so the component arrives here
            // already moved from the setter's prvalue.
            driver.set_evaluator(std::move(component));
#endif
#ifdef ATV_HAS_SEMIEMPIRICAL
        else if constexpr (std::is_same_v<P, cpu::task::AM1EvaluatorParams>)
            driver.set_evaluator(std::move(component));
#endif
#ifdef ATV_HAS_PW
        else if constexpr (std::is_same_v<P, cpu::task::PwEvaluatorParams>)
            // Move-only for the same reason as KSpacePME.
            driver.set_evaluator(std::move(component));
#endif
        else if constexpr (std::is_same_v<P, cpu::task::NeighborListParams>)
            driver.set_neighbor(std::move(component));
        else if constexpr (
            std::is_same_v<P, cpu::task::LBFGSParams>
            || std::is_same_v<P, cpu::task::CGParams>
            || std::is_same_v<P, cpu::task::FIREParams>
            || std::is_same_v<P, cpu::task::IsotropicCellRelaxParams>
            || std::is_same_v<P, cpu::task::AnisotropicCellRelaxParams>
            || std::is_same_v<P, cpu::task::TriclinicCellRelaxParams>
            || std::is_same_v<P, cpu::task::JointIsotropicRelaxParams>)
            // The optimizer slot is singular: LBFGS, CG, FIRE and the four
            // variable-cell relaxers are alternatives, never a stack.
            driver.set_optimizer(std::move(component));
        else
            static_assert(cpu::task::detail::unsupported_component_v<P>);

        if constexpr (wire_carries_v<P>)
            mirror.push_back(MinimizeComponentEntry{p, std::nullopt});
    }

    /// Escape hatch: explicit phase + order (maps to Driver::on(phase).add).
    template <typename Params, typename Component>
    void add_on_phase(const Params& p, Component component,
                      MinimizeBinding binding) {
        require_configurable_("on(phase).add");
        driver.on(binding.phase).add(std::move(component), binding.order);
        mirror.push_back(MinimizeComponentEntry{p, binding});
    }

    /// @brief Install the single stop policy (`Driver::add_criterion`).
    ///
    /// The same installer the launcher replay uses, so a criterion cannot
    /// reach the driver differently from Python than from a `MinimizeTask`.
    /// It CONSTRUCTS the C++ criterion, which is what makes its own invariants
    /// (`ptol > 0`, `stol > 0`, at least one unmasked axis) fire at this call
    /// site rather than at build.
    ///
    /// @param c Typed criterion mirror — force, pressure or stress.
    void add_criterion(const MinimizeCriterionParams& c) {
        require_configurable_("add_criterion");
        cpu::task::internal::add_minimize_criterion(driver, c);
        convergence = c;
        user_criterion = true;
    }

    /// @brief Refuse an MPI-launcher dispatch this handle cannot mirror.
    ///
    /// The launcher re-sends the configuration as a `TaskSpec` per run, and
    /// the Minimize payload's component allow-list (`MinimizeComponentParams`)
    /// is narrower than the set of force sources bound here — every Coulomb,
    /// bonded and electronic-structure evaluator is MD-side wire vocabulary.
    /// Dispatching anyway would run a Minimize with a silently missing force
    /// term, so it fails loud with the components it cannot carry named.
    ///
    /// @throws std::runtime_error when a registered component has no mirror.
    void require_wire_representable_() const {
        if (unmirrored_.empty())
            return;
        std::string names;
        for (const std::string& name : unmirrored_) {
            if (!names.empty())
                names += ", ";
            names += name;
        }
        throw std::runtime_error(
            "Minimize.run: the MPI launcher task payload cannot carry " + names
            + " — run this configuration without the launcher, or use a "
              "force source the Minimize payload mirrors (PairLJCut).");
    }

    /// @brief Reject a chargeless frame before an electrostatic run starts.
    ///
    /// Same guard, same rationale as `MdHandle::require_charge_column`
    /// (`bind_md.cpp`), because both handles take the same nine force sources:
    /// `charge` is an *optional* per-atom column at the C++ boundary, and
    /// `cpu::from_molrs` zero-fills an absent one, so past that point a
    /// forgotten charge column is indistinguishable from a deliberately
    /// neutral system — the optimization converges instantly on a Hamiltonian
    /// with no electrostatics in it. The molrs Frame is the last place the
    /// absence is visible, so the refusal happens here, with a how-to.
    ///
    /// Intentional zeros stay expressible: insert an explicit zero `charge`
    /// column and the check passes.
    ///
    /// Both registration paths count. The mirror is the whole story on the MD
    /// side, but `MinimizeComponentParams` carries no Coulomb alternative, so
    /// every electrostatic evaluator bound here lands in `unmirrored_` and is
    /// remembered by `unmirrored_coulomb_` instead — a CoulLong the launcher
    /// would refuse must still trip this guard on the serial path, which is
    /// the path that actually ran it.
    ///
    /// @param frame Borrowed molrs Frame about to be optimized.
    /// @throws std::runtime_error when a Coulomb evaluator is registered and
    ///         the frame's `atoms` block has no `charge` column.
    void require_charge_column(const molrs::FrameRef& frame) const {
        const bool coulomb =
            unmirrored_coulomb_
            || std::any_of(mirror.begin(), mirror.end(),
                           is_coulomb_entry<MinimizeComponentEntry>);
        if (!coulomb || has_molrs_column(frame, "atoms", "charge"))
            return;
        throw std::runtime_error(
            "this Minimize run registers a Coulomb evaluator but the "
            "molrs.Frame has no per-atom 'charge' column, so every charge "
            "would be taken as 0 e. Add the charges [e] before run(), e.g. "
            "frame[\"atoms\"].insert(\"charge\", q).");
    }

    TaskResult run(nb::object frame_obj, int max_iter) {
        const cpu::task::LauncherGate gate;
        if (!gate.active()) {
            auto frame = borrow_molrs_frame(frame_obj);
            require_charge_column(*frame);
            int iterations = 0;
            {
                nb::gil_scoped_release release;
                iterations = driver.run(*frame, max_iter);
            }
            // Armed only on a COMPLETED run: a run that threw left no
            // compiled plan behind, so the caller must still be able to
            // finish configuring.
            ran_ = true;
            return TaskResult{iterations, driver.converged(),
                              driver.stop_reason()};
        }

#ifdef ATV_HAS_MPI
        require_wire_representable_();
        // TaskSpec always carries a convergence policy (defaults if unset).
        const MinimizeCriterionParams conv =
            user_criterion ? convergence : MinimizeCriterionParams{};
        const cpu::task::TaskSpec spec{cpu::task::MinimizeTask{
            mirror, static_cast<std::int64_t>(max_iter), conv}};
        auto frame = borrow_molrs_frame(frame_obj);
        require_charge_column(*frame);
        cpu::Frame work = cpu::from_molrs(*frame);
        cpu::task::DispatchResult dispatched{cpu::Frame(0), {}};
        {
            nb::gil_scoped_release release;
            cpu::task::TaskDispatcher dispatcher(MPI_COMM_WORLD,
                                                 cpu::task::kProtocolVersion,
                                                 /*do_handshake=*/false);
            dispatched = dispatcher.dispatch(spec, std::move(work));
        }
        cpu::to_molrs(dispatched.frame, *frame);
        ran_ = true;
        return std::move(dispatched.control);
#else
        throw std::runtime_error(
            "launcher mode requires an MPI-enabled Atomiverse build");
#endif
    }

  private:
    /// `Component::name()` of every registration the Minimize wire cannot
    /// mirror; see `require_wire_representable_`.
    std::vector<std::string> unmirrored_;
    /// True once an unmirrored registration was electrostatic; see
    /// `require_charge_column`.
    bool unmirrored_coulomb_ = false;
    /// True once a `run()` has completed; see `require_configurable_`.
    bool ran_ = false;
};

struct MinimizePhaseBuilder {
    MinimizeHandle* handle;
    ATV::minimizer::Phase phase;
};

} // namespace

void bind_minimizer(nb::module_& m) {
    nb::enum_<minimizer::Phase>(m, "MinimizePhase",
                                "Geometry-minimization phase sequence.")
        .value("search_direction", minimizer::Phase::search_direction)
        .value("line_search", minimizer::Phase::line_search)
        .value("force", minimizer::Phase::force)
        .value("convergence_check", minimizer::Phase::convergence_check)
        .value("end_of_step", minimizer::Phase::end_of_step);

    // ── Optimizers (the singular set_optimizer slot) ──

    nb::class_<cpu::task::LBFGSParams>(
        m, "LBFGS",
        "Limited-memory BFGS optimizer — the usual first choice for a "
        "fixed-cell geometry optimization.\n\n"
        "BFGS (Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton method: it "
        "never builds the matrix of second derivatives, it reconstructs that "
        "matrix's action on the gradient from how the gradient has changed "
        "over the last few steps. The limited-memory variant keeps only the "
        "last `memory` such pairs, then backtracks along the resulting "
        "direction until the energy has dropped enough (the Armijo "
        "condition). Fastest of the three fixed-cell optimizers near a "
        "minimum, where that curvature estimate is meaningful.\n\n"
        "Convergence tolerances are NOT this component's business: they live "
        "on the criterion (ForceConverged), which is what makes LBFGS, CG and "
        "FIRE interchangeable under one stop policy.")
        .def(
            "__init__",
            [](cpu::task::LBFGSParams* p, int memory, double alpha0,
               int max_ls) {
                new (p) cpu::task::LBFGSParams{memory, alpha0, max_ls};
            },
            nb::arg("memory") = 5, nb::arg("alpha0") = 0.01,
            nb::arg("max_ls") = 10,
            "Construct with the C++ defaults:\n"
            "  memory   stored curvature pairs [count]; more buys a better "
            "estimate at a linear cost in storage and work.\n"
            "  alpha0   initial step size [A^2/eV] — the trial displacement "
            "per unit force, before the line search backtracks.\n"
            "  max_ls   cap on backtracking attempts within one step "
            "[count].");

    nb::class_<cpu::task::CGParams>(
        m, "CG",
        "Polak-Ribiere+ conjugate-gradient optimizer (LAMMPS `min_style "
        "cg`).\n\n"
        "The search direction is steepest descent plus a beta-weighted memory "
        "of the previous direction, with beta clamped at zero so an "
        "ill-conditioned region restarts into plain steepest descent.\n\n"
        "Lower memory than LBFGS, and the better choice when L-BFGS keeps "
        "discarding its curvature history.\n\n"
        "Convergence tolerances are NOT this component's business: they live "
        "on the criterion (ForceConverged), which is exactly what makes "
        "LBFGS, CG and FIRE interchangeable under one stop policy.\n\n"
        "The C++ line-search selector (`CG::line_search`) is deliberately not "
        "mirrored here: it is a post-construction setter rather than a "
        "constructor argument, so this binding — which carries constructor "
        "arguments only — leaves it at its C++ default (strong Wolfe, "
        "c1 = 1e-4, c2 = 0.1, max_step = 0.2 A, min_alpha = 1e-8). A line "
        "search is the 1-D search for how far to move along the chosen "
        "direction; the Wolfe conditions are the standard pair of tests "
        "(enough energy drop, and enough flattening of the slope) that decide "
        "when a trial step is good enough.")
        .def(
            "__init__",
            [](cpu::task::CGParams* p, double alpha0) {
                new (p) cpu::task::CGParams{alpha0};
            },
            nb::arg("alpha0") = 0.01,
            "Construct with the trial step size alpha0 [A^2/eV] taken along "
            "the conjugate direction before the line search refines it.");

    nb::class_<cpu::task::FIREParams>(
        m, "FIRE",
        "Fast Inertial Relaxation Engine, FIRE 1.0 (Bitzek et al., Phys. Rev. "
        "Lett. 97, 170201 (2006)).\n\n"
        "Damped dynamics rather than a line-search method: it integrates a "
        "fictitious unit-mass trajectory, mixes the velocity towards the "
        "force direction by alpha, grows the pseudo-timestep while the power "
        "F.v stays positive, and zeroes the velocity outright when it turns "
        "negative. Cheap per step and robust far from the minimum.\n\n"
        "The defaults are ASE's, which assume O(1) eV/A forces. A soft "
        "system — an Ar Lennard-Jones cluster has a residual of ~1e-2 eV/A — "
        "wants a larger pseudo-time scale, e.g. dt0 = 1.0, dt_max = 20.0 "
        "(regressions/minimizer/fire_lj_dimer).")
        .def(
            "__init__",
            [](cpu::task::FIREParams* p, double dt0, double dt_max, int n_min,
               double f_inc, double f_dec, double alpha0, double f_alpha,
               double max_step) {
                new (p) cpu::task::FIREParams{dt0,   dt_max, n_min,   f_inc,
                                              f_dec, alpha0, f_alpha, max_step};
            },
            nb::arg("dt0") = 0.1, nb::arg("dt_max") = 1.0, nb::arg("n_min") = 5,
            nb::arg("f_inc") = 1.1, nb::arg("f_dec") = 0.5,
            nb::arg("alpha0") = 0.1, nb::arg("f_alpha") = 0.99,
            nb::arg("max_step") = 0.2,
            "Construct with the FIRE 1.0 knobs, all defaulted to the C++ "
            "(ASE-aligned) values. Note alpha0 here is a mixing FRACTION, not "
            "the step size LBFGS and CG call alpha0:\n"
            "  dt0       initial pseudo-timestep [fs].\n"
            "  dt_max    ceiling on the pseudo-timestep [fs].\n"
            "  n_min     accelerating steps before dt grows [count].\n"
            "  f_inc     dt growth factor while accelerating "
            "(dimensionless, > 1).\n"
            "  f_dec     dt shrink factor when F.v turns non-positive "
            "(dimensionless, < 1).\n"
            "  alpha0    initial velocity-mixing parameter alpha_start "
            "(dimensionless, in [0, 1]): the fraction of the velocity "
            "redirected onto the force direction each step.\n"
            "  f_alpha   decay factor applied to alpha while accelerating "
            "(dimensionless, < 1).\n"
            "  max_step  cap on the per-atom displacement of one step [A].");

    nb::class_<cpu::task::IsotropicCellRelaxParams>(
        m, "IsotropicCellRelax",
        "Isotropic variable-cell relaxation optimizer.\n\n"
        "Reads the extensive Voigt virial W [eV] the evaluators wrote to the "
        "frame, forms the isotropic pressure P = tr(W) / (3V) [eV/A^3] from "
        "the cell volume V, and scales the whole cell — the atom coordinates "
        "riding along affinely — towards P_target. It never moves an atom "
        "RELATIVE to the cell, so pair it with a separate geometry relaxation "
        "or use JointIsotropicRelax.\n\n"
        "MECHANICS (shared by all four relaxers). One iteration computes the "
        "scale factor s = 1 + alpha0 (P - P_target) / bulk_mod, clamped to "
        "|s - 1| <= max_scale, then applies it to the box and to every "
        "coordinate. The pressure it uses is the one left on the frame by the "
        "PREVIOUS iteration's force phase, because the minimizer phase order "
        "is search_direction -> line_search -> force -> convergence_check. "
        "Iteration 0 has had no force phase of its own, so it reads whatever "
        "virial the frame arrived with — zero on a freshly built frame, which "
        "with the default P_target = 0 makes that first scale exactly 1, so "
        "the cell starts moving on iteration 1. Budget one extra iteration "
        "accordingly.\n\n"
        "STRESS SOURCE (the honest gate). A relaxer can only follow a virial "
        "something actually wrote. The classical evaluators — PairLJCut, "
        "PairCoulCut, CoulLong/KSpacePME and the bonded styles — write it by "
        "default. AM1Evaluator writes none at all (it is a molecular method "
        "with no cell), and set_evaluator_pw installs a gradient-only "
        "plane-wave copy, so neither drives a cell step from Python: the "
        "virial stays zero, the scale stays 1, and a PressureConverged at "
        "P_target = 0 reports success on an untouched cell.\n\n"
        "A variable-cell run does not terminate on forces: add "
        "PressureConverged (or StressConverged), not ForceConverged. The "
        "relaxer publishes its pressure residual [eV/A^3] into the metric "
        "slot ForceConverged reads as a force [eV/A], so that pairing does "
        "not fail loudly — it silently compares the wrong quantity.")
        .def(
            "__init__",
            [](cpu::task::IsotropicCellRelaxParams* p, double P_target,
               double alpha0, double max_scale, double bulk_mod) {
                new (p) cpu::task::IsotropicCellRelaxParams{
                    P_target, alpha0, max_scale, bulk_mod};
            },
            nb::arg("P_target") = 0.0, nb::arg("alpha0") = 0.05,
            nb::arg("max_scale") = 0.05, nb::arg("bulk_mod") = 1.0,
            "Construct with the target pressure P_target [eV/A^3] (0 is a "
            "relaxed cell; 1 bar ~ 6.2415e-7 eV/A^3).\n\n"
            "  alpha0     relative scale-step gain, i.e. how strongly the "
            "pressure residual translates into a requested |s-1|.\n"
            "  max_scale  cap on |s-1| per step — how far one iteration may "
            "breathe the cell.\n"
            "  bulk_mod   soft bulk modulus [eV/A^3] dividing the step; "
            "larger means a gentler cell change per iteration.");

    nb::class_<cpu::task::AnisotropicCellRelaxParams>(
        m, "AnisotropicCellRelax",
        "Orthorhombic variable-cell relaxation optimizer.\n\n"
        "The three box edges scale independently from their own diagonal "
        "stress components W_dd / V, so the cell changes shape as well as "
        "volume. The tilt factors stay put — shear is TriclinicCellRelax. "
        "Step mechanics, the iteration-0 zero-virial read and the stress-"
        "source gate are IsotropicCellRelax's; read those first.\n\n"
        "The per-axis mask is the CP2K `CELL_OPT` / `CONSTRAINT` twin: a "
        "False axis is never scaled and never enters the residual. Mask the "
        "same axes on a StressConverged criterion, or a constrained "
        "direction's reaction stress blocks convergence forever.")
        .def(
            "__init__",
            [](cpu::task::AnisotropicCellRelaxParams* p, double P_target,
               double alpha0, double max_scale, double bulk_mod, bool relax_x,
               bool relax_y, bool relax_z) {
                new (p) cpu::task::AnisotropicCellRelaxParams{
                    P_target, alpha0,  max_scale, bulk_mod,
                    relax_x,  relax_y, relax_z};
            },
            nb::arg("P_target") = 0.0, nb::arg("alpha0") = 0.05,
            nb::arg("max_scale") = 0.05, nb::arg("bulk_mod") = 1.0,
            nb::arg("relax_x") = true, nb::arg("relax_y") = true,
            nb::arg("relax_z") = true,
            "Construct with the target pressure P_target [eV/A^3], the same "
            "on every relaxed axis, plus the per-axis scale gain alpha0, the "
            "per-step cap max_scale on |s-1|, the soft bulk modulus bulk_mod "
            "[eV/A^3] and the relax_x / relax_y / relax_z mask (False freezes "
            "that box edge).");

    nb::class_<cpu::task::TriclinicCellRelaxParams>(
        m, "TriclinicCellRelax",
        "Full-cell (shearable) stress-driven relaxation optimizer.\n\n"
        "A strain tensor eps proportional to the stress is formed from the "
        "WHOLE virial, off-diagonals included (eps_ab = alpha0 W_ab / "
        "(bulk_mod V), the off-diagonals halved), and the cell matrix h is "
        "updated as h <- (I + eps) h with the atoms mapped affinely. Strain "
        "is the dimensionless relative deformation of the cell, so the cap is "
        "max_strain rather than the cubic relaxers' max_scale: the capped "
        "quantity is a strain component, not a length ratio. Step mechanics, "
        "the iteration-0 zero-virial read and the stress-source gate are "
        "IsotropicCellRelax's.\n\n"
        "There is no P_target — this relaxer drives the cell towards ZERO "
        "stress (the 0 K geometry), so its matching criterion is "
        "StressConverged.")
        .def(
            "__init__",
            [](cpu::task::TriclinicCellRelaxParams* p, double alpha0,
               double max_strain, double bulk_mod) {
                new (p) cpu::task::TriclinicCellRelaxParams{alpha0, max_strain,
                                                            bulk_mod};
            },
            nb::arg("alpha0") = 0.02, nb::arg("max_strain") = 0.05,
            nb::arg("bulk_mod") = 1.0,
            "Construct with the strain gain alpha0, the per-step cap "
            "max_strain on |eps_ab|, and the soft modulus bulk_mod [eV/A^3] "
            "dividing the step.");

    nb::class_<cpu::task::JointIsotropicRelaxParams>(
        m, "JointIsotropicRelax",
        "Joint atom + isotropic cell steepest-descent optimizer.\n\n"
        "Each iteration does both halves of a variable-cell optimization: a "
        "steepest-descent atom step along the unit force, then an isotropic "
        "cell scale from the virial pressure. `atom_step` is what makes it "
        "joint; the other four knobs — and the step mechanics, the "
        "iteration-0 zero-virial read and the stress-source gate — are "
        "IsotropicCellRelax's.\n\n"
        "Terminating it honestly wants both a force and a pressure test, and "
        "add_criterion is plural — the driver stops on the first criterion "
        "that fires (severity Stop > Converged > Continue).")
        .def(
            "__init__",
            [](cpu::task::JointIsotropicRelaxParams* p, double P_target,
               double atom_step, double alpha0, double max_scale,
               double bulk_mod) {
                new (p) cpu::task::JointIsotropicRelaxParams{
                    P_target, atom_step, alpha0, max_scale, bulk_mod};
            },
            nb::arg("P_target") = 0.0, nb::arg("atom_step") = 0.01,
            nb::arg("alpha0") = 0.05, nb::arg("max_scale") = 0.05,
            nb::arg("bulk_mod") = 1.0,
            "Construct with the target pressure P_target [eV/A^3], the atom "
            "displacement atom_step [A] taken along the unit force each "
            "iteration, the relative cell-scale gain alpha0, the per-step cap "
            "max_scale on |s-1| and the soft bulk modulus bulk_mod [eV/A^3].");

    // ── Criteria (the plural add_criterion slot) ──

    nb::class_<ForceConvergedParams>(
        m, "ForceConverged",
        "Five-tolerance stop policy for a FIXED-cell minimization (the "
        "build-time default).\n\n"
        "All five tests must hold at once: a geometry is converged when the "
        "forces are small, when the last step moved almost nothing, and when "
        "the energy stopped changing. The displacement pair is skipped on an "
        "iteration that has no previous geometry to difference against.\n\n"
        "It judges the metrics the optimizer publishes, which is why the same "
        "criterion serves LBFGS, CG and FIRE. It is the WRONG policy for a "
        "variable-cell relaxer — that moves the box, not the atoms, and can "
        "sit at zero force while far from the target pressure; use "
        "PressureConverged or StressConverged there.")
        .def(
            "__init__",
            [](ForceConvergedParams* p, double ftol, double rms_ftol,
               double max_dtol, double rms_dtol, double etol) {
                new (p) ForceConvergedParams{ftol, rms_ftol, max_dtol, rms_dtol,
                                             etol};
            },
            nb::arg("ftol") = 0.05, nb::arg("rms_ftol") = 0.03,
            nb::arg("max_dtol") = 0.002, nb::arg("rms_dtol") = 0.001,
            nb::arg("etol") = 1e-5,
            "Construct with the five tolerances, defaulted to the C++ "
            "(LAMMPS/ASE-shaped) values:\n"
            "  ftol      largest per-atom force magnitude [eV/A].\n"
            "  rms_ftol  root-mean-square force over all atoms [eV/A].\n"
            "  max_dtol  largest per-atom displacement of the last step [A].\n"
            "  rms_dtol  root-mean-square displacement of that step [A].\n"
            "  etol      energy change between the last two iterations [eV].");

    nb::class_<cpu::task::PressureConvergedParams>(
        m, "PressureConverged",
        "Isotropic-pressure stop policy for a variable-cell run.\n\n"
        "The virial is the configurational stress an evaluator accumulates "
        "alongside the forces; the frame carries it as the 'stress' metadata "
        "entry in Voigt order — the six independent components of a symmetric "
        "3x3 tensor, listed (xx, yy, zz, xy, xz, yz). It is EXTENSIVE: a "
        "whole-cell energy W [eV], not a stress, for every evaluator that "
        "writes it. This criterion therefore forms P = (W_xx + W_yy + W_zz) / "
        "(3V) [eV/A^3] from the box volume V and declares convergence once "
        "|P - P_target| falls below ptol.\n\n"
        "StressConverged reads the identical entry and divides by the "
        "identical volume; it differs only in judging each diagonal component "
        "on its own — with an axis mask — instead of their mean.")
        .def(
            "__init__",
            [](cpu::task::PressureConvergedParams* p, double P_target,
               double ptol) {
                new (p) cpu::task::PressureConvergedParams{P_target, ptol};
            },
            nb::arg("P_target") = 0.0, nb::arg("ptol") = 1e-5,
            "Construct with the target pressure P_target [eV/A^3] (0 is a "
            "relaxed cell; negative is a tensile setpoint) and the absolute "
            "tolerance ptol [eV/A^3], which must be > 0 — a non-positive "
            "tolerance can never be met, and add_criterion raises ValueError "
            "on it.");

    nb::class_<cpu::task::StressConvergedParams>(
        m, "StressConverged",
        "Per-axis diagonal-stress stop policy for a variable-cell run.\n\n"
        "Same input as PressureConverged — the extensive Voigt virial W [eV] "
        "on the frame's 'stress' entry — divided by the same cell volume V, "
        "but judged per axis: it compares the intensive stress "
        "sigma_dd = W_dd / V [eV/A^3] of every UNMASKED axis d against "
        "sigma_target and converges once the largest of those deviations "
        "falls below stol.\n\n"
        "Pair the mask with AnisotropicCellRelax's relax_x / relax_y / "
        "relax_z: mask the same axes on both and a constrained direction's "
        "reaction stress — the stress a frozen axis pushes back with, which "
        "no cell step is driving to the target — cannot block convergence "
        "(the CP2K `CELL_OPT` `CONSTRAINT` twin).\n\n"
        "Masking every axis, or a non-positive stol, makes add_criterion "
        "raise ValueError (the C++ criterion is constructed there, so its "
        "invariants fire at the call site) — a criterion that judges nothing "
        "would declare victory on iteration zero.")
        .def(
            "__init__",
            [](cpu::task::StressConvergedParams* p, double sigma_target,
               double stol, bool check_x, bool check_y, bool check_z) {
                new (p) cpu::task::StressConvergedParams{
                    sigma_target, stol, check_x, check_y, check_z};
            },
            nb::arg("sigma_target") = 0.0, nb::arg("stol") = 1e-4,
            nb::arg("check_x") = true, nb::arg("check_y") = true,
            nb::arg("check_z") = true,
            "Construct with the target diagonal stress sigma_target [eV/A^3] "
            "(0 is a relaxed cell), the absolute tolerance stol [eV/A^3] "
            "(must be > 0) and the check_x / check_y / check_z axis mask.");

    nb::class_<TaskResult>(m, "MinimizeResult",
                           "Loop-control result; state is written to Frame.")
        .def_ro("iterations", &TaskResult::iterations,
                "Iterations actually executed [count], at most max_iter.")
        .def_ro("converged", &TaskResult::converged,
                "True when a criterion ended the run, False when the budget "
                "did.")
        .def_ro("stop_reason", &TaskResult::stop_reason,
                "Name of the criterion that ended the run; empty when the "
                "iteration budget ended it.");

    // ── Minimize driver (categorical setters) ──

    nb::class_<MinimizeHandle> minimize(
        m, "Minimize",
        "CPU geometry-minimization driver.\n\n"
        "Configure it fully before the first run(): the execution plan is "
        "compiled exactly once, so every setter — and on(phase) — raises "
        "RuntimeError afterwards. A different configuration means a new "
        "Minimize object. Running again with the same configuration stays "
        "legal and re-optimizes from the frame's current state.");
    minimize
        .def(nb::init<>(),
             "Default construct; ForceConverged is auto-installed at build "
             "unless add_criterion is called.")
        .def(
            "set_optimizer",
            [](MinimizeHandle& self, const cpu::task::LBFGSParams& p) {
                self.set_categorical(
                    "set_optimizer", p,
                    cpu::minimizer::LBFGS{p.memory, p.alpha0, p.max_ls});
            },
            nb::arg("comp"), "LAMMPS-style set_optimizer (LBFGS).")
        .def(
            "set_optimizer",
            [](MinimizeHandle& self, const cpu::task::CGParams& p) {
                self.set_categorical("set_optimizer", p,
                                     cpu::minimizer::CG{p.alpha0});
            },
            nb::arg("comp"), "LAMMPS-style set_optimizer (CG).")
        .def(
            "set_optimizer",
            [](MinimizeHandle& self, const cpu::task::FIREParams& p) {
                // Built by the same params-mirror builder the launcher replay
                // uses, so a knob cannot reach one path and not the other.
                self.set_categorical("set_optimizer", p,
                                     cpu::task::internal::tuned_fire(p));
            },
            nb::arg("comp"), "LAMMPS-style set_optimizer (FIRE).")
        .def(
            "set_optimizer",
            [](MinimizeHandle& self,
               const cpu::task::IsotropicCellRelaxParams& p) {
                self.set_categorical(
                    "set_optimizer", p,
                    cpu::task::internal::tuned_isotropic_cell_relax(p));
            },
            nb::arg("comp"),
            "LAMMPS-style set_optimizer (IsotropicCellRelax). Add "
            "PressureConverged rather than ForceConverged: a variable-cell "
            "run terminates on pressure.")
        .def(
            "set_optimizer",
            [](MinimizeHandle& self,
               const cpu::task::AnisotropicCellRelaxParams& p) {
                self.set_categorical(
                    "set_optimizer", p,
                    cpu::task::internal::tuned_anisotropic_cell_relax(p));
            },
            nb::arg("comp"),
            "LAMMPS-style set_optimizer (AnisotropicCellRelax).")
        .def(
            "set_optimizer",
            [](MinimizeHandle& self,
               const cpu::task::TriclinicCellRelaxParams& p) {
                self.set_categorical(
                    "set_optimizer", p,
                    cpu::task::internal::tuned_triclinic_cell_relax(p));
            },
            nb::arg("comp"), "LAMMPS-style set_optimizer (TriclinicCellRelax).")
        .def(
            "set_optimizer",
            [](MinimizeHandle& self,
               const cpu::task::JointIsotropicRelaxParams& p) {
                self.set_categorical(
                    "set_optimizer", p,
                    cpu::task::internal::tuned_joint_isotropic_relax(p));
            },
            nb::arg("comp"),
            "LAMMPS-style set_optimizer (JointIsotropicRelax).");

    // The set_evaluator slot is plural: every call appends another
    // energy/force source and they are summed. The typed spellings are what
    // Python's single `Minimize.set_evaluator` dispatches to by component
    // type — the SAME method names `MD` exposes, registered from the one
    // header that knows the nine force sources (`evaluator_setters.h`),
    // because both facades read one dispatch table.
    bind_evaluator_setters(
        minimize,
        // set_evaluator_am1 — the geometry-optimization reading of the same
        // component.
        "Append the AM1 semiempirical evaluator as the force source of a "
        "geometry optimization — one self-consistent-field solve per "
        "iteration, forces from the fixed-density central finite "
        "difference (step 1e-4 A). The C++ twin is "
        "regressions/semiempirical/am1_water_minimize. Takes no "
        "NeighborList (AM1 is molecular, not a pair style) and reads the "
        "frame's 'atomic_number' column [Z].\n\n"
        "The driver installs a FRESH gradient-enabled copy of the five "
        "SCF parameters: the object you pass keeps its own state, so "
        "last_result() on it is NOT populated by run().",
        // set_evaluator_pw — the geometry-optimization reading of the same
        // component.
        "Append the plane-wave DFT evaluator as the force source of a "
        "geometry optimization: the electrons are re-converged at every "
        "geometry, with analytic ionic forces taken at the converged "
        "density. Budget it deliberately — an iteration costs TWO "
        "self-consistent-field solves. Takes no NeighborList and needs a "
        "box.\n\n"
        "The driver installs a FRESH gradient-enabled copy of the fifteen "
        "method parameters: the object you pass keeps its own state.");

    minimize
        .def(
            "set_neighbor",
            [](MinimizeHandle& self, const cpu::task::NeighborListParams& p) {
                self.set_categorical("set_neighbor", p,
                                     cpu::md::NeighborList<>{p.cutoff, p.skin});
            },
            nb::arg("comp"), "LAMMPS-style set_neighbor.")
        .def(
            "add_criterion",
            [](MinimizeHandle& self, const ForceConvergedParams& c) {
                self.add_criterion(MinimizeCriterionParams{c});
            },
            nb::arg("criterion"),
            "Attach the five-tolerance force stop policy (replaces the build "
            "default).")
        .def(
            "add_criterion",
            [](MinimizeHandle& self,
               const cpu::task::PressureConvergedParams& c) {
                self.add_criterion(MinimizeCriterionParams{c});
            },
            nb::arg("criterion"),
            "Attach the isotropic-pressure stop policy of a variable-cell "
            "run (replaces the build default).")
        .def(
            "add_criterion",
            [](MinimizeHandle& self,
               const cpu::task::StressConvergedParams& c) {
                self.add_criterion(MinimizeCriterionParams{c});
            },
            nb::arg("criterion"),
            "Attach the per-axis diagonal-stress stop policy (replaces the "
            "build default).")
        .def(
            "on",
            [](MinimizeHandle& self, minimizer::Phase phase) {
                // Refused HERE rather than at the builder's add(): on() is the
                // registration entry point, so a caller who has already run
                // gets the contract before constructing a component.
                self.require_configurable_("on");
                return MinimizePhaseBuilder{&self, phase};
            },
            nb::arg("phase"),
            "Escape hatch: explicit phase binding (prefer categorical "
            "setters).")
        .def(
            "build", [](MinimizeHandle& self) { self.driver.build(); },
            "Compile the execution plan from the registrations made so far. "
            "The `atomiverse.minimizer.Minimize` facade calls this for you "
            "before the first run(); at this level it is explicit.")
        .def("run", &MinimizeHandle::run, nb::arg("frame"), nb::arg("max_iter"),
             "Optimize a molrs.Frame for at most max_iter iterations. GIL "
             "released during compute.\n\n"
             "Geometry, energy, forces, the stress entry and — for a "
             "variable-cell run — the box are written back into the frame you "
             "passed. The return value carries loop control only "
             "(MinimizeResult).\n\n"
             "A completed run() freezes the configuration: the plan is "
             "compiled once, so every setter and on(phase) raises "
             "RuntimeError from here on. Running again is still allowed and "
             "re-optimizes from the frame's current state.\n\n"
             "UNDER THE MPI LAUNCHER (a multi-rank world hosting this "
             "interpreter) the configuration is re-sent to the worker ranks "
             "as a task payload every run, and that payload's component "
             "vocabulary is narrower than what this handle accepts: it "
             "carries PairLJCut, the NeighborList and the seven optimizers, "
             "and nothing else. A run configured with any other force source "
             "raises RuntimeError naming the components it cannot carry, "
             "rather than silently optimizing without them. Outside the "
             "launcher every bound force source runs normally.")
        .def(
            "converged",
            [](const MinimizeHandle& self) { return self.driver.converged(); },
            "True when a criterion ended the last run, False when the "
            "iteration budget did.")
        .def(
            "stop_reason",
            [](const MinimizeHandle& self) {
                return self.driver.stop_reason();
            },
            "Name of the criterion that ended the last run (e.g. "
            "'force_converged', 'pressure_converged'); empty when the "
            "iteration budget ended it.");

    nb::class_<MinimizePhaseBuilder>(
        m, "MinimizePhaseBuilder",
        "Fluent builder for typed minimizer phase bindings.")
        .def(
            "add_pair_lj_cut",
            [](MinimizePhaseBuilder& self, const cpu::task::PairLJCutParams& p,
               int order) -> MinimizePhaseBuilder& {
                self.handle->add_on_phase(
                    p, cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff},
                    MinimizeBinding{self.phase, order});
                return self;
            },
            nb::arg("component"), nb::arg("order") = 0,
            nb::rv_policy::reference,
            "Bind a Lennard-Jones pair evaluator to this phase at `order` "
            "(lower runs first). Returns self for chaining.")
        .def(
            "add_neighbor_list",
            [](MinimizePhaseBuilder& self,
               const cpu::task::NeighborListParams& p,
               int order) -> MinimizePhaseBuilder& {
                self.handle->add_on_phase(
                    p, cpu::md::NeighborList<>{p.cutoff, p.skin},
                    MinimizeBinding{self.phase, order});
                return self;
            },
            nb::arg("component"), nb::arg("order") = 0,
            nb::rv_policy::reference,
            "Bind a neighbor list to this phase at `order` (lower runs "
            "first). Returns self for chaining.");
}
