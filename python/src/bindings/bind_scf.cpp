/// @file python/src/bindings/bind_scf.cpp
/// @brief Gaussian SCF single-point bindings: the concrete mean-field methods
///        `RHF` / `UHF` / `RKS` / `UKS`.
///
/// **SCF** = *self-consistent field*. A **mean-field** electronic-structure
/// model replaces the mutual repulsion of the electrons by an averaged field
/// that each electron moves in. That averaged field is itself built from the
/// orbitals it determines, so the equations are solved by iterating —
/// build the field, diagonalize, rebuild — until the total energy and the
/// residual stop changing. That loop is the SCF.
///
/// The four bound methods differ along two axes. **R**estricted means the
/// spin-up (α) and spin-down (β) electrons share one set of spatial orbitals
/// (closed shell, every occupied orbital doubly filled); **U**nrestricted
/// gives α and β their own orbitals, which is what an open-shell system with
/// unpaired electrons needs. **HF** = Hartree–Fock treats electron exchange
/// exactly and omits correlation; **KS** = Kohn–Sham density-functional
/// theory replaces both by an approximate exchange–correlation *functional*
/// (a recipe mapping the electron density to an energy), named by the
/// `functional` argument.
///
/// One Python class per concrete method — there is no `SCF` facade, no
/// `SCFConfig` bag and no factory. Parameters are constructor arguments;
/// basis set (the fixed set of Gaussian functions the orbitals are expanded
/// in) and exchange–correlation functional are strings resolved against the
/// C++ enums (unknown names throw `ValueError` listing the supported ones —
/// no silent fallback).
///
/// `run(frame)` writes the total energy [eV] to the borrowed molpy.Frame's
/// typed meta and leaves fx/fy/fz untouched; `forces(frame)` additionally
/// returns an (N, 3) array [eV/Å] and writes fx/fy/fz back. Both are
/// **local** to this process: unlike the AM1 `single_point` entry there is
/// no `cpu::task::TaskSpec` mirror for a mean-field method, so nothing is
/// dispatched to MPI launcher workers.
///
/// Error surface seen from Python: `ValueError` for an unknown basis /
/// functional name (at construction) and for an α/β occupation the electron
/// count cannot support (at `run` / `forces`); `RuntimeError` from the
/// molpy.Frame boundary when a required column is missing or has the wrong
/// dtype.

#include "force_array.h"
#include "molrs_frame_bridge.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#ifdef ATV_HAS_HF
// The functional-name table is shared with the plane-wave binding (`PW.xc`),
// so it lives in its own header rather than twice in this file.
#include "xc_names.h"

#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/molrs_exchange.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/internal/scf_types.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/scf/xc_functional.h"

namespace {

using ATV::cpu::scf::BasisSet;

/// @brief Basis-set name → `BasisSet` (names verbatim from `basis_set_name`).
/// @param name Basis-set name, e.g. "STO-3G" or "def2-SVP".
/// @return The matching enumerator.
/// @throws std::invalid_argument (Python `ValueError`) on an unsupported
///         name, listing every supported name — no fallback basis.
BasisSet basis_from_name(const std::string& name) {
    static constexpr std::pair<const char*, BasisSet> kTable[] = {
        {"STO-3G", BasisSet::STO3G},
        {"6-31G", BasisSet::Pople631G},
        {"6-31G*", BasisSet::Pople631Gs},
        {"6-31+G*", BasisSet::Pople631pGs},
        {"def2-SVP", BasisSet::Def2SVP},
        {"def2-TZVP", BasisSet::Def2TZVP},
        {"aug-cc-pVDZ", BasisSet::AugCcPVDZ},
    };
    std::string supported;
    for (const auto& [key, value] : kTable) {
        if (name == key)
            return value;
        if (!supported.empty())
            supported += ", ";
        supported += key;
    }
    throw std::invalid_argument("unknown basis set '" + name
                                + "'; supported: " + supported);
}

/// @brief Electron count N_e = Σ_A Z_A − q of a materialized Frame: the
///        nuclear charges of all atoms less the molecular charge.
/// @param frame  Frame whose `kAtom_type` column carries the atomic number Z
///               per atom (imported from molrs `atoms.atomic_number`).
/// @param charge Molecular charge q [e].
/// @return Electron count (dimensionless).
int electron_count(const ATV::cpu::Frame& frame, int charge) {
    const auto z = frame.span(ATV::kAtom_type);
    int total = 0;
    for (int i = 0; i < frame.nlocal(); ++i)
        total += z[i];
    return total - charge;
}

/// @brief A method whose occupation is the explicit α / β split (UHF, UKS),
///        as opposed to the restricted methods that derive it from N_e alone.
/// @tparam Method Concrete C++ mean-field method being probed.
template <typename Method>
concept OpenShellMethod = requires(Method& method) {
    method.n_alpha;
    method.n_beta;
};

/// @brief Python-side owner of one concrete mean-field method.
///
/// Holds the C++ method (whose parameters are its own members) plus the spin
/// multiplicity, which only becomes an α / β split once the electron count
/// N_e = ΣZ − q is known — i.e. at `run` / `forces` time, from the Frame.
/// @tparam Method Concrete C++ method owned by value (`RHF` / `UHF` / `RKS`
///         / `UKS`).
template <typename Method>
class MethodHandle {
  public:
    Method method; ///< Concrete C++ method; parameters are members.

    /// @brief Spin multiplicity M = 2S + 1, where S is the total spin
    ///        (M = 1 singlet, 2 doublet, 3 triplet; dimensionless). Read by
    ///        the open-shell methods only.
    int multiplicity = 1;

    /// @brief Construct the owned method in place from its C++ ctor args.
    ///        Constrained so it never hijacks copy / move construction.
    /// @param args Arguments forwarded verbatim to the `Method` constructor.
    template <typename... Args>
        requires std::is_constructible_v<Method, Args...>
    explicit MethodHandle(Args&&... args)
        : method(std::forward<Args>(args)...) {}

    /// @brief Assign the four parameters every mean-field method shares, for
    ///        the methods that have no forwarding constructor to take them.
    /// @param basis    Resolved basis set.
    /// @param charge   Molecular charge q [e].
    /// @param tol      SCF tolerance on the total-energy change between
    ///                 iterations, |ΔE| [Hartree]. Convergence also requires
    ///                 the Frobenius norm of the DIIS error matrix (DIIS =
    ///                 *direct inversion of the iterative subspace*, the
    ///                 standard SCF accelerator; its error matrix measures
    ///                 how far the current density is from self-consistency,
    ///                 dimensionless) to fall below `conv_tol_grad`, which
    ///                 this binding never sets, so it keeps the C++ default
    ///                 of √tol (`MeanFieldMethod::conv_tol_grad` < 0).
    /// @param max_iter SCF iteration cap (dimensionless count).
    void configure(BasisSet basis, int charge, double tol, int max_iter) {
        method.basis = basis;
        method.charge = charge;
        method.tol = tol;
        method.max_iter = max_iter;
    }

    /// @brief Run one SCF on a borrowed molpy.Frame.
    /// @param frame_obj A `molpy.Frame`; total energy [eV] lands on its meta.
    ///        Forces are not computed here — fx/fy/fz round-trip unchanged.
    /// @throws std::runtime_error (Python `RuntimeError`) from the Frame
    ///         boundary on a missing / wrong-dtype column.
    /// @throws std::invalid_argument (Python `ValueError`) when the requested
    ///         multiplicity cannot split this electron count (open shell).
    void run(nb::object frame_obj) {
        auto fref = borrow_molrs_frame(frame_obj);
        ATV::cpu::Frame work = ATV::cpu::from_molrs(*fref);
        resolve_occupation_(work);
        {
            nb::gil_scoped_release release;
            method.run(work);
        }
        ATV::cpu::to_molrs(work, *fref);
    }

    /// @brief Run one SCF and return the analytic nuclear forces
    ///        F = −∂E/∂R (analytic = differentiated integrals, not a finite
    ///        difference of energies).
    ///
    /// Same error surface as `run`.
    /// @param frame_obj A `molpy.Frame`; energy [eV] and fx/fy/fz [eV/Å] land
    ///                  on it.
    /// @return Forces [eV/Å], shape (n_atoms, 3).
    ForceArray forces(nb::object frame_obj) {
        auto fref = borrow_molrs_frame(frame_obj);
        ATV::cpu::Frame work = ATV::cpu::from_molrs(*fref);
        resolve_occupation_(work);
        ATV::cpu::scf::GradientResult result;
        {
            nb::gil_scoped_release release;
            result = method.forces(work);
        }
        ATV::cpu::to_molrs(work, *fref);

        // `result.forces` is a row-major [n_atoms, 3] container, so its flat
        // storage is already the (fx0, fy0, fz0, …) layout to_force_array
        // copies from.
        return to_force_array(std::span<const double>(result.forces.data(),
                                                      result.forces.size()));
    }

  private:
    /// @brief Resolve the α / β occupation of an open-shell method from the
    ///        Frame's electron count and the requested multiplicity.
    ///
    /// No-op for the restricted methods, which fix their own occupation from
    /// the electron count alone.
    /// @param work Materialized Frame (Z per atom in `kAtom_type`).
    /// @throws std::invalid_argument (Python `ValueError`) when N_e and M have
    ///         mismatched parity, when M > N_e + 1, or when the charge leaves
    ///         a negative electron count — `SpinResolution` states the rule in
    ///         the message. No fallback occupation.
    void resolve_occupation_(const ATV::cpu::Frame& work) {
        if constexpr (OpenShellMethod<Method>) {
            const ATV::cpu::scf::SpinResolution occupation(
                electron_count(work, method.charge), multiplicity);
            method.n_alpha = occupation.n_alpha;
            method.n_beta = occupation.n_beta;
        }
    }
};

/// @brief Bind the four methods' shared single-point surface.
/// @tparam Method Concrete C++ method the handle owns.
/// @param cls Class binding of a `MethodHandle` specialization.
template <typename Method>
void bind_single_point(nb::class_<MethodHandle<Method>>& cls) {
    using Handle = MethodHandle<Method>;
    cls.def("run", &Handle::run, nb::arg("frame"),
            "Iterate the SCF (self-consistent field) loop to convergence at "
            "the frame's geometry. Writes the total energy [eV] to "
            "frame.meta['energy']; the same number in Hartree, the atomic "
            "unit the solver works in (1 Ha = 27.211386 eV), is on energy(). "
            "Forces are NOT computed here: fx/fy/fz come back unchanged "
            "(zeros on a frame that had none) — call forces() for those. "
            "The GIL is released during the solve. Local to this process — "
            "a mean-field method has no MPI launcher task, so nothing is "
            "dispatched to worker ranks. Raises RuntimeError when the frame "
            "lacks a required column (atoms.x/y/z/atomic_number) and, for "
            "UHF/UKS, ValueError when the multiplicity does not match the "
            "electron count N_e = sum(Z) - charge.")
        .def("forces", &Handle::forces, nb::arg("frame"),
             "Run one SCF and return the analytic nuclear forces "
             "F = -dE/dR — the negative gradient of the total energy with "
             "respect to the nuclear positions, obtained from differentiated "
             "integrals rather than a finite difference — as an "
             "(n_atoms, 3) array [eV/Å]; the same values are written back to "
             "the frame's fx/fy/fz columns and the energy [eV] to "
             "frame.meta['energy']. Same error surface as run().")
        .def(
            "energy", [](const Handle& self) { return self.method.energy(); },
            "Total energy (electronic + nuclear repulsion) of the most recent "
            "run [Hartree]; frame.meta['energy'] carries the same number in "
            "eV. 0.0 before the first run.")
        .def(
            "converged",
            [](const Handle& self) { return self.method.converged(); },
            "True when the most recent SCF loop met BOTH thresholds: the "
            "energy change between iterations |dE| below tol [Hartree], and "
            "the Frobenius norm of the DIIS error matrix (DIIS = direct "
            "inversion of the iterative subspace, the standard SCF "
            "accelerator; its error matrix measures the distance from "
            "self-consistency) below sqrt(tol). False before the first run "
            "and when max_iter was reached first.")
        .def(
            "iterations",
            [](const Handle& self) { return self.method.iterations(); },
            "SCF iterations performed by the most recent run "
            "(dimensionless count); 0 before the first run.");
}

} // namespace
#endif // ATV_HAS_HF

/// @brief Register the Gaussian SCF classes on the `_atomiverse` module.
///        This translation unit is added to the extension only when
///        `ATV_BUILD_HF=ON`, so on any other build the module carries no
///        `RHF` attribute — which is exactly what `atomiverse.scf` probes
///        for before populating its `__all__`.
/// @param m The nanobind module under construction.
void bind_scf(nb::module_& m) {

#ifdef ATV_HAS_HF
    using ATV::cpu::scf::RHF;
    using ATV::cpu::scf::RKS;
    using ATV::cpu::scf::UHF;
    using ATV::cpu::scf::UKS;

    nb::class_<MethodHandle<RHF>> rhf(
        m, "RHF",
        "Restricted Hartree-Fock (RHF): the closed-shell mean-field method. "
        "Every electron moves in the averaged field of all the others, "
        "electron exchange is treated exactly and correlation is omitted, and "
        "'restricted' means the spin-up (alpha) and spin-down (beta) "
        "electrons share one set of spatial orbitals — so every occupied "
        "orbital holds exactly two electrons. Use it when the system has no "
        "unpaired electrons; see UHF otherwise.");
    rhf.def(
        "__init__",
        [](MethodHandle<RHF>* self, const std::string& basis, int charge,
           double tol, int max_iter) {
            new (self) MethodHandle<RHF>(basis_from_name(basis), charge, tol,
                                         max_iter);
        },
        nb::arg("basis") = "STO-3G", nb::arg("charge") = 0,
        nb::arg("tol") = 1.0e-9, nb::arg("max_iter") = 100,
        "Construct with basis-set name (the fixed set of Gaussian functions "
        "the orbitals are expanded in — a bigger set is more accurate and "
        "more expensive), molecular charge [e], SCF tolerance on the "
        "energy change per iteration [Hartree], and the iteration cap. "
        "An unknown basis name raises ValueError listing the supported ones.");
    bind_single_point(rhf);

    nb::class_<MethodHandle<UHF>> uhf(
        m, "UHF",
        "Unrestricted Hartree-Fock (UHF): the same exact-exchange mean-field "
        "model as RHF, but the spin-up (alpha) and spin-down (beta) electrons "
        "get independent spatial orbitals. That freedom is what an open-shell "
        "system needs — one with unpaired electrons, such as a radical or a "
        "singly ionized molecule — because there the two spin channels really "
        "do see different fields.");
    uhf.def(
        "__init__",
        [](MethodHandle<UHF>* self, const std::string& basis, int charge,
           int multiplicity, double tol, int max_iter) {
            new (self) MethodHandle<UHF>();
            self->configure(basis_from_name(basis), charge, tol, max_iter);
            self->multiplicity = multiplicity;
        },
        nb::arg("basis") = "STO-3G", nb::arg("charge") = 0,
        nb::arg("multiplicity") = 1, nb::arg("tol") = 1.0e-8,
        nb::arg("max_iter") = 100,
        "Construct with basis-set name, molecular charge [e], spin "
        "multiplicity, SCF tolerance on the energy change per iteration "
        "[Hartree] and the iteration cap. The multiplicity is M = 2S + 1 "
        "with S the total spin: 1 = singlet (all electrons paired), "
        "2 = doublet (one unpaired), 3 = triplet (two unpaired). It is not a "
        "count of electrons, so the alpha / beta split is only resolved at "
        "run() / forces() time, once the frame fixes the electron count "
        "N_e = sum(Z) - charge: n_alpha = (N_e + M - 1)/2, "
        "n_beta = (N_e - M + 1)/2. Those are integers only for an even N_e "
        "with an odd M or an odd N_e with an even M; any other pairing, or "
        "M > N_e + 1, raises ValueError at run() / forces(), not at "
        "construction.");
    bind_single_point(uhf);

    nb::class_<MethodHandle<RKS>> rks(
        m, "RKS",
        "Restricted Kohn-Sham (RKS) density-functional theory (DFT): the "
        "electrons are described by the orbitals of a fictitious "
        "non-interacting system chosen to reproduce the real electron "
        "density, and everything the fiction leaves out — exchange and "
        "correlation — is supplied by an exchange-correlation functional, a "
        "recipe mapping the density to an energy (named by `functional`). "
        "'Restricted' means closed shell: alpha and beta electrons share one "
        "set of spatial orbitals.");
    rks.def(
        "__init__",
        [](MethodHandle<RKS>* self, const std::string& functional,
           const std::string& basis, int charge, double tol, int max_iter) {
            new (self) MethodHandle<RKS>(xc_from_name(functional));
            self->configure(basis_from_name(basis), charge, tol, max_iter);
        },
        nb::arg("functional") = "PBE", nb::arg("basis") = "STO-3G",
        nb::arg("charge") = 0, nb::arg("tol") = 1.0e-9,
        nb::arg("max_iter") = 100,
        "Construct with exchange-correlation functional name, basis-set "
        "name, molecular charge [e], SCF tolerance on the energy change per "
        "iteration [Hartree] and the iteration cap. An unknown functional or "
        "basis name raises ValueError listing the supported ones.");
    bind_single_point(rks);

    nb::class_<MethodHandle<UKS>> uks(
        m, "UKS",
        "Unrestricted Kohn-Sham (UKS) density-functional theory: RKS with "
        "independent spatial orbitals for the spin-up (alpha) and spin-down "
        "(beta) channels, for open-shell systems with unpaired electrons "
        "(radicals, triplet ground states such as O2).");
    uks.def(
        "__init__",
        [](MethodHandle<UKS>* self, const std::string& functional,
           const std::string& basis, int charge, int multiplicity, double tol,
           int max_iter) {
            new (self) MethodHandle<UKS>();
            self->method.xc = xc_from_name(functional);
            self->configure(basis_from_name(basis), charge, tol, max_iter);
            self->multiplicity = multiplicity;
        },
        nb::arg("functional") = "LSDA", nb::arg("basis") = "STO-3G",
        nb::arg("charge") = 0, nb::arg("multiplicity") = 1,
        nb::arg("tol") = 1.0e-8, nb::arg("max_iter") = 100,
        "Construct with exchange-correlation functional name, basis-set "
        "name, molecular charge [e], spin multiplicity M = 2S + 1 (S = total "
        "spin; 1 singlet, 2 doublet, 3 triplet), SCF tolerance on the energy "
        "change per iteration [Hartree] and the iteration cap. As for UHF the "
        "alpha / beta split is resolved at run() / forces() from "
        "N_e = sum(Z) - charge and M, and an impossible pairing raises "
        "ValueError there.");
    bind_single_point(uks);
#else
    (void)m;
#endif
}
