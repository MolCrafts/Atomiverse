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
/// `xc` argument.
///
/// One Python class per concrete method — there is no `SCF` facade, no
/// `SCFConfig` bag and no factory. Parameters are constructor arguments
/// whose keywords match the C++ parameter names; the identity (basis,
/// charge, α / β counts, functional) is read-only afterwards. The basis set
/// (the fixed set of Gaussian functions the orbitals are expanded in) and the
/// exchange–correlation functional are strings resolved against the C++
/// enums (unknown names throw `ValueError` listing the supported ones — no
/// silent fallback).
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
#include "molpy_frame_bridge.h"

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
#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/gradient_result.h"
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
        {"cc-pVDZ", BasisSet::CcPVDZ},
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

/// @brief Python-side owner of one concrete mean-field method.
///
/// Holds the C++ method by value; its identity (basis, charge, occupation,
/// functional) is fixed by the constructor and its knobs are its own members.
/// @tparam Method Concrete C++ method owned by value (`RHF` / `UHF` / `RKS`
///         / `UKS`).
template <typename Method>
class MethodHandle : public ATV::cpu::DriverNode {
  public:
    Method method; ///< Concrete C++ method; parameters are members.

    bool provides_energy() const override { return method.provides_energy(); }
    bool provides_force() const override { return method.provides_force(); }
    int default_budget() const override { return method.default_budget(); }
    bool requires_tag_index() const override {
        return method.requires_tag_index();
    }
    ATV::cpu::TimestepConsumer* timestep_consumer() override { return &method; }
    void build_on(ATV::cpu::Runtime& runtime) override {
        method.build_on(runtime);
    }
    int run_nested_erased(ATV::cpu::Context& context, ATV::cpu::Frame& frame,
                          int budget) override {
        return method.run_nested_erased(context, frame, budget);
    }
    void destroy_on(ATV::cpu::Runtime& runtime) override {
        method.destroy_on(runtime);
    }

    /// @brief Construct the owned method in place from its C++ ctor args.
    ///        Constrained so it never hijacks copy / move construction.
    /// @param args Arguments forwarded verbatim to the `Method` constructor.
    template <typename... Args>
        requires std::is_constructible_v<Method, Args...>
    explicit MethodHandle(Args&&... args)
        : method(std::forward<Args>(args)...) {}

    /// @brief Run one SCF on a borrowed molpy.Frame.
    /// @param frame_obj A `molpy.Frame`; total energy [eV] lands on its meta.
    ///        Forces are not computed here — fx/fy/fz round-trip unchanged.
    /// @throws std::runtime_error (Python `RuntimeError`) from the Frame
    ///         boundary on a missing / wrong-dtype column.
    /// @throws std::invalid_argument (Python `ValueError`) when the α / β
    ///         occupation does not sum to the electron count ΣZ − charge.
    void run(nb::object frame_obj) {
        auto fref = borrow_molpy_frame(frame_obj);
        ATV::cpu::Frame work = ATV::cpu::from_molpy(*fref);
        {
            nb::gil_scoped_release release;
            method.run(work);
        }
        ATV::cpu::to_molpy(work, *fref);
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
        auto fref = borrow_molpy_frame(frame_obj);
        ATV::cpu::Frame work = ATV::cpu::from_molpy(*fref);
        ATV::cpu::scf::GradientResult result;
        {
            nb::gil_scoped_release release;
            result = method.forces(work);
        }
        ATV::cpu::to_molpy(work, *fref);

        // `result.forces` is a row-major [n_atoms, 3] container, so its flat
        // storage is already the (fx0, fy0, fz0, …) layout to_force_array
        // copies from.
        return to_force_array(std::span<const double>(result.forces.data(),
                                                      result.forces.size()));
    }
};

/// @brief Bind the four methods' shared single-point surface.
/// @tparam Method Concrete C++ method the handle owns.
/// @param cls Class binding of a `MethodHandle` specialization.
template <typename Method>
void bind_single_point(
    nb::class_<MethodHandle<Method>, ATV::cpu::DriverNode>& cls) {
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
            "UHF/UKS, ValueError when n_alpha + n_beta does not equal the "
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
            "(dimensionless count); 0 before the first run.")
        .def_prop_ro(
            "basis",
            [](const Handle& self) {
                return std::string(
                    ATV::cpu::scf::basis_set_name(self.method.basis));
            },
            "Basis-set name fixed at construction (read-only).")
        .def_prop_ro(
            "charge", [](const Handle& self) { return self.method.charge; },
            "Molecular charge [e] fixed at construction (read-only).");
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

    nb::class_<MethodHandle<RHF>, ATV::cpu::DriverNode> rhf(
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
        nb::arg("basis"), nb::arg("charge") = 0, nb::arg("tol") = 1.0e-9,
        nb::arg("max_iter") = 100,
        "Construct with basis-set name (the fixed set of Gaussian functions "
        "the orbitals are expanded in — a bigger set is more accurate and "
        "more expensive), molecular charge [e], SCF tolerance on the "
        "energy change per iteration [Hartree], and the iteration cap. "
        "An unknown basis name raises ValueError listing the supported ones.");
    bind_single_point(rhf);

    nb::class_<MethodHandle<UHF>, ATV::cpu::DriverNode> uhf(
        m, "UHF",
        "Unrestricted Hartree-Fock (UHF): the same exact-exchange mean-field "
        "model as RHF, but the spin-up (alpha) and spin-down (beta) electrons "
        "get independent spatial orbitals. That freedom is what an open-shell "
        "system needs — one with unpaired electrons, such as a radical or a "
        "singly ionized molecule — because there the two spin channels really "
        "do see different fields.");
    uhf.def(
           "__init__",
           [](MethodHandle<UHF>* self, const std::string& basis, int n_alpha,
              int n_beta, int charge, double tol, int max_iter) {
               new (self) MethodHandle<UHF>(basis_from_name(basis), n_alpha,
                                            n_beta, charge, tol, max_iter);
           },
           nb::arg("basis"), nb::arg("n_alpha"), nb::arg("n_beta"),
           nb::arg("charge") = 0, nb::arg("tol") = 1.0e-8,
           nb::arg("max_iter") = 100,
           "Construct with basis-set name, the spin-up (alpha) and spin-down "
           "(beta) electron counts, molecular charge [e], SCF tolerance on "
           "the energy change per iteration [Hartree] and the iteration cap. "
           "n_alpha - n_beta = M - 1 for spin multiplicity M = 2S + 1 "
           "(1 singlet, 2 doublet, 3 triplet), and n_alpha + n_beta must "
           "equal the electron count N_e = sum(Z) - charge of the frame — a "
           "mismatch raises ValueError at run() / forces(), when the frame "
           "fixes N_e. An unknown basis name raises ValueError here.")
        .def_prop_ro(
            "n_alpha",
            [](const MethodHandle<UHF>& self) { return self.method.n_alpha; },
            "Spin-up electron count fixed at construction (read-only).")
        .def_prop_ro(
            "n_beta",
            [](const MethodHandle<UHF>& self) { return self.method.n_beta; },
            "Spin-down electron count fixed at construction (read-only).");
    bind_single_point(uhf);

    nb::class_<MethodHandle<RKS>, ATV::cpu::DriverNode> rks(
        m, "RKS",
        "Restricted Kohn-Sham (RKS) density-functional theory (DFT): the "
        "electrons are described by the orbitals of a fictitious "
        "non-interacting system chosen to reproduce the real electron "
        "density, and everything the fiction leaves out — exchange and "
        "correlation — is supplied by an exchange-correlation functional, a "
        "recipe mapping the density to an energy (named by `xc`). "
        "'Restricted' means closed shell: alpha and beta electrons share one "
        "set of spatial orbitals.");
    rks.def(
           "__init__",
           [](MethodHandle<RKS>* self, const std::string& basis,
              const std::string& xc, int charge, double tol, int max_iter) {
               new (self) MethodHandle<RKS>(basis_from_name(basis),
                                            xc_from_name(xc), charge);
               self->method.tol = tol;
               self->method.max_iter = max_iter;
           },
           nb::arg("basis"), nb::arg("xc"), nb::arg("charge") = 0,
           nb::arg("tol") = 1.0e-9, nb::arg("max_iter") = 100,
           "Construct with basis-set name, exchange-correlation functional "
           "name, molecular charge [e], SCF tolerance on the energy change "
           "per iteration [Hartree] and the iteration cap. An unknown "
           "functional or basis name raises ValueError listing the supported "
           "ones.")
        .def_prop_ro(
            "xc",
            [](const MethodHandle<RKS>& self) {
                return xc_name(self.method.xc);
            },
            "Exchange-correlation functional name fixed at construction "
            "(read-only).");
    bind_single_point(rks);

    nb::class_<MethodHandle<UKS>, ATV::cpu::DriverNode> uks(
        m, "UKS",
        "Unrestricted Kohn-Sham (UKS) density-functional theory: RKS with "
        "independent spatial orbitals for the spin-up (alpha) and spin-down "
        "(beta) channels, for open-shell systems with unpaired electrons "
        "(radicals, triplet ground states such as O2).");
    uks.def(
           "__init__",
           [](MethodHandle<UKS>* self, const std::string& basis, int n_alpha,
              int n_beta, const std::string& xc, int charge, double tol,
              int max_iter) {
               new (self) MethodHandle<UKS>(basis_from_name(basis), n_alpha,
                                            n_beta, xc_from_name(xc), charge);
               self->method.tol = tol;
               self->method.max_iter = max_iter;
           },
           nb::arg("basis"), nb::arg("n_alpha"), nb::arg("n_beta"),
           nb::arg("xc") = "LSDA", nb::arg("charge") = 0,
           nb::arg("tol") = 1.0e-8, nb::arg("max_iter") = 100,
           "Construct with basis-set name, spin-up (alpha) and spin-down "
           "(beta) electron counts, exchange-correlation functional name, "
           "molecular charge [e], SCF tolerance on the energy change per "
           "iteration [Hartree] and the iteration cap. As for UHF, "
           "n_alpha + n_beta must equal N_e = sum(Z) - charge, checked at "
           "run() / forces().")
        .def_prop_ro(
            "n_alpha",
            [](const MethodHandle<UKS>& self) { return self.method.n_alpha; },
            "Spin-up electron count fixed at construction (read-only).")
        .def_prop_ro(
            "n_beta",
            [](const MethodHandle<UKS>& self) { return self.method.n_beta; },
            "Spin-down electron count fixed at construction (read-only).")
        .def_prop_ro(
            "xc",
            [](const MethodHandle<UKS>& self) {
                return xc_name(self.method.xc);
            },
            "Exchange-correlation functional name fixed at construction "
            "(read-only).");
    bind_single_point(uks);
#else
    (void)m;
#endif
}
