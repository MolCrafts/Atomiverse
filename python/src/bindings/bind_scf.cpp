/// @file python/src/bindings/bind_scf.cpp
/// @brief Gaussian SCF single-point bindings: the concrete mean-field methods
///        `RHF` / `UHF` / `ROHF` / `GHF` / `RKS` / `UKS`.
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
/// local by default. With an MPI build and `mpi=True`, every initialized
/// MPI_COMM_WORLD rank must call with the same full frame; J/K is collective.
/// This does not dispatch tasks to launcher workers.
///
/// Error surface seen from Python: `ValueError` for an unknown basis /
/// functional name (at construction) and for an α/β occupation the electron
/// count cannot support (at `run` / `forces`); `RuntimeError` from the
/// molpy.Frame boundary when a required column is missing or has the wrong
/// dtype.

#include "force_array.h"
#include "molpy_frame_bridge.h"
#include "result_array.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#ifdef ATV_HAS_HF
// The functional-name table is shared with the plane-wave binding (`PW.xc`),
// so it lives in its own header rather than twice in this file.
#include "basis_names.h"
#include "xc_names.h"

#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/molpy_exchange.h"
#include "atomiverse/cpu/scf/basis_set.h"
#include "atomiverse/cpu/scf/ghf.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/rks.h"
#include "atomiverse/cpu/scf/rohf.h"
#include "atomiverse/cpu/scf/uhf.h"
#include "atomiverse/cpu/scf/uks.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

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

/// @brief Bind the concrete methods' shared single-point surface.
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
            "Molecular charge [e] fixed at construction (read-only).")
        .def_prop_ro("tol", [](const Handle& self) { return self.method.tol; })
        .def_prop_ro("max_iter",
                     [](const Handle& self) { return self.method.max_iter; })
        .def_prop_rw(
            "level_shift",
            [](const Handle& self) { return self.method.level_shift; },
            [](Handle& self, double value) { self.method.level_shift = value; })
        .def_prop_rw(
            "max_memory_mb",
            [](const Handle& self) { return self.method.max_memory_mb; },
            [](Handle& self, double value) {
                if (!std::isfinite(value) || value < 0)
                    throw std::invalid_argument(
                        "max_memory_mb must be finite and nonnegative");
                self.method.max_memory_mb = value;
            },
            "Process memory budget in MB; exact-ERI execution is selected "
            "automatically.")
        .def(
            "set_threads",
            [](Handle& self, int n) { self.method.set_threads(n); },
            nb::arg("threads"), "Override the default OpenMP thread count.")
        .def_prop_rw(
            "mpi", [](const Handle& self) { return self.method.mpi; },
            [](Handle& self, bool enabled) { self.method.mpi = enabled; },
            "Collective HF J/K on initialized MPI_COMM_WORLD; all ranks must "
            "participate.")
        .def_prop_rw(
            "init_guess",
            [](const Handle& self) {
                return self.method.init_guess == ATV::cpu::scf::InitGuess::Hcore
                           ? "hcore"
                           : "minao";
            },
            [](Handle& self, const std::string& guess) {
                if (guess != "hcore" && guess != "minao")
                    throw std::invalid_argument(
                        "init_guess must be hcore or minao");
                self.method.init_guess = guess == "hcore"
                                             ? ATV::cpu::scf::InitGuess::Hcore
                                             : ATV::cpu::scf::InitGuess::MINAO;
            },
            "Initial density policy; no converged state is reused.")
        .def_prop_rw(
            "damping",
            [](const Handle& self) { return self.method.damping_factor; },
            [](Handle& self, double value) {
                self.method.damping_factor = value;
            })
        .def_prop_rw(
            "conv_tol_grad",
            [](const Handle& self) { return self.method.conv_tol_grad; },
            [](Handle& self, double value) {
                self.method.conv_tol_grad = value;
            })
        .def_prop_rw(
            "diis_space",
            [](const Handle& self) { return self.method.diis_max_vectors; },
            [](Handle& self, int value) {
                self.method.diis_max_vectors = value;
            })
        .def_prop_rw(
            "diis_start_cycle",
            [](const Handle& self) { return self.method.diis_start_iter; },
            [](Handle& self, int value) {
                self.method.diis_start_iter = value;
            });
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

    using ATV::cpu::scf::ROHF;
    nb::class_<MethodHandle<ROHF>, ATV::cpu::DriverNode> rohf(
        m, "ROHF",
        "Restricted open-shell Hartree-Fock, Guest-Saunders convention.");
    rohf.def(
            "__init__",
            [](MethodHandle<ROHF>* self, const std::string& basis, int n_alpha,
               int n_beta, int charge, double tol, int max_iter) {
                new (self) MethodHandle<ROHF>(basis_from_name(basis), n_alpha,
                                              n_beta, charge);
                self->method.tol = tol;
                self->method.max_iter = max_iter;
            },
            nb::arg("basis"), nb::arg("n_alpha"), nb::arg("n_beta"),
            nb::arg("charge") = 0, nb::arg("tol") = 1e-10,
            nb::arg("max_iter") = 200)
        .def_prop_ro(
            "n_alpha",
            [](const MethodHandle<ROHF>& self) { return self.method.n_alpha; })
        .def_prop_ro("n_beta", [](const MethodHandle<ROHF>& self) {
            return self.method.n_beta;
        });
    bind_single_point(rohf);

    using ATV::cpu::scf::GHF;
    nb::class_<MethodHandle<GHF>, ATV::cpu::DriverNode> ghf(
        m, "GHF", "Real two-component generalized Hartree-Fock.");
    ghf.def(
           "__init__",
           [](MethodHandle<GHF>* self, const std::string& basis,
              int n_electrons, double tol, int max_iter, double spin_rotation) {
               new (self)
                   MethodHandle<GHF>(basis_from_name(basis), n_electrons);
               self->method.tol = tol;
               self->method.max_iter = max_iter;
               self->method.guess_spin_rotation = spin_rotation;
           },
           nb::arg("basis"), nb::arg("n_electrons"), nb::arg("tol") = 1e-10,
           nb::arg("max_iter") = 200, nb::arg("spin_rotation") = 0.2)
        .def_prop_ro("n_electrons",
                     [](const MethodHandle<GHF>& self) {
                         return self.method.n_electrons;
                     })
        .def_prop_rw(
            "spin_rotation",
            [](const MethodHandle<GHF>& self) {
                return self.method.guess_spin_rotation;
            },
            [](MethodHandle<GHF>& self, double value) {
                self.method.guess_spin_rotation = value;
            })
        .def(
            "spin_density_block",
            [](const MethodHandle<GHF>& self, const std::string& bra,
               const std::string& ket) {
                if (!self.method.converged())
                    throw std::logic_error(
                        "GHF.spin_density_block requires a converged solve");
                auto spin = [](const std::string& name) {
                    if (name == "alpha")
                        return ATV::cpu::scf::SlaterDeterminant::Spin::alpha;
                    if (name == "beta")
                        return ATV::cpu::scf::SlaterDeterminant::Spin::beta;
                    throw std::invalid_argument("spin must be alpha or beta");
                };
                // C++ SCF matrices are column-major; the shared numpy helper
                // expects row-major storage. Let xtensor perform that layout copy.
                xt::xtensor<double, 2, xt::layout_type::row_major> block;
                {
                    nb::gil_scoped_release release;
                    block = self.method.determinant().spin_block(spin(bra),
                                                                 spin(ket));
                }
                return to_matrix_array(
                    std::span<const double>(block.data(), block.size()),
                    block.shape()[0], block.shape()[1]);
            },
            nb::arg("bra") = "alpha", nb::arg("ket") = "beta",
            "Owned AO spin-density block from the converged real spinor "
            "determinant.");
    bind_single_point(ghf);

    m.def(
        "hf_backend_info",
        []() {
            nb::dict info;
            info["backend"] = "cpu";
            info["compiler_id"] = ATV_HF_CXX_COMPILER_ID;
            info["compiler_version"] = ATV_HF_CXX_COMPILER_VERSION;
            info["compiler_executable"] = ATV_HF_CXX_COMPILER_PATH;
#ifdef __VERSION__
            info["compiler"] = __VERSION__;
#else
            info["compiler"] =
                ATV_HF_CXX_COMPILER_ID " " ATV_HF_CXX_COMPILER_VERSION;
#endif
#ifdef _OPENMP
            int observed = 1;
            {
                nb::gil_scoped_release release;
#pragma omp parallel
                {
#pragma omp single
                    observed = omp_get_num_threads();
                }
            }
            info["openmp"] = true;
            info["openmp_version"] = _OPENMP;
            info["openmp_threads"] = observed;
            info["openmp_max_threads"] = omp_get_max_threads();
#else
            info["openmp"] = false;
            info["openmp_version"] = 0;
            info["openmp_threads"] = 1;
            info["openmp_max_threads"] = 1;
#endif
            return info;
        },
        "Actual CPU HF compiler and OpenMP runtime configuration.");

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
