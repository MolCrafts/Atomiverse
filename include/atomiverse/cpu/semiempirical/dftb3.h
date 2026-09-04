/// @file cpu/semiempirical/dftb3.h
/// @brief SCC-DFTB3 multi-element method (static in-tree SK tables, CPU).
#pragma once

#include <array>
#include <string>

#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_3ob_hubbard_derivs.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

/// @brief SCC-DFTB3 energy (+ optional analytic nuclear forces) for H/C/N/O.
///
/// Third-order self-consistent-charge DFTB (Gaus et al., J. Chem. Theory
/// Comput. **7**, 931 (2011), DOI 10.1021/ct100684s) on top of the
/// Elstner DFTB2 skeleton: electronic tight-binding + second-order SCC +
/// third-order charge fluctuation (\(U^{\mathrm{d}}\), \(\Gamma_{AB}\)) +
/// pair repulsive. Integrals, onsite energies and Hubbard \(U\) come from
/// static mio-style tables in `basis_data/dftb_mio_tables.h` (no runtime
/// `.skf` parser, no public factory). Third-order Hubbard derivatives
/// \(U^{\mathrm d}=dU/dq\) are **not** in that table — they are explicit
/// `hubbard_du_ha` input
/// [Ha] (atomic units, as in DFTB+ HubbardDerivs / SpinConstants — NOT eV).
/// Supported Z: 1, 6, 7, 8; any other element or missing pair throws.
///
/// The SCC potential is the exact derivative
/// \f$V_A=\partial(E^{(2)}+E^{(3)})/\partial\Delta q_A\f$, so the band
/// energy double-counts by \f$E^{(2)}+2E^{(3)}\f$ (Euler:
/// \f$\sum_A\Delta q_A V_A = 2E^{(2)}+3E^{(3)}\f$). Both live in
/// `internal/dftb3_scc.h`.
///
/// Nuclear forces are the **closed-form** gradient at the converged SCC
/// density — Hellmann–Feynman \f$\partial H^0/\partial R\f$, Pulay
/// \f$-W\,\partial S/\partial R\f$ with the energy-weighted density
/// \f$W_{\mu\nu}=\sum_o n_o\varepsilon_o C_{\mu o}C_{\nu o}\f$, the SCC shift
/// on \f$\partial S/\partial R\f$, the fixed-Δq
/// \f$\partial(E^{(2)}+E^{(3)})/\partial R\f$ (both the \f$\gamma\f$ and the
/// non-symmetric \f$\Gamma\f$ radial derivatives), and the pair repulsive
/// gradient. Because \f$H=\partial E/\partial P\f$ exactly, the density is
/// stationary and no coupled-perturbed charge response is needed; gated
/// against re-SCC finite differences by
/// `regressions/semiempirical/dftb3-analytic-forces/`. This is **not** a
/// finite-difference path: one SCC solve per force evaluation, not \f$6N+1\f$.
///
/// **PBC:** Γ-point lattice-image walk for H⁰/S, γ/Γ, repulsive, and analytic
/// forces (same shape as DFTB2 MioSketch). Closed shell only; no multi-k,
/// no cell stress, no dispersion.
///
/// Units on the public surface: energy [eV], positions [Å], forces [eV/Å].
///
/// @code
///   cpu::semiempirical::DFTB3 dftb3;
///   dftb3.sk_set = cpu::semiempirical::DftbSkSet::MioSketch;
///   dftb3.hubbard_du_ha = basis_data::k3obHubbardDerivsHa;
///   dftb3.with_forces = true;
///   eval.on(evaluate::Phase::evaluate).add(std::move(dftb3));
/// @endcode
class DFTB3 : public cpu::Semiempirical {
  public:
    /// @brief SCC residual tolerance on max |Δq| (dimensionless e).
    double scc_tol = 1e-8;
    /// @brief Maximum SCC iterations (dimensionless). Exhausting it throws.
    int max_scc = 50;
    /// @brief Anderson charge-mixing relaxation β ∈ (0,1] (dimensionless).
    ///
    /// Undamped Picard (no mixing) is a period-2 limit cycle for polar
    /// systems: it never meets `scc_tol` and the reported energy is picked by
    /// the parity of `max_scc`. Charges go through `internal::ChargeMixer`.
    /// The default converges every in-tree case in ≤ 14 iterations; β ≳ 0.7
    /// stops converging for hydrogen-bonded systems.
    double mix_beta = 0.3;
    /// @brief When true, write forces F=−∇E [eV/Å] on launch (closed form).
    bool with_forces = true;

    /// @brief Static SK parameter set. Default `Unset` — DFTB3 will not pick
    ///        one. `launch()` throws `std::invalid_argument` if left `Unset`.
    ///        Declare `DftbSkSet::MioSketch` (mixed-set warning: 3ob
    ///        \(U^{\mathrm d}\) on mio SK must not be labelled `oracle: dftb+`)
    ///        or `ThreeOb` (`std::runtime_error` until `dftb-gamma-h-damping`).
    DftbSkSet sk_set = DftbSkSet::Unset;

    /// @brief Per-Z DFTB3 Hubbard derivatives \(U^{\mathrm d}=dU/dq\)
    ///        [Ha] (atomic units, as in DFTB+ HubbardDerivs / SpinConstants — NOT eV).
    ///
    /// Index = atomic number \(Z\) (dimensionless). Zero / non-finite = unset.
    /// Required on `launch()` for every \(Z\) present in the Frame; missing
    /// \(Z\) throws `std::invalid_argument`. `launch()` converts Ha→eV with
    /// `units::kHartreeInEv` before building \(E^{(3)}\) (public energy [eV]).
    /// The published 3ob-3-1 table is `basis_data::k3obHubbardDerivsHa`.
    std::array<double, 96> hubbard_du_ha{};

    DFTB3() = default;

    /// @brief Component name token `"dftb3"` (dimensionless).
    std::string name() const override { return "dftb3"; }

    /// @brief No-op; DFTB3 allocates no per-run workspace in `build`.
    /// @param ctx Driver build context (unused).
    void build(const BuildContext& ctx) override;

    /// @brief One SCC-DFTB3 evaluation; writes `kEnergy` [eV] to `frame`.
    ///
    /// Optional: `with_forces` writes \(F=-\nabla E\) [eV/Å] (closed form).
    ///
    /// @param ctx Driver launch context (non-owning `frame` + services).
    /// @throws std::invalid_argument when `sk_set` is `Unset`, when
    ///         `hubbard_du_ha` is unset for a present \(Z\), when the Frame
    ///         is empty, or when `mix_beta` is out of range.
    /// @throws std::runtime_error when `sk_set` is `ThreeOb`.
    void launch(const LaunchContext& ctx) override;

    /// @brief Last electronic+repulsive energy [eV].
    double last_energy() const { return energy_; }

    /// @brief Last second-order SCC double-counting term \(E^{(2)}\) [eV].
    double last_e2() const { return e2_; }

    /// @brief Last third-order term \(E^{(3)}\) [eV].
    double last_e3() const { return e3_; }

    /// @brief SCC residual max|Δq_out − Δq_in| at exit [e]; below `scc_tol`
    ///        whenever `launch` returned (it throws otherwise).
    double last_scc_residual() const { return scc_residual_; }

    /// @brief SCC iterations the last solve took (dimensionless).
    int last_scc_iterations() const { return scc_iterations_; }

  private:
    double energy_ = 0;
    double e2_ = 0;
    double e3_ = 0;
    double scc_residual_ = 0;
    int scc_iterations_ = 0;
    bool warned_mixed_set_ = false;
};

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
