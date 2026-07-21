/// @file cpu/scf/slater_determinant.h
/// @brief `cpu::scf::SlaterDeterminant` — the mean-field wavefunction as a
///        validating public value type: per-spin-channel {C, ε, f}.
///
/// Storage is per spin channel (P^σ = Σ_i f_i^σ c_i c_iᵀ keeps the clean
/// idempotency invariant P^σ S P^σ = P^σ for integer occupations); the
/// factor-2 closed-shell total density is only ever a derived value
/// (`density_total()`). Restricted (RHF/RKS) determinants store one channel
/// and answer every β query with the α data (view semantics, no copy).
/// The generalized (GHF spinor) form stores ONE channel whose rows are the
/// stacked (α; β) components of two-component real spinors — C is 2n×N_mo,
/// the spinor density `density()` is 2n×2n with spin blocks reachable via
/// `spin_block`, and the overlap metric is S⊗1₂ (`validate`).
///
/// Construction validates the S-free invariants and throws
/// `std::invalid_argument` with numeric context (no fallback values);
/// overlap-dependent invariants (Tr(P^σS) = N^σ, idempotency) live in
/// `validate(S)` because the constructor has no S.
///
/// References: Szabo & Ostlund, *Modern Quantum Chemistry* (Dover 1996),
/// Ch. 3 (density matrices, Pople–Nesbet); Koopmans, Physica 1, 104 (1934),
/// DOI 10.1016/S0031-8914(34)90011-2 (homo() as −IP estimate); Gilbert,
/// Besley, Gill, JPCA 112, 13164 (2008), DOI 10.1021/jp801738f (ΔSCF/MOM —
/// `with_occupation` is the occupation entry point).
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/cpu/linalg/density.h"
#include "atomiverse/cpu/scf/dense_matrix.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief A single Slater determinant: per-spin-channel MO coefficients,
///        orbital energies, and occupations, with the derived densities and
///        one-electron expectation values as methods.
///
/// Value type (copyable, movable); mutation only through methods that return
/// a re-validated new value (`with_occupation`), following the `cpu::Box`
/// mutate-through-method discipline.
class SlaterDeterminant {
  public:
    /// @brief Spin channel selector. Restricted determinants answer both
    ///        values with the single stored channel; so does the generalized
    ///        (spinor) form, whose single channel carries both components.
    enum class Spin { alpha, beta };

    /// @brief Tag type selecting the generalized (two-component real spinor)
    ///        construction form (dimensionless; disambiguates the otherwise
    ///        identical constructor signature — std::in_place_t precedent).
    struct Generalized {};

    /// @brief Restricted (RHF/RKS) constructor: one spatial channel shared
    ///        by both spins.
    /// @param C MO coefficient matrix [n x n_mo] (dimensionless).
    /// @param epsilon Orbital energies [Hartree], one per MO column.
    /// @param occupation Per-spin occupations f_i in [0, 1] (dimensionless;
    ///        a doubly occupied spatial orbital is f_i = 1 per spin).
    /// @param n_electrons_per_spin Electron count per spin channel N^σ.
    /// @throws std::invalid_argument on dimension mismatch, f_i outside
    ///         [0, 1], or |Σf − N^σ| > 1e-12·max(1, N^σ).
    SlaterDeterminant(RealMatrix C, RealVector epsilon, RealVector occupation,
                      int n_electrons_per_spin)
        : alpha_{std::move(C), std::move(epsilon), std::move(occupation)},
          n_alpha_(n_electrons_per_spin), n_beta_(n_electrons_per_spin),
          restricted_(true) {
        check_channel_("alpha(restricted)", alpha_, n_alpha_);
    }

    /// @brief Unrestricted (UHF/UKS) constructor: independent α/β channels.
    /// @param C_a Alpha MO coefficients [n x n_mo] (dimensionless).
    /// @param eps_a Alpha orbital energies [Hartree].
    /// @param f_a Alpha occupations in [0, 1] (dimensionless).
    /// @param n_alpha Alpha electron count N^α.
    /// @param C_b Beta MO coefficients [n x n_mo] (dimensionless).
    /// @param eps_b Beta orbital energies [Hartree].
    /// @param f_b Beta occupations in [0, 1] (dimensionless).
    /// @param n_beta Beta electron count N^β.
    /// @throws std::invalid_argument on dimension mismatch, occupations
    ///         outside [0, 1], or per-channel |Σf − N^σ| > 1e-12·max(1, N^σ).
    SlaterDeterminant(RealMatrix C_a, RealVector eps_a, RealVector f_a,
                      int n_alpha, RealMatrix C_b, RealVector eps_b,
                      RealVector f_b, int n_beta)
        : alpha_{std::move(C_a), std::move(eps_a), std::move(f_a)},
          beta_{std::move(C_b), std::move(eps_b), std::move(f_b)},
          n_alpha_(n_alpha), n_beta_(n_beta), restricted_(false) {
        check_channel_("alpha", alpha_, n_alpha_);
        check_channel_("beta", beta_, n_beta_);
    }

    /// @brief Generalized (GHF) constructor: one channel of two-component
    ///        real spinor MOs. Row μ < n is the α component on AO μ, row
    ///        n + μ the β component (blocked spinor layout).
    /// @param tag Generalized-form selector (dimensionless tag).
    /// @param C Spinor MO coefficient matrix [2n x n_mo] (dimensionless);
    ///        the row count must be even.
    /// @param epsilon Spinor orbital energies [Hartree], one per MO column.
    /// @param occupation Spinor occupations f_i in [0, 1] (dimensionless;
    ///        each spinor holds at most one electron).
    /// @param n_electrons Total electron count N.
    /// @throws std::invalid_argument on dimension mismatch, odd spinor row
    ///         count, f_i outside [0, 1], or |Σf − N| > 1e-12·max(1, N).
    SlaterDeterminant(Generalized tag, RealMatrix C, RealVector epsilon,
                      RealVector occupation, int n_electrons)
        : alpha_{std::move(C), std::move(epsilon), std::move(occupation)},
          n_alpha_(n_electrons), n_beta_(0), restricted_(false),
          generalized_(true) {
        (void)tag;
        check_channel_("spinor", alpha_, n_alpha_);
        if (alpha_.C.shape()[0] % 2 != 0)
            throw std::invalid_argument(
                "SlaterDeterminant: generalized (spinor) C has "
                + std::to_string(alpha_.C.shape()[0])
                + " rows — must be even (2n: stacked alpha/beta components)");
    }

    /// @brief Whether this is a restricted (single spatial channel)
    ///        determinant.
    /// @return true for the restricted form (β accessors forward to α).
    bool restricted() const { return restricted_; }

    /// @brief Whether this is the generalized (two-component spinor) form.
    /// @return true when constructed with the `Generalized` tag: C is
    ///         2n×N_mo and `density()` is the 2n×2n spinor density.
    bool generalized() const { return generalized_; }

    /// @brief MO coefficient matrix of channel σ.
    /// @param s Spin channel.
    /// @return C^σ [n x n_mo] (dimensionless).
    const RealMatrix& coefficients(Spin s) const { return channel_(s).C; }

    /// @brief Orbital energies of channel σ.
    /// @param s Spin channel.
    /// @return ε^σ, one per MO column [Hartree].
    const RealVector& orbital_energies(Spin s) const {
        return channel_(s).epsilon;
    }

    /// @brief Per-spin occupations of channel σ.
    /// @param s Spin channel.
    /// @return f^σ in [0, 1] per spin orbital (dimensionless).
    const RealVector& occupations(Spin s) const {
        return channel_(s).occupation;
    }

    /// @brief Electron count of channel σ.
    /// @param s Spin channel.
    /// @return N^σ (dimensionless count).
    /// @throws std::invalid_argument for the generalized form — a spinor
    ///         determinant has no sharp per-spin electron count (use
    ///         `n_electrons_total()`).
    int n_electrons(Spin s) const {
        if (generalized_)
            throw std::invalid_argument(
                "SlaterDeterminant::n_electrons(Spin): the generalized "
                "(spinor) form has no sharp per-spin electron count — use "
                "n_electrons_total()");
        return (s == Spin::beta) ? n_beta_ : n_alpha_;
    }

    /// @brief Total electron count N = N^α + N^β (the spinor count N for
    ///        the generalized form).
    /// @return N (dimensionless count).
    int n_electrons_total() const { return n_alpha_ + n_beta_; }

    /// @brief Cached spin-contamination diagnostic ⟨S²⟩ − S(S+1)
    ///        (Amos-Hall; 0 for a pure spin state and for the restricted
    ///        form). Filled by the producing method's epilogue — evaluation
    ///        needs the AO overlap S, which is method-private workspace.
    /// @return ⟨S²⟩ deviation (dimensionless).
    double spin_contamination() const { return spin_contamination_; }

    /// @brief Cache the spin-contamination diagnostic (producer-side).
    /// @param value ⟨S²⟩ − S(S+1) deviation (dimensionless).
    void set_spin_contamination(double value) { spin_contamination_ = value; }

    /// @brief Spin density matrix P^σ = Σ_i f_i^σ c_i c_iᵀ (AO basis,
    ///        dimensionless), built by delegating the GEMM to
    ///        `linalg::build_density_matrix`. For the generalized form this
    ///        is the σσ diagonal spin block of the spinor density
    ///        (`spin_block(s, s)`).
    /// @param s Spin channel.
    /// @return P^σ [n x n] (dimensionless). For integer Aufbau occupations
    ///         this is bit-identical to C_occ C_occᵀ.
    RealMatrix density(Spin s) const {
        if (generalized_)
            return spin_block(s, s);
        return density_of_(channel_(s));
    }

    /// @brief Spinor density matrix P = Σ_i f_i c_i c_iᵀ over the
    ///        two-component spinor MOs (generalized form only).
    /// @return P [2n x 2n] (dimensionless), spin-blocked
    ///         [[P^αα, P^αβ], [P^βα, P^ββ]].
    /// @throws std::invalid_argument when the determinant is not
    ///         generalized (use `density(Spin)` / `density_total()`).
    RealMatrix density() const {
        if (!generalized_)
            throw std::invalid_argument(
                "SlaterDeterminant::density(): the 2n x 2n spinor density "
                "is defined for the generalized form only — use "
                "density(Spin) or density_total()");
        return density_of_(alpha_);
    }

    /// @brief Spin block P^{σσ'} of the spinor density (generalized form
    ///        only): rows from the σ component, columns from the σ'
    ///        component. P^βα = (P^αβ)ᵀ for real spinors.
    /// @param bra Row spin component σ.
    /// @param ket Column spin component σ'.
    /// @return P^{σσ'} [n x n] (dimensionless).
    /// @throws std::invalid_argument when the determinant is not generalized.
    RealMatrix spin_block(Spin bra, Spin ket) const {
        if (!generalized_)
            throw std::invalid_argument(
                "SlaterDeterminant::spin_block: spin blocks are defined for "
                "the generalized (spinor) form only");
        const RealMatrix P = density_of_(alpha_);
        const std::size_t n = P.shape()[0] / 2;
        const std::size_t r = (bra == Spin::beta) ? n : 0;
        const std::size_t c = (ket == Spin::beta) ? n : 0;
        return xt::view(P, xt::range(r, r + n), xt::range(c, c + n));
    }

    /// @brief Total density P^α + P^β (AO basis, dimensionless). For the
    ///        restricted form this is the factor-2 closed-shell density
    ///        2·C_occ C_occᵀ — a derived value, not the storage convention.
    ///        For the generalized form it is the n×n charge density
    ///        P^αα + P^ββ (the spin-off-diagonal blocks carry no charge).
    /// @return P_total [n x n] (dimensionless).
    RealMatrix density_total() const {
        if (generalized_)
            return spin_block(Spin::alpha, Spin::alpha)
                   + spin_block(Spin::beta, Spin::beta);
        if (restricted_)
            return 2.0 * density(Spin::alpha);
        return density(Spin::alpha) + density(Spin::beta);
    }

    /// @brief One-electron expectation value ⟨Φ|Ô|Φ⟩ = Tr(P_total·O).
    ///
    /// @warning Valid for **one-electron** operators only: a two-electron
    ///          operator needs the two-particle density matrix and is not
    ///          expressible as Tr(P·G) for any fixed matrix G. In
    ///          particular the mean-field total energy is **not**
    ///          `expect(F)` — Σε double-counts the electron repulsion;
    ///          E_HF = ½Tr[P(H_core + F)] + E_nn.
    /// @param O AO matrix of the one-electron operator [n x n]; units of O.
    /// @return Tr(P_total·O), in the units of O.
    /// @throws std::invalid_argument when O's shape does not match the AO
    ///         dimension.
    double expect(const RealMatrix& O) const {
        const RealMatrix P = density_total();
        if (O.shape()[0] != P.shape()[0] || O.shape()[1] != P.shape()[1])
            throw std::invalid_argument(
                "SlaterDeterminant::expect: operator shape ("
                + std::to_string(O.shape()[0]) + "x"
                + std::to_string(O.shape()[1]) + ") != AO dimension ("
                + std::to_string(P.shape()[0]) + ")");
        const RealMatrix PO = xt::linalg::dot(P, O);
        double tr = 0.0;
        for (std::size_t i = 0; i < PO.shape()[0]; ++i)
            tr += PO(i, i);
        return tr;
    }

    /// @brief Orbital energy ε_i of channel σ.
    /// @param s Spin channel.
    /// @param i MO index (0-based).
    /// @return ε_i [Hartree].
    /// @throws std::out_of_range when i is out of range.
    double orbital_energy(Spin s, int i) const {
        const Channel& ch = channel_(s);
        if (i < 0 || i >= static_cast<int>(ch.epsilon.size()))
            throw std::out_of_range(
                "SlaterDeterminant::orbital_energy: index "
                + std::to_string(i) + " out of range [0, "
                + std::to_string(ch.epsilon.size()) + ")");
        return ch.epsilon(i);
    }

    /// @brief Highest occupied orbital energy of channel σ (occupation-
    ///        defined: max ε_i over {i : f_i > 0}). Per Koopmans, −homo()
    ///        estimates the ionization potential (frozen orbitals).
    /// @param s Spin channel.
    /// @return ε_HOMO [Hartree].
    /// @throws std::invalid_argument when the channel has no occupied
    ///         orbital.
    double homo(Spin s) const {
        const Channel& ch = channel_(s);
        bool found = false;
        double e = 0.0;
        for (std::size_t i = 0; i < ch.occupation.size(); ++i) {
            if (ch.occupation(i) > 0.0 && (!found || ch.epsilon(i) > e)) {
                e = ch.epsilon(i);
                found = true;
            }
        }
        if (!found)
            throw std::invalid_argument(
                "SlaterDeterminant::homo: channel has no occupied orbital");
        return e;
    }

    /// @brief Lowest unoccupied orbital energy of channel σ (min ε_i over
    ///        {i : f_i = 0}); returns homo() when the channel has no
    ///        virtual orbital (existing observer semantics).
    /// @param s Spin channel.
    /// @return ε_LUMO [Hartree] (or ε_HOMO when no virtuals exist).
    double lumo(Spin s) const {
        const Channel& ch = channel_(s);
        bool found = false;
        double e = 0.0;
        for (std::size_t i = 0; i < ch.occupation.size(); ++i) {
            if (ch.occupation(i) == 0.0 && (!found || ch.epsilon(i) < e)) {
                e = ch.epsilon(i);
                found = true;
            }
        }
        return found ? e : homo(s);
    }

    /// @brief ΔSCF/MOM entry point: move one electron of channel σ from an
    ///        occupied orbital to a virtual one, returning a re-validated
    ///        new determinant (this object is unchanged).
    /// @param s Spin channel (the restricted form edits its single shared
    ///        channel — one spatial-orbital promotion per spin).
    /// @param from Source MO index; must have f_from = 1.
    /// @param to Target MO index; must have f_to = 0.
    /// @return The promoted determinant.
    /// @throws std::invalid_argument on out-of-range indices or when
    ///         f_from != 1 or f_to != 0 (only integer transfers are defined).
    SlaterDeterminant with_occupation(Spin s, int from, int to) const& {
        SlaterDeterminant copy(*this);
        copy.promote_(s, from, to);
        return copy;
    }

    /// @brief Rvalue overload of `with_occupation` — mutates this expiring
    ///        value in place and moves it out (chainable).
    /// @param s Spin channel.
    /// @param from Source MO index (f_from = 1 required).
    /// @param to Target MO index (f_to = 0 required).
    /// @return The promoted determinant (moved from *this).
    /// @throws std::invalid_argument as the const& overload.
    SlaterDeterminant with_occupation(Spin s, int from, int to) && {
        promote_(s, from, to);
        return std::move(*this);
    }

    /// @brief Validate the overlap-dependent invariants: Tr(P^σS) = N^σ per
    ///        channel, and — only when every f is integer (a true single
    ///        determinant) — idempotency P^σSP^σ = P^σ. Fractional
    ///        occupations skip the idempotency check (it fails exactly
    ///        then, by construction). The generalized form validates the
    ///        spinor density against the spinor metric S⊗1₂ built from the
    ///        given AO overlap: Tr(P·S⊗1₂) = N and P(S⊗1₂)P = P.
    /// @param S AO overlap matrix [n x n] (dimensionless) — always the
    ///        n-dimensional AO overlap, also for the generalized form.
    /// @param tol Absolute tolerance on both checks (dimensionless).
    /// @throws std::invalid_argument when S does not match the AO dimension
    ///         (generalized: 2·n_S must equal the spinor row count).
    /// @throws std::runtime_error naming the channel and deviation on
    ///         violation (a runtime numerical invariant, not an argument
    ///         error — CanonicalOrth precedent).
    void validate(const RealMatrix& S, double tol = 1e-8) const {
        if (generalized_) {
            const std::size_t two_n = alpha_.C.shape()[0];
            if (S.shape()[0] * 2 != two_n || S.shape()[1] * 2 != two_n)
                throw std::invalid_argument(
                    "SlaterDeterminant::validate: AO overlap is "
                    + std::to_string(S.shape()[0]) + "x"
                    + std::to_string(S.shape()[1])
                    + " but the spinor dimension is "
                    + std::to_string(two_n) + " (expected 2n x N_mo C)");
            const std::size_t n = two_n / 2;
            RealMatrix S_spinor = xt::zeros<double>({two_n, two_n});
            xt::view(S_spinor, xt::range(0, n), xt::range(0, n)) = S;
            xt::view(S_spinor, xt::range(n, two_n), xt::range(n, two_n)) = S;
            validate_density_("spinor", alpha_, density_of_(alpha_), S_spinor,
                              tol);
            return;
        }
        validate_density_("alpha", channel_(Spin::alpha),
                          density_of_(channel_(Spin::alpha)), S, tol);
        if (!restricted_)
            validate_density_("beta", channel_(Spin::beta),
                              density_of_(channel_(Spin::beta)), S, tol);
    }

  private:
    /// One spin channel: coefficients, energies, occupations (AO basis).
    struct Channel {
        RealMatrix C;          ///< MO coefficients [n x n_mo].
        RealVector epsilon;    ///< Orbital energies [Hartree].
        RealVector occupation; ///< Per-spin occupations in [0, 1].
    };

    // Restricted AND generalized forms hold a single channel — β queries
    // forward to it; only the unrestricted form has a distinct β channel.
    const Channel& channel_(Spin s) const {
        return (s == Spin::beta && !restricted_ && !generalized_) ? beta_
                                                                  : alpha_;
    }
    Channel& channel_(Spin s) {
        return (s == Spin::beta && !restricted_ && !generalized_) ? beta_
                                                                  : alpha_;
    }

    /// Density of one stored channel: P = Σ_i f_i c_i c_iᵀ (spinor-dimension
    /// 2n×2n when the channel holds spinor coefficients).
    static RealMatrix density_of_(const Channel& ch) {
        const int n = static_cast<int>(ch.C.shape()[0]);
        const int n_mo = static_cast<int>(ch.C.shape()[1]);
        // Gather occupied columns weighted by sqrt(f) so that
        // C' C'^T = sum_i f_i c_i c_i^T; integer occupations copy columns
        // unchanged (sqrt(1) = 1) and reproduce the plain GEMM bitwise.
        int n_occ = 0;
        for (int i = 0; i < n_mo; ++i)
            if (ch.occupation(i) > 0.0)
                ++n_occ;
        RealMatrix C_occ = xt::zeros<double>({n, n_occ});
        int col = 0;
        for (int i = 0; i < n_mo; ++i) {
            const double f = ch.occupation(i);
            if (f <= 0.0)
                continue;
            const double w = (f == 1.0) ? 1.0 : std::sqrt(f);
            xt::view(C_occ, xt::all(), col) = w * xt::view(ch.C, xt::all(), i);
            ++col;
        }
        return linalg::build_density_matrix(C_occ, n_occ, n, 1.0);
    }

    static void check_channel_(const char* name, const Channel& ch, int n_el) {
        const std::size_t n_mo = ch.C.shape().size() == 2 ? ch.C.shape()[1] : 0;
        if (ch.C.shape().size() != 2 || ch.epsilon.size() != n_mo
            || ch.occupation.size() != n_mo)
            throw std::invalid_argument(
                std::string("SlaterDeterminant: channel ") + name
                + " dimension mismatch: C is "
                + (ch.C.shape().size() == 2
                       ? std::to_string(ch.C.shape()[0]) + "x"
                             + std::to_string(ch.C.shape()[1])
                       : std::string("not a matrix"))
                + ", epsilon size " + std::to_string(ch.epsilon.size())
                + ", occupation size " + std::to_string(ch.occupation.size()));
        double sum = 0.0;
        for (std::size_t i = 0; i < n_mo; ++i) {
            const double f = ch.occupation(i);
            if (f < 0.0 || f > 1.0)
                throw std::invalid_argument(
                    std::string("SlaterDeterminant: channel ") + name
                    + " occupation f[" + std::to_string(i) + "] = "
                    + std::to_string(f) + " outside [0, 1]");
            sum += f;
        }
        const double dev = std::abs(sum - static_cast<double>(n_el));
        if (dev > 1e-12 * std::max(1.0, static_cast<double>(n_el)))
            throw std::invalid_argument(
                std::string("SlaterDeterminant: channel ") + name
                + " occupation sum " + std::to_string(sum)
                + " != electron count " + std::to_string(n_el)
                + " (deviation " + std::to_string(dev) + ")");
    }

    void promote_(Spin s, int from, int to) {
        Channel& ch = channel_(s);
        const int n_mo = static_cast<int>(ch.occupation.size());
        if (from < 0 || from >= n_mo || to < 0 || to >= n_mo)
            throw std::invalid_argument(
                "SlaterDeterminant::with_occupation: index out of range "
                "(from=" + std::to_string(from) + ", to=" + std::to_string(to)
                + ", n_mo=" + std::to_string(n_mo) + ")");
        if (ch.occupation(from) != 1.0 || ch.occupation(to) != 0.0)
            throw std::invalid_argument(
                "SlaterDeterminant::with_occupation: requires f[from] = 1 "
                "and f[to] = 0 (got f[" + std::to_string(from) + "] = "
                + std::to_string(ch.occupation(from)) + ", f["
                + std::to_string(to) + "] = "
                + std::to_string(ch.occupation(to)) + ")");
        ch.occupation(from) = 0.0;
        ch.occupation(to) = 1.0;
        const bool single_channel = restricted_ || generalized_;
        check_channel_(generalized_
                           ? "spinor"
                           : (restricted_ ? "alpha(restricted)"
                                          : (s == Spin::beta ? "beta"
                                                             : "alpha")),
                       ch,
                       (s == Spin::beta && !single_channel) ? n_beta_
                                                            : n_alpha_);
    }

    void validate_density_(const char* name, const Channel& ch,
                           const RealMatrix& P, const RealMatrix& S,
                           double tol) const {
        const RealMatrix PS = xt::linalg::dot(P, S);
        double tr = 0.0;
        for (std::size_t i = 0; i < PS.shape()[0]; ++i)
            tr += PS(i, i);
        double expected = 0.0;
        for (std::size_t i = 0; i < ch.occupation.size(); ++i)
            expected += ch.occupation(i);
        if (std::abs(tr - expected) > tol)
            throw std::runtime_error(
                std::string("SlaterDeterminant::validate: channel ") + name
                + " Tr(PS) = " + std::to_string(tr) + " != sum(f) = "
                + std::to_string(expected) + " (deviation "
                + std::to_string(std::abs(tr - expected)) + ")");
        bool integer_occ = true;
        for (std::size_t i = 0; i < ch.occupation.size(); ++i) {
            const double f = ch.occupation(i);
            if (std::abs(f) > 1e-12 && std::abs(f - 1.0) > 1e-12) {
                integer_occ = false;
                break;
            }
        }
        if (!integer_occ)
            return; // idempotency holds iff f in {0,1}; skip for fractional
        const RealMatrix PSP = xt::linalg::dot(PS, P);
        double max_dev = 0.0;
        for (std::size_t i = 0; i < P.shape()[0]; ++i)
            for (std::size_t j = 0; j < P.shape()[1]; ++j)
                max_dev = std::max(max_dev, std::abs(PSP(i, j) - P(i, j)));
        if (max_dev > tol)
            throw std::runtime_error(
                std::string("SlaterDeterminant::validate: channel ") + name
                + " idempotency |PSP - P|_max = " + std::to_string(max_dev)
                + " > tol = " + std::to_string(tol));
    }

    Channel alpha_;    ///< Alpha channel (the only channel when restricted).
    Channel beta_;     ///< Beta channel (empty when restricted).
    int n_alpha_ = 0;  ///< Alpha electron count N^α.
    int n_beta_ = 0;   ///< Beta electron count N^β.
    bool restricted_ = true; ///< Restricted (β forwards to α) or not.
    bool generalized_ = false; ///< Generalized (two-component spinor) form.
    double spin_contamination_ = 0; ///< Cached ⟨S²⟩ − S(S+1) deviation.
};

} // namespace scf
} // namespace cpu
} // namespace ATV
