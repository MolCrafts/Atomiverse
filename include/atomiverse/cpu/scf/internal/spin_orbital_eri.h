/// @file cpu/scf/internal/spin_orbital_eri.h
/// @brief Single authoritative spin-orbital ERI antisymmetrization + the MP
///        aufbau energy-denominator guard (private post-HF workspace).
///
/// The four post-HF methods that build a spin-orbital residual (CCSD, CISD,
/// MP3, MP4) each used to carry a private copy of the same two helpers: a
/// spatial+spin label and the physicist-notation antisymmetrized integral
/// \f$\langle pq\Vert rs\rangle\f$ from the chemist MO tensor \f$(pq|rs)\f$.
/// They collapse here to one inline definition. Two spin-orbital enumeration
/// orders coexist unchanged: CCSD/CISD walk block-spin order (all α then all
/// β) with `SpinOrbital` labels; MP3/MP4 walk interleaved order
/// (`so = 2·spat + spin`) with bit-packed integer indices — the packed
/// overload decodes via `SpinOrbital::from_packed` and forwards to the one
/// struct implementation, so both call paths run identical float ops.
///
/// Reference: Szabo & Ostlund, *Modern Quantum Chemistry*, Dover (1996),
/// §2.4 (spin-orbital ERIs) and §6.5 (Møller–Plesset aufbau denominators).
#pragma once

#include <stdexcept>

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Spin-orbital label: spatial MO index plus spin channel. A two-int
///        tag value (not a parameter bag), used to gather the chemist MO ERI
///        and compare spins in `antisym_eri`.
struct SpinOrbital {
    int spat; ///< Spatial MO index.
    int spin; ///< Spin channel (0 = α, 1 = β).

    /// @brief Decode a bit-packed spin-orbital index `so = 2·spat + spin`.
    /// @param so Packed spin-orbital index (`spat = so>>1`, `spin = so&1`).
    /// @return The `{spat, spin}` label (zero-cost, usable in a constant
    ///         expression).
    static constexpr SpinOrbital from_packed(int so) {
        return SpinOrbital{so >> 1, so & 1};
    }
};

/// @brief Antisymmetrized spin-orbital ERI
///        \f$\langle pq\Vert rs\rangle = (pr|qs)\,\delta_{\sigma_p\sigma_r}
///        \delta_{\sigma_q\sigma_s} - (ps|qr)\,\delta_{\sigma_p\sigma_s}
///        \delta_{\sigma_q\sigma_r}\f$ from the chemist MO tensor.
/// @param g Chemist-notation MO two-electron integrals \f$(ab|cd)\f$ [Hartree].
/// @param p First bra spin-orbital.
/// @param q Second bra spin-orbital.
/// @param r First ket spin-orbital.
/// @param s Second ket spin-orbital.
/// @return \f$\langle pq\Vert rs\rangle\f$ [Hartree].
inline double antisym_eri(const xt::xtensor<double, 4>& g, const SpinOrbital& p,
                          const SpinOrbital& q, const SpinOrbital& r,
                          const SpinOrbital& s) {
    double v = 0.0;
    if (p.spin == r.spin && q.spin == s.spin)
        v += g(p.spat, r.spat, q.spat, s.spat); // (pr|qs)
    if (p.spin == s.spin && q.spin == r.spin)
        v -= g(p.spat, s.spat, q.spat, r.spat); // (ps|qr)
    return v;
}

/// @brief Bit-packed overload of `antisym_eri` for producers that carry
///        interleaved `so = 2·spat + spin` loop indices (MP3/MP4). Decodes
///        each index with `SpinOrbital::from_packed` and forwards to the sole
///        struct implementation above.
/// @param g Chemist-notation MO two-electron integrals \f$(ab|cd)\f$ [Hartree].
/// @param p First bra packed spin-orbital index.
/// @param q Second bra packed spin-orbital index.
/// @param r First ket packed spin-orbital index.
/// @param s Second ket packed spin-orbital index.
/// @return \f$\langle pq\Vert rs\rangle\f$ [Hartree].
inline double antisym_eri(const xt::xtensor<double, 4>& g, int p, int q, int r,
                          int s) {
    return antisym_eri(g, SpinOrbital::from_packed(p),
                       SpinOrbital::from_packed(q), SpinOrbital::from_packed(r),
                       SpinOrbital::from_packed(s));
}

/// @brief Møller–Plesset aufbau energy denominator
///        \f$\Delta_{ij}^{ab} = \varepsilon_i + \varepsilon_j
///        - \varepsilon_a - \varepsilon_b\f$, guarding the aufbau invariant.
///
/// A closed-shell aufbau reference keeps every virtual above every occupied
/// orbital, so \f$\Delta < 0\f$ strictly. A non-negative value signals a
/// degenerate or non-aufbau reference and is rejected rather than emitting
/// garbage.
/// @param e_i Occupied orbital energy \f$\varepsilon_i\f$ [Hartree].
/// @param e_j Occupied orbital energy \f$\varepsilon_j\f$ [Hartree].
/// @param e_a Virtual orbital energy \f$\varepsilon_a\f$ [Hartree].
/// @param e_b Virtual orbital energy \f$\varepsilon_b\f$ [Hartree].
/// @return The strictly negative denominator \f$\Delta_{ij}^{ab}\f$ [Hartree].
/// @throws std::invalid_argument when \f$\Delta \ge 0\f$ (degenerate /
///         non-aufbau reference).
inline double mp_denominator(double e_i, double e_j, double e_a, double e_b) {
    const double denom = e_i + e_j - e_a - e_b;
    if (denom >= 0.0) {
        throw std::invalid_argument(
            "mp_denominator: non-negative orbital-energy denominator; "
            "reference is degenerate or non-aufbau.");
    }
    return denom;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
