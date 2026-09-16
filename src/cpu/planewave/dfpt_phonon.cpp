/// @file cpu/planewave/dfpt_phonon.cpp
/// @brief DfptPhonon — mass weighting and diagonalization over the internal
///        DFPT force-constant assembly (Γ; commensurate finite q via
///        supercell Γ-DFPT + IFC Fourier; incommensurate finite q via direct
///        Sternheimer `dfpt_q_force_constants`).

#include "atomiverse/cpu/planewave/dfpt_phonon.h"

#include <cmath>
#include <complex>
#include <optional>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/planewave/internal/dfpt.h"
#include "atomiverse/cpu/planewave/internal/pw_subspace.h"
#include "atomiverse/cpu/vibration/atomic_masses.h"
#include "atomiverse/cpu/vibration/nist_constants.h"
#include "atomiverse/cpu/vibration/phonon_dispersion.h"
#include "atomiverse/cpu/vibration/supercell.h"

namespace ATV {
namespace cpu {
namespace pw {

namespace {

bool is_gamma(const std::array<double, 3>& q) {
    return std::abs(q[0]) < 1e-14 && std::abs(q[1]) < 1e-14
           && std::abs(q[2]) < 1e-14;
}

/// @brief Smallest positive integer N ≤ nmax with N·x ≈ integer, or nullopt.
std::optional<int> commensurate_factor(double x, int nmax = 6) {
    const double ax = std::abs(x);
    if (ax < 1e-14)
        return 1;
    // Reduce to [0,1).
    double f = ax - std::floor(ax);
    if (f > 0.5)
        f = 1.0 - f;
    if (f < 1e-14)
        return 1;
    for (int n = 1; n <= nmax; ++n) {
        const double y = f * static_cast<double>(n);
        if (std::abs(y - std::round(y)) < 1e-8)
            return n;
    }
    return std::nullopt;
}

/// @brief Mass-weight Hermitian C [Ha/Bohr²], diagonalize, fill frequencies.
void diagonalize_complex_C(const std::vector<std::complex<AccReal>>& C,
                           int n, int dim, const int* types,
                           DfptPhononResult& result) {
    const double to_ev_ang2 =
        1.0 / vib::nist::EV_PER_ANG2_TO_HARTREE_PER_BOHR2;

    std::vector<double> inv_sqrt_m(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        inv_sqrt_m[static_cast<std::size_t>(i)] =
            1.0 / std::sqrt(vib::mass_amu(types[i]));

    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            const auto& cij =
                C[static_cast<std::size_t>(i) * dim + j];
            result.force_constants(i, j) =
                static_cast<double>(cij.real()) * to_ev_ang2;
            result.force_constants_imag(i, j) =
                static_cast<double>(cij.imag()) * to_ev_ang2;
        }

    std::vector<std::complex<AccReal>> dmat(
        static_cast<std::size_t>(dim) * dim);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            const AccReal scale = static_cast<AccReal>(
                inv_sqrt_m[static_cast<std::size_t>(i / 3)]
                * inv_sqrt_m[static_cast<std::size_t>(j / 3)]);
            dmat[static_cast<std::size_t>(i) * dim + j] =
                C[static_cast<std::size_t>(i) * dim + j] * scale;
        }

    std::vector<AccReal> eig(static_cast<std::size_t>(dim));
    std::vector<std::complex<AccReal>> vecs(
        static_cast<std::size_t>(dim) * dim);
    internal::jacobi_eigh_hermitian(dim, dmat.data(), eig.data(),
                                    vecs.data(), 50);

    const double au2cm = vib::nist::au2wavenumber();
    result.frequencies =
        xt::zeros<double>({static_cast<std::size_t>(dim)});
    for (int k = 0; k < dim; ++k) {
        const double lambda =
            static_cast<double>(eig[static_cast<std::size_t>(k)]);
        result.frequencies(k) =
            (lambda < 0.0) ? -std::sqrt(-lambda) * au2cm
                           : std::sqrt(lambda) * au2cm;
    }
}

} // namespace

DfptPhononResult DfptPhonon::compute(const PW& method,
                                     const cpu::Frame& frame) const {
    const int n = frame.nlocal();
    const int dim = 3 * n;
    const double to_ev_ang2 =
        1.0 / vib::nist::EV_PER_ANG2_TO_HARTREE_PER_BOHR2;

    DfptPhononResult result;
    result.n_atoms = n;
    result.force_constants = xt::zeros<double>(
        {static_cast<std::size_t>(dim), static_cast<std::size_t>(dim)});
    result.force_constants_imag = xt::zeros<double>(
        {static_cast<std::size_t>(dim), static_cast<std::size_t>(dim)});

    // ── Γ: native DFPT assembly ──────────────────────────────────────
    if (is_gamma(q_frac)) {
        const auto out = internal::dfpt_gamma_force_constants(
            frame, method, AccReal(response_tol), response_max_iter,
            AccReal(cg_tol), cg_max_iter, AccReal(mix_beta));
        result.scf_converged = out.scf_converged;
        result.response_converged = out.response_converged;
        result.max_asymmetry =
            static_cast<double>(out.max_asymmetry) * to_ev_ang2;

        std::vector<std::complex<AccReal>> C_c(
            static_cast<std::size_t>(dim) * dim, {0, 0});
        for (int i = 0; i < dim * dim; ++i)
            C_c[static_cast<std::size_t>(i)] =
                out.force_constants[static_cast<std::size_t>(i)];

        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j) {
                result.force_constants(i, j) =
                    static_cast<double>(
                        out.force_constants[static_cast<std::size_t>(i) * dim
                                            + j])
                    * to_ev_ang2;
            }

        const int* types = frame.span(kAtom_type);
        std::vector<double> inv_sqrt_m(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            inv_sqrt_m[static_cast<std::size_t>(i)] =
                1.0 / std::sqrt(vib::mass_amu(types[i]));

        std::vector<AccReal> dmat(static_cast<std::size_t>(dim) * dim);
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                dmat[static_cast<std::size_t>(i) * dim + j] =
                    out.force_constants[static_cast<std::size_t>(i) * dim + j]
                    * static_cast<AccReal>(
                        inv_sqrt_m[static_cast<std::size_t>(i / 3)])
                    * static_cast<AccReal>(
                        inv_sqrt_m[static_cast<std::size_t>(j / 3)]);

        std::vector<AccReal> eig(static_cast<std::size_t>(dim));
        std::vector<AccReal> vecs(static_cast<std::size_t>(dim) * dim);
        internal::jacobi_eigh(dim, dmat.data(), eig.data(), vecs.data(), 50);

        const double au2cm = vib::nist::au2wavenumber();
        result.frequencies =
            xt::zeros<double>({static_cast<std::size_t>(dim)});
        for (int k = 0; k < dim; ++k) {
            const double lambda =
                static_cast<double>(eig[static_cast<std::size_t>(k)]);
            result.frequencies(k) =
                (lambda < 0.0) ? -std::sqrt(-lambda) * au2cm
                               : std::sqrt(lambda) * au2cm;
        }
        return result;
    }

    // ── Commensurate finite q: supercell Γ-DFPT + IFC Fourier ─────────
    // A phonon at rational q = (h/na, k/nb, l/nc) is exactly the Γ phonon
    // of the na×nb×nc supercell folded back by e^{iq·R}. Reuses the
    // proven Sternheimer assembly (`dfpt_gamma_force_constants`) and the
    // PhononDispersion Fourier (same as IFC path, different Φ source).
    const auto fa = commensurate_factor(q_frac[0]);
    const auto fb = commensurate_factor(q_frac[1]);
    const auto fc = commensurate_factor(q_frac[2]);
    if (fa && fb && fc) {
        const int na = *fa;
        const int nb = *fb;
        const int nc = *fc;
        if (na * nb * nc <= 8) {
            const cpu::Frame sc = vib::expand_supercell(frame, na, nb, nc);
            const auto out = internal::dfpt_gamma_force_constants(
                sc, method, AccReal(response_tol), response_max_iter,
                AccReal(cg_tol), cg_max_iter, AccReal(mix_beta));
            result.scf_converged = out.scf_converged;
            result.response_converged = out.response_converged;
            result.max_asymmetry =
                static_cast<double>(out.max_asymmetry) * to_ev_ang2;

            const int n_sc = sc.nlocal();
            const int dim_sc = 3 * n_sc;
            xt::xtensor<double, 2> hess_sc = xt::zeros<double>(
                {static_cast<std::size_t>(dim_sc),
                 static_cast<std::size_t>(dim_sc)});
            for (int i = 0; i < dim_sc; ++i)
                for (int j = 0; j < dim_sc; ++j)
                    hess_sc(i, j) =
                        static_cast<double>(
                            out.force_constants[static_cast<std::size_t>(i)
                                                    * dim_sc
                                                + j])
                        * to_ev_ang2;

            vib::PhononDispersion path;
            const std::vector<double> freqs = path.frequencies_at_q(
                hess_sc, frame, na, nb, nc, q_frac);
            result.frequencies =
                xt::zeros<double>({static_cast<std::size_t>(dim)});
            for (int k = 0; k < dim; ++k)
                result.frequencies(k) = freqs[static_cast<std::size_t>(k)];

            for (int i = 0; i < n; ++i)
                for (int a = 0; a < 3; ++a)
                    for (int j = 0; j < n; ++j)
                        for (int b = 0; b < 3; ++b)
                            result.force_constants(3 * i + a, 3 * j + b) =
                                hess_sc(3 * i + a, 3 * j + b);

            return result;
        }
    }

    // ── Incommensurate / oversize: direct finite-q Sternheimer ────────
    const Vec3 qv{static_cast<AccReal>(q_frac[0]),
                  static_cast<AccReal>(q_frac[1]),
                  static_cast<AccReal>(q_frac[2])};
    const auto out = internal::dfpt_q_force_constants(
        frame, method, qv, AccReal(response_tol), response_max_iter,
        AccReal(cg_tol), cg_max_iter, AccReal(mix_beta));
    result.scf_converged = out.scf_converged;
    result.response_converged = out.response_converged;
    result.max_asymmetry =
        static_cast<double>(out.max_asymmetry) * to_ev_ang2;

    const int* types = frame.span(kAtom_type);
    diagonalize_complex_C(out.force_constants, n, dim, types, result);
    return result;
}

} // namespace pw
} // namespace cpu
} // namespace ATV
