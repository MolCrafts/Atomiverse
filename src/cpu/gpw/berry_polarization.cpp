/// @file cpu/gpw/berry_polarization.cpp
/// @brief King–Smith–Vanderbilt Berry-phase polarization on GPW multi-k
///        RKS/UKS orbitals (string-mesh SCF + occupied MO link dets).

#include "atomiverse/cpu/gpw/berry_polarization.h"

#include "atomiverse/core/log.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/kpoints.h"
#include "atomiverse/cpu/gpw/internal/periodic.h"

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ATV {
namespace cpu {
namespace gpw {

using internal::GpwAtom;

namespace {

using Cdouble = std::complex<double>;
constexpr double kTwoPi = 6.283185307179586476925286766559;

/// Determinant of a dense complex n x n matrix (in-place LU, partial pivot).
Cdouble det_lu(std::vector<Cdouble>& a, int n) {
    Cdouble det{1.0, 0.0};
    for (int col = 0; col < n; ++col) {
        int piv = col;
        double best = std::abs(a[static_cast<std::size_t>(col * n + col)]);
        for (int r = col + 1; r < n; ++r) {
            const double m = std::abs(a[static_cast<std::size_t>(r * n + col)]);
            if (m > best) {
                best = m;
                piv = r;
            }
        }
        if (!(best > 0.0))
            return {0.0, 0.0};
        if (piv != col) {
            for (int c = 0; c < n; ++c)
                std::swap(a[static_cast<std::size_t>(col * n + c)],
                          a[static_cast<std::size_t>(piv * n + c)]);
            det = -det;
        }
        const auto pivot = a[static_cast<std::size_t>(col * n + col)];
        det *= pivot;
        for (int r = col + 1; r < n; ++r) {
            const auto f = a[static_cast<std::size_t>(r * n + col)] / pivot;
            for (int c = col; c < n; ++c)
                a[static_cast<std::size_t>(r * n + c)] -=
                    f * a[static_cast<std::size_t>(col * n + c)];
        }
    }
    return det;
}

/// M = C†(k) S_cross C(k+b); C is nb × n_occ (columns = occupied MOs).
void mo_overlap(const std::vector<Cdouble>& Ck, const std::vector<Cdouble>& Ckb,
                const std::vector<Cdouble>& S, int nbas, int n_occ,
                std::vector<Cdouble>& M) {
    std::vector<Cdouble> T(static_cast<std::size_t>(nbas * n_occ),
                           Cdouble{0, 0});
    for (int mu = 0; mu < nbas; ++mu)
        for (int n = 0; n < n_occ; ++n) {
            Cdouble s{0, 0};
            for (int nu = 0; nu < nbas; ++nu)
                s += S[static_cast<std::size_t>(mu * nbas + nu)]
                     * Ckb[static_cast<std::size_t>(nu * n_occ + n)];
            T[static_cast<std::size_t>(mu * n_occ + n)] = s;
        }
    M.assign(static_cast<std::size_t>(n_occ * n_occ), Cdouble{0, 0});
    for (int m = 0; m < n_occ; ++m)
        for (int n = 0; n < n_occ; ++n) {
            Cdouble s{0, 0};
            for (int mu = 0; mu < nbas; ++mu)
                s += std::conj(Ck[static_cast<std::size_t>(mu * n_occ + m)])
                     * T[static_cast<std::size_t>(mu * n_occ + n)];
            M[static_cast<std::size_t>(m * n_occ + n)] = s;
        }
}

std::vector<GridBasisFunction>
flatten_basis(const std::vector<GpwAtom>& atoms) {
    std::vector<GridBasisFunction> basis;
    for (const auto& a : atoms)
        for (const auto& bf : a.basis)
            basis.push_back(bf);
    return basis;
}

/// Flatten occupied C_k (nb × n_occ) from GpwResult payload.
std::vector<Cdouble> flatten_C_occ(const GpwResult& res, int ik, int nbas,
                                   int n_occ, int spin) {
    std::vector<Cdouble> C(static_cast<std::size_t>(nbas * n_occ),
                           Cdouble{0, 0});
    const auto& occ_stack = (spin == 0) ? res.C_occ_k : res.C_occ_k_beta;
    const auto& full_stack = (spin == 0) ? res.C_k : res.C_k_beta;
    if (!occ_stack.empty() && static_cast<int>(occ_stack.size()) > ik
        && !occ_stack[static_cast<std::size_t>(ik)].empty()) {
        const auto& raw = occ_stack[static_cast<std::size_t>(ik)];
        const int ncol = static_cast<int>(raw.size()) / std::max(nbas, 1);
        const int nuse = std::min(n_occ, ncol);
        for (int mu = 0; mu < nbas; ++mu)
            for (int o = 0; o < nuse; ++o)
                C[static_cast<std::size_t>(mu * n_occ + o)] =
                    raw[static_cast<std::size_t>(mu * ncol + o)];
        return C;
    }
    if (static_cast<int>(full_stack.size()) <= ik)
        throw std::runtime_error(
            "BerryPolarization: missing C_k at string k-point");
    const auto& Ck = full_stack[static_cast<std::size_t>(ik)];
    const int nb0 = static_cast<int>(Ck.shape(0));
    const int nmo = static_cast<int>(Ck.shape(1));
    if (nb0 != nbas)
        throw std::runtime_error("BerryPolarization: C_k row count mismatch");
    const int nuse = std::min(n_occ, nmo);
    for (int mu = 0; mu < nbas; ++mu)
        for (int o = 0; o < nuse; ++o)
            C[static_cast<std::size_t>(mu * n_occ + o)] = Ck(mu, o);
    return C;
}

/// String-averaged phases of the occupied channels on one string mesh.
/// The AO cross-k operator is spin independent; build it once per link.
std::array<double, 2> string_phases(const GpwResult& res,
                                    const std::array<int, 2>& n_occ, int dir,
                                    int k_string, int k_perp,
                                    const std::vector<GridBasisFunction>& basis,
                                    const std::vector<double>& radii,
                                    const double cell[3], const int mimg[3],
                                    double screen) {
    const int J = k_string;
    const int n_strings = k_perp * k_perp;
    const int nbas = static_cast<int>(basis.size());
    const int nk = res.n_kpts;
    if (nk != n_strings * J)
        throw std::runtime_error(
            "BerryPolarization: string-mesh n_kpts mismatch (got "
            + std::to_string(nk) + ", want " + std::to_string(n_strings * J)
            + ")");

    // Short reciprocal step along `dir` [1/Bohr]: b = 2π / (L_dir * J).
    double b_bohr[3] = {0, 0, 0};
    b_bohr[dir] = kTwoPi / (cell[dir] * static_cast<double>(J));

    std::array<std::vector<std::vector<Cdouble>>, 2> C;
    std::array<std::vector<double>, 2> phis;
    for (int spin = 0; spin < 2; ++spin) {
        if (n_occ[spin] < 1)
            continue;
        C[spin].resize(static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik)
            C[spin][static_cast<std::size_t>(ik)] =
                flatten_C_occ(res, ik, nbas, n_occ[spin], spin);
        phis[spin].reserve(static_cast<std::size_t>(n_strings));
    }

    std::vector<internal::FourierOverlapLink> links(
        static_cast<std::size_t>(nk));
    for (int s = 0; s < n_strings; ++s)
        for (int j = 0; j < J; ++j) {
            const int ib = s * J + j;
            const int ik = s * J + (j + 1) % J;
            links[static_cast<std::size_t>(ib)] = {
                res.k_frac[static_cast<std::size_t>(ik)],
                {b_bohr[0], b_bohr[1], b_bohr[2]}};
        }
    const auto S_cross = internal::build_fourier_overlap_links(
        basis, radii, cell, mimg, screen, links);

    for (int s = 0; s < n_strings; ++s) {
        std::array<std::vector<Cdouble>, 2> dets;
        for (int spin = 0; spin < 2; ++spin)
            if (n_occ[spin] > 0)
                dets[spin].resize(static_cast<std::size_t>(J));
        for (int j = 0; j < J; ++j) {
            const int ib = s * J + j;
            const int ik = s * J + (j + 1) % J;
            const auto& S = S_cross[static_cast<std::size_t>(ib)];
            for (int spin = 0; spin < 2; ++spin) {
                if (n_occ[spin] < 1)
                    continue;
                std::vector<Cdouble> M;
                mo_overlap(C[spin][static_cast<std::size_t>(ib)],
                           C[spin][static_cast<std::size_t>(ik)], S, nbas,
                           n_occ[spin], M);
                dets[spin][static_cast<std::size_t>(j)] =
                    det_lu(M, n_occ[spin]);
            }
        }
        for (int spin = 0; spin < 2; ++spin) {
            if (n_occ[spin] < 1)
                continue;
            phis[spin].push_back(phase_from_link_dets(dets[spin]));
            ATV_DEBUG_CAT("gpw",
                          "berry dir={} spin={} n_occ={} string={} phi={:+.6f}",
                          dir, spin, n_occ[spin], s, phis[spin].back());
        }
    }

    std::array<double, 2> phases{};
    for (int spin = 0; spin < 2; ++spin) {
        if (n_occ[spin] < 1)
            continue;
        const double phi0 = phis[spin][0];
        double acc = 0.0;
        for (const double phi : phis[spin])
            acc += phi0 + wrap_centered(phi - phi0, kTwoPi);
        phases[spin] = acc / static_cast<double>(n_strings);
    }
    return phases;
}

void fill_ionic(BerryPolarizationResult& out, const std::vector<GpwAtom>& atoms,
                const double cell[3]) {
    for (const auto& a : atoms) {
        for (int d = 0; d < 3; ++d) {
            if (!(cell[d] > 0.0))
                continue;
            double f = a.pos[d] / cell[d];
            f -= std::floor(f);
            if (f < 0.0)
                f += 1.0;
            out.p_ion_frac[static_cast<std::size_t>(d)] += a.z_val * f;
        }
    }
}

void assemble_totals(BerryPolarizationResult& out, const double cell[3],
                     double quantum) {
    const double a0 = units::kBohrInAngstrom;
    for (int d = 0; d < 3; ++d) {
        if (!out.computed[static_cast<std::size_t>(d)])
            continue;
        out.p_frac[static_cast<std::size_t>(d)] =
            wrap_centered(out.p_ion_frac[static_cast<std::size_t>(d)]
                              + out.p_el_frac[static_cast<std::size_t>(d)],
                          quantum);
        // Orthorhombic: a_i along Cartesian e_i.
        out.dipole_e_ang[static_cast<std::size_t>(d)] =
            out.p_frac[static_cast<std::size_t>(d)] * cell[d] * a0;
    }
}

} // namespace

double phase_from_link_dets(const std::vector<std::complex<double>>& dets) {
    if (dets.empty())
        throw std::invalid_argument("phase_from_link_dets: empty string");
    std::complex<double> z{1.0, 0.0};
    for (const auto& d : dets) {
        const double m = std::abs(d);
        if (m <= 1e-12)
            throw std::invalid_argument(
                "phase_from_link_dets: |det| <= 1e-12 — vanishing string "
                "overlap (metallic occupation or broken string)");
        z *= d / m;
    }
    return std::atan2(z.imag(), z.real());
}

double wrap_centered(double p, double quantum) {
    if (!(quantum > 0.0))
        throw std::invalid_argument("wrap_centered: quantum must be > 0");
    double w = p - quantum * std::round(p / quantum);
    if (w <= -0.5 * quantum)
        w += quantum;
    return w;
}

std::vector<std::array<double, 3>>
BerryPolarization::string_kpoints(int dir) const {
    if (dir < 0 || dir > 2)
        throw std::invalid_argument(
            "BerryPolarization::string_kpoints: dir must be in [0,2]");
    if (k_perp < 1)
        throw std::invalid_argument(
            "BerryPolarization::string_kpoints: k_perp must be >= 1");
    if (k_string < 2)
        throw std::invalid_argument(
            "BerryPolarization::string_kpoints: k_string must be >= 2");

    const int p1 = (dir + 1) % 3;
    const int p2 = (dir + 2) % 3;
    std::vector<std::array<double, 3>> kf;
    kf.reserve(static_cast<std::size_t>(k_perp * k_perp * k_string));
    for (int i1 = 0; i1 < k_perp; ++i1) {
        for (int i2 = 0; i2 < k_perp; ++i2) {
            for (int j = 0; j < k_string; ++j) {
                std::array<double, 3> f{0.0, 0.0, 0.0};
                f[static_cast<std::size_t>(p1)] =
                    static_cast<double>(i1) / static_cast<double>(k_perp);
                f[static_cast<std::size_t>(p2)] =
                    static_cast<double>(i2) / static_cast<double>(k_perp);
                f[static_cast<std::size_t>(dir)] =
                    static_cast<double>(j) / static_cast<double>(k_string);
                kf.push_back(f);
            }
        }
    }
    return kf;
}

BerryPolarizationResult BerryPolarization::compute(GpwDft& method,
                                                   Frame& frame) const {
    internal::GpwDftAccess::set_cell(method,
                                     internal::cell_from_box(frame.box));
    const std::vector<GpwAtom> atoms =
        internal::GpwDftAccess::atoms_from_frame(method, frame);
    if (method.smearing != Smearing::none)
        throw std::invalid_argument(
            "BerryPolarization::compute: Berry-phase polarization requires "
            "an insulator (no metallic smearing)");
    if (k_perp < 1 || k_string < 2)
        throw std::invalid_argument(
            "BerryPolarization::compute: need k_perp >= 1 and k_string >= 2");
    if (method.nonperiodic)
        throw std::invalid_argument(
            "BerryPolarization::compute: requires a periodic cell");
    const bool uks = method.is_uks();
    if (!uks && method.n_occ < 1)
        throw std::invalid_argument(
            "BerryPolarization::compute: n_occ must be >= 1");
    if (uks && method.n_alpha + method.n_beta < 1)
        throw std::invalid_argument(
            "BerryPolarization::compute: n_alpha + n_beta must be >= 1");

    BerryPolarizationResult out;
    fill_ionic(out, atoms, method.cell_bohr().data());
    out.scf_converged = true;

    const auto basis = flatten_basis(atoms);
    const int nbas = static_cast<int>(basis.size());
    std::vector<double> radii(static_cast<std::size_t>(nbas));
    double rmax = 0.0;
    for (int i = 0; i < nbas; ++i) {
        radii[static_cast<std::size_t>(i)] = basis_function_radius(
            basis[static_cast<std::size_t>(i)], method.eps_default);
        rmax = std::max(rmax, radii[static_cast<std::size_t>(i)]);
    }
    int mimg[3];
    pair_image_counts(method.cell_bohr().data(), /*periodic=*/true, 2.0 * rmax,
                      mimg);
    const double screen = method.eps_default;

    const int n_up = uks ? method.n_alpha : method.n_occ;
    const int n_dn = uks ? method.n_beta : 0;
    const double quantum = uks ? 1.0 : 2.0;

    // Each direction solves its own physical k-mesh through the canonical
    // public method path. Only its spin-independent AO links are shared.
    const bool prof = std::getenv("ATV_GPW_PROF") != nullptr;
    for (int dir = 0; dir < 3; ++dir) {
        if (!directions[static_cast<std::size_t>(dir)])
            continue;

        const auto kf = string_kpoints(dir);
        const int nk = static_cast<int>(kf.size());
        method.k_mesh[0] = method.k_mesh[1] = method.k_mesh[2] = 1;
        method.k_time_reversal = false;
        method.k_points = kf;
        method.k_point_weights.assign(static_cast<std::size_t>(nk),
                                      1.0 / static_cast<double>(nk));

        const auto scf_start = std::chrono::steady_clock::now();
        const GpwResult& res = method.run(frame);
        const auto scf_end = std::chrono::steady_clock::now();
        out.scf_converged = out.scf_converged && res.converged;
        out.energy_ha = res.e_total;
        if (!res.converged) {
            ATV_WARN("BerryPolarization: SCF did not converge on string "
                     "mesh dir={}; continuing with last orbitals",
                     dir);
        }
        if (res.n_kpts != nk || static_cast<int>(res.k_frac.size()) != nk)
            throw std::runtime_error(
                "BerryPolarization::compute: multi-k payload missing after "
                "string-mesh run");

        const auto phases =
            string_phases(res, {n_up, n_dn}, dir, k_string, k_perp, basis,
                          radii, method.cell_bohr().data(), mimg, screen);
        if (!uks) {
            out.phase_el[static_cast<std::size_t>(dir)] = phases[0];
            out.p_el_frac[static_cast<std::size_t>(dir)] =
                2.0 * phases[0] / kTwoPi;
        } else {
            const double phase_sum = phases[0] + phases[1];
            out.phase_el[static_cast<std::size_t>(dir)] = 0.5 * phase_sum;
            out.p_el_frac[static_cast<std::size_t>(dir)] = phase_sum / kTwoPi;
        }

        if (prof) {
            const auto end = std::chrono::steady_clock::now();
            std::fprintf(
                stderr,
                "[gpw berry] dir=%d scf=%.0f ms links=%.0f ms "
                "iterations=%d energy=%.12f Ha\n",
                dir,
                std::chrono::duration<double, std::milli>(scf_end - scf_start)
                    .count(),
                std::chrono::duration<double, std::milli>(end - scf_end)
                    .count(),
                res.iters, res.e_total);
        }
        out.computed[static_cast<std::size_t>(dir)] = true;
    }

    assemble_totals(out, method.cell_bohr().data(), quantum);
    return out;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
