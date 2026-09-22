/// @file cpu/gpw/berry_polarization.cpp
/// @brief King–Smith–Vanderbilt Berry-phase polarization on GPW multi-k
///        RKS/UKS orbitals (string-mesh SCF + occupied MO link dets).

#include "atomiverse/cpu/gpw/berry_polarization.h"

#include "atomiverse/core/log.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/one_electron.h"
#include "atomiverse/cpu/gpw/internal/periodic.h"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ATV {
namespace cpu {
namespace gpw {

namespace {

using Cdouble = std::complex<double>;
constexpr Cdouble I{0.0, 1.0};
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

/// ⟨a|e^{-i b·r}|b⟩ for Cartesian Gaussian primitives (product-center phase).
Cdouble prim_overlap_phase(const GridPrimitive& a, const GridPrimitive& b,
                           const double bvec[3]) {
    const double s = prim_overlap(a, b);
    if (s == 0.0)
        return Cdouble{0.0, 0.0};
    const double p = a.alpha + b.alpha;
    if (!(p > 0.0))
        return Cdouble{s, 0.0};
    const double Rp[3] = {(a.alpha * a.center[0] + b.alpha * b.center[0]) / p,
                          (a.alpha * a.center[1] + b.alpha * b.center[1]) / p,
                          (a.alpha * a.center[2] + b.alpha * b.center[2]) / p};
    const double b2 = bvec[0] * bvec[0] + bvec[1] * bvec[1] + bvec[2] * bvec[2];
    const double phase = bvec[0] * Rp[0] + bvec[1] * Rp[1] + bvec[2] * Rp[2];
    const double damp = std::exp(-b2 / (4.0 * p));
    return Cdouble{s, 0.0} * damp * std::exp(-I * phase);
}

/// Cross-k AO overlap S^{k, k+b}_{μν} = Σ_T e^{i q·T} ⟨φ_μ|e^{-ib·r}|φ_ν^T⟩
/// with q = k+b (phase on ket image) and b = q−k.
void build_S_cross(const std::vector<GridBasisFunction>& basis,
                   const std::vector<double>& radii, const double L[3],
                   const int m[3], double screen,
                   const std::array<double, 3>& k_frac_nb,
                   const double b_bohr[3], int nb,
                   std::vector<Cdouble>& S /* nb*nb row-major */) {
    S.assign(static_cast<std::size_t>(nb * nb), Cdouble{0, 0});
    for (int i = 0; i < nb; ++i) {
        const auto& a = basis[static_cast<std::size_t>(i)];
        if (a.empty())
            continue;
        for (int j = 0; j < nb; ++j) {
            const auto& b = basis[static_cast<std::size_t>(j)];
            if (b.empty())
                continue;
            double best = 0.0;
            for (const auto& p : a)
                for (const auto& q : b) {
                    double d2 = 0.0;
                    for (int d = 0; d < 3; ++d) {
                        double e = p.center[d] - q.center[d];
                        if (L[d] > 0.0)
                            e -= L[d] * std::nearbyint(e / L[d]);
                        d2 += e * e;
                    }
                    best = std::max(best,
                                    std::abs(p.coeff * q.coeff)
                                        * std::exp(-p.alpha * q.alpha
                                                   / (p.alpha + q.alpha) * d2));
                }
            if (best < screen)
                continue;
            const double reach = radii[static_cast<std::size_t>(i)]
                                 + radii[static_cast<std::size_t>(j)];
            const double reach2 = reach * reach;
            const double* ac = a[0].center;
            const double* bc = b[0].center;
            Cdouble s{0, 0};
            for (int tx = -m[0]; tx <= m[0]; ++tx)
                for (int ty = -m[1]; ty <= m[1]; ++ty)
                    for (int tz = -m[2]; tz <= m[2]; ++tz) {
                        const double T[3] = {tx * L[0], ty * L[1], tz * L[2]};
                        const double dx = ac[0] - bc[0] - T[0];
                        const double dy = ac[1] - bc[1] - T[1];
                        const double dz = ac[2] - bc[2] - T[2];
                        if (dx * dx + dy * dy + dz * dz > reach2)
                            continue;
                        const double arg =
                            kTwoPi
                            * (k_frac_nb[0] * static_cast<double>(tx)
                               + k_frac_nb[1] * static_cast<double>(ty)
                               + k_frac_nb[2] * static_cast<double>(tz));
                        const Cdouble phase(std::cos(arg), std::sin(arg));
                        Cdouble local{0, 0};
                        for (const auto& pa : a)
                            for (const auto& pb : b) {
                                GridPrimitive q = pb;
                                q.center[0] += T[0];
                                q.center[1] += T[1];
                                q.center[2] += T[2];
                                local += prim_overlap_phase(pa, q, b_bohr);
                            }
                        s += phase * local;
                    }
            S[static_cast<std::size_t>(i * nb + j)] = s;
        }
    }
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

/// String-averaged Berry phase for one occupied channel on a string mesh.
double string_phase_channel(const GpwResult& res, int n_occ, int spin, int dir,
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

    std::vector<std::vector<Cdouble>> C(static_cast<std::size_t>(nk));
    for (int ik = 0; ik < nk; ++ik)
        C[static_cast<std::size_t>(ik)] =
            flatten_C_occ(res, ik, nbas, n_occ, spin);

    std::vector<double> phis;
    phis.reserve(static_cast<std::size_t>(n_strings));
    for (int s = 0; s < n_strings; ++s) {
        std::vector<Cdouble> dets(static_cast<std::size_t>(J));
        for (int j = 0; j < J; ++j) {
            const int ib = s * J + j;
            const int ik = s * J + (j + 1) % J;
            const auto& kfn = res.k_frac[static_cast<std::size_t>(ik)];
            std::vector<Cdouble> S;
            build_S_cross(basis, radii, cell, mimg, screen, kfn, b_bohr, nbas,
                          S);
            std::vector<Cdouble> M;
            mo_overlap(C[static_cast<std::size_t>(ib)],
                       C[static_cast<std::size_t>(ik)], S, nbas, n_occ, M);
            dets[static_cast<std::size_t>(j)] = det_lu(M, n_occ);
        }
        phis.push_back(phase_from_link_dets(dets));
        ATV_DEBUG_CAT("gpw",
                      "berry dir={} spin={} n_occ={} string={} phi={:+.6f}",
                      dir, spin, n_occ, s, phis.back());
    }

    const double phi0 = phis[0];
    double acc = 0.0;
    for (const double phi : phis)
        acc += phi0 + wrap_centered(phi - phi0, kTwoPi);
    return acc / static_cast<double>(n_strings);
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

BerryPolarizationResult
BerryPolarization::compute(GpwDft& method,
                           const std::vector<GpwAtom>& atoms) const {
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
    fill_ionic(out, atoms, method.cell);
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
    pair_image_counts(method.cell, /*periodic=*/true, 2.0 * rmax, mimg);
    const double screen = method.eps_default;

    const int n_up = uks ? method.n_alpha : method.n_occ;
    const int n_dn = uks ? method.n_beta : 0;
    const double quantum = uks ? 1.0 : 2.0;

    // One multi-k SCF per requested direction (string mesh). Directions share
    // no common mesh in general (string-major layouts differ), so each active
    // direction gets its own run. Callers that only need one axis (e.g. Born
    // Z*) set directions accordingly.
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

        const GpwResult& res = method.run(atoms);
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

        if (!uks) {
            const double phase = string_phase_channel(
                res, n_up, /*spin=*/0, dir, k_string, k_perp, basis, radii,
                method.cell, mimg, screen);
            out.phase_el[static_cast<std::size_t>(dir)] = phase;
            out.p_el_frac[static_cast<std::size_t>(dir)] = 2.0 * phase / kTwoPi;
        } else {
            double phase_sum = 0.0;
            for (int s = 0; s < 2; ++s) {
                const int n_ch = (s == 0) ? n_up : n_dn;
                if (n_ch < 1)
                    continue;
                const double phi_s = string_phase_channel(
                    res, n_ch, s, dir, k_string, k_perp, basis, radii,
                    method.cell, mimg, screen);
                phase_sum += phi_s;
                ATV_DEBUG_CAT("gpw", "berry-spin dir={} spin={} phi={:+.6f}",
                              dir, s, phi_s);
            }
            out.phase_el[static_cast<std::size_t>(dir)] = 0.5 * phase_sum;
            out.p_el_frac[static_cast<std::size_t>(dir)] = phase_sum / kTwoPi;
        }
        out.computed[static_cast<std::size_t>(dir)] = true;
    }

    assemble_totals(out, method.cell, quantum);
    return out;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
