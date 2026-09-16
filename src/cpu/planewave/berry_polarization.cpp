/// @file cpu/planewave/berry_polarization.cpp
/// @brief King–Smith–Vanderbilt Berry-phase polarization on frozen PW
///        \(V_\mathrm{eff}\) (NSCF strings + occupied overlap determinants).
///        Closed-shell (f=2) and collinear-spin (f=1 per channel).

#include "atomiverse/cpu/planewave/berry_polarization.h"

#include "atomiverse/core/log.h"
#include "atomiverse/core/units.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/restricted_scf.h"
#include "atomiverse/cpu/planewave/internal/veff_builder.h"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {

namespace {

/// Pack an integer G-triple into one map key (|h|,|k|,|l| < 2^20).
long long pack_hkl(int h, int k, int l) {
    const long long off = 1 << 20;
    return ((static_cast<long long>(h) + off) << 42)
           | ((static_cast<long long>(k) + off) << 21)
           | (static_cast<long long>(l) + off);
}

/// Determinant of a dense complex n x n matrix (in-place LU, partial pivot).
std::complex<double> det_lu(std::vector<std::complex<double>>& a, int n) {
    std::complex<double> det{1.0, 0.0};
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
            return {0.0, 0.0}; // singular
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

/// \(\det S\), \(S_{mn}=\sum_G c^*_{m,\mathrm{bra}}(G)\,
/// c_{n,\mathrm{ket}}(G+\Delta)\) with \(\Delta\) one reciprocal vector
/// along `shift_dir` (or none when `shift_dir < 0`). Coefficients are
/// matched by integer (h,k,l) triple; G outside the partner sphere
/// contributes zero.
std::complex<double> overlap_det(const internal::GVectorList& bra,
                                 const std::vector<std::complex<AccReal>>& cb,
                                 const internal::GVectorList& ket,
                                 const std::vector<std::complex<AccReal>>& ck,
                                 int n_occ, int shift_dir) {
    const int ng_bra = bra.count();
    const int ng_ket = ket.count();

    std::unordered_map<long long, int> ket_idx;
    ket_idx.reserve(static_cast<std::size_t>(2 * ng_ket));
    for (int jg = 0; jg < ng_ket; ++jg)
        ket_idx.emplace(pack_hkl(ket.h[static_cast<std::size_t>(jg)],
                                 ket.k[static_cast<std::size_t>(jg)],
                                 ket.l[static_cast<std::size_t>(jg)]),
                        jg);

    const int dh = (shift_dir == 0) ? 1 : 0;
    const int dk = (shift_dir == 1) ? 1 : 0;
    const int dl = (shift_dir == 2) ? 1 : 0;

    std::vector<std::complex<double>> S(static_cast<std::size_t>(n_occ * n_occ),
                                        {0.0, 0.0});
    for (int ig = 0; ig < ng_bra; ++ig) {
        const auto it =
            ket_idx.find(pack_hkl(bra.h[static_cast<std::size_t>(ig)] + dh,
                                  bra.k[static_cast<std::size_t>(ig)] + dk,
                                  bra.l[static_cast<std::size_t>(ig)] + dl));
        if (it == ket_idx.end())
            continue;
        const int jg = it->second;
        for (int m = 0; m < n_occ; ++m) {
            const auto bm =
                std::conj(cb[static_cast<std::size_t>(m * ng_bra + ig)]);
            for (int n = 0; n < n_occ; ++n)
                S[static_cast<std::size_t>(m * n_occ + n)] +=
                    static_cast<std::complex<double>>(
                        bm * ck[static_cast<std::size_t>(n * ng_ket + jg)]);
        }
    }
    return det_lu(S, n_occ);
}

/// String-averaged Berry phase for one spin channel on frozen V_eff.
double string_phase_for_channel(
    internal::PWProblem& prob, internal::PWWorkspace& ws,
    const std::vector<std::array<double, 3>>& kf, int dir, int k_string,
    int k_perp, int n_occ, double tol, double two_pi) {
    using internal::DavidsonSolver;

    std::vector<Vec3> kvec;
    kvec.reserve(kf.size());
    for (const auto& f : kf)
        kvec.push_back({static_cast<AccReal>(f[0]),
                        static_cast<AccReal>(f[1]),
                        static_cast<AccReal>(f[2])});
    // Note: rebuild_k_tables is called by the outer loop once per direction
    // before spin channels share the same k-mesh tables.

    const int n_k = static_cast<int>(kf.size());
    std::vector<std::vector<std::complex<AccReal>>> coef(
        static_cast<std::size_t>(n_k));
    DavidsonSolver davidson(n_occ, AccReal(tol) * AccReal(0.1), 50);
    davidson.prepare(ws);
    auto* saved_gvec = ws.gvec;
    const int n_g0 = ws.n_g;
    for (int ik = 0; ik < n_k; ++ik) {
        auto& gvk = prob.gv_k[static_cast<std::size_t>(ik)];
        ws.gvec = &gvk;
        ws.n_g = gvk.count();
        ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
        for (int ig = 0; ig < ws.n_g; ++ig)
            ws.T_G[ig] = gvk.ekin[ig];
        if (!prob.kb_proj_k.empty())
            ws.kb_proj = prob.kb_proj_k[static_cast<std::size_t>(ik)];
        if (!prob.atom_wfc_k.empty())
            ws.atom_wfc = prob.atom_wfc_k[static_cast<std::size_t>(ik)];

        davidson.step(ws);

        auto& c = coef[static_cast<std::size_t>(ik)];
        c.resize(static_cast<std::size_t>(n_occ * ws.n_g));
        for (int b = 0; b < n_occ; ++b)
            for (int ig = 0; ig < ws.n_g; ++ig)
                c[static_cast<std::size_t>(b * ws.n_g + ig)] = ws.wfc(b, ig);
    }
    ws.gvec = saved_gvec;
    ws.n_g = n_g0;

    const int J = k_string;
    const int n_strings = k_perp * k_perp;
    std::vector<double> phis;
    phis.reserve(static_cast<std::size_t>(n_strings));
    for (int s = 0; s < n_strings; ++s) {
        std::vector<std::complex<double>> dets(static_cast<std::size_t>(J));
        for (int j = 0; j < J; ++j) {
            const int ib = s * J + j;
            const int ik = s * J + (j + 1) % J;
            const int shift = (j == J - 1) ? dir : -1;
            dets[static_cast<std::size_t>(j)] = overlap_det(
                prob.gv_k[static_cast<std::size_t>(ib)],
                coef[static_cast<std::size_t>(ib)],
                prob.gv_k[static_cast<std::size_t>(ik)],
                coef[static_cast<std::size_t>(ik)], n_occ, shift);
        }
        phis.push_back(phase_from_link_dets(dets));
        ATV_DEBUG_CAT("pw", "berry dir={} n_occ={} string={} phi={:+.6f}", dir,
                      n_occ, s, phis.back());
    }

    const double phi0 = phis[0];
    double acc = 0.0;
    for (const double phi : phis)
        acc += phi0 + wrap_centered(phi - phi0, two_pi);
    return acc / static_cast<double>(n_strings);
}

void fill_ionic(BerryPolarizationResult& out, const cpu::Frame& frame) {
    const auto recip = frame.box.recip_cell();
    const StorageReal* __restrict sx = frame.span(kAtom_x);
    const StorageReal* __restrict sy = frame.span(kAtom_y);
    const StorageReal* __restrict sz = frame.span(kAtom_z);
    const int* __restrict st = frame.span(kAtom_type);
    const double two_pi = 2.0 * units::pi;
    for (int a = 0; a < frame.nlocal(); ++a) {
        const double zv = static_cast<double>(internal::valence_from_z(st[a]));
        for (int d = 0; d < 3; ++d) {
            double f = (static_cast<double>(recip(d, 0)) * sx[a]
                        + static_cast<double>(recip(d, 1)) * sy[a]
                        + static_cast<double>(recip(d, 2)) * sz[a])
                       / two_pi;
            f -= std::floor(f);
            out.p_ion_frac[static_cast<std::size_t>(d)] += zv * f;
        }
    }
}

void assemble_totals(BerryPolarizationResult& out, const cpu::Frame& frame,
                     double quantum) {
    const auto& cell = frame.box.cell();
    for (int d = 0; d < 3; ++d) {
        if (!out.computed[static_cast<std::size_t>(d)])
            continue;
        out.p_frac[static_cast<std::size_t>(d)] =
            wrap_centered(out.p_ion_frac[static_cast<std::size_t>(d)]
                              + out.p_el_frac[static_cast<std::size_t>(d)],
                          quantum);
        for (int c = 0; c < 3; ++c)
            out.dipole_e_ang[static_cast<std::size_t>(c)] +=
                out.p_frac[static_cast<std::size_t>(d)]
                * static_cast<double>(cell(d, c));
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
BerryPolarization::compute(const PW& method, const cpu::Frame& frame) const {
    if (method.smearing != Smearing::none)
        throw std::invalid_argument(
            "BerryPolarization::compute: Berry-phase polarization requires "
            "an insulator (no metallic smearing)");
    if (k_perp < 1 || k_string < 2)
        throw std::invalid_argument(
            "BerryPolarization::compute: need k_perp >= 1 and k_string >= 2");

    const double two_pi = 2.0 * units::pi;
    BerryPolarizationResult out;

    if (!method.spin_polarized) {
        // ── Closed-shell restricted path (spin factor f = 2) ───────────
        internal::PWProblem prob;
        const PWResult scf = internal::run_restricted_scf(frame, method, prob);
        const int n_occ = prob.n_bands;
        out.scf_converged = scf.converged;
        out.energy_ry = scf.energy;
        fill_ionic(out, frame);

        for (int dir = 0; dir < 3; ++dir) {
            if (!directions[static_cast<std::size_t>(dir)])
                continue;
            const auto kf = string_kpoints(dir);
            std::vector<Vec3> kvec;
            kvec.reserve(kf.size());
            for (const auto& f : kf)
                kvec.push_back({static_cast<AccReal>(f[0]),
                                static_cast<AccReal>(f[1]),
                                static_cast<AccReal>(f[2])});
            internal::rebuild_k_tables(prob, frame, method.upf_path,
                                       method.gth_name, method.ecut_ry, kvec);
            const double phase =
                string_phase_for_channel(prob, prob.ws, kf, dir, k_string,
                                         k_perp, n_occ, method.tol, two_pi);
            out.phase_el[static_cast<std::size_t>(dir)] = phase;
            out.p_el_frac[static_cast<std::size_t>(dir)] = 2.0 * phase / two_pi;
            out.computed[static_cast<std::size_t>(dir)] = true;
        }
        assemble_totals(out, frame, /*quantum=*/2.0);
        return out;
    }

    // ── Collinear spin path (spin factor f = 1 per channel) ────────────
    internal::PWProblem prob;
    int n_alpha = 0;
    int n_beta = 0;
    const PWResult scf =
        internal::run_collinear_scf(frame, method, prob, n_alpha, n_beta);
    out.scf_converged = scf.converged;
    out.energy_ry = scf.energy;
    fill_ionic(out, frame);

    internal::PWWorkspace& ws = prob.ws;
    const int ng0 = prob.gv0.count();
    // Freeze V_eff(r) per spin for NSCF strings.
    internal::VeffBuilder veff;
    veff.prepare(ws);
    std::vector<AccReal> veff_r[2];
    for (int s = 0; s < 2; ++s) {
        for (int ig = 0; ig < ng0; ++ig)
            ws.V_xc_G[ig] = ws.V_xc_spin_G[static_cast<std::size_t>(s)][ig];
        veff.step(ws);
        veff_r[s].assign(ws.V_eff_r.begin(), ws.V_eff_r.begin() + ws.nr);
    }
    const int n_occ[2] = {n_alpha, n_beta};

    for (int dir = 0; dir < 3; ++dir) {
        if (!directions[static_cast<std::size_t>(dir)])
            continue;
        const auto kf = string_kpoints(dir);
        std::vector<Vec3> kvec;
        kvec.reserve(kf.size());
        for (const auto& f : kf)
            kvec.push_back({static_cast<AccReal>(f[0]),
                            static_cast<AccReal>(f[1]),
                            static_cast<AccReal>(f[2])});
        internal::rebuild_k_tables(prob, frame, method.upf_path, method.gth_name,
                     method.ecut_ry,
                                   kvec);

        double phase_sum = 0.0;
        for (int s = 0; s < 2; ++s) {
            if (n_occ[s] <= 0)
                continue;
            // Bind frozen H_σ.
            for (int i = 0; i < ws.nr; ++i)
                ws.V_eff_r[i] = veff_r[s][static_cast<std::size_t>(i)];
            const double phi_s =
                string_phase_for_channel(prob, ws, kf, dir, k_string, k_perp,
                                         n_occ[s], method.tol, two_pi);
            phase_sum += phi_s;
            ATV_DEBUG_CAT("pw", "berry-spin dir={} spin={} phi={:+.6f}", dir, s,
                          phi_s);
        }
        // Mean phase so closed-shell identity p_el = 2*phase/2π holds when
        // N↑=N↓; total electronic polarization is always (φ↑+φ↓)/2π.
        out.phase_el[static_cast<std::size_t>(dir)] = 0.5 * phase_sum;
        out.p_el_frac[static_cast<std::size_t>(dir)] = phase_sum / two_pi;
        out.computed[static_cast<std::size_t>(dir)] = true;
    }
    // nspin=2 quantum is e (not 2e).
    assemble_totals(out, frame, /*quantum=*/1.0);
    return out;
}

} // namespace pw
} // namespace cpu
} // namespace ATV
