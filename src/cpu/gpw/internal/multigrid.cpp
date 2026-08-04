/// @file src/cpu/gpw/internal/multigrid.cpp
/// @brief Implementation of the CP2K-style CUTOFF/REL_CUTOFF multigrid helpers
///        (see multigrid.h). Level construction, exponent->level assignment, and
///        the adjoint trilinear prolong/restrict transfer operators.

#include "atomiverse/cpu/gpw/internal/multigrid.h"

#include "atomiverse/cpu/device/chunked_range.h"
#include "atomiverse/cpu/gpw/internal/product_collint.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <unordered_map>

#include <xsimd/xsimd.hpp>

#ifdef ATV_GPW_HAS_FFT
#include "atomiverse/cpu/gpw/internal/fft_cache.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {
namespace gpw {

MultiGrid make_multigrid(const double cell_abc[3], double cutoff_ry, int ngrids,
                         double rel_cutoff) {
    MultiGrid mg;
    const int ng = std::max(1, ngrids);
    if (ng == 1 || rel_cutoff <= 0.0 || rel_cutoff >= cutoff_ry) {
        mg.levels.push_back(make_uniform_grid(cell_abc, cutoff_ry));
        mg.level_cutoff_ry.push_back(cutoff_ry);
        return mg;
    }
    // Geometric ladder cutoff_ry = c0 > c1 > ... > c_{ng-1} ~ rel_cutoff.
    const double ratio = std::pow(cutoff_ry / rel_cutoff, 1.0 / (ng - 1));
    double c = cutoff_ry;
    for (int l = 0; l < ng; ++l) {
        mg.levels.push_back(make_uniform_grid(cell_abc, c));
        mg.level_cutoff_ry.push_back(c);
        c /= ratio;
    }
    return mg;
}

std::vector<int> assign_levels(const MultiGrid& mg,
                               const std::vector<GridBasisFunction>& basis,
                               double rel_cutoff) {
    std::vector<int> level(basis.size(), 0);
    const int nlev = static_cast<int>(mg.levels.size());
    if (nlev <= 1)
        return level;
    // A primitive of exponent alpha has density exponent 2*alpha; the plane-wave
    // cutoff needed to resolve it scales linearly with alpha. Calibrate so the
    // reference exponent alpha_ref = rel_cutoff (in Ry, as an exponent proxy)
    // lands a Gaussian at the rel_cutoff level: need_cutoff = rel_cutoff*alpha.
    // A basis function goes on the COARSEST level whose cutoff still meets the
    // need of its SHARPEST primitive (so the function is never under-resolved).
    for (std::size_t u = 0; u < basis.size(); ++u) {
        double amax = 0.0;
        for (const auto& p : basis[u])
            amax = std::max(amax, p.alpha);
        const double need = rel_cutoff * amax; // [Ry] resolution need
        // Coarsest level whose cutoff still resolves the Gaussian (most
        // economical). If none does (a very sharp Gaussian needs more than the
        // finest cutoff), clamp to the finest level (0).
        int lev = 0;
        for (int l = nlev - 1; l >= 0; --l) {
            if (mg.level_cutoff_ry[l] >= need) {
                lev = l;
                break;
            }
        }
        level[u] = lev;
    }
    return level;
}

namespace {
// Periodic wrap of an integer grid index into [0, n).
inline int wrap(int i, int n) {
    i %= n;
    return i < 0 ? i + n : i;
}
} // namespace

std::vector<double> prolong_trilinear(const UniformGrid& coarse,
                                      const UniformGrid& fine,
                                      const std::vector<double>& f_coarse,
                                      bool periodic) {
    std::vector<double> out(static_cast<std::size_t>(fine.npts()), 0.0);
    const int cn0 = coarse.n[0], cn1 = coarse.n[1], cn2 = coarse.n[2];
    const double cstep0 = coarse.L[0] / cn0, cstep1 = coarse.L[1] / cn1,
                 cstep2 = coarse.L[2] / cn2;
#pragma omp parallel for schedule(static)
    for (long g = 0; g < fine.npts(); ++g) {
        double r[3];
        fine.point(g, r);
        // Fractional position in coarse-grid index space.
        double fx = (r[0] - coarse.origin[0]) / cstep0;
        double fy = (r[1] - coarse.origin[1]) / cstep1;
        double fz = (r[2] - coarse.origin[2]) / cstep2;
        int i0 = static_cast<int>(std::floor(fx));
        int j0 = static_cast<int>(std::floor(fy));
        int k0 = static_cast<int>(std::floor(fz));
        double tx = fx - i0, ty = fy - j0, tz = fz - k0;
        double val = 0.0;
        for (int di = 0; di < 2; ++di)
            for (int dj = 0; dj < 2; ++dj)
                for (int dk = 0; dk < 2; ++dk) {
                    const double w = (di ? tx : 1 - tx) * (dj ? ty : 1 - ty)
                                     * (dk ? tz : 1 - tz);
                    int ci = periodic ? wrap(i0 + di, cn0)
                                      : std::clamp(i0 + di, 0, cn0 - 1);
                    int cj = periodic ? wrap(j0 + dj, cn1)
                                      : std::clamp(j0 + dj, 0, cn1 - 1);
                    int ck = periodic ? wrap(k0 + dk, cn2)
                                      : std::clamp(k0 + dk, 0, cn2 - 1);
                    const long ci_lin =
                        (static_cast<long>(ci) * cn1 + cj) * cn2 + ck;
                    val += w * f_coarse[static_cast<std::size_t>(ci_lin)];
                }
        out[static_cast<std::size_t>(g)] = val;
    }
    return out;
}

std::vector<double> restrict_transpose(const UniformGrid& fine,
                                       const UniformGrid& coarse,
                                       const std::vector<double>& f_fine,
                                       bool periodic) {
    // Exact transpose of prolong_trilinear, times the per-point volume ratio so
    // the operator is the adjoint w.r.t. the grid integration inner products:
    // sum_fine w_f V_f (P g)_f = sum_coarse w_c (R V)_c g_c.
    std::vector<double> out(static_cast<std::size_t>(coarse.npts()), 0.0);
    const int cn0 = coarse.n[0], cn1 = coarse.n[1], cn2 = coarse.n[2];
    const double cstep0 = coarse.L[0] / cn0, cstep1 = coarse.L[1] / cn1,
                 cstep2 = coarse.L[2] / cn2;
    const double vol_ratio = fine.weight() / coarse.weight();
    // Serial scatter (atomic adds would dominate); coarse grids are small.
    for (long g = 0; g < fine.npts(); ++g) {
        double r[3];
        fine.point(g, r);
        double fx = (r[0] - coarse.origin[0]) / cstep0;
        double fy = (r[1] - coarse.origin[1]) / cstep1;
        double fz = (r[2] - coarse.origin[2]) / cstep2;
        int i0 = static_cast<int>(std::floor(fx));
        int j0 = static_cast<int>(std::floor(fy));
        int k0 = static_cast<int>(std::floor(fz));
        double tx = fx - i0, ty = fy - j0, tz = fz - k0;
        const double vf = f_fine[static_cast<std::size_t>(g)] * vol_ratio;
        for (int di = 0; di < 2; ++di)
            for (int dj = 0; dj < 2; ++dj)
                for (int dk = 0; dk < 2; ++dk) {
                    const double w = (di ? tx : 1 - tx) * (dj ? ty : 1 - ty)
                                     * (dk ? tz : 1 - tz);
                    int ci = periodic ? wrap(i0 + di, cn0)
                                      : std::clamp(i0 + di, 0, cn0 - 1);
                    int cj = periodic ? wrap(j0 + dj, cn1)
                                      : std::clamp(j0 + dj, 0, cn1 - 1);
                    int ck = periodic ? wrap(k0 + dk, cn2)
                                      : std::clamp(k0 + dk, 0, cn2 - 1);
                    const long ci_lin =
                        (static_cast<long>(ci) * cn1 + cj) * cn2 + ck;
                    out[static_cast<std::size_t>(ci_lin)] += w * vf;
                }
    }
    return out;
}

#ifdef ATV_GPW_HAS_FFT
namespace {

// Forward complex DFT (FFTW c2c, FFTW_FORWARD: F(m) = Σ_r f(r) e^{-iG·r}) of a
// real field laid out as nx*ny*nz row-major (z fastest). Plans + scratch are
// cached by grid shape (see fft_cache.h).
std::vector<std::complex<double>> fwd_dft(const std::vector<double>& f,
                                          const int n[3]) {
    C2cFft& fft = C2cFft::for_shape(n[0], n[1], n[2]);
    const long ntot = fft.ntot();
    fftw_complex* in = fft.in();
    fftw_complex* out = fft.out();
    for (long i = 0; i < ntot; ++i) {
        in[i][0] = f[static_cast<std::size_t>(i)];
        in[i][1] = 0.0;
    }
    fft.execute_fwd();
    std::vector<std::complex<double>> F(static_cast<std::size_t>(ntot));
    for (long i = 0; i < ntot; ++i)
        F[static_cast<std::size_t>(i)] = {out[i][0], out[i][1]};
    return F;
}

// Inverse complex DFT (FFTW c2c, FFTW_BACKWARD: g(r) = Σ_m F(m) e^{+iG·r}); FFTW
// does NOT normalize, so this returns the UNnormalized sum (caller divides by N).
// Returns the real part (the spectra here are Hermitian by construction).
std::vector<double> inv_dft_real(const std::vector<std::complex<double>>& F,
                                 const int n[3]) {
    C2cFft& fft = C2cFft::for_shape(n[0], n[1], n[2]);
    const long ntot = fft.ntot();
    fftw_complex* in = fft.in();
    fftw_complex* out = fft.out();
    for (long i = 0; i < ntot; ++i) {
        in[i][0] = F[static_cast<std::size_t>(i)].real();
        in[i][1] = F[static_cast<std::size_t>(i)].imag();
    }
    fft.execute_bwd();
    std::vector<double> g(static_cast<std::size_t>(ntot));
    for (long i = 0; i < ntot; ++i)
        g[static_cast<std::size_t>(i)] = out[i][0];
    return g;
}

// Per-axis mapping of a COARSE c2c frequency index `ic` (in [0,nc)) to the set of
// FINE frequency indices that carry the SAME physical frequency G = 2π m / L
// (signed m). At most two destinations: a Nyquist coarse mode (|m| == nc/2)
// splits half/half between the +nc/2 and -nc/2 fine bins so the interpolant stays
// real; every other mode maps one-to-one. Writes (fine_index, weight) into out
// and returns the count (1 or 2). Requires nf >= nc, both even.
inline int axis_coarse_to_fine(int ic, int nc, int nf, int dst[2],
                               double wt[2]) {
    const int m = (ic <= nc / 2) ? ic : ic - nc; // signed coarse frequency
    if (std::abs(m) == nc / 2) {
        // Nyquist of the coarse grid: split into +nc/2 and -nc/2 on the fine grid.
        dst[0] = nc / 2;      // +nc/2 (in [0,nf), since nf>=nc)
        dst[1] = nf - nc / 2; // -nc/2
        wt[0] = 0.5;
        wt[1] = 0.5;
        return 2;
    }
    dst[0] = (m >= 0) ? m : nf + m; // same signed frequency on the fine grid
    wt[0] = 1.0;
    return 1;
}

} // namespace

std::vector<double> prolong_fft(const UniformGrid& coarse,
                                const UniformGrid& fine,
                                const std::vector<double>& f_coarse,
                                bool periodic) {
    if (!periodic)
        return prolong_trilinear(coarse, fine, f_coarse, periodic);
    const int nc[3] = {coarse.n[0], coarse.n[1], coarse.n[2]};
    const int nf[3] = {fine.n[0], fine.n[1], fine.n[2]};
    const long ntot_f = fine.npts();
    const double Nc = static_cast<double>(coarse.npts());
    const double Nf = static_cast<double>(ntot_f);

    // Forward DFT on the coarse grid.
    const std::vector<std::complex<double>> Fc = fwd_dft(f_coarse, nc);

    // Embed each coarse mode into the fine spectrum at the same physical
    // frequency, scaled by Nf/Nc so values (and ∫) are preserved on inversion.
    std::vector<std::complex<double>> Ff(static_cast<std::size_t>(ntot_f),
                                         std::complex<double>(0.0, 0.0));
    const double scale = Nf / Nc;
    for (int i0 = 0; i0 < nc[0]; ++i0) {
        int d0[2];
        double w0[2];
        const int c0 = axis_coarse_to_fine(i0, nc[0], nf[0], d0, w0);
        for (int i1 = 0; i1 < nc[1]; ++i1) {
            int d1[2];
            double w1[2];
            const int c1 = axis_coarse_to_fine(i1, nc[1], nf[1], d1, w1);
            for (int i2 = 0; i2 < nc[2]; ++i2) {
                int d2[2];
                double w2[2];
                const int c2 = axis_coarse_to_fine(i2, nc[2], nf[2], d2, w2);
                const std::complex<double> v =
                    scale
                    * Fc[(static_cast<std::size_t>(i0) * nc[1] + i1) * nc[2]
                         + i2];
                for (int a = 0; a < c0; ++a)
                    for (int b = 0; b < c1; ++b)
                        for (int c = 0; c < c2; ++c) {
                            const long idx =
                                (static_cast<long>(d0[a]) * nf[1] + d1[b])
                                    * nf[2]
                                + d2[c];
                            Ff[static_cast<std::size_t>(idx)] +=
                                (w0[a] * w1[b] * w2[c]) * v;
                        }
            }
        }
    }

    // Inverse DFT on the fine grid (FFTW unnormalized -> divide by Nf).
    std::vector<double> g = inv_dft_real(Ff, nf);
    const double inv_nf = 1.0 / Nf;
    for (auto& x : g)
        x *= inv_nf;
    return g;
}

std::vector<double> restrict_fft(const UniformGrid& fine,
                                 const UniformGrid& coarse,
                                 const std::vector<double>& f_fine,
                                 bool periodic) {
    if (!periodic)
        return restrict_transpose(fine, coarse, f_fine, periodic);
    const int nc[3] = {coarse.n[0], coarse.n[1], coarse.n[2]};
    const int nf[3] = {fine.n[0], fine.n[1], fine.n[2]};
    const long ntot_c = coarse.npts();
    const double Nc = static_cast<double>(ntot_c);
    const double Nf = static_cast<double>(fine.npts());

    // Forward DFT on the fine grid.
    const std::vector<std::complex<double>> Ff = fwd_dft(f_fine, nf);

    // Transpose of the prolong embedding: each coarse mode GATHERS the fine modes
    // it scattered to (same (index,weight) map). Adjointness w.r.t. the grid inner
    // products fixes the overall scalar: with P_{rc} = (1/Nc) Σ E_{m'm} e^{iG'r}
    // e^{-iGc}, R = (w_f/w_c) Pᵀ, and w_f/w_c = Nc/Nf (same box), the coarse field
    // is (1/Nf)·IDFT_coarse(gather(FV)) — i.e. a plain 1/Nf normalization on the
    // unnormalized inverse DFT (NO extra Nf/Nc spectral factor here; that factor
    // lives only on prolong's forward path).
    std::vector<std::complex<double>> Fc(static_cast<std::size_t>(ntot_c),
                                         std::complex<double>(0.0, 0.0));
    for (int i0 = 0; i0 < nc[0]; ++i0) {
        int d0[2];
        double w0[2];
        const int c0 = axis_coarse_to_fine(i0, nc[0], nf[0], d0, w0);
        for (int i1 = 0; i1 < nc[1]; ++i1) {
            int d1[2];
            double w1[2];
            const int c1 = axis_coarse_to_fine(i1, nc[1], nf[1], d1, w1);
            for (int i2 = 0; i2 < nc[2]; ++i2) {
                int d2[2];
                double w2[2];
                const int c2 = axis_coarse_to_fine(i2, nc[2], nf[2], d2, w2);
                std::complex<double> acc(0.0, 0.0);
                for (int a = 0; a < c0; ++a)
                    for (int b = 0; b < c1; ++b)
                        for (int c = 0; c < c2; ++c) {
                            const long idx =
                                (static_cast<long>(d0[a]) * nf[1] + d1[b])
                                    * nf[2]
                                + d2[c];
                            acc += (w0[a] * w1[b] * w2[c])
                                   * Ff[static_cast<std::size_t>(idx)];
                        }
                Fc[(static_cast<std::size_t>(i0) * nc[1] + i1) * nc[2] + i2] =
                    acc;
            }
        }
    }

    // Inverse DFT on the coarse grid; the adjoint normalization is 1/Nf (see
    // above), NOT the usual 1/Nc, because w_f/w_c = Nc/Nf cancels the 1/Nc.
    std::vector<double> g = inv_dft_real(Fc, nc);
    const double inv_nf = 1.0 / Nf;
    for (auto& x : g)
        x *= inv_nf;
    (void)Nc;
    return g;
}
#endif // ATV_GPW_HAS_FFT

namespace {

// Coarsest level whose cutoff resolves a Gaussian of (density) exponent need.
// Mirrors assign_levels: a product Gaussian of combined exponent p needs an
// effective cutoff rel_cutoff*p; pick the coarsest level meeting it, clamp finest.
int level_for_exponent(const MultiGrid& mg, double rel_cutoff, double p) {
    const int nlev = static_cast<int>(mg.levels.size());
    const double need = rel_cutoff * p;
    int lev = 0;
    for (int l = nlev - 1; l >= 0; --l) {
        if (mg.level_cutoff_ry[static_cast<std::size_t>(l)] >= need) {
            lev = l;
            break;
        }
    }
    return lev;
}

// Derived screen threshold for the PRODUCT path. The caller's eps
// (eps_default) is a per-point VALUE threshold — the compact single-grid path
// drops individual grid-point values below it, so its density bias stays
// ~eps. The product path instead drops WHOLE products (build pair drop,
// per-iteration weight drop) and truncates windows, discarding INTEGRATED
// mass up to (π/p)^{3/2} ≈ 70× the value threshold per diffuse MOLOPT
// product — and a periodic cell holds thousands of image pairs sitting at
// threshold. Without the margin the collocated density loses charge:
// periodic Si bct DZVP-MOLOPT at 95 Ry lost 0.090 of 16 e⁻ at eps 1e-7,
// shifting the SCF energy 4.1e-2 Ha off the single-grid path; at eps·1e-4
// the gap is 3.5e-5 Ha — the same order as the single grid's own eps
// sensitivity (gpw-multigrid-kpoints, 2026-07-31).
constexpr double kProductMassMargin = 1e-4;

// Radius [Bohr] beyond which |gw·r^(la+lb)·exp(-p r²)| < eps for a product
// Gaussian. Same fixed-point root as basis_function_radius, with the combined
// polynomial order L = |la|+|lb| and prefactor magnitude |gw|.
double product_radius(double gw, double p, int Ltot, double eps) {
    if (gw == 0.0 || p <= 0.0)
        return 0.0;
    const double leps = std::log(eps);
    const double lc = std::log(std::abs(gw));
    double r2 = std::max(0.0, (lc - leps) / p);
    for (int it = 0; it < 3; ++it) {
        const double r = std::sqrt(std::max(r2, 1e-12));
        r2 = std::max(0.0, (lc + Ltot * std::log(r) - leps) / p);
    }
    return std::sqrt(r2);
}

} // namespace

ProductSet build_products(const MultiGrid& mg,
                          const std::vector<GridBasisFunction>& basis,
                          const double L[3], double rel_cutoff, double eps,
                          bool periodic) {
    eps *= kProductMassMargin; // whole-product drops discard integrated mass
    const int nb = static_cast<int>(basis.size());
    const int nlev = static_cast<int>(mg.levels.size());

    ProductSet ps;
    ps.by_level.resize(static_cast<std::size_t>(nlev));

    // ---- 1. Deduplicate cartesian primitives -------------------------------
    // Each spherical basis function μ is stored as its real-solid-harmonic
    // cartesian expansion, so the SAME (α, center, l) primitive recurs across the
    // shell (e.g. x² in both d_z² and d_x²-y²). Collapse to UNIQUE cartesian
    // primitives, each tagged with the (μ, coeff) spherical functions it feeds, so
    // a product between two unique primitives is collocated ONCE and scattered to
    // all contributor (μ,ν) pairs — no per-monomial window recompute, no sort.
    struct PrimKey {
        double a, cx, cy, cz;
        int lx, ly, lz;
        bool operator==(const PrimKey& o) const {
            return a == o.a && cx == o.cx && cy == o.cy && cz == o.cz
                   && lx == o.lx && ly == o.ly && lz == o.lz;
        }
    };
    struct KeyHash {
        std::size_t operator()(const PrimKey& k) const {
            std::size_t h = std::hash<double>()(k.a);
            auto mix = [&h](std::size_t x) {
                h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            };
            mix(std::hash<double>()(k.cx));
            mix(std::hash<double>()(k.cy));
            mix(std::hash<double>()(k.cz));
            mix(std::hash<int>()(k.lx));
            mix(std::hash<int>()(k.ly));
            mix(std::hash<int>()(k.lz));
            return h;
        }
    };
    std::unordered_map<PrimKey, int, KeyHash> uniq;
    std::vector<std::vector<PrimContrib>> tmp_contribs;
    for (int mu = 0; mu < nb; ++mu) {
        for (const auto& p : basis[static_cast<std::size_t>(mu)]) {
            if (p.coeff == 0.0)
                continue;
            PrimKey key{p.alpha, p.center[0], p.center[1], p.center[2],
                        p.l[0],  p.l[1],      p.l[2]};
            auto it = uniq.find(key);
            int pi;
            if (it == uniq.end()) {
                pi = static_cast<int>(ps.prims.size());
                PrimRef pr;
                pr.alpha = p.alpha;
                for (int d = 0; d < 3; ++d) {
                    pr.center[d] = p.center[d];
                    pr.l[d] = p.l[d];
                }
                ps.prims.push_back(pr);
                tmp_contribs.emplace_back();
                uniq.emplace(key, pi);
            } else {
                pi = it->second;
            }
            tmp_contribs[static_cast<std::size_t>(pi)].push_back({mu, p.coeff});
        }
    }
    const int nprim = static_cast<int>(ps.prims.size());
    // Flatten contributors; per-prim max|coeff| (build screen) and support radius.
    std::vector<double> prad(static_cast<std::size_t>(nprim), 0.0);
    for (int pi = 0; pi < nprim; ++pi) {
        PrimRef& pr = ps.prims[static_cast<std::size_t>(pi)];
        pr.cbeg = static_cast<int>(ps.contribs.size());
        pr.cn =
            static_cast<int>(tmp_contribs[static_cast<std::size_t>(pi)].size());
        double mc = 0.0;
        for (const auto& c : tmp_contribs[static_cast<std::size_t>(pi)]) {
            ps.contribs.push_back(c);
            mc = std::max(mc, std::abs(c.coeff));
        }
        pr.maxc = mc;
        const int Lsum = pr.l[0] + pr.l[1] + pr.l[2];
        prad[static_cast<std::size_t>(pi)] =
            product_radius(mc, pr.alpha, Lsum, eps);
    }

    // ---- 2. Generate UNIQUE-shape products over cartesian-primitive pairs ----
    // One bucket PER BRA PRIMITIVE, not per thread: the product list order is
    // the summation order of every downstream density/KS contraction, and a
    // per-thread bucket makes it a function of which thread took which `ua`
    // chunk — which a dynamic schedule varies between two runs of the same
    // binary. Bucketing by the loop index and concatenating in ascending `ua`
    // gives one canonical list at any thread count.
    std::vector<std::vector<std::vector<GridProduct>>> per_bra(
        static_cast<std::size_t>(nprim)); // [ua][level]
    for (auto& t : per_bra)
        t.resize(static_cast<std::size_t>(nlev));

#pragma omp parallel for schedule(dynamic, 4)
    for (int ua = 0; ua < nprim; ++ua) {
        auto& out = per_bra[static_cast<std::size_t>(ua)];
        const PrimRef& A = ps.prims[static_cast<std::size_t>(ua)];
        for (int ub = 0; ub < nprim; ++ub) {
            const PrimRef& B = ps.prims[static_cast<std::size_t>(ub)];
            const double reach = prad[static_cast<std::size_t>(ua)]
                                 + prad[static_cast<std::size_t>(ub)];
            const double pa = A.alpha, pb = B.alpha, p = pa + pb;
            const double mc = A.maxc * B.maxc;
            // Image shifts of the KET prim (B); the bra A stays home (mirrors
            // the Γ-point cross-image product Σ_T φ_A φ_B(·−T)).
            int m[3] = {0, 0, 0};
            if (periodic)
                for (int d = 0; d < 3; ++d)
                    m[d] = (L[d] > 0.0) ? static_cast<int>(std::ceil(
                                              reach / L[d] - 1e-12))
                                        : 0;
            for (int tx = -m[0]; tx <= m[0]; ++tx)
                for (int ty = -m[1]; ty <= m[1]; ++ty)
                    for (int tz = -m[2]; tz <= m[2]; ++tz) {
                        const double tL[3] = {tx * L[0], ty * L[1],
                                              tz * L[2]};
                        const double sx = A.center[0] - B.center[0] - tL[0];
                        const double sy = A.center[1] - B.center[1] - tL[1];
                        const double sz = A.center[2] - B.center[2] - tL[2];
                        if (sx * sx + sy * sy + sz * sz > reach * reach)
                            continue;
                        double rb[3], dpq2 = 0.0, P[3];
                        for (int d = 0; d < 3; ++d) {
                            rb[d] = B.center[d] + tL[d];
                            const double diff = A.center[d] - rb[d];
                            dpq2 += diff * diff;
                            P[d] = (pa * A.center[d] + pb * rb[d]) / p;
                        }
                        const double K = std::exp(-pa * pb / p * dpq2);
                        if (mc * K < eps)
                            continue; // no contributor pair survives
                        GridProduct prod;
                        prod.p = p;
                        prod.k = K;
                        prod.ua = ua;
                        prod.ub = ub;
                        for (int d = 0; d < 3; ++d) {
                            prod.pc[d] = P[d];
                            prod.ra[d] = A.center[d];
                            prod.rb[d] = rb[d];
                            prod.la[d] = A.l[d];
                            prod.lb[d] = B.l[d];
                        }
                        prod.t[0] = tx;
                        prod.t[1] = ty;
                        prod.t[2] = tz;
                        prod.level = level_for_exponent(mg, rel_cutoff, p);
                        out[static_cast<std::size_t>(prod.level)].push_back(
                            prod);
                    }
        }
    }

    for (auto& t : per_bra)
        for (int l = 0; l < nlev; ++l) {
            auto& dst = ps.by_level[static_cast<std::size_t>(l)];
            auto& src = t[static_cast<std::size_t>(l)];
            dst.insert(dst.end(), src.begin(), src.end());
        }

    // ---- 3. Index the distinct ket image shifts (Bloch phase cells) --------
    // Serial post-pass over the canonical concatenated lists: `cells` is
    // lexicographically sorted (deterministic at any thread count) and every
    // product's `tcell` points at its shift. Γ consumers ignore both.
    std::map<std::array<int, 3>, int> cell_of;
    for (auto& lev : ps.by_level)
        for (auto& prod : lev)
            cell_of.emplace(std::array<int, 3>{prod.t[0], prod.t[1], prod.t[2]},
                            0);
    ps.cells.reserve(cell_of.size());
    for (auto& [tvec, idx] : cell_of) {
        idx = static_cast<int>(ps.cells.size());
        ps.cells.push_back(tvec);
    }
    for (auto& lev : ps.by_level)
        for (auto& prod : lev)
            prod.tcell = cell_of.at(
                std::array<int, 3>{prod.t[0], prod.t[1], prod.t[2]});
    return ps;
}

namespace {
// Grow-only per-OpenMP-thread 1D-axis scratch for collocate_one. Product
// collocate/integrate call this once per product per SCF rebuild — millions of
// tiny heap allocs of fbuf/ebuf/wbuf. File-scope thread_local (not inside the
// template) shares one buffer set across all lambda instantiations; capacity
// only grows. Only the prefix [0,total) is written each call.
thread_local std::vector<double> g_colloc_fbuf;
thread_local std::vector<double> g_colloc_ebuf;
thread_local std::vector<long> g_colloc_wbuf;

inline void ensure_colloc_axis(int total) {
    const std::size_t n = static_cast<std::size_t>(total);
    if (g_colloc_fbuf.size() < n) {
        g_colloc_fbuf.resize(n);
        g_colloc_ebuf.resize(n);
        g_colloc_wbuf.resize(n);
    }
}

// xsimd kernels for the contiguous z-axis runs collocate_one emits (portable
// batch<double>, unaligned loads, scalar tail — same idiom as the ss|ss ket
// loop in scf/native_integrals.cpp). z is the stride-1 grid axis, so both the
// density scatter (axpy) and the KS gather (dot) run vector-wide along it.
// Vector accumulation reorders the FP sums vs. the scalar loop; results stay
// within the regression tolerances and remain run-to-run deterministic (the
// order is fixed by the batch width, not the schedule).
inline double axis_dot(const double* v, const double* f, long cnt) {
    using batch_t = xsimd::batch<double>;
    constexpr long W = static_cast<long>(batch_t::size);
    batch_t acc_v(0.0);
    long t = 0;
    for (; t + W <= cnt; t += W)
        acc_v += batch_t::load_unaligned(v + t) * batch_t::load_unaligned(f + t);
    // Horizontal reduce (portable xsimd); scalar tail for remainder.
    double s = xsimd::reduce_add(acc_v);
    for (; t < cnt; ++t)
        s += v[t] * f[t];
    return s;
}

inline void axis_axpy(double a, const double* f, double* out, long cnt) {
    using batch_t = xsimd::batch<double>;
    constexpr long W = static_cast<long>(batch_t::size);
    const batch_t a_v(a);
    long t = 0;
    for (; t + W <= cnt; t += W) {
        batch_t r = batch_t::load_unaligned(out + t);
        r += a_v * batch_t::load_unaligned(f + t);
        r.store_unaligned(out + t);
    }
    for (; t < cnt; ++t)
        out[t] += a * f[t];
}

/// @brief Fill F[t] *= (r−ra)^la (r−rb)^lb, E2[t]=(r−Pc)², W[t]=wrap(lo+t)
///        over a 1D product window. Gaussian part of F is already written.
///        Portable xsimd on the poly/E2 path; wrap stays scalar (integer mod).
inline void fill_axis_poly_e2_w(double* F, double* E2, long* W, int len,
                                long lo, long n_d, double origin, double L_d,
                                double pc, double ra, double rb, int la,
                                int lb) {
    using batch_t = xsimd::batch<double>;
    constexpr int Wd = static_cast<int>(batch_t::size);
    const double scale = L_d / static_cast<double>(n_d);
    const batch_t v_origin(origin), v_scale(scale), v_pc(pc), v_ra(ra),
        v_rb(rb);
    // Sequential indices for the first lane; broadcast + offset per lane.
    alignas(64) double lane_off[16];
    for (int k = 0; k < Wd; ++k)
        lane_off[k] = static_cast<double>(k);
    const batch_t v_lane = batch_t::load_unaligned(lane_off);
    int t = 0;
    for (; t + Wd <= len; t += Wd) {
        const batch_t v_idx =
            batch_t(static_cast<double>(lo + t)) + v_lane;
        const batch_t r = v_origin + v_scale * v_idx;
        const batch_t e = r - v_pc;
        batch_t v = batch_t::load_unaligned(F + t);
        const batch_t a = r - v_ra;
        const batch_t b = r - v_rb;
        for (int q = 0; q < la; ++q)
            v *= a;
        for (int q = 0; q < lb; ++q)
            v *= b;
        v.store_unaligned(F + t);
        (e * e).store_unaligned(E2 + t);
    }
    for (; t < len; ++t) {
        const long idx = lo + t;
        const double r =
            origin + L_d * static_cast<double>(idx) / static_cast<double>(n_d);
        const double e = r - pc;
        double v = F[t];
        const double a = r - ra, b = r - rb;
        for (int q = 0; q < la; ++q)
            v *= a;
        for (int q = 0; q < lb; ++q)
            v *= b;
        F[t] = v;
        E2[t] = e * e;
    }
    // Wrapped storage index (positive mod). Scalar: cheap vs poly, and
    // integer batch mod is awkward without a native long batch path.
    for (int u = 0; u < len; ++u) {
        long idx = lo + u;
        W[u] = ((idx % n_d) + n_d) % n_d;
    }
}

// Accumulate one product over an unwrapped window into a per-point sink `sink`,
// matching build_compact_basis_grid: window [ic-ih, ic+ih] in grid cells around
// the product center, evaluate at the UNWRAPPED real position (image offset in r,
// no min-image), wrap the storage index, accumulate. `add(g0, fz, cnt, fxy)`
// receives one CONTIGUOUS stride-1 z-run: wrapped linear start index g0, the
// per-point z factors fz[0..cnt), and the column prefactor fxy — the point
// value at g0+t is fxy·fz[t]. Runs split only at the periodic z wrap, so the
// sink can process them vector-wide (xsimd axpy / dot along the fast axis).
// SEPARABLE collocation: the product Gaussian factorizes as
//   exp(-p|r-P|²)·(r-ra)^la·(r-rb)^lb = Fx(x)·Fy(y)·Fz(z),
// so the per-axis factor (Gaussian × 1D polynomial) is precomputed ONCE over the
// 1D window (exp evaluated O(window) times, not O(window³)) and the 3D window is
// filled by the outer product — no per-point exp, no gather, vectorizable inner.
// `mag` is the EFFECTIVE magnitude that sizes the window: the build coefficient
// gweight for the potential integral, but the per-iter weight |P_μν·gweight| for
// the density collocation — the latter is often ≪ gweight (small density-matrix
// element), so sizing by it shrinks the window to where the actual contribution
// exceeds eps (CP2K's density screening), instead of the build-time worst case.
template <typename Add>
inline void collocate_one(const UniformGrid& grid, const GridProduct& prod,
                          double mag, double eps, Add add) {
    const long n[3] = {grid.n[0], grid.n[1], grid.n[2]};
    const double dx[3] = {grid.L[0] / n[0], grid.L[1] / n[1], grid.L[2] / n[2]};
    const int Ltot = prod.la[0] + prod.la[1] + prod.la[2] + prod.lb[0]
                     + prod.lb[1] + prod.lb[2];
    const double rc = product_radius(mag, prod.p, Ltot, eps);
    long lo[3], hi[3];
    for (int d = 0; d < 3; ++d) {
        const long ih = static_cast<long>(std::ceil(rc / dx[d]));
        const long ic = static_cast<long>(std::floor(prod.pc[d] / dx[d]));
        lo[d] = ic - ih;
        hi[d] = ic + ih;
    }
    // Per-axis 1D factor F_d[t] (t = index − lo[d]) = exp(-p (r−Pc)²) (r−ra)^la (r−rb)^lb,
    // the squared distance E2_d[t] = (r−Pc)² (for the sphere cull), and the wrapped
    // storage index w_d[t]. Grow-only thread_local scratch (no per-call heap).
    const int la[3] = {prod.la[0], prod.la[1], prod.la[2]};
    const int lb[3] = {prod.lb[0], prod.lb[1], prod.lb[2]};
    double* F[3];
    double* E2[3];
    long* W[3];
    int len[3];
    int total = 0;
    for (int d = 0; d < 3; ++d) {
        len[d] = static_cast<int>(hi[d] - lo[d] + 1);
        total += len[d];
    }
    ensure_colloc_axis(total);
    int off = 0;
    for (int d = 0; d < 3; ++d) {
        F[d] = g_colloc_fbuf.data() + off;
        E2[d] = g_colloc_ebuf.data() + off;
        W[d] = g_colloc_wbuf.data() + off;
        off += len[d];
        // Gaussian via exp recurrence (CP2K), propagated OUTWARD from the centre so
        // it stays bounded (starting at an edge would overflow B=exp(-2p e dx) for
        // sharp functions): exp(-p(e±dx)²)=exp(-p e²)·B^{±}·A, A=exp(-p dx²),
        // B^{±}=exp(∓2p e dx), advanced by ·A². 3 exps/axis instead of len[d].
        const double dxd = grid.L[d] / n[d];
        const double A = std::exp(-prod.p * dxd * dxd);
        const double A2 = A * A;
        const int tc =
            (len[d] - 1) / 2; // centre offset (window is 2·ih+1, odd)
        const double rc_ =
            grid.origin[d] + grid.L[d] * static_cast<double>(lo[d] + tc) / n[d];
        const double ec = rc_ - prod.pc[d];
        F[d][tc] =
            std::exp(-prod.p * ec * ec); // centre Gaussian (ec sub-grid, small)
        {                                // outward to the right (e increasing)
            double g = F[d][tc], B = std::exp(-2.0 * prod.p * ec * dxd);
            for (int t = tc + 1; t < len[d]; ++t) {
                g *= B * A;
                F[d][t] = g;
                B *= A2;
            }
        }
        { // outward to the left (e decreasing)
            double g = F[d][tc], C = std::exp(2.0 * prod.p * ec * dxd);
            for (int t = tc - 1; t >= 0; --t) {
                g *= C * A;
                F[d][t] = g;
                C *= A2;
            }
        }
        // Poly · Gaussian, |r−Pc|², wrap — xsimd on the float path.
        fill_axis_poly_e2_w(F[d], E2[d], W[d], len[d], lo[d], n[d],
                            grid.origin[d], grid.L[d], prod.pc[d], prod.ra[d],
                            prod.rb[d], la[d], lb[d]);
    }
    // SPHERE-screened outer-product fill (CP2K-style): the product Gaussian is
    // negligible outside radius rc, so skip the cube corners — for each (i,j)
    // column only the z-range with (x−Px)²+(y−Py)²+(z−Pz)² ≤ rc² is collocated
    // (~2× fewer points than the cube). Dropped points have value < eps already.
    const double rc2 = rc * rc;
    const double invdz =
        static_cast<double>(n[2]) / grid.L[2]; // grid-index per Bohr, z
    // Precompute z-sphere half-width scale: klo/khi = round(pc_z ± sqrt(rc2−r2xy)).
    // Hoist origin/pc/lo constants out of the (i,j) hot nest.
    const double pc_z = prod.pc[2];
    const double origin_z = grid.origin[2];
    const long lo_z = lo[2];
    const long n_z = n[2];
    const int len_z = len[2];
    const double* Fz = F[2];
    const long* Wz = W[2];
    for (int ti = 0; ti < len[0]; ++ti) {
        const double fx = F[0][ti], ex2 = E2[0][ti];
        if (fx == 0.0 || ex2 > rc2)
            continue;
        const long ii = W[0][ti];
        const long base_i = ii * n[1];
        for (int tj = 0; tj < len[1]; ++tj) {
            const double r2xy = ex2 + E2[1][tj];
            if (r2xy > rc2)
                continue;
            const double fxy = fx * F[1][tj];
            if (fxy == 0.0)
                continue;
            const long base = (base_i + W[1][tj]) * n_z;
            // Sphere z-bounds; one sqrt per surviving (i,j) column.
            const double zr = std::sqrt(rc2 - r2xy);
            long klo = static_cast<long>(
                          std::ceil((pc_z - zr - origin_z) * invdz))
                      - lo_z;
            long khi = static_cast<long>(
                           std::floor((pc_z + zr - origin_z) * invdz))
                       - lo_z;
            if (klo < 0)
                klo = 0;
            if (khi > static_cast<long>(len_z) - 1)
                khi = len_z - 1;
            // Emit contiguous stride-1 runs (split only at periodic z wrap).
            long tk = klo;
            while (tk <= khi) {
                const long z0 = Wz[tk];
                const long run = std::min(khi - tk + 1, n_z - z0);
                add(base + z0, Fz + tk, run, fxy);
                tk += run;
            }
        }
    }
}

/// Floor on the chunk count of the grid-sized density scatter. Below this the
/// kernel loses its parallelism on large grids; at this value the accumulator
/// footprint matches what an 8-thread run already paid for per-thread buffers.
constexpr long kMinScatterChunks = 8;

// Grow-only flat fixed-chunk scatter pools (levels sequential). Layout
// chunk c starts at c*stride. Capacity grows across SCF iters; workers zero the
// active prefix. Avoids per-call allocate/free of nc full-grid / nb² buffers.
struct FlatChunkPool {
    std::vector<double> data;
    std::size_t stride = 0;
    void prepare(long nc, std::size_t n) {
        const std::size_t need = static_cast<std::size_t>(nc) * n;
        if (data.size() < need)
            data.resize(need);
        stride = n;
    }
    double* chunk(long c) {
        return data.data() + static_cast<std::size_t>(c) * stride;
    }
};
FlatChunkPool g_rho_scatter;
FlatChunkPool g_ks_scatter;
FlatChunkPool g_ks_cell_scatter;

} // namespace

namespace {
/// Geometry key for collint batching (same p/ra/rb/pc ⇒ shared pol table).
/// Stable total order so chunk partitions of a sorted product list are
/// reproducible across thread counts.
inline bool geom_less(const GridProduct& a, const GridProduct& b) {
    if (a.p != b.p)
        return a.p < b.p;
    for (int d = 0; d < 3; ++d)
        if (a.pc[d] != b.pc[d])
            return a.pc[d] < b.pc[d];
    for (int d = 0; d < 3; ++d)
        if (a.ra[d] != b.ra[d])
            return a.ra[d] < b.ra[d];
    for (int d = 0; d < 3; ++d)
        if (a.rb[d] != b.rb[d])
            return a.rb[d] < b.rb[d];
    return false;
}

/// Density weight w = K · Σ P[μ,ν] coeff_μ coeff_ν for one product shape.
inline double product_density_weight(const GridProduct& prod,
                                     const std::vector<PrimRef>& prims,
                                     const std::vector<PrimContrib>& contribs,
                                     int nb, const std::vector<double>& P) {
    const PrimRef& A = prims[static_cast<std::size_t>(prod.ua)];
    const PrimRef& B = prims[static_cast<std::size_t>(prod.ub)];
    double s = 0.0;
    for (int i = 0; i < A.cn; ++i) {
        const PrimContrib& ca = contribs[static_cast<std::size_t>(A.cbeg + i)];
        const std::size_t row = static_cast<std::size_t>(ca.mu) * nb;
        double rs = 0.0;
        for (int j = 0; j < B.cn; ++j) {
            const PrimContrib& cb =
                contribs[static_cast<std::size_t>(B.cbeg + j)];
            rs += P[row + static_cast<std::size_t>(cb.mu)] * cb.coeff;
        }
        s += ca.coeff * rs;
    }
    return s * prod.k;
}
} // namespace

std::vector<double> collocate_products(const UniformGrid& grid,
                                       const std::vector<GridProduct>& prods,
                                       const std::vector<PrimRef>& prims,
                                       const std::vector<PrimContrib>& contribs,
                                       int nb, const std::vector<double>& P,
                                       double eps) {
    eps *= kProductMassMargin; // window/weight drops discard integrated mass
    const std::size_t npts = static_cast<std::size_t>(grid.npts());
    std::vector<double> rho(npts, 0.0);
    const long np = static_cast<long>(prods.size());
    if (std::getenv("ATV_GPW_PROF")) {
        static long printed = 0;
        if (printed < 5) {
            std::fprintf(stderr, "[mg coll] level npts=%ld  products=%ld\n",
                         grid.npts(), np);
            ++printed;
        }
    }
    // Geometry order: products that share p/ra/rb/pc (cartesian monoms of one
    // shell pair) become consecutive so collint_collocate_group shares one pol.
    std::vector<int> order(static_cast<std::size_t>(np));
    for (long i = 0; i < np; ++i)
        order[static_cast<std::size_t>(i)] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        const auto& pa = prods[static_cast<std::size_t>(a)];
        const auto& pb = prods[static_cast<std::size_t>(b)];
        if (geom_less(pa, pb))
            return true;
        if (geom_less(pb, pa))
            return false;
        return a < b;
    });
    // Scatter buffers indexed by a FIXED product partition of the sorted order.
    const device::ChunkedRange range(
        np, std::max<long>(kMinScatterChunks,
                           device::block_reduce_chunks(npts * sizeof(double))));
    const long nc = range.chunks();
    g_rho_scatter.prepare(nc, npts);
#pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < nc; ++c) {
        double* loc = g_rho_scatter.chunk(c);
        std::fill_n(loc, npts, 0.0);
        const long iplo = range.begin(c), iphi = range.end(c);
        // Thread-local group buffers (grow-only via vector).
        std::vector<GridProduct> gprods;
        std::vector<double> gweights;
        gprods.reserve(32);
        gweights.reserve(32);
        auto flush_group = [&]() {
            if (gprods.empty())
                return;
            double mag = 0.0;
            for (double w : gweights)
                mag += std::abs(w);
            if (mag >= eps)
                internal::collint_collocate_group(grid, gprods.data(),
                                                  gweights.data(),
                                                  static_cast<int>(gprods.size()),
                                                  mag, eps, loc);
            gprods.clear();
            gweights.clear();
        };
        for (long ii = iplo; ii < iphi; ++ii) {
            const int ip = order[static_cast<std::size_t>(ii)];
            const GridProduct& prod = prods[static_cast<std::size_t>(ip)];
            const double w =
                product_density_weight(prod, prims, contribs, nb, P);
            if (std::abs(w) < eps)
                continue;
            if (!gprods.empty()
                && !internal::collint_same_geometry(gprods.back(), prod))
                flush_group();
            gprods.push_back(prod);
            gweights.push_back(w);
        }
        flush_group();
    }
#pragma omp parallel for schedule(static)
    for (long g = 0; g < static_cast<long>(npts); ++g) {
        double s = 0.0;
        for (long c = 0; c < nc; ++c)
            s += g_rho_scatter.chunk(c)[static_cast<std::size_t>(g)];
        rho[static_cast<std::size_t>(g)] = s;
    }
    return rho;
}

namespace {

/// @brief |V| at the product Gaussian centre (wrapped grid index).
///
/// Used to |V|-scale integrate windows: point value is shape·V, so a product
/// far from nuclei (small |V|) gets a smaller window than the density-
/// independent worst case max|coeff|·K (implicit |V|≡1). Floor avoids
/// over-shrinking in near-zero V pockets. O(1) per product.
inline double v_at_product_center(const UniformGrid& grid,
                                  const std::vector<double>& V_level,
                                  const GridProduct& prod) {
    const long n0 = grid.n[0], n1 = grid.n[1], n2 = grid.n[2];
    auto wrap = [](long i, long n) { return ((i % n) + n) % n; };
    long ix = wrap(static_cast<long>(std::floor(prod.pc[0] / (grid.L[0] / n0))),
                   n0);
    long iy = wrap(static_cast<long>(std::floor(prod.pc[1] / (grid.L[1] / n1))),
                   n1);
    long iz = wrap(static_cast<long>(std::floor(prod.pc[2] / (grid.L[2] / n2))),
                   n2);
    const std::size_t g = static_cast<std::size_t>((ix * n1 + iy) * n2 + iz);
    return std::abs(V_level[g]);
}

/// @brief Window magnitude for integrate: geom · max(|V(pc)|, floor).
/// Floor 0.05 keeps a safety margin vs pure local |V| without reverting to
/// whole-cell sweeps when |V| is large only near nuclei (global max|V| would
/// inflate every product).
inline double integrate_window_mag(const UniformGrid& grid,
                                   const std::vector<double>& V_level,
                                   const GridProduct& prod, double geom) {
    constexpr double kVFloor = 0.05;
    const double vsc = std::max(v_at_product_center(grid, V_level, prod), kVFloor);
    return geom * vsc;
}

} // namespace

std::vector<double> integrate_products(
    const UniformGrid& grid, const std::vector<GridProduct>& prods,
    const std::vector<PrimRef>& prims, const std::vector<PrimContrib>& contribs,
    int nb, const std::vector<double>& V_level, double eps) {
    eps *= kProductMassMargin; // window truncation discards integrated mass
    const std::size_t nn = static_cast<std::size_t>(nb) * nb;
    std::vector<double> V(nn, 0.0);
    const double w = grid.weight();
    const long np = static_cast<long>(prods.size());
    // Geometry order first (shared monom table), then spatial cell as tie-break
    // for V-cache locality across different centres.
    std::vector<int> order(static_cast<std::size_t>(np));
    for (long i = 0; i < np; ++i)
        order[static_cast<std::size_t>(i)] = static_cast<int>(i);
    {
        const double invdx = grid.n[0] / grid.L[0];
        const double invdy = grid.n[1] / grid.L[1];
        const double invdz = grid.n[2] / grid.L[2];
        const long n1 = grid.n[1], n2 = grid.n[2];
        auto cell_key = [&](int ip) {
            const auto& p = prods[static_cast<std::size_t>(ip)];
            const long ix = static_cast<long>(std::floor(p.pc[0] * invdx));
            const long iy = static_cast<long>(std::floor(p.pc[1] * invdy));
            const long iz = static_cast<long>(std::floor(p.pc[2] * invdz));
            return (ix * n1 + iy) * n2 + iz;
        };
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            const auto& pa = prods[static_cast<std::size_t>(a)];
            const auto& pb = prods[static_cast<std::size_t>(b)];
            if (geom_less(pa, pb))
                return true;
            if (geom_less(pb, pa))
                return false;
            const long ca = cell_key(a), cb = cell_key(b);
            if (ca != cb)
                return ca < cb;
            return a < b;
        });
    }
    const device::ChunkedRange range(
        np, device::block_reduce_chunks(nn * sizeof(double)));
    const long nc = range.chunks();
    g_ks_scatter.prepare(nc, nn);
#pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < nc; ++c) {
        double* Vloc = g_ks_scatter.chunk(c);
        std::fill_n(Vloc, nn, 0.0);
        const long iplo = range.begin(c), iphi = range.end(c);
        std::vector<GridProduct> gprods;
        std::vector<int> gidx; // index into prods for scatter
        std::vector<double> gacc;
        gprods.reserve(32);
        gidx.reserve(32);
        gacc.reserve(32);
        auto flush_group = [&]() {
            if (gprods.empty())
                return;
            double mag = 0.0;
            for (const auto& prod : gprods) {
                const PrimRef& A = prims[static_cast<std::size_t>(prod.ua)];
                const PrimRef& B = prims[static_cast<std::size_t>(prod.ub)];
                const double geom = A.maxc * B.maxc * std::abs(prod.k);
                mag = std::max(
                    mag, integrate_window_mag(grid, V_level, prod, geom));
            }
            gacc.resize(gprods.size());
            if (mag >= eps)
                internal::collint_integrate_group(
                    grid, gprods.data(), static_cast<int>(gprods.size()), mag,
                    eps, V_level.data(), gacc.data());
            else
                std::fill(gacc.begin(), gacc.end(), 0.0);
            for (std::size_t gi = 0; gi < gprods.size(); ++gi) {
                const GridProduct& prod = gprods[gi];
                const PrimRef& A = prims[static_cast<std::size_t>(prod.ua)];
                const PrimRef& B = prims[static_cast<std::size_t>(prod.ub)];
                const double sw = prod.k * w * gacc[gi];
                for (int i = 0; i < A.cn; ++i) {
                    const PrimContrib& ca =
                        contribs[static_cast<std::size_t>(A.cbeg + i)];
                    const std::size_t row =
                        static_cast<std::size_t>(ca.mu) * nb;
                    const double cas = ca.coeff * sw;
                    for (int j = 0; j < B.cn; ++j) {
                        const PrimContrib& cb =
                            contribs[static_cast<std::size_t>(B.cbeg + j)];
                        Vloc[row + static_cast<std::size_t>(cb.mu)] +=
                            cas * cb.coeff;
                    }
                }
            }
            gprods.clear();
            gidx.clear();
        };
        for (long ii = iplo; ii < iphi; ++ii) {
            const int ip = order[static_cast<std::size_t>(ii)];
            const GridProduct& prod = prods[static_cast<std::size_t>(ip)];
            if (!gprods.empty()
                && !internal::collint_same_geometry(gprods.back(), prod))
                flush_group();
            gprods.push_back(prod);
            gidx.push_back(ip);
        }
        flush_group();
    }
#pragma omp parallel for schedule(static)
    for (long kk = 0; kk < static_cast<long>(nn); ++kk) {
        double s = 0.0;
        for (long c = 0; c < nc; ++c)
            s += g_ks_scatter.chunk(c)[static_cast<std::size_t>(kk)];
        V[static_cast<std::size_t>(kk)] = s;
    }
    return V;
}

std::vector<double> collocate_products_cells(
    const UniformGrid& grid, const std::vector<GridProduct>& prods,
    const std::vector<PrimRef>& prims, const std::vector<PrimContrib>& contribs,
    int nb, const std::vector<std::vector<double>>& D, double eps) {
    eps *= kProductMassMargin; // window/weight drops discard integrated mass
    // `collocate_products` with the density weight resolved per image cell:
    // w = K · Σ D[tcell][μ,ν]·coeff. Geometry-batched collint (shared pol).
    const std::size_t npts = static_cast<std::size_t>(grid.npts());
    std::vector<double> rho(npts, 0.0);
    const long np = static_cast<long>(prods.size());
    std::vector<int> order(static_cast<std::size_t>(np));
    for (long i = 0; i < np; ++i)
        order[static_cast<std::size_t>(i)] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        const auto& pa = prods[static_cast<std::size_t>(a)];
        const auto& pb = prods[static_cast<std::size_t>(b)];
        if (geom_less(pa, pb))
            return true;
        if (geom_less(pb, pa))
            return false;
        return a < b;
    });
    const device::ChunkedRange range(
        np, std::max<long>(kMinScatterChunks,
                           device::block_reduce_chunks(npts * sizeof(double))));
    const long nc = range.chunks();
    g_rho_scatter.prepare(nc, npts);
#pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < nc; ++c) {
        double* loc = g_rho_scatter.chunk(c);
        std::fill_n(loc, npts, 0.0);
        const long iplo = range.begin(c), iphi = range.end(c);
        std::vector<GridProduct> gprods;
        std::vector<double> gweights;
        gprods.reserve(32);
        gweights.reserve(32);
        auto flush_group = [&]() {
            if (gprods.empty())
                return;
            double mag = 0.0;
            for (double ww : gweights)
                mag += std::abs(ww);
            if (mag >= eps)
                internal::collint_collocate_group(
                    grid, gprods.data(), gweights.data(),
                    static_cast<int>(gprods.size()), mag, eps, loc);
            gprods.clear();
            gweights.clear();
        };
        for (long ii = iplo; ii < iphi; ++ii) {
            const int ip = order[static_cast<std::size_t>(ii)];
            const GridProduct& prod = prods[static_cast<std::size_t>(ip)];
            const double ww = product_density_weight(
                prod, prims, contribs, nb,
                D[static_cast<std::size_t>(prod.tcell)]);
            if (std::abs(ww) < eps)
                continue;
            if (!gprods.empty()
                && !internal::collint_same_geometry(gprods.back(), prod))
                flush_group();
            gprods.push_back(prod);
            gweights.push_back(ww);
        }
        flush_group();
    }
#pragma omp parallel for schedule(static)
    for (long g = 0; g < static_cast<long>(npts); ++g) {
        double s = 0.0;
        for (long c = 0; c < nc; ++c)
            s += g_rho_scatter.chunk(c)[static_cast<std::size_t>(g)];
        rho[static_cast<std::size_t>(g)] = s;
    }
    return rho;
}

std::vector<std::vector<double>> integrate_products_cells(
    const UniformGrid& grid, const std::vector<GridProduct>& prods,
    const std::vector<PrimRef>& prims, const std::vector<PrimContrib>& contribs,
    int nb, const std::vector<double>& V_level, std::size_t ncells,
    double eps) {
    eps *= kProductMassMargin; // window truncation discards integrated mass
    // KS scatter per image cell; geometry-batched monom integrate.
    const std::size_t nn = static_cast<std::size_t>(nb) * nb;
    const std::size_t nnall = nn * ncells;
    std::vector<std::vector<double>> V(
        ncells, std::vector<double>(nn, 0.0));
    const double w = grid.weight();
    const long np = static_cast<long>(prods.size());
    std::vector<int> order(static_cast<std::size_t>(np));
    for (long i = 0; i < np; ++i)
        order[static_cast<std::size_t>(i)] = static_cast<int>(i);
    {
        const double invdx = grid.n[0] / grid.L[0];
        const double invdy = grid.n[1] / grid.L[1];
        const double invdz = grid.n[2] / grid.L[2];
        const long n1 = grid.n[1], n2 = grid.n[2];
        auto cell_key = [&](int ip) {
            const auto& p = prods[static_cast<std::size_t>(ip)];
            const long ix = static_cast<long>(std::floor(p.pc[0] * invdx));
            const long iy = static_cast<long>(std::floor(p.pc[1] * invdy));
            const long iz = static_cast<long>(std::floor(p.pc[2] * invdz));
            return (ix * n1 + iy) * n2 + iz;
        };
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            const auto& pa = prods[static_cast<std::size_t>(a)];
            const auto& pb = prods[static_cast<std::size_t>(b)];
            if (geom_less(pa, pb))
                return true;
            if (geom_less(pb, pa))
                return false;
            const long ca = cell_key(a), cb = cell_key(b);
            if (ca != cb)
                return ca < cb;
            return a < b;
        });
    }
    const device::ChunkedRange range(
        np, device::block_reduce_chunks(nnall * sizeof(double)));
    const long nc = range.chunks();
    g_ks_cell_scatter.prepare(nc, nnall);
#pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < nc; ++c) {
        double* Vloc = g_ks_cell_scatter.chunk(c);
        std::fill_n(Vloc, nnall, 0.0);
        const long iplo = range.begin(c), iphi = range.end(c);
        std::vector<GridProduct> gprods;
        std::vector<double> gacc;
        gprods.reserve(32);
        gacc.reserve(32);
        auto flush_group = [&]() {
            if (gprods.empty())
                return;
            double mag = 0.0;
            for (const auto& prod : gprods) {
                const PrimRef& A = prims[static_cast<std::size_t>(prod.ua)];
                const PrimRef& B = prims[static_cast<std::size_t>(prod.ub)];
                const double geom = A.maxc * B.maxc * std::abs(prod.k);
                mag = std::max(
                    mag, integrate_window_mag(grid, V_level, prod, geom));
            }
            gacc.resize(gprods.size());
            if (mag >= eps)
                internal::collint_integrate_group(
                    grid, gprods.data(), static_cast<int>(gprods.size()), mag,
                    eps, V_level.data(), gacc.data());
            else
                std::fill(gacc.begin(), gacc.end(), 0.0);
            for (std::size_t gi = 0; gi < gprods.size(); ++gi) {
                const GridProduct& prod = gprods[gi];
                const PrimRef& A = prims[static_cast<std::size_t>(prod.ua)];
                const PrimRef& B = prims[static_cast<std::size_t>(prod.ub)];
                const double sw = prod.k * w * gacc[gi];
                double* Vcell =
                    Vloc + static_cast<std::size_t>(prod.tcell) * nn;
                for (int i = 0; i < A.cn; ++i) {
                    const PrimContrib& ca =
                        contribs[static_cast<std::size_t>(A.cbeg + i)];
                    const std::size_t row =
                        static_cast<std::size_t>(ca.mu) * nb;
                    const double cas = ca.coeff * sw;
                    for (int j = 0; j < B.cn; ++j) {
                        const PrimContrib& cb =
                            contribs[static_cast<std::size_t>(B.cbeg + j)];
                        Vcell[row + static_cast<std::size_t>(cb.mu)] +=
                            cas * cb.coeff;
                    }
                }
            }
            gprods.clear();
        };
        for (long ii = iplo; ii < iphi; ++ii) {
            const int ip = order[static_cast<std::size_t>(ii)];
            const GridProduct& prod = prods[static_cast<std::size_t>(ip)];
            if (!gprods.empty()
                && !internal::collint_same_geometry(gprods.back(), prod))
                flush_group();
            gprods.push_back(prod);
        }
        flush_group();
    }
#pragma omp parallel for schedule(static)
    for (long kk = 0; kk < static_cast<long>(nnall); ++kk) {
        double s = 0.0;
        for (long c = 0; c < nc; ++c)
            s += g_ks_cell_scatter.chunk(c)[static_cast<std::size_t>(kk)];
        V[static_cast<std::size_t>(kk) / nn]
         [static_cast<std::size_t>(kk) % nn] = s;
    }
    return V;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
