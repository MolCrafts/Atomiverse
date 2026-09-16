/// @file src/cpu/gpw/internal/kpoints.cpp
/// @brief Monkhorst-Pack k-point machinery for the GPW RKS SCF (see header).

#include "atomiverse/cpu/gpw/internal/kpoints.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/generators/xbuilder.hpp>

#include "atomiverse/cpu/device/chunked_range.h"
#include "atomiverse/cpu/gpw/internal/gth_integrals.h" // Prim, gth_projector_overlap
#include "atomiverse/cpu/gpw/internal/one_electron.h" // prim_overlap, prim_kinetic
#include "atomiverse/cpu/linalg/orbital_tangent.h" // c_mul, c_dagger_mul (zgemm)

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {
namespace gpw {
namespace internal {

namespace {

using Cplx = std::complex<double>;

constexpr double kTwoPi = 6.283185307179586476925286766559;

// Per-axis Bloch phase table e^{2 pi i kappa_d t} for t in [-m, m], indexed
// t + m. One table per (k, axis); the image phase is the 3-axis product.
struct PhaseTables {
    std::vector<std::vector<Cplx>> ax[3]; // ax[d][ik][t + m[d]]

    PhaseTables(const std::vector<KPoint>& kpts, const int m[3]) {
        for (int d = 0; d < 3; ++d) {
            ax[d].resize(kpts.size());
            for (std::size_t ik = 0; ik < kpts.size(); ++ik) {
                std::vector<Cplx>& tab = ax[d][ik];
                tab.resize(static_cast<std::size_t>(2 * m[d] + 1));
                for (int t = -m[d]; t <= m[d]; ++t) {
                    const double arg = kTwoPi * kpts[ik].frac[d] * t;
                    tab[static_cast<std::size_t>(t + m[d])] =
                        Cplx(std::cos(arg), std::sin(arg));
                }
            }
        }
    }
};

// Max Gaussian-product prefactor over the primitive pairs of (mu, nu) with
// min-image displacement — the same pair screen as the Γ build
// (`pair_overlap_bound` in gpw_solver.cpp): below the screen the WHOLE
// (i, j) element is negligible for every k (|phase| = 1).
double pair_bound(const GridBasisFunction& mu, const GridBasisFunction& nu,
                  const double L[3]) {
    double best = 0.0;
    for (const auto& p : mu)
        for (const auto& q : nu) {
            const double a = p.alpha, b = q.alpha;
            double d2 = 0.0;
            for (int d = 0; d < 3; ++d) {
                double e = p.center[d] - q.center[d];
                if (L[d] > 0.0)
                    e -= L[d] * std::nearbyint(e / L[d]);
                d2 += e * e;
            }
            best = std::max(best, std::abs(p.coeff * q.coeff)
                                      * std::exp(-a * b / (a + b) * d2));
        }
    return best;
}

// Re Tr(A B) for row-major complex matrices (no conjugation — the physical
// traces Tr(P T) pair a Hermitian density with a Hermitian operator).
// (defined in the outer anonymous namespace for reuse by the SCF loop through
// the header consumers; kept local here — the solver computes its own.)

} // namespace

std::vector<KPoint> monkhorst_pack(const int mesh[3], bool time_reversal) {
    for (int d = 0; d < 3; ++d)
        if (mesh[d] < 1)
            throw std::invalid_argument(
                "monkhorst_pack: mesh subdivision must be >= 1 (axis "
                + std::to_string(d) + " got " + std::to_string(mesh[d]) + ")");

    std::vector<KPoint> full;
    const double w0 = 1.0 / (static_cast<double>(mesh[0]) * mesh[1] * mesh[2]);
    for (int r0 = 1; r0 <= mesh[0]; ++r0)
        for (int r1 = 1; r1 <= mesh[1]; ++r1)
            for (int r2 = 1; r2 <= mesh[2]; ++r2) {
                KPoint kp;
                const int r[3] = {r0, r1, r2};
                for (int d = 0; d < 3; ++d) {
                    // CP2K MONKHORST-PACK: kappa = (2r - N - 1) / (2N).
                    double f = (2.0 * r[d] - mesh[d] - 1.0)
                               / (2.0 * static_cast<double>(mesh[d]));
                    // Canonicalize to (-1/2, 1/2].
                    f -= std::floor(f + 0.5);
                    if (f <= -0.5)
                        f += 1.0;
                    kp.frac[d] = f;
                }
                kp.weight = w0;
                full.push_back(kp);
            }

    if (!time_reversal)
        return full;

    // Fold kappa and -kappa (mod 1): keep the first of each pair, double its
    // weight. Self-paired points (kappa == -kappa mod 1: components 0 or 1/2)
    // keep weight w0.
    std::vector<KPoint> folded;
    std::vector<bool> used(full.size(), false);
    const double tol = 1e-12;
    auto same = [&](const double a[3], const double b[3]) {
        for (int d = 0; d < 3; ++d) {
            double diff = a[d] - b[d];
            diff -= std::nearbyint(diff); // mod 1
            if (std::abs(diff) > tol)
                return false;
        }
        return true;
    };
    for (std::size_t i = 0; i < full.size(); ++i) {
        if (used[i])
            continue;
        used[i] = true;
        KPoint kp = full[i];
        double neg[3] = {-kp.frac[0], -kp.frac[1], -kp.frac[2]};
        for (std::size_t j = i + 1; j < full.size(); ++j) {
            if (used[j])
                continue;
            if (same(full[j].frac, neg)) {
                used[j] = true;
                kp.weight += full[j].weight;
                break; // MP meshes have no duplicate points
            }
        }
        folded.push_back(kp);
    }
    return folded;
}

void build_overlap_kinetic_k(const std::vector<GridBasisFunction>& basis,
                             const std::vector<double>& radii,
                             const double L[3], const int m[3], double screen,
                             const std::vector<KPoint>& kpts,
                             std::vector<CplxMatrix>& S_k,
                             std::vector<CplxMatrix>& T_k) {
    const int nb = static_cast<int>(basis.size());
    const int nk = static_cast<int>(kpts.size());
    S_k.assign(static_cast<std::size_t>(nk),
               CplxMatrix(xt::zeros<Cplx>({nb, nb})));
    T_k.assign(static_cast<std::size_t>(nk),
               CplxMatrix(xt::zeros<Cplx>({nb, nb})));
    const PhaseTables ph(kpts, m);

    // Flattened image list + pre-multiplied Bloch phases (one Cplx per k·image).
    // Saves two complex multiplies per (image, k) in the pair loop.
    const int nx = 2 * m[0] + 1, ny = 2 * m[1] + 1, nz = 2 * m[2] + 1;
    const int nimg = nx * ny * nz;
    struct Img {
        int tx, ty, tz;
        double T[3];
    };
    std::vector<Img> imgs(static_cast<std::size_t>(nimg));
    std::vector<Cplx> phase_ik(static_cast<std::size_t>(nk) *
                               static_cast<std::size_t>(nimg));
    int ii = 0;
    for (int tx = -m[0]; tx <= m[0]; ++tx)
        for (int ty = -m[1]; ty <= m[1]; ++ty)
            for (int tz = -m[2]; tz <= m[2]; ++tz, ++ii) {
                imgs[static_cast<std::size_t>(ii)].tx = tx;
                imgs[static_cast<std::size_t>(ii)].ty = ty;
                imgs[static_cast<std::size_t>(ii)].tz = tz;
                imgs[static_cast<std::size_t>(ii)].T[0] = tx * L[0];
                imgs[static_cast<std::size_t>(ii)].T[1] = ty * L[1];
                imgs[static_cast<std::size_t>(ii)].T[2] = tz * L[2];
                for (int ik = 0; ik < nk; ++ik) {
                    phase_ik[static_cast<std::size_t>(ik) *
                                 static_cast<std::size_t>(nimg)
                             + static_cast<std::size_t>(ii)] =
                        ph.ax[0][static_cast<std::size_t>(ik)]
                             [static_cast<std::size_t>(tx + m[0])]
                        * ph.ax[1][static_cast<std::size_t>(ik)]
                               [static_cast<std::size_t>(ty + m[1])]
                        * ph.ax[2][static_cast<std::size_t>(ik)]
                               [static_cast<std::size_t>(tz + m[2])];
                }
            }

// Upper-triangle pair list for load-balanced OMP (row-parallel left the
    // last rows starved; dynamic pair chunks match CP2K pair tasks better).
    const long npairs = static_cast<long>(nb) * (static_cast<long>(nb) + 1) / 2;
#pragma omp parallel for schedule(dynamic, 8)
    for (long p = 0; p < npairs; ++p) {
        // Decode upper-tri (i,j): p = i*nb - i*(i-1)/2 + (j-i) with j>=i.
        // Solve i from p via quadratic; small-nb linear scan is fine too.
        int i = 0;
        long rem = p;
        long row = nb;
        while (rem >= row) {
            rem -= row;
            --row;
            ++i;
        }
        const int j = i + static_cast<int>(rem);
        const GridBasisFunction& a = basis[static_cast<std::size_t>(i)];
        const GridBasisFunction& b = basis[static_cast<std::size_t>(j)];
        if (a.empty() || b.empty())
            continue;
        if (pair_bound(a, b, L) < screen)
            continue;
        const double reach = radii[static_cast<std::size_t>(i)]
                             + radii[static_cast<std::size_t>(j)];
        const double reach2 = reach * reach;
        const double* ac = a[0].center;
        const double* bc = b[0].center;
        // Pack live images first (branch-free ST + phase over a short list).
        // For diamond Si + MOLOPT most of nimg is dead — packing cuts mispredicts.
        thread_local std::vector<int> live;
        live.clear();
        live.reserve(32);
        for (int im = 0; im < nimg; ++im) {
            const Img& img = imgs[static_cast<std::size_t>(im)];
            const double dx = ac[0] - bc[0] - img.T[0];
            const double dy = ac[1] - bc[1] - img.T[1];
            const double dz = ac[2] - bc[2] - img.T[2];
            if (dx * dx + dy * dy + dz * dz <= reach2)
                live.push_back(im);
        }
        std::vector<Cplx> s_acc(static_cast<std::size_t>(nk), Cplx(0, 0));
        std::vector<Cplx> t_acc(static_cast<std::size_t>(nk), Cplx(0, 0));
        for (int im : live) {
            const Img& img = imgs[static_cast<std::size_t>(im)];
            double s_im = 0.0, t_im = 0.0;
            contracted_ST_shift(a, b, img.T, s_im, t_im);
            for (int ik = 0; ik < nk; ++ik) {
                const Cplx phase =
                    phase_ik[static_cast<std::size_t>(ik) *
                                 static_cast<std::size_t>(nimg)
                             + static_cast<std::size_t>(im)];
                s_acc[static_cast<std::size_t>(ik)] += phase * s_im;
                t_acc[static_cast<std::size_t>(ik)] += phase * t_im;
            }
        }
        for (int ik = 0; ik < nk; ++ik) {
            S_k[static_cast<std::size_t>(ik)](i, j) =
                s_acc[static_cast<std::size_t>(ik)];
            T_k[static_cast<std::size_t>(ik)](i, j) =
                t_acc[static_cast<std::size_t>(ik)];
            if (j != i) {
                S_k[static_cast<std::size_t>(ik)](j, i) =
                    std::conj(s_acc[static_cast<std::size_t>(ik)]);
                T_k[static_cast<std::size_t>(ik)](j, i) =
                    std::conj(t_acc[static_cast<std::size_t>(ik)]);
            }
        }
    }
}

void add_nonlocal_kb_k(std::vector<CplxMatrix>& V_k,
                       const std::vector<GridBasisFunction>& basis,
                       const std::vector<GpwAtom>& atoms,
                       const std::vector<double>& radii, const double L[3],
                       const int mimg[3], double screen,
                       const std::vector<KPoint>& kpts) {
    const int nb = static_cast<int>(basis.size());
    const int nk = static_cast<int>(kpts.size());
    const PhaseTables ph(kpts, mimg);
    const double proj_eps = std::sqrt(2.0 * std::log(1.0 / screen));

    for (const auto& at : atoms) {
        if (at.nonlocal.empty())
            continue;
        double rmax_ch = 0.0;
        for (const auto& ch : at.nonlocal)
            rmax_ch = std::max(rmax_ch, ch.r);
        const double proj_reach = rmax_ch * proj_eps;
        // Flat slot layout over (channel, m, iproj).
        int nslot = 0;
        std::vector<int> choff(at.nonlocal.size());
        for (std::size_t c = 0; c < at.nonlocal.size(); ++c) {
            choff[c] = nslot;
            nslot += (2 * at.nonlocal[c].l + 1) * at.nonlocal[c].nproj;
        }
        const double* A = at.pos;
        // O_mu^k[slot] = <chi_mu^k | p_A> = sum_T e^{-i k T} <phi_mu(.-T)|p_A>.
        // per-k, per-basis slot vectors (complex); (nb x nslot) per k.
        std::vector<std::vector<Cplx>> O(
            static_cast<std::size_t>(nk),
            std::vector<Cplx>(static_cast<std::size_t>(nb) * nslot,
                              Cplx(0, 0)));
        std::vector<char> any(static_cast<std::size_t>(nb), 0);

#pragma omp parallel for schedule(dynamic, 4)
        for (int mu = 0; mu < nb; ++mu) {
            const GridBasisFunction& bf = basis[static_cast<std::size_t>(mu)];
            if (bf.empty())
                continue;
            const double reach =
                radii[static_cast<std::size_t>(mu)] + proj_reach;
            const double reach2 = reach * reach;
            std::vector<double> o_im(static_cast<std::size_t>(nslot));
            bool got = false;
            for (int tx = -mimg[0]; tx <= mimg[0]; ++tx)
                for (int ty = -mimg[1]; ty <= mimg[1]; ++ty)
                    for (int tz = -mimg[2]; tz <= mimg[2]; ++tz) {
                        const double T[3] = {tx * L[0], ty * L[1], tz * L[2]};
                        const double dx = bf[0].center[0] + T[0] - A[0];
                        const double dy = bf[0].center[1] + T[1] - A[1];
                        const double dz = bf[0].center[2] + T[2] - A[2];
                        if (dx * dx + dy * dy + dz * dz > reach2)
                            continue;
                        std::fill(o_im.begin(), o_im.end(), 0.0);
                        bool nz = false;
                        for (const auto& p : bf) {
                            if (p.coeff == 0.0)
                                continue;
                            Prim g;
                            g.alpha = p.alpha;
                            g.center[0] = p.center[0] + T[0];
                            g.center[1] = p.center[1] + T[1];
                            g.center[2] = p.center[2] + T[2];
                            g.l[0] = p.l[0];
                            g.l[1] = p.l[1];
                            g.l[2] = p.l[2];
                            for (std::size_t c = 0; c < at.nonlocal.size();
                                 ++c) {
                                const auto& ch = at.nonlocal[c];
                                const int nm = 2 * ch.l + 1;
                                for (int mm = 0; mm < nm; ++mm)
                                    for (int ip = 0; ip < ch.nproj; ++ip) {
                                        const double v =
                                            p.coeff
                                            * gth_projector_overlap(
                                                g, A, ch.l, ip + 1, ch.r, mm);
                                        if (v != 0.0) {
                                            o_im[static_cast<std::size_t>(
                                                choff[c] + mm * ch.nproj
                                                + ip)] += v;
                                            nz = true;
                                        }
                                    }
                            }
                        }
                        if (!nz)
                            continue;
                        got = true;
                        for (int ik = 0; ik < nk; ++ik) {
                            // Slot gather with the PLAIN phase e^{+2πi κ·t}:
                            // summing the projector images T″ collapses the
                            // double Bloch sum to V_nl^k(μν) = Σ_A O_μ h
                            // conj(O_ν) with O_μ = Σ_u e^{−iku}⟨φ_μ(·−u)|p⟩;
                            // storing O′ = conj(O) here lets the assembly
                            // below keep the ⟨bra|·|ket⟩ shape
                            // conj(O′_μ)·h·O′_ν ≡ O_μ·h·conj(O_ν). Putting
                            // the conjugate on the gather instead (the naive
                            // ⟨χ_μ|p_A⟩ reading) evaluates V_nl at −k — it
                            // stays Hermitian and Γ/zone-boundary exact, but
                            // broke the supercell folding identity by 2 Ha at
                            // κ = ±1/4 (caught by acceptance gate A2).
                            const Cplx phase =
                                ph.ax[0][static_cast<std::size_t>(ik)]
                                     [static_cast<std::size_t>(tx + mimg[0])]
                                * ph.ax[1][static_cast<std::size_t>(ik)]
                                       [static_cast<std::size_t>(ty + mimg[1])]
                                * ph.ax[2][static_cast<std::size_t>(ik)]
                                       [static_cast<std::size_t>(tz + mimg[2])];
                            Cplx* dst = O[static_cast<std::size_t>(ik)].data()
                                        + static_cast<std::size_t>(mu) * nslot;
                            for (int t = 0; t < nslot; ++t)
                                dst[t] +=
                                    phase * o_im[static_cast<std::size_t>(t)];
                        }
                    }
            if (got)
                any[static_cast<std::size_t>(mu)] = 1;
        }

        // hO[slot] = sum_j h_ij O[j] within each (channel, m); then
        // V_nl^k(mu, nu) += conj(O_mu) . hO_nu.
        for (int ik = 0; ik < nk; ++ik) {
            const std::vector<Cplx>& Ok = O[static_cast<std::size_t>(ik)];
            std::vector<Cplx> hO(static_cast<std::size_t>(nb) * nslot,
                                 Cplx(0, 0));
            for (int mu = 0; mu < nb; ++mu) {
                if (!any[static_cast<std::size_t>(mu)])
                    continue;
                const Cplx* src =
                    Ok.data() + static_cast<std::size_t>(mu) * nslot;
                Cplx* dst = hO.data() + static_cast<std::size_t>(mu) * nslot;
                for (std::size_t c = 0; c < at.nonlocal.size(); ++c) {
                    const auto& ch = at.nonlocal[c];
                    const int nm = 2 * ch.l + 1;
                    for (int mm = 0; mm < nm; ++mm) {
                        const int base = choff[c] + mm * ch.nproj;
                        for (int p1 = 0; p1 < ch.nproj; ++p1) {
                            Cplx s(0, 0);
                            for (int p2 = 0; p2 < ch.nproj; ++p2)
                                s += ch.hij(p1, p2)
                                     * src[static_cast<std::size_t>(base + p2)];
                            dst[static_cast<std::size_t>(base + p1)] = s;
                        }
                    }
                }
            }
            CplxMatrix& V = V_k[static_cast<std::size_t>(ik)];
#pragma omp parallel for schedule(static)
            for (int mu = 0; mu < nb; ++mu) {
                if (!any[static_cast<std::size_t>(mu)])
                    continue;
                const Cplx* o1 =
                    Ok.data() + static_cast<std::size_t>(mu) * nslot;
                for (int nu = 0; nu < nb; ++nu) {
                    if (!any[static_cast<std::size_t>(nu)])
                        continue;
                    const Cplx* h2 =
                        hO.data() + static_cast<std::size_t>(nu) * nslot;
                    Cplx s(0, 0);
                    for (int t = 0; t < nslot; ++t)
                        s += std::conj(o1[t]) * h2[t];
                    V(mu, nu) += s;
                }
            }
        }
    }
}

KpointBasisGrid
build_kpoint_basis_grid(const UniformGrid& grid,
                        const std::vector<GridBasisFunction>& basis,
                        const std::vector<KPoint>& kpts, double eps) {
    KpointBasisGrid kbg;
    const int nb = static_cast<int>(basis.size());
    const int nk = static_cast<int>(kpts.size());
    const long npts = grid.npts();
    kbg.nb = nb;
    kbg.npts = npts;
    kbg.weight = grid.weight();
    // Diffuse bases in few-Bohr cells touch every point through some image, so
    // every grid point is active (dense per-point storage; no CSR pruning).
    kbg.gidx.resize(static_cast<std::size_t>(npts));
    for (long g = 0; g < npts; ++g)
        kbg.gidx[static_cast<std::size_t>(g)] = g;
    kbg.chi.assign(
        static_cast<std::size_t>(nk),
        std::vector<Cplx>(static_cast<std::size_t>(npts) * nb, Cplx(0, 0)));

    const long n0 = grid.n[0], n1 = grid.n[1], n2 = grid.n[2];
    const double d0 = grid.L[0] / n0, d1 = grid.L[1] / n1, d2 = grid.L[2] / n2;

#pragma omp parallel for schedule(dynamic, 1)
    for (int u = 0; u < nb; ++u) {
        const GridBasisFunction& bf = basis[static_cast<std::size_t>(u)];
        if (bf.empty())
            continue;
        const double rc = basis_function_radius(bf, eps);
        const double c[3] = {bf[0].center[0], bf[0].center[1], bf[0].center[2]};
        const long n[3] = {n0, n1, n2};
        const double dx[3] = {d0, d1, d2};
        long lo[3], hi[3];
        for (int d = 0; d < 3; ++d) {
            const long ih = static_cast<long>(std::ceil(rc / dx[d]));
            const long ic = static_cast<long>(std::floor(c[d] / dx[d]));
            lo[d] = ic - ih;
            hi[d] = ic + ih;
        }
        // Per-axis image index range of the unwrapped window and its phase
        // table: cell index i lies in image t = (i - wrap(i)) / n.
        auto img_of = [](long i, long nn) {
            const long ii = ((i % nn) + nn) % nn;
            return (i - ii) / nn;
        };
        std::vector<std::vector<Cplx>> px(3);
        long tlo[3], thi[3];
        for (int d = 0; d < 3; ++d) {
            tlo[d] = img_of(lo[d], n[d]);
            thi[d] = img_of(hi[d], n[d]);
        }
        std::vector<std::vector<std::vector<Cplx>>> ptab(
            static_cast<std::size_t>(nk));
        for (int ik = 0; ik < nk; ++ik) {
            ptab[static_cast<std::size_t>(ik)].resize(3);
            for (int d = 0; d < 3; ++d) {
                auto& tab = ptab[static_cast<std::size_t>(ik)]
                                [static_cast<std::size_t>(d)];
                tab.resize(static_cast<std::size_t>(thi[d] - tlo[d] + 1));
                for (long t = tlo[d]; t <= thi[d]; ++t) {
                    // chi accumulates with e^{-2 pi i kappa . t'} (the window
                    // cell t' holds the image T = -T').
                    const double arg =
                        -kTwoPi * kpts[static_cast<std::size_t>(ik)].frac[d]
                        * static_cast<double>(t);
                    tab[static_cast<std::size_t>(t - tlo[d])] =
                        Cplx(std::cos(arg), std::sin(arg));
                }
            }
        }
        for (long i = lo[0]; i <= hi[0]; ++i) {
            const long ii = ((i % n0) + n0) % n0;
            const long ti = img_of(i, n0) - tlo[0];
            const double r0 =
                grid.origin[0] + grid.L[0] * static_cast<double>(i) / n0;
            for (long j = lo[1]; j <= hi[1]; ++j) {
                const long jj = ((j % n1) + n1) % n1;
                const long tj = img_of(j, n1) - tlo[1];
                const double r1 =
                    grid.origin[1] + grid.L[1] * static_cast<double>(j) / n1;
                const long base = (ii * n1 + jj) * n2;
                for (long kk2 = lo[2]; kk2 <= hi[2]; ++kk2) {
                    const long kkw = ((kk2 % n2) + n2) % n2;
                    const long tk = img_of(kk2, n2) - tlo[2];
                    const double r2 =
                        grid.origin[2]
                        + grid.L[2] * static_cast<double>(kk2) / n2;
                    // Contracted Gaussian at the UNWRAPPED point (image offset
                    // carried by the raw displacement).
                    double val = 0.0;
                    for (const auto& p : bf) {
                        const double ddx = r0 - p.center[0];
                        const double ddy = r1 - p.center[1];
                        const double ddz = r2 - p.center[2];
                        const double rr = ddx * ddx + ddy * ddy + ddz * ddz;
                        double t = p.coeff * std::exp(-p.alpha * rr);
                        for (int q = 0; q < p.l[0]; ++q)
                            t *= ddx;
                        for (int q = 0; q < p.l[1]; ++q)
                            t *= ddy;
                        for (int q = 0; q < p.l[2]; ++q)
                            t *= ddz;
                        val += t;
                    }
                    if (std::fabs(val) <= eps)
                        continue;
                    const long g = base + kkw;
                    for (int ik = 0; ik < nk; ++ik) {
                        const auto& pt = ptab[static_cast<std::size_t>(ik)];
                        const Cplx phase =
                            pt[0][static_cast<std::size_t>(ti)]
                            * pt[1][static_cast<std::size_t>(tj)]
                            * pt[2][static_cast<std::size_t>(tk)];
                        kbg.chi[static_cast<std::size_t>(ik)]
                               [static_cast<std::size_t>(g) * nb + u] +=
                            phase * val;
                    }
                }
            }
        }
    }
    return kbg;
}

void collocate_density_k(const KpointBasisGrid& kbg, int ik,
                         const CplxMatrix& P_k, double wk,
                         std::vector<double>& rho) {
    const int nb = kbg.nb;
    const long npts = kbg.npts;
    const std::vector<Cplx>& chi = kbg.chi[static_cast<std::size_t>(ik)];
#pragma omp parallel
    {
        std::vector<Cplx> y(static_cast<std::size_t>(nb));
#pragma omp for schedule(static)
        for (long g = 0; g < npts; ++g) {
            const Cplx* v = chi.data() + static_cast<std::size_t>(g) * nb;
            // rho_k(g) = sum_uv P_uv v_u conj(v_v) = sum_u v_u (P conj(v))_u.
            for (int u = 0; u < nb; ++u) {
                Cplx s(0, 0);
                const Cplx* Pu = &P_k(u, 0);
                for (int w = 0; w < nb; ++w)
                    s += Pu[w] * std::conj(v[w]);
                y[static_cast<std::size_t>(u)] = s;
            }
            double acc = 0.0;
            for (int u = 0; u < nb; ++u)
                acc += (v[u] * y[static_cast<std::size_t>(u)]).real();
            rho[static_cast<std::size_t>(g)] += wk * acc;
        }
    }
}

CplxMatrix integrate_potential_k(const KpointBasisGrid& kbg, int ik,
                                 const std::vector<double>& V_r) {
    const int nb = kbg.nb;
    const long npts = kbg.npts;
    const double w = kbg.weight;
    const std::vector<Cplx>& chi = kbg.chi[static_cast<std::size_t>(ik)];
    CplxMatrix V = xt::zeros<Cplx>({nb, nb});
    // Reproducible reduction: a FIXED partition of the grid into chunks, one
    // nb² accumulator each, folded in ascending chunk order. The per-thread
    // block merged under `#pragma omp critical` this replaces mixed partials in
    // thread-completion order, so S_k-metric quantities built from V_k differed
    // run to run — which is what made the multi-k GW Newton QP solve
    // intermittently miss its root. See cpu/device/chunked_range.h.
    const std::size_t nn = static_cast<std::size_t>(nb) * nb;
    const device::ChunkedRange range(
        npts, device::block_reduce_chunks(nn * sizeof(Cplx)));
    const long nc = range.chunks();
    std::vector<Cplx> part(static_cast<std::size_t>(nc) * nn, Cplx(0, 0));
#pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < nc; ++c) {
        Cplx* Vloc = &part[static_cast<std::size_t>(c) * nn];
        const long glo = range.begin(c), ghi = range.end(c);
        for (long g = glo; g < ghi; ++g) {
            const double vg = w * V_r[static_cast<std::size_t>(g)];
            if (vg == 0.0)
                continue;
            const Cplx* v = chi.data() + static_cast<std::size_t>(g) * nb;
            for (int u = 0; u < nb; ++u) {
                const Cplx cu = vg * std::conj(v[u]);
                Cplx* row = Vloc + static_cast<std::size_t>(u) * nb;
                for (int t = 0; t < nb; ++t)
                    row[t] += cu * v[t];
            }
        }
    }
    for (long c = 0; c < nc; ++c) {
        const Cplx* Vloc = &part[static_cast<std::size_t>(c) * nn];
        for (int u = 0; u < nb; ++u)
            for (int t = 0; t < nb; ++t)
                V(u, t) += Vloc[static_cast<std::size_t>(u) * nb + t];
    }
    return V;
}

KOrth::KOrth(const CplxMatrix& S_k, double tau) {
    const int n = static_cast<int>(S_k.shape(0));
    std::vector<double> s;
    CplxMatrix U;
    linalg::eigh_hermitian(S_k, s, U); // ascending
    s_min = s.empty() ? 0.0 : s[0];
    // A Bloch overlap is a Gram matrix at every k — a negative eigenvalue
    // means the phased lattice sum is WRONG, not ill-conditioned (same guard
    // as the Γ path).
    constexpr double kPsdTol = -1e-8;
    if (s_min < kPsdTol)
        throw std::runtime_error(
            "GPW kpoints: S_k is not positive semidefinite (s_min="
            + std::to_string(s_min)
            + ") — the phased lattice sum is incomplete");
    int first = 0;
    if (tau > 0)
        while (first < n && s[static_cast<std::size_t>(first)] < tau)
            ++first;
    n_dropped = first;
    const int m = n - first;
    if (m == 0)
        throw std::runtime_error(
            "GPW kpoints: every overlap eigenvalue below lindep threshold "
            "at this k — no basis left");
    X = xt::zeros<Cplx>({n, m});
    for (int j = 0; j < m; ++j) {
        const double inv_sqrt =
            1.0 / std::sqrt(s[static_cast<std::size_t>(first + j)]);
        for (int i = 0; i < n; ++i)
            X(i, j) = U(i, first + j) * inv_sqrt;
    }
}

std::pair<std::vector<double>, CplxMatrix> diagonalize_k(const CplxMatrix& H_k,
                                                         const KOrth& orth) {
    // Dense complex products go through BLAS zgemm (linalg::c_mul /
    // linalg::c_dagger_mul, same row-major CplxMatrix type).
    const CplxMatrix Hp = linalg::c_dagger_mul(orth.X, linalg::c_mul(H_k, orth.X));
    std::vector<double> eps;
    CplxMatrix Cp;
    linalg::eigh_hermitian(Hp, eps, Cp);
    return {std::move(eps), linalg::c_mul(orth.X, Cp)};
}

CplxMatrix k_diis_error(const CplxMatrix& H, const CplxMatrix& P,
                        const CplxMatrix& S, const CplxMatrix& X) {
    const CplxMatrix comm = linalg::c_mul(H, linalg::c_mul(P, S))
                            - linalg::c_mul(S, linalg::c_mul(P, H));
    return linalg::c_dagger_mul(X, linalg::c_mul(comm, X));
}

std::vector<CplxMatrix> KDiis::step(const std::vector<CplxMatrix>& H,
                                    const std::vector<CplxMatrix>& e) {
    h_hist_.push_back(H);
    e_hist_.push_back(e);
    while (static_cast<int>(h_hist_.size()) > max_vectors_) {
        h_hist_.erase(h_hist_.begin());
        e_hist_.erase(e_hist_.begin());
    }
    const int m = static_cast<int>(h_hist_.size());
    if (m < 2)
        return h_hist_.back();

    const int nk = static_cast<int>(H.size());
    const int dim = m + 1;
    xt::xtensor<double, 2> B = xt::zeros<double>({dim, dim});
    for (int i = 0; i < m; ++i)
        for (int j = 0; j <= i; ++j) {
            double val = 0.0;
            for (int ik = 0; ik < nk; ++ik) {
                const CplxMatrix& ei = e_hist_[static_cast<std::size_t>(i)]
                                              [static_cast<std::size_t>(ik)];
                const CplxMatrix& ej = e_hist_[static_cast<std::size_t>(j)]
                                              [static_cast<std::size_t>(ik)];
                // Empty (zero-size) slots are non-owned k under MPI RR —
                // their contribution is Allreduced in from the owner rank.
                if (ei.size() == 0 || ej.size() == 0)
                    continue;
                double s = 0.0;
                const std::size_t nn = ei.size();
                const Cplx* pi = ei.data();
                const Cplx* pj = ej.data();
                for (std::size_t t = 0; t < nn; ++t)
                    s += (std::conj(pi[t]) * pj[t]).real();
                val += weights_[static_cast<std::size_t>(ik)] * s;
            }
            B(i, j) = val;
            B(j, i) = val;
        }
#ifdef ATV_HAS_MPI
    // Multi-rank k-RR: each rank only filled local e^k in the history; sum the
    // m×m Pulay block so every rank solves the same B system (serial-identical
    // DIIS coefficients). Lagrange row/col is set after the reduction.
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        int nprocs = 1;
        if (initialized)
            MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        if (initialized && nprocs > 1) {
            std::vector<double> pack(static_cast<std::size_t>(m * m), 0.0);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j)
                    pack[static_cast<std::size_t>(i * m + j)] = B(i, j);
            MPI_Allreduce(MPI_IN_PLACE, pack.data(), m * m, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j)
                    B(i, j) = pack[static_cast<std::size_t>(i * m + j)];
        }
    }
#endif
    for (int i = 0; i < m; ++i) {
        B(m, i) = -1.0;
        B(i, m) = -1.0;
    }
    B(m, m) = 0.0;

    // Solve the bordered Pulay system by Gaussian elimination with partial
    // pivoting (small dense real system; falls back to the latest H when the
    // subspace is singular).
    std::vector<double> rhs(static_cast<std::size_t>(dim), 0.0);
    rhs[static_cast<std::size_t>(m)] = -1.0;
    std::vector<double> M(static_cast<std::size_t>(dim) * dim);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            M[static_cast<std::size_t>(i) * dim + j] = B(i, j);
    bool singular = false;
    for (int col = 0; col < dim && !singular; ++col) {
        int piv = col;
        double best = std::abs(M[static_cast<std::size_t>(col) * dim + col]);
        for (int r = col + 1; r < dim; ++r) {
            const double v =
                std::abs(M[static_cast<std::size_t>(r) * dim + col]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }
        if (best < 1e-14) {
            singular = true;
            break;
        }
        if (piv != col) {
            for (int j = 0; j < dim; ++j)
                std::swap(M[static_cast<std::size_t>(col) * dim + j],
                          M[static_cast<std::size_t>(piv) * dim + j]);
            std::swap(rhs[static_cast<std::size_t>(col)],
                      rhs[static_cast<std::size_t>(piv)]);
        }
        const double d = M[static_cast<std::size_t>(col) * dim + col];
        for (int r = 0; r < dim; ++r) {
            if (r == col)
                continue;
            const double f = M[static_cast<std::size_t>(r) * dim + col] / d;
            if (f == 0.0)
                continue;
            for (int j = col; j < dim; ++j)
                M[static_cast<std::size_t>(r) * dim + j] -=
                    f * M[static_cast<std::size_t>(col) * dim + j];
            rhs[static_cast<std::size_t>(r)] -=
                f * rhs[static_cast<std::size_t>(col)];
        }
    }
    if (singular)
        return h_hist_.back();
    std::vector<double> c(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i)
        c[static_cast<std::size_t>(i)] =
            rhs[static_cast<std::size_t>(i)]
            / M[static_cast<std::size_t>(i) * dim + i];

    std::vector<CplxMatrix> out;
    out.reserve(static_cast<std::size_t>(nk));
    for (int ik = 0; ik < nk; ++ik) {
        CplxMatrix Hd = xt::zeros<Cplx>(
            h_hist_.back()[static_cast<std::size_t>(ik)].shape());
        for (int i = 0; i < m; ++i)
            Hd += c[static_cast<std::size_t>(i)]
                  * h_hist_[static_cast<std::size_t>(i)]
                           [static_cast<std::size_t>(ik)];
        out.push_back(std::move(Hd));
    }
    return out;
}

} // namespace internal
} // namespace gpw
} // namespace cpu
} // namespace ATV
