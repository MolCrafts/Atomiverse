/// @file cpu/scf/internal/kpoint_hf.cpp
/// @brief Periodic k-point RHF and UHF. See kpoint_hf.h for the PySCF functions.
///        KUHF iteration is `cpu::scf::KUHF`; this file only steps one cycle.

#include "atomiverse/cpu/scf/internal/kpoint_hf.h"

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/scf/internal/basis_workspace.h"
#include "atomiverse/cpu/scf/native_integrals.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kPrecision = 1e-8;

int fft_bin(int i, int n) {
    // np.fft.fftfreq(n)[i] * n, odd or even.
    if (i <= (n - 1) / 2)
        return i;
    return i - n;
}

double estimate_rcut_one(double alpha, int l, double c, double precision) {
    // pyscf.pbc.gto.cell._estimate_rcut
    const double theta = alpha * 0.5;
    const double a1 = 1.0 / std::sqrt(alpha * 2.0);
    const double norm_ang = (2.0 * l + 1.0) / (4.0 * kPi);
    double fac = 2.0 * kPi * c * c * norm_ang / theta / precision;
    fac *= 4.0 * alpha * alpha;
    double r0 = 20.0;
    const double pl = static_cast<double>(2 * l + 2);
    for (int it = 0; it < 2; ++it) {
        const double mid = r0 * 0.5 + a1;
        r0 = std::sqrt(std::log(fac * r0 * std::pow(mid, pl) + 1.0) / theta);
    }
    return r0;
}

double estimate_ke_one(double alpha, int l, double c, double precision) {
    // pyscf.pbc.gto.cell._estimate_ke_cutoff, omega = 0.
    const double norm_ang = (2.0 * l + 1.0) / (4.0 * kPi);
    const double fac = 32.0 * kPi * kPi * std::pow(2.0 * kPi, 1.5) * c * c
                       * norm_ang / std::pow(2.0 * alpha, 2.0 * l + 0.5)
                       / precision;
    double ecut = 20.0;
    const double pl = static_cast<double>(l) - 0.5;
    for (int it = 0; it < 2; ++it)
        ecut = std::log(fac * std::pow(ecut * 2.0, pl) + 1.0) * 4.0 * alpha;
    return ecut;
}

struct CMat {
    int n = 0;
    std::vector<std::complex<double>> a;
    std::complex<double>& operator()(int i, int j) { return a[i * n + j]; }
    std::complex<double> operator()(int i, int j) const { return a[i * n + j]; }
    static CMat zeros(int n) {
        CMat m;
        m.n = n;
        m.a.assign(static_cast<std::size_t>(n * n), {0.0, 0.0});
        return m;
    }
    void hermitize() {
        for (int i = 0; i < n; ++i) {
            (*this)(i, i) = {(*this)(i, i).real(), 0.0};
            for (int j = i + 1; j < n; ++j) {
                const std::complex<double> h =
                    0.5 * ((*this)(i, j) + std::conj((*this)(j, i)));
                (*this)(i, j) = h;
                (*this)(j, i) = std::conj(h);
            }
        }
    }
    linalg::CplxMatrix to_xt() const {
        linalg::CplxMatrix m = xt::zeros<std::complex<double>>({n, n});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                m(i, j) = (*this)(i, j);
        return m;
    }
};

void dft_1d(std::complex<double>* v, int n, int stride, bool inverse) {
    std::vector<std::complex<double>> in(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j)
        in[static_cast<std::size_t>(j)] = v[j * stride];
    const double sign = inverse ? 1.0 : -1.0;
    const double inv = inverse ? 1.0 / static_cast<double>(n) : 1.0;
    for (int k = 0; k < n; ++k) {
        std::complex<double> s{0.0, 0.0};
        for (int j = 0; j < n; ++j) {
            const double ang =
                sign * 2.0 * kPi * static_cast<double>(k * j) / n;
            s += in[static_cast<std::size_t>(j)]
                 * std::complex<double>(std::cos(ang), std::sin(ang));
        }
        v[k * stride] = s * inv;
    }
}

std::vector<std::array<double, 3>> image_vectors(const double L[3],
                                                 double rcut) {
    int m[3];
    for (int d = 0; d < 3; ++d) {
        if (!(L[d] > 0.0))
            throw std::runtime_error(
                "k-point HF: orthorhombic edge must be positive");
        m[d] = static_cast<int>(std::ceil(rcut / L[d]));
    }
    std::vector<std::array<double, 3>> ts;
    ts.reserve(static_cast<std::size_t>((2 * m[0] + 1) * (2 * m[1] + 1)
                                        * (2 * m[2] + 1)));
    for (int ix = -m[0]; ix <= m[0]; ++ix)
        for (int iy = -m[1]; iy <= m[1]; ++iy)
            for (int iz = -m[2]; iz <= m[2]; ++iz) {
                const double t[3] = {ix * L[0], iy * L[1], iz * L[2]};
                const double r =
                    std::sqrt(t[0] * t[0] + t[1] * t[1] + t[2] * t[2]);
                if (r <= rcut + 1e-12)
                    ts.push_back({t[0], t[1], t[2]});
            }
    if (ts.empty())
        ts.push_back({0.0, 0.0, 0.0});
    return ts;
}

void grid_geometry(const double L[3], const int mesh[3],
                   std::vector<double>& coords, std::vector<double>& gv) {
    const int nx = mesh[0], ny = mesh[1], nz = mesh[2];
    const int ng = nx * ny * nz;
    coords.resize(static_cast<std::size_t>(ng) * 3);
    gv.resize(static_cast<std::size_t>(ng) * 3);
    const double b[3] = {2.0 * kPi / L[0], 2.0 * kPi / L[1], 2.0 * kPi / L[2]};
    for (int ix = 0; ix < nx; ++ix) {
        const double fx = static_cast<double>(fft_bin(ix, nx)) / nx;
        const double gx = static_cast<double>(fft_bin(ix, nx)) * b[0];
        for (int iy = 0; iy < ny; ++iy) {
            const double fy = static_cast<double>(fft_bin(iy, ny)) / ny;
            const double gy = static_cast<double>(fft_bin(iy, ny)) * b[1];
            for (int iz = 0; iz < nz; ++iz) {
                const int g = iz + nz * (iy + ny * ix);
                const double fz = static_cast<double>(fft_bin(iz, nz)) / nz;
                const double gz = static_cast<double>(fft_bin(iz, nz)) * b[2];
                coords[static_cast<std::size_t>(g) * 3 + 0] = fx * L[0];
                coords[static_cast<std::size_t>(g) * 3 + 1] = fy * L[1];
                coords[static_cast<std::size_t>(g) * 3 + 2] = fz * L[2];
                gv[static_cast<std::size_t>(g) * 3 + 0] = gx;
                gv[static_cast<std::size_t>(g) * 3 + 1] = gy;
                gv[static_cast<std::size_t>(g) * 3 + 2] = gz;
            }
        }
    }
}

std::vector<double> coulomb_kernel(const double L[3], const int mesh[3],
                                   const double* gv, const double kshift[3],
                                   bool ewald_g0, int nk, double madelung,
                                   double vol) {
    // pyscf.pbc.tools.pbc.get_coulG, dimension 3, exxdiv 'ewald' or None.
    const int ng = mesh[0] * mesh[1] * mesh[2];
    const double b[3] = {2.0 * kPi / L[0], 2.0 * kPi / L[1], 2.0 * kPi / L[2]};
    const bool shift =
        std::abs(kshift[0]) + std::abs(kshift[1]) + std::abs(kshift[2]) > 1e-9;
    std::vector<double> coul(static_cast<std::size_t>(ng), 0.0);
    for (int g = 0; g < ng; ++g) {
        double kg[3] = {gv[g * 3] + (shift ? kshift[0] : 0.0),
                        gv[g * 3 + 1] + (shift ? kshift[1] : 0.0),
                        gv[g * 3 + 2] + (shift ? kshift[2] : 0.0)};
        if (shift) {
            for (int d = 0; d < 3; ++d) {
                const double edge = mesh[d] * b[d];
                const double red = kg[d] / edge;
                if (red > 0.5)
                    kg[d] -= edge;
                if (red < -0.5)
                    kg[d] += edge;
            }
        }
        const double g2 = kg[0] * kg[0] + kg[1] * kg[1] + kg[2] * kg[2];
        if (g2 == 0.0) {
            coul[static_cast<std::size_t>(g)] =
                (ewald_g0 ? static_cast<double>(nk) * vol * madelung : 0.0);
        } else {
            coul[static_cast<std::size_t>(g)] = 4.0 * kPi / g2;
        }
    }
    return coul;
}

void eval_ao_s(const std::vector<native::Shell>& shells,
               const std::vector<std::array<double, 3>>& images,
               const std::vector<double>& coords, int ngrids,
               const double kpt[3], std::vector<std::complex<double>>& ao) {
    const int nao = static_cast<int>(shells.size());
    ao.assign(static_cast<std::size_t>(ngrids * nao), {0.0, 0.0});
    for (int mu = 0; mu < nao; ++mu) {
        const native::Shell& sh = shells[static_cast<std::size_t>(mu)];
        if (sh.l != 0)
            throw std::runtime_error("KRHF: l>0 collocation is not implemented "
                                     "(missing kernel krhf_ao_lgt0)");
        for (const auto& T : images) {
            const std::complex<double> phase = std::exp(std::complex<double>(
                0.0, kpt[0] * T[0] + kpt[1] * T[1] + kpt[2] * T[2]));
            const double cx = sh.center[0] + T[0];
            const double cy = sh.center[1] + T[1];
            const double cz = sh.center[2] + T[2];
            for (int g = 0; g < ngrids; ++g) {
                const double dx = coords[g * 3] - cx;
                const double dy = coords[g * 3 + 1] - cy;
                const double dz = coords[g * 3 + 2] - cz;
                const double r2 = dx * dx + dy * dy + dz * dz;
                double val = 0.0;
                for (int p = 0; p < sh.n_prim; ++p)
                    val += sh.coeff[p] * std::exp(-sh.alpha[p] * r2);
                ao[static_cast<std::size_t>(g * nao + mu)] += phase * val;
            }
        }
    }
}

CMat bloch_1e(const std::vector<native::Shell>& shells,
              const std::vector<std::array<double, 3>>& images,
              const double kpt[3], bool kinetic) {
    const int nao = static_cast<int>(shells.size());
    CMat m = CMat::zeros(nao);
    for (int i = 0; i < nao; ++i) {
        for (int j = 0; j < nao; ++j) {
            std::complex<double> acc{0.0, 0.0};
            for (const auto& T : images) {
                native::Shell b = shells[static_cast<std::size_t>(j)];
                b.center[0] += T[0];
                b.center[1] += T[1];
                b.center[2] += T[2];
                const double integ =
                    kinetic
                        ? native::kinetic(shells[static_cast<std::size_t>(i)],
                                          b)
                        : native::overlap(shells[static_cast<std::size_t>(i)],
                                          b);
                const std::complex<double> phase =
                    std::exp(std::complex<double>(
                        0.0, kpt[0] * T[0] + kpt[1] * T[1] + kpt[2] * T[2]));
                acc += phase * integ;
            }
            m(i, j) = acc;
        }
    }
    m.hermitize();
    return m;
}

struct Orth {
    linalg::CplxMatrix X; // nao x nmo
    int nmo = 0;
};

Orth canonical_x(const CMat& S) {
    std::vector<double> eval;
    linalg::CplxMatrix U;
    linalg::eigh_hermitian(S.to_xt(), eval, U);
    const int n = S.n;
    int first = 0;
    constexpr double kDrop = 1e-8;
    while (first < n && eval[static_cast<std::size_t>(first)] < kDrop)
        ++first;
    if (first < n && eval[static_cast<std::size_t>(first)] < -1e-8)
        throw std::runtime_error(
            "KRHF: Bloch overlap is not positive semidefinite");
    const int m = n - first;
    if (m <= 0)
        throw std::runtime_error("KRHF: overlap has no positive eigenvalue");
    Orth o;
    o.nmo = m;
    o.X = xt::zeros<std::complex<double>>({n, m});
    for (int j = 0; j < m; ++j) {
        const double inv =
            1.0 / std::sqrt(eval[static_cast<std::size_t>(first + j)]);
        for (int i = 0; i < n; ++i)
            o.X(i, j) = U(i, first + j) * inv;
    }
    return o;
}

void eigh_fock(const CMat& F, const Orth& orth, std::vector<double>& eps,
               linalg::CplxMatrix& C) {
    const int n = F.n;
    const int m = orth.nmo;
    linalg::CplxMatrix Fm = xt::zeros<std::complex<double>>({n, m});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            std::complex<double> s{0.0, 0.0};
            for (int k = 0; k < n; ++k)
                s += F(i, k) * orth.X(k, j);
            Fm(i, j) = s;
        }
    linalg::CplxMatrix Hp = xt::zeros<std::complex<double>>({m, m});
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j) {
            std::complex<double> s{0.0, 0.0};
            for (int k = 0; k < n; ++k)
                s += std::conj(orth.X(k, i)) * Fm(k, j);
            Hp(i, j) = s;
        }
    std::vector<double> ev;
    linalg::CplxMatrix Cp;
    linalg::eigh_hermitian(Hp, ev, Cp);
    eps.assign(static_cast<std::size_t>(n), 1.0e8);
    C = xt::zeros<std::complex<double>>({n, n});
    for (int j = 0; j < m; ++j) {
        eps[static_cast<std::size_t>(j)] = ev[static_cast<std::size_t>(j)];
        for (int i = 0; i < n; ++i) {
            std::complex<double> s{0.0, 0.0};
            for (int k = 0; k < m; ++k)
                s += orth.X(i, k) * Cp(k, j);
            C(i, j) = s;
        }
    }
}

CMat make_rdm1(const linalg::CplxMatrix& C, const std::vector<double>& occ) {
    const int n = static_cast<int>(occ.size());
    CMat dm = CMat::zeros(n);
    for (int a = 0; a < n; ++a) {
        if (!(occ[static_cast<std::size_t>(a)] > 0.0))
            continue;
        const double o = occ[static_cast<std::size_t>(a)];
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                dm(i, j) += C(i, a) * o * std::conj(C(j, a));
    }
    dm.hermitize();
    return dm;
}

void occupy(const std::vector<std::vector<double>>& eps, int nocc, double fill,
            std::vector<std::vector<double>>& occ) {
    // khf.get_occ / kuhf.get_occ: lowest `nocc` orbitals across k.
    // Restricted fill is 2; unrestricted fill is 1.
    struct Item {
        double e;
        int k;
        int a;
    };
    std::vector<Item> all;
    const int nk = static_cast<int>(eps.size());
    for (int k = 0; k < nk; ++k)
        for (int a = 0; a < static_cast<int>(eps[k].size()); ++a)
            all.push_back(
                {eps[static_cast<std::size_t>(k)][static_cast<std::size_t>(a)],
                 k, a});
    if (nocc < 0 || nocc > static_cast<int>(all.size()))
        throw std::runtime_error("k-point HF: occupied count exceeds the basis "
                                 "(pyscf.pbc.scf.khf.get_occ / kuhf.get_occ)");
    std::stable_sort(all.begin(), all.end(),
                     [](const Item& x, const Item& y) { return x.e < y.e; });
    occ.assign(static_cast<std::size_t>(nk), {});
    for (int k = 0; k < nk; ++k)
        occ[static_cast<std::size_t>(k)].assign(eps[k].size(), 0.0);
    for (int i = 0; i < nocc; ++i) {
        const Item& it = all[static_cast<std::size_t>(i)];
        occ[static_cast<std::size_t>(it.k)][static_cast<std::size_t>(it.a)] =
            fill;
    }
}

} // namespace

std::array<int, 3> fft_mesh_orthorhombic(const double L_bohr[3],
                                         const PgtoSpec* prims, int nprim,
                                         double precision) {
    if (nprim <= 0)
        throw std::runtime_error("fft_mesh_orthorhombic: no primitives");
    double ecut = 0.0;
    for (int i = 0; i < nprim; ++i)
        ecut = std::max(ecut, estimate_ke_one(prims[i].alpha, prims[i].l,
                                              std::abs(prims[i].c_libcint),
                                              precision));
    std::array<int, 3> mesh{1, 1, 1};
    const double gcut = std::sqrt(2.0 * ecut);
    for (int d = 0; d < 3; ++d) {
        if (!(L_bohr[d] > 0.0))
            throw std::runtime_error(
                "fft_mesh_orthorhombic: edge must be positive");
        const double bd = 2.0 * kPi / L_bohr[d];
        const double gmax = gcut / std::abs(bd);
        mesh[static_cast<std::size_t>(d)] =
            static_cast<int>(std::ceil(gmax - 1e-12)) * 2 + 1;
    }
    return mesh;
}

double lattice_rcut(const PgtoSpec* prims, int nprim, double precision) {
    double r = 0.0;
    for (int i = 0; i < nprim; ++i)
        r = std::max(r, estimate_rcut_one(prims[i].alpha, prims[i].l,
                                          std::abs(prims[i].c_libcint),
                                          precision));
    return r;
}

double ewald_nuclear(const double* Z, const double (*R)[3], int natm,
                     const double L_bohr[3], double precision) {
    if (natm <= 0)
        return 0.0;
    double vol = L_bohr[0] * L_bohr[1] * L_bohr[2];
    if (!(vol > 0.0))
        throw std::runtime_error("ewald_nuclear: cell volume must be positive");
    double zsum = 0.0;
    double z2 = 0.0;
    for (int a = 0; a < natm; ++a) {
        zsum += Z[a];
        z2 += Z[a] * Z[a];
    }
    // pyscf get_ewald_params, dimension 3: eta = vol^{-1/6}.
    const double eta = 1.0 / std::pow(vol, 1.0 / 6.0);
    const double ew_cut = estimate_rcut_one(eta * eta, 0, 1.0, precision);
    const auto images = image_vectors(L_bohr, ew_cut);

    double ewovrl = 0.0;
    for (const auto& T : images) {
        for (int i = 0; i < natm; ++i) {
            for (int j = 0; j < natm; ++j) {
                const double dx = R[i][0] - R[j][0] + T[0];
                const double dy = R[i][1] - R[j][1] + T[1];
                const double dz = R[i][2] - R[j][2] + T[2];
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (r < 1e-16)
                    continue;
                ewovrl += Z[i] * Z[j] * std::erfc(eta * r) / r;
            }
        }
    }
    ewovrl *= 0.5;

    double ewself = -0.5 * z2 * 2.0 * eta / std::sqrt(kPi);
    ewself += -0.5 * zsum * zsum * kPi / (eta * eta * vol);

    const double log_precision =
        std::log(precision / (zsum * 16.0 * kPi * kPi));
    const double ke = -2.0 * eta * eta * log_precision;
    // Mesh from the same ortho cutoff rule, no basis primitives: Gmax.
    std::array<int, 3> mesh{1, 1, 1};
    const double gcut = std::sqrt(2.0 * ke);
    for (int d = 0; d < 3; ++d) {
        const double bd = 2.0 * kPi / L_bohr[d];
        mesh[static_cast<std::size_t>(d)] =
            static_cast<int>(std::ceil(gcut / std::abs(bd) - 1e-12)) * 2 + 1;
    }
    std::vector<double> coords, gv;
    grid_geometry(L_bohr, mesh.data(), coords, gv);
    const int ng = mesh[0] * mesh[1] * mesh[2];
    double ewg = 0.0;
    for (int g = 0; g < ng; ++g) {
        const double gx = gv[static_cast<std::size_t>(g) * 3];
        const double gy = gv[static_cast<std::size_t>(g) * 3 + 1];
        const double gz = gv[static_cast<std::size_t>(g) * 3 + 2];
        const double g2 = gx * gx + gy * gy + gz * gz;
        if (g2 == 0.0)
            continue;
        std::complex<double> zsi{0.0, 0.0};
        for (int a = 0; a < natm; ++a) {
            const double phase = gx * R[a][0] + gy * R[a][1] + gz * R[a][2];
            zsi +=
                Z[a] * std::complex<double>(std::cos(-phase), std::sin(-phase));
        }
        const double coul = 4.0 * kPi / g2 / vol;
        ewg += std::norm(zsi) * std::exp(-g2 / (4.0 * eta * eta)) * coul;
    }
    ewg *= 0.5;
    return ewovrl + ewself + ewg;
}

void monkhorst_kpts(const int nmesh[3], const double L_bohr[3],
                    std::vector<std::array<double, 3>>& k_out) {
    for (int d = 0; d < 3; ++d)
        if (nmesh[d] < 1)
            throw std::runtime_error(
                "monkhorst_kpts: each mesh count must be >= 1");
    const double b[3] = {2.0 * kPi / L_bohr[0], 2.0 * kPi / L_bohr[1],
                         2.0 * kPi / L_bohr[2]};
    k_out.clear();
    k_out.reserve(static_cast<std::size_t>(nmesh[0] * nmesh[1] * nmesh[2]));
    // with_gamma, wrap_around false: s = i/n, k = s * b (row of reciprocal).
    for (int ix = 0; ix < nmesh[0]; ++ix)
        for (int iy = 0; iy < nmesh[1]; ++iy)
            for (int iz = 0; iz < nmesh[2]; ++iz) {
                const double sx = static_cast<double>(ix) / nmesh[0];
                const double sy = static_cast<double>(iy) / nmesh[1];
                const double sz = static_cast<double>(iz) / nmesh[2];
                k_out.push_back({sx * b[0], sy * b[1], sz * b[2]});
            }
}

void fft3(std::vector<std::complex<double>>& data, int nx, int ny, int nz,
          bool inverse) {
    const int ng = nx * ny * nz;
    if (static_cast<int>(data.size()) < ng)
        throw std::runtime_error("fft3: buffer shorter than the mesh");
    // z contiguous.
    for (int ix = 0; ix < nx; ++ix)
        for (int iy = 0; iy < ny; ++iy)
            dft_1d(&data[static_cast<std::size_t>(nz * (iy + ny * ix))], nz, 1,
                   inverse);
    for (int ix = 0; ix < nx; ++ix)
        for (int iz = 0; iz < nz; ++iz)
            dft_1d(&data[static_cast<std::size_t>(iz + nz * ny * ix)], ny, nz,
                   inverse);
    for (int iy = 0; iy < ny; ++iy)
        for (int iz = 0; iz < nz; ++iz)
            dft_1d(&data[static_cast<std::size_t>(iz + nz * iy)], nx, ny * nz,
                   inverse);
}

namespace {

std::vector<CMat>
build_j(const std::vector<std::vector<std::complex<double>>>& ao,
        const std::vector<CMat>& dm, const std::vector<double>& coul,
        const int mesh[3], double vol) {
    const int nk = static_cast<int>(dm.size());
    const int nao = dm[0].n;
    const int ng = mesh[0] * mesh[1] * mesh[2];
    std::vector<std::complex<double>> rho(static_cast<std::size_t>(ng),
                                          {0.0, 0.0});
    for (int k = 0; k < nk; ++k) {
        const auto& aok = ao[static_cast<std::size_t>(k)];
        for (int g = 0; g < ng; ++g) {
            std::complex<double> s{0.0, 0.0};
            for (int nu = 0; nu < nao; ++nu) {
                std::complex<double> aod{0.0, 0.0};
                for (int mu = 0; mu < nao; ++mu)
                    aod += aok[static_cast<std::size_t>(g * nao + mu)]
                           * dm[static_cast<std::size_t>(k)](mu, nu);
                s += aod
                     * std::conj(aok[static_cast<std::size_t>(g * nao + nu)]);
            }
            rho[static_cast<std::size_t>(g)] += s;
        }
    }
    const double inv_nk = 1.0 / static_cast<double>(nk);
    for (int g = 0; g < ng; ++g)
        rho[static_cast<std::size_t>(g)] =
            rho[static_cast<std::size_t>(g)].real() * inv_nk;
    fft3(rho, mesh[0], mesh[1], mesh[2], false);
    for (int g = 0; g < ng; ++g)
        rho[static_cast<std::size_t>(g)] *= coul[static_cast<std::size_t>(g)];
    fft3(rho, mesh[0], mesh[1], mesh[2], true);
    const double w = vol / static_cast<double>(ng);
    std::vector<CMat> vj(static_cast<std::size_t>(nk), CMat::zeros(nao));
    for (int k = 0; k < nk; ++k) {
        const auto& aok = ao[static_cast<std::size_t>(k)];
        for (int g = 0; g < ng; ++g) {
            const double vr = rho[static_cast<std::size_t>(g)].real() * w;
            for (int mu = 0; mu < nao; ++mu) {
                const std::complex<double> bra =
                    std::conj(aok[static_cast<std::size_t>(g * nao + mu)]) * vr;
                for (int nu = 0; nu < nao; ++nu)
                    vj[static_cast<std::size_t>(k)](mu, nu) +=
                        bra * aok[static_cast<std::size_t>(g * nao + nu)];
            }
        }
        vj[static_cast<std::size_t>(k)].hermitize();
    }
    return vj;
}

std::vector<CMat>
build_k(const std::vector<std::vector<std::complex<double>>>& ao,
        const std::vector<CMat>& dm, const std::vector<double>& coords,
        const std::vector<std::array<double, 3>>& kpts,
        const std::vector<double>& coul_k, const int mesh[3], double vol) {
    // fft_jk.get_k_kpts, exxdiv='ewald', kpts_band = kpts.
    const int nk = static_cast<int>(dm.size());
    const int nao = dm[0].n;
    const int ng = mesh[0] * mesh[1] * mesh[2];
    const double weight =
        (1.0 / static_cast<double>(nk)) * (vol / static_cast<double>(ng));
    std::vector<CMat> vk(static_cast<std::size_t>(nk), CMat::zeros(nao));
    std::vector<std::complex<double>> rho1(
        static_cast<std::size_t>(nao * nao * ng));
    std::vector<std::complex<double>> vrdm(static_cast<std::size_t>(nao * ng));
    for (int k2 = 0; k2 < nk; ++k2) {
        const auto& ao2 = ao[static_cast<std::size_t>(k2)];
        std::vector<std::complex<double>> aodm(
            static_cast<std::size_t>(nao * ng), {0.0, 0.0});
        for (int g = 0; g < ng; ++g) {
            for (int mu = 0; mu < nao; ++mu) {
                std::complex<double> s{0.0, 0.0};
                for (int nu = 0; nu < nao; ++nu)
                    s += dm[static_cast<std::size_t>(k2)](mu, nu)
                         * std::conj(
                             ao2[static_cast<std::size_t>(g * nao + nu)]);
                aodm[static_cast<std::size_t>(mu * ng + g)] = s;
            }
        }
        for (int k1 = 0; k1 < nk; ++k1) {
            const auto& ao1 = ao[static_cast<std::size_t>(k1)];
            const double kd[3] = {kpts[k2][0] - kpts[k1][0],
                                  kpts[k2][1] - kpts[k1][1],
                                  kpts[k2][2] - kpts[k1][2]};
            const bool zero =
                std::abs(kd[0]) + std::abs(kd[1]) + std::abs(kd[2]) < 1e-9;
            const double* coul =
                coul_k.data() + static_cast<std::size_t>((k2 * nk + k1) * ng);
            for (int i = 0; i < nao; ++i) {
                for (int j = 0; j < nao; ++j) {
                    for (int g = 0; g < ng; ++g) {
                        std::complex<double> phase{1.0, 0.0};
                        if (!zero) {
                            const double dot = coords[g * 3] * kd[0]
                                               + coords[g * 3 + 1] * kd[1]
                                               + coords[g * 3 + 2] * kd[2];
                            phase = std::complex<double>(std::cos(-dot),
                                                         std::sin(-dot));
                        }
                        rho1[static_cast<std::size_t>((i * nao + j) * ng + g)] =
                            std::conj(
                                ao1[static_cast<std::size_t>(g * nao + i)])
                            * phase
                            * ao2[static_cast<std::size_t>(g * nao + j)];
                    }
                }
            }
            for (int ij = 0; ij < nao * nao; ++ij) {
                std::vector<std::complex<double>> slice(
                    rho1.begin() + static_cast<std::ptrdiff_t>(ij * ng),
                    rho1.begin() + static_cast<std::ptrdiff_t>((ij + 1) * ng));
                fft3(slice, mesh[0], mesh[1], mesh[2], false);
                for (int g = 0; g < ng; ++g)
                    slice[static_cast<std::size_t>(g)] *=
                        coul[static_cast<std::size_t>(g)];
                fft3(slice, mesh[0], mesh[1], mesh[2], true);
                std::copy(slice.begin(), slice.end(),
                          rho1.begin() + static_cast<std::ptrdiff_t>(ij * ng));
            }
            std::fill(vrdm.begin(), vrdm.end(), std::complex<double>{0.0, 0.0});
            for (int i = 0; i < nao; ++i) {
                for (int g = 0; g < ng; ++g) {
                    std::complex<double> s{0.0, 0.0};
                    for (int j = 0; j < nao; ++j)
                        s += rho1[static_cast<std::size_t>((i * nao + j) * ng
                                                           + g)]
                             * aodm[static_cast<std::size_t>(j * ng + g)];
                    if (!zero) {
                        const double dot = coords[g * 3] * kd[0]
                                           + coords[g * 3 + 1] * kd[1]
                                           + coords[g * 3 + 2] * kd[2];
                        s *= std::complex<double>(std::cos(dot), std::sin(dot));
                    }
                    vrdm[static_cast<std::size_t>(i * ng + g)] = s;
                }
            }
            for (int i = 0; i < nao; ++i)
                for (int p = 0; p < nao; ++p) {
                    std::complex<double> s{0.0, 0.0};
                    for (int g = 0; g < ng; ++g)
                        s += vrdm[static_cast<std::size_t>(i * ng + g)]
                             * ao1[static_cast<std::size_t>(g * nao + p)];
                    vk[static_cast<std::size_t>(k1)](i, p) += weight * s;
                }
        }
    }
    for (int k = 0; k < nk; ++k)
        vk[static_cast<std::size_t>(k)].hermitize();
    return vk;
}

double energy_elec(const std::vector<CMat>& dm, const std::vector<CMat>& H,
                   const std::vector<CMat>& vhf) {
    const int nk = static_cast<int>(dm.size());
    const int nao = dm[0].n;
    std::complex<double> e1{0.0, 0.0};
    std::complex<double> e2{0.0, 0.0};
    for (int k = 0; k < nk; ++k)
        for (int i = 0; i < nao; ++i)
            for (int j = 0; j < nao; ++j) {
                // einsum kij,kji -> dm[k](i,j) * H[k](j,i)
                e1 += dm[k](i, j) * H[k](j, i);
                e2 += dm[k](i, j) * vhf[k](j, i);
            }
    return (e1.real() + 0.5 * e2.real()) / static_cast<double>(nk);
}

} // namespace

struct Setup {
    int mesh[3]{};
    int nk = 0, nao = 0, ng = 0, nocc = 0, ne_total = 0;
    double vol = 0, enuc = 0;
    std::vector<std::array<double, 3>> kpts;
    std::vector<double> coords, coul_j, coul_k;
    std::vector<std::vector<std::complex<double>>> ao;
    std::vector<CMat> H;
    std::vector<Orth> orth;
};

Setup setup_krhf(const Frame& frame, BasisSet basis, const int k_mesh[3],
                 int charge, const char* tag = "KRHF", bool restricted = true) {
    if (!frame.box.is_orthorhombic())
        throw std::runtime_error(
            std::string(tag)
            + ": Frame.box must be orthorhombic "
              "(PySCF FFTDF v1; triclinic cells are not supported)");
    for (int d = 0; d < 3; ++d)
        if (k_mesh[d] < 1)
            throw std::runtime_error(std::string(tag)
                                     + ": k_mesh entries must be >= 1");

    const double L[3] = {frame.box.length(0) * units::kAngstromInBohr,
                         frame.box.length(1) * units::kAngstromInBohr,
                         frame.box.length(2) * units::kAngstromInBohr};
    const double vol = L[0] * L[1] * L[2];

    AtomList atoms(frame, charge);
    ShellList shells(atoms, basis);
    for (const auto& sh : shells.shells)
        if (sh.l != 0)
            throw std::runtime_error(
                std::string(tag)
                + ": l>0 Gaussian collocation on the FFT mesh is not "
                  "implemented (missing kernel krhf_ao_lgt0)");
    const int nao = static_cast<int>(shells.shells.size());
    if (nao < 1)
        throw std::runtime_error(std::string(tag)
                                 + ": basis produced no functions");

    std::vector<PgtoSpec> prims;
    for (const auto& sh : shells.shells) {
        for (int p = 0; p < sh.n_prim; ++p) {
            // l = 0: libcint coeff = 3D prefactor * sqrt(4π).
            PgtoSpec spec;
            spec.l = 0;
            spec.alpha = sh.alpha[p];
            spec.c_libcint = std::abs(sh.coeff[p]) * std::sqrt(4.0 * kPi);
            prims.push_back(spec);
        }
    }
    const auto mesh = fft_mesh_orthorhombic(
        L, prims.data(), static_cast<int>(prims.size()), kPrecision);
    const double rcut =
        lattice_rcut(prims.data(), static_cast<int>(prims.size()), kPrecision);
    const auto images = image_vectors(L, rcut);

    std::vector<std::array<double, 3>> kpts;
    monkhorst_kpts(k_mesh, L, kpts);
    const int nk = static_cast<int>(kpts.size());

    // PySCF tot_electrons(nk) = (sum Z)*nk - charge.
    int zsum = 0;
    for (const auto& at : atoms.atoms)
        zsum += at.Z;
    const int ne_total = zsum * nk - charge;
    if (ne_total <= 0 || (restricted && ne_total % 2 != 0))
        throw std::runtime_error(
            std::string(tag)
            + (restricted
                   ? ": electron count across the k-mesh must be positive and "
                     "even "
                     "(pyscf.pbc.gto.cell.tot_electrons)"
                   : ": electron count across the k-mesh must be positive "
                     "(pyscf.pbc.gto.cell.tot_electrons)"));
    const int nocc = restricted ? ne_total / 2 : 0;

    std::vector<double> coords, gv;
    grid_geometry(L, mesh.data(), coords, gv);
    const int ng = mesh[0] * mesh[1] * mesh[2];

    // Probe-charge Madelung on the Monkhorst–Pack supercell (tools.madelung).
    const double Ls[3] = {L[0] * k_mesh[0], L[1] * k_mesh[1], L[2] * k_mesh[2]};
    const double zprobe = 1.0;
    const double rprobe[1][3] = {{0.0, 0.0, 0.0}};
    const double madelung =
        -2.0 * ewald_nuclear(&zprobe, rprobe, 1, Ls, kPrecision);

    const double k0[3] = {0.0, 0.0, 0.0};
    const std::vector<double> coul_j =
        coulomb_kernel(L, mesh.data(), gv.data(), k0, false, nk, madelung, vol);

    std::vector<double> coul_k(static_cast<std::size_t>(nk * nk * ng));
    for (int k2 = 0; k2 < nk; ++k2) {
        for (int k1 = 0; k1 < nk; ++k1) {
            const double kd[3] = {kpts[k2][0] - kpts[k1][0],
                                  kpts[k2][1] - kpts[k1][1],
                                  kpts[k2][2] - kpts[k1][2]};
            auto ck = coulomb_kernel(L, mesh.data(), gv.data(), kd, true, nk,
                                     madelung, vol);
            std::copy(ck.begin(), ck.end(),
                      coul_k.begin()
                          + static_cast<std::ptrdiff_t>((k2 * nk + k1) * ng));
        }
    }

    std::vector<double> Z(static_cast<std::size_t>(atoms.atoms.size()));
    std::vector<std::array<double, 3>> Rarr(atoms.atoms.size());
    for (int a = 0; a < static_cast<int>(atoms.atoms.size()); ++a) {
        Z[static_cast<std::size_t>(a)] = static_cast<double>(atoms.atoms[a].Z);
        Rarr[static_cast<std::size_t>(a)] = {atoms.atoms[a].pos[0],
                                             atoms.atoms[a].pos[1],
                                             atoms.atoms[a].pos[2]};
    }
    const double enuc = ewald_nuclear(
        Z.data(), reinterpret_cast<const double (*)[3]>(Rarr.data()),
        static_cast<int>(atoms.atoms.size()), L, kPrecision);

    // Nuclear attraction, fft.get_nuc: charge = -Z, no quadrature weight.
    std::vector<std::complex<double>> vnuc_g(static_cast<std::size_t>(ng),
                                             {0.0, 0.0});
    for (int g = 0; g < ng; ++g) {
        const double gx = gv[g * 3], gy = gv[g * 3 + 1], gz = gv[g * 3 + 2];
        std::complex<double> si_dot{0.0, 0.0};
        for (int a = 0; a < static_cast<int>(atoms.atoms.size()); ++a) {
            const double phase = gx * atoms.atoms[a].pos[0]
                                 + gy * atoms.atoms[a].pos[1]
                                 + gz * atoms.atoms[a].pos[2];
            const std::complex<double> si(std::cos(-phase), std::sin(-phase));
            si_dot += (-static_cast<double>(atoms.atoms[a].Z)) * si;
        }
        vnuc_g[static_cast<std::size_t>(g)] =
            si_dot * coul_j[static_cast<std::size_t>(g)];
    }
    fft3(vnuc_g, mesh[0], mesh[1], mesh[2], true);

    std::vector<std::vector<std::complex<double>>> ao(
        static_cast<std::size_t>(nk));
    std::vector<CMat> S(static_cast<std::size_t>(nk));
    std::vector<CMat> H(static_cast<std::size_t>(nk));
    std::vector<Orth> orth(static_cast<std::size_t>(nk));
    for (int k = 0; k < nk; ++k) {
        eval_ao_s(shells.shells, images, coords, ng, kpts[k].data(),
                  ao[static_cast<std::size_t>(k)]);
        const CMat Tm = bloch_1e(shells.shells, images, kpts[k].data(), true);
        S[static_cast<std::size_t>(k)] =
            bloch_1e(shells.shells, images, kpts[k].data(), false);
        CMat V = CMat::zeros(nao);
        const auto& aok = ao[static_cast<std::size_t>(k)];
        for (int g = 0; g < ng; ++g) {
            const double vr = vnuc_g[static_cast<std::size_t>(g)].real();
            for (int mu = 0; mu < nao; ++mu) {
                const std::complex<double> bra =
                    std::conj(aok[static_cast<std::size_t>(g * nao + mu)]) * vr;
                for (int nu = 0; nu < nao; ++nu)
                    V(mu, nu) +=
                        bra * aok[static_cast<std::size_t>(g * nao + nu)];
            }
        }
        V.hermitize();
        H[static_cast<std::size_t>(k)] = CMat::zeros(nao);
        for (int i = 0; i < nao; ++i)
            for (int j = 0; j < nao; ++j)
                H[k](i, j) = Tm(i, j) + V(i, j);
        H[k].hermitize();
        orth[static_cast<std::size_t>(k)] =
            canonical_x(S[static_cast<std::size_t>(k)]);
    }

    Setup out;
    out.mesh[0] = mesh[0];
    out.mesh[1] = mesh[1];
    out.mesh[2] = mesh[2];
    out.nk = nk;
    out.nao = nao;
    out.ng = ng;
    out.nocc = nocc;
    out.ne_total = ne_total;
    out.vol = vol;
    out.enuc = enuc;
    out.kpts = std::move(kpts);
    out.coords = std::move(coords);
    out.coul_j = std::move(coul_j);
    out.coul_k = std::move(coul_k);
    out.ao = std::move(ao);
    out.H = std::move(H);
    out.orth = std::move(orth);
    return out;
}

KrhfStatic krhf_assemble(const Frame& frame, BasisSet basis,
                         const int k_mesh[3], int charge) {
    const Setup s = setup_krhf(frame, basis, k_mesh, charge);
    KrhfStatic o;
    o.mesh[0] = s.mesh[0];
    o.mesh[1] = s.mesh[1];
    o.mesh[2] = s.mesh[2];
    o.nk = s.nk;
    o.nao = s.nao;
    o.ng = s.ng;
    o.nocc = s.nocc;
    o.vol = s.vol;
    o.enuc = s.enuc;
    o.kpts = s.kpts;
    o.coords = s.coords;
    o.coul_j = s.coul_j;
    o.coul_k = s.coul_k;
    o.nmo = std::vector<int>(static_cast<std::size_t>(s.nk));
    o.ao.resize(static_cast<std::size_t>(s.nk * s.ng * s.nao));
    o.H.resize(static_cast<std::size_t>(s.nk * s.nao * s.nao));
    o.X.resize(static_cast<std::size_t>(s.nk * s.nao * s.nao));
    for (int k = 0; k < s.nk; ++k) {
        o.nmo[static_cast<std::size_t>(k)] =
            s.orth[static_cast<std::size_t>(k)].nmo;
        const auto& aok = s.ao[static_cast<std::size_t>(k)];
        for (int i = 0; i < static_cast<int>(aok.size()); ++i)
            o.ao[static_cast<std::size_t>((k * s.ng) * s.nao + i)] =
                aok[static_cast<std::size_t>(i)];
        for (int i = 0; i < s.nao; ++i)
            for (int j = 0; j < s.nao; ++j)
                o.H[static_cast<std::size_t>((k * s.nao + i) * s.nao + j)] =
                    s.H[static_cast<std::size_t>(k)](i, j);
        const int m = s.orth[static_cast<std::size_t>(k)].nmo;
        for (int i = 0; i < s.nao; ++i)
            for (int j = 0; j < m; ++j)
                o.X[static_cast<std::size_t>((k * s.nao + i) * s.nao + j)] =
                    s.orth[static_cast<std::size_t>(k)].X(i, j);
    }
    return o;
}

void eigh_all(const Setup& s, const std::vector<CMat>& f,
              std::vector<std::vector<double>>& eps,
              std::vector<linalg::CplxMatrix>& cmo) {
    const int nk = s.nk;
    eps.resize(static_cast<std::size_t>(nk));
    cmo.resize(static_cast<std::size_t>(nk));
    for (int k = 0; k < nk; ++k)
        eigh_fock(
            f[static_cast<std::size_t>(k)], s.orth[static_cast<std::size_t>(k)],
            eps[static_cast<std::size_t>(k)], cmo[static_cast<std::size_t>(k)]);
}

void dm_from_eps(const std::vector<std::vector<double>>& eps,
                 const std::vector<linalg::CplxMatrix>& cmo, int nocc,
                 double fill, std::vector<CMat>& dm) {
    std::vector<std::vector<double>> occ;
    occupy(eps, nocc, fill, occ);
    const int nk = static_cast<int>(eps.size());
    dm.resize(static_cast<std::size_t>(nk));
    for (int k = 0; k < nk; ++k)
        dm[static_cast<std::size_t>(k)] = make_rdm1(
            cmo[static_cast<std::size_t>(k)], occ[static_cast<std::size_t>(k)]);
}

struct KrhfCycle {
    Setup setup;
    std::vector<CMat> dm;
    std::vector<CMat> fock;
    std::vector<std::vector<double>> eps;
    std::vector<linalg::CplxMatrix> cmo;
    double energy = 0.0;
    double previous = 0.0;
    bool have_previous = false;
};

void krhf_cycle_delete(KrhfCycle* cycle) { delete cycle; }

KrhfCycleOwner krhf_cycle_open(const Frame& frame, BasisSet basis,
                               const int k_mesh[3], int charge) {
    KrhfCycleOwner own(new KrhfCycle, &krhf_cycle_delete);
    own->setup = setup_krhf(frame, basis, k_mesh, charge);
    // init_guess_by_1e: occupy the core Hamiltonian (hf.kernel dm0).
    own->fock = own->setup.H;
    krhf_cycle_diagonalize(*own);
    krhf_cycle_density(*own);
    return own;
}

void krhf_cycle_fock(KrhfCycle& c) {
    // KSCF.get_fock with diis=None: F = H + J[dm] − ½ K[dm].
    // energy_elec is the symmetric trace of that density, not the next one.
    const Setup& s = c.setup;
    const int nk = s.nk;
    const int nao = s.nao;
    const auto vj = build_j(s.ao, c.dm, s.coul_j, s.mesh, s.vol);
    const auto vk =
        build_k(s.ao, c.dm, s.coords, s.kpts, s.coul_k, s.mesh, s.vol);
    std::vector<CMat> vhf(static_cast<std::size_t>(nk), CMat::zeros(nao));
    c.fock.assign(static_cast<std::size_t>(nk), CMat::zeros(nao));
    for (int k = 0; k < nk; ++k) {
        for (int i = 0; i < nao; ++i)
            for (int j = 0; j < nao; ++j)
                vhf[static_cast<std::size_t>(k)](i, j) =
                    vj[static_cast<std::size_t>(k)](i, j)
                    - 0.5 * vk[static_cast<std::size_t>(k)](i, j);
        vhf[static_cast<std::size_t>(k)].hermitize();
        for (int i = 0; i < nao; ++i)
            for (int j = 0; j < nao; ++j)
                c.fock[static_cast<std::size_t>(k)](i, j) =
                    s.H[static_cast<std::size_t>(k)](i, j)
                    + vhf[static_cast<std::size_t>(k)](i, j);
        c.fock[static_cast<std::size_t>(k)].hermitize();
    }
    c.energy = energy_elec(c.dm, s.H, vhf) + s.enuc;
}

void krhf_cycle_diagonalize(KrhfCycle& c) {
    eigh_all(c.setup, c.fock, c.eps, c.cmo);
}

void krhf_cycle_density(KrhfCycle& c) {
    // khf.get_occ (fill 2) then make_rdm1.
    dm_from_eps(c.eps, c.cmo, c.setup.nocc, 2.0, c.dm);
}

double krhf_cycle_energy(const KrhfCycle& c) { return c.energy; }

double krhf_cycle_delta(KrhfCycle& c) {
    const double delta = c.have_previous
                             ? std::abs(c.energy - c.previous)
                             : std::numeric_limits<double>::infinity();
    c.previous = c.energy;
    c.have_previous = true;
    return delta;
}

struct KuhfCycle {
    Setup setup;
    int n_alpha = 0;
    int n_beta = 0;
    std::vector<CMat> dm[2];
    std::vector<CMat> fock[2];
    std::vector<std::vector<double>> eps[2];
    std::vector<linalg::CplxMatrix> cmo[2];
    double energy = 0.0;
    double previous = 0.0;
    bool have_previous = false;
};

void kuhf_cycle_delete(KuhfCycle* cycle) { delete cycle; }

void kuhf_cycle_diagonalize(KuhfCycle& c) {
    // KUHF.eig → KSCF.eig, one Hermitian diagonalization per spin.
    eigh_all(c.setup, c.fock[0], c.eps[0], c.cmo[0]);
    eigh_all(c.setup, c.fock[1], c.eps[1], c.cmo[1]);
}

void kuhf_cycle_density(KuhfCycle& c) {
    // kuhf.get_occ (fill 1) then make_rdm1, each spin.
    dm_from_eps(c.eps[0], c.cmo[0], c.n_alpha, 1.0, c.dm[0]);
    dm_from_eps(c.eps[1], c.cmo[1], c.n_beta, 1.0, c.dm[1]);
}

KuhfCycleOwner kuhf_cycle_open(const Frame& frame, BasisSet basis,
                               const int k_mesh[3], int n_alpha, int n_beta,
                               int charge) {
    KuhfCycleOwner own(new KuhfCycle, &kuhf_cycle_delete);
    own->setup = setup_krhf(frame, basis, k_mesh, charge, "KUHF", false);
    if (n_alpha < 0 || n_beta < 0 || n_alpha + n_beta != own->setup.ne_total)
        throw std::runtime_error(
            "KUHF: n_alpha + n_beta must equal tot_electrons(nk) = "
            "(sum Z)*nk - charge (pyscf.pbc.scf.kuhf.KUHF.nelec). "
            "cell.spin = n_alpha - n_beta is the mesh total and is not "
            "multiplied by nk; got n_alpha="
            + std::to_string(n_alpha) + " n_beta=" + std::to_string(n_beta)
            + " ne=" + std::to_string(own->setup.ne_total));
    own->n_alpha = n_alpha;
    own->n_beta = n_beta;
    // init_guess_by_1e: occupy the core Hamiltonian (hf.kernel dm0).
    own->fock[0] = own->setup.H;
    own->fock[1] = own->setup.H;
    kuhf_cycle_diagonalize(*own);
    kuhf_cycle_density(*own);
    return own;
}

void kuhf_cycle_fock(KuhfCycle& c) {
    // KUHF.get_veff: J = Ja + Jb, vhf = J − K on each spin.
    // KUHF.energy_elec sums both spins (1/nk, half the vhf trace).
    // The trace is this density, not the one the following eig builds.
    const Setup& s = c.setup;
    const int nk = s.nk;
    const int nao = s.nao;
    const int* mesh = s.mesh;
    std::vector<CMat> dm_j(static_cast<std::size_t>(nk));
    for (int k = 0; k < nk; ++k) {
        dm_j[static_cast<std::size_t>(k)] = CMat::zeros(nao);
        for (int i = 0; i < nao; ++i)
            for (int j = 0; j < nao; ++j)
                dm_j[static_cast<std::size_t>(k)](i, j) =
                    c.dm[0][static_cast<std::size_t>(k)](i, j)
                    + c.dm[1][static_cast<std::size_t>(k)](i, j);
    }
    const auto vj = build_j(s.ao, dm_j, s.coul_j, mesh, s.vol);
    double e_elec = 0.0;
    for (int sp = 0; sp < 2; ++sp) {
        const auto vk =
            build_k(s.ao, c.dm[sp], s.coords, s.kpts, s.coul_k, mesh, s.vol);
        std::vector<CMat> vhf(static_cast<std::size_t>(nk), CMat::zeros(nao));
        c.fock[sp].assign(static_cast<std::size_t>(nk), CMat::zeros(nao));
        for (int k = 0; k < nk; ++k) {
            for (int i = 0; i < nao; ++i)
                for (int j = 0; j < nao; ++j)
                    vhf[static_cast<std::size_t>(k)](i, j) =
                        vj[static_cast<std::size_t>(k)](i, j)
                        - vk[static_cast<std::size_t>(k)](i, j);
            vhf[static_cast<std::size_t>(k)].hermitize();
            for (int i = 0; i < nao; ++i)
                for (int j = 0; j < nao; ++j)
                    c.fock[sp][static_cast<std::size_t>(k)](i, j) =
                        s.H[static_cast<std::size_t>(k)](i, j)
                        + vhf[static_cast<std::size_t>(k)](i, j);
            c.fock[sp][static_cast<std::size_t>(k)].hermitize();
        }
        e_elec += energy_elec(c.dm[sp], s.H, vhf);
    }
    c.energy = e_elec + s.enuc;
}

double kuhf_cycle_energy(const KuhfCycle& c) { return c.energy; }

double kuhf_cycle_delta(KuhfCycle& c) {
    const double delta = c.have_previous
                             ? std::abs(c.energy - c.previous)
                             : std::numeric_limits<double>::infinity();
    c.previous = c.energy;
    c.have_previous = true;
    return delta;
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
