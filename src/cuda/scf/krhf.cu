/// @file cuda/scf/krhf.cu
/// @brief Device SCF step for cuda::scf::KRHF.
///
/// Setup (mesh, AO, Hcore, canonical X, Coulomb kernels) is
/// `cpu::scf::internal::krhf_assemble` — geometry and basis, not the
/// density. The iteration builds J (`fft_jk.get_j_kpts`), K
/// (`get_k_kpts`), the Fock, `zheevd`, occupation and the density on
/// device. Orbital coefficients stay on device.

#include "atomiverse/cuda/scf/krhf.cuh"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/core/verdict.h"
#include "atomiverse/cpu/scf/internal/kpoint_hf.h"
#include "atomiverse/cuda/frame/precision.cuh"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ATV {
namespace cuda {
namespace scf {

namespace {

struct Z {
    double r;
    double i;
};

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("cuda::scf::KRHF: ") + what + ": "
                                 + cudaGetErrorString(e));
}

void check_sol(cusolverStatus_t s, const char* what) {
    if (s != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuda::scf::KRHF: ") + what
                                 + " cusolver status="
                                 + std::to_string(int(s)));
}

__global__ void k_dft_line(const Z* in, Z* out, int n, int nlines, int kind,
                           int nx, int ny, int nz, int inverse) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nlines * n)
        return;
    const int line = idx / n;
    const int k = idx - line * n;
    int base = 0;
    int stride = 1;
    if (kind == 0) {
        base = line * nz;
        stride = 1;
    } else if (kind == 1) {
        const int ix = line / nz;
        const int iz = line - ix * nz;
        base = iz + nz * ny * ix;
        stride = nz;
    } else {
        const int iy = line / nz;
        const int iz = line - iy * nz;
        base = iz + nz * iy;
        stride = ny * nz;
    }
    const double sign = inverse ? 1.0 : -1.0;
    const double inv = inverse ? 1.0 / static_cast<double>(n) : 1.0;
    double sr = 0.0, si = 0.0;
    for (int j = 0; j < n; ++j) {
        const Z v = in[base + j * stride];
        const double ang = sign * 2.0 * 3.14159265358979323846
                           * static_cast<double>(k * j)
                           / static_cast<double>(n);
        const double c = cos(ang);
        const double s = sin(ang);
        sr += v.r * c - v.i * s;
        si += v.r * s + v.i * c;
    }
    out[base + k * stride] = Z{sr * inv, si * inv};
}

void fft3_dev(Z* data, Z* tmp, int nx, int ny, int nz, bool inverse,
              cudaStream_t stream) {
    auto launch = [&](int kind, int n, int nlines) {
        const int ntot = nlines * n;
        const int blk = 128;
        const int grid = (ntot + blk - 1) / blk;
        k_dft_line<<<grid, blk, 0, stream>>>(data, tmp, n, nlines, kind, nx, ny,
                                             nz, inverse ? 1 : 0);
        check_cuda(cudaMemcpyAsync(data, tmp, sizeof(Z) * nx * ny * nz,
                                   cudaMemcpyDeviceToDevice, stream),
                   "fft copy");
    };
    launch(0, nz, nx * ny);
    launch(1, ny, nx * nz);
    launch(2, nx, ny * nz);
}

__global__ void k_rho(const Z* ao, const Z* dm, Z* rho, int nk, int nao,
                      int ng) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ng)
        return;
    double rr = 0.0;
    for (int k = 0; k < nk; ++k) {
        const Z* a = ao + (static_cast<long long>(k) * ng + g) * nao;
        const Z* d = dm + static_cast<long long>(k) * nao * nao;
        for (int nu = 0; nu < nao; ++nu) {
            double ar = 0.0, ai = 0.0;
            for (int mu = 0; mu < nao; ++mu) {
                const Z am = a[mu];
                const Z dj = d[mu * nao + nu];
                ar += am.r * dj.r - am.i * dj.i;
                ai += am.r * dj.i + am.i * dj.r;
            }
            const Z an = a[nu];
            rr += ar * an.r + ai * an.i;
        }
    }
    rho[g] = Z{rr / static_cast<double>(nk), 0.0};
}

__global__ void k_scale(Z* v, const double* coul, int ng) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ng)
        return;
    v[g].r *= coul[g];
    v[g].i *= coul[g];
}

__global__ void k_j_contract(const Z* ao, const Z* vr, Z* vj, int nk, int nao,
                             int ng, double weight) {
    const int job = blockIdx.x * blockDim.x + threadIdx.x;
    const int njob = nk * nao * nao;
    if (job >= njob)
        return;
    const int nu = job % nao;
    const int mu = (job / nao) % nao;
    const int k = job / (nao * nao);
    double sr = 0.0, si = 0.0;
    for (int g = 0; g < ng; ++g) {
        const Z a_mu = ao[(static_cast<long long>(k) * ng + g) * nao + mu];
        const Z a_nu = ao[(static_cast<long long>(k) * ng + g) * nao + nu];
        const double w = vr[g].r * weight;
        // conj(a_mu) * w * a_nu
        sr += (a_mu.r * a_nu.r + a_mu.i * a_nu.i) * w;
        si += (a_mu.r * a_nu.i - a_mu.i * a_nu.r) * w;
    }
    vj[static_cast<long long>(k) * nao * nao + mu * nao + nu] = Z{sr, si};
}

__global__ void k_hermitize(Z* a, int nk, int nao) {
    const int k = blockIdx.x;
    if (k >= nk)
        return;
    Z* m = a + static_cast<long long>(k) * nao * nao;
    for (int i = threadIdx.x; i < nao; i += blockDim.x) {
        m[i * nao + i].i = 0.0;
        for (int j = i + 1; j < nao; ++j) {
            const Z ij = m[i * nao + j];
            const Z ji = m[j * nao + i];
            const double hr = 0.5 * (ij.r + ji.r);
            const double hi = 0.5 * (ij.i - ji.i);
            m[i * nao + j] = Z{hr, hi};
            m[j * nao + i] = Z{hr, -hi};
        }
    }
}

__global__ void k_vhf(const Z* vj, const Z* vk, Z* v, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    v[i] = Z{vj[i].r - 0.5 * vk[i].r, vj[i].i - 0.5 * vk[i].i};
}

__global__ void k_fock(const Z* H, const Z* vhf, Z* F, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    F[i] = Z{H[i].r + vhf[i].r, H[i].i + vhf[i].i};
}

__global__ void k_fill_d(double* a, int n, double v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        a[i] = v;
}

// khf.get_occ: lowest nocc across k. Stable in (k, orbital) index order.
__global__ void k_occupy(const double* eps, int* idx, double* occ, int n,
                         int nocc) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    for (int i = 0; i < n; ++i)
        idx[i] = i;
    for (int i = 1; i < n; ++i) {
        const int key = idx[i];
        const double e = eps[key];
        int j = i - 1;
        while (j >= 0 && eps[idx[j]] > e) {
            idx[j + 1] = idx[j];
            --j;
        }
        idx[j + 1] = key;
    }
    for (int i = 0; i < n; ++i)
        occ[i] = 0.0;
    const int nfill = nocc < n ? nocc : n;
    for (int i = 0; i < nfill; ++i)
        occ[idx[i]] = 2.0;
}

__global__ void k_trace_energy(const Z* dm, const Z* H, const Z* vhf, int nk,
                               int nao, double* e_out) {
    // One thread: nk and nao are tiny on this path.
    double e1 = 0.0, e2 = 0.0;
    for (int k = 0; k < nk; ++k) {
        const Z* d = dm + static_cast<long long>(k) * nao * nao;
        const Z* h = H + static_cast<long long>(k) * nao * nao;
        const Z* v = vhf + static_cast<long long>(k) * nao * nao;
        for (int i = 0; i < nao; ++i)
            for (int j = 0; j < nao; ++j) {
                const Z dij = d[i * nao + j];
                const Z hij = h[j * nao + i];
                const Z vij = v[j * nao + i];
                e1 += dij.r * hij.r - dij.i * hij.i;
                e2 += dij.r * vij.r - dij.i * vij.i;
            }
    }
    e_out[0] = (e1 + 0.5 * e2) / static_cast<double>(nk);
}

__global__ void k_aodm(const Z* ao, const Z* dm, Z* aodm, int nao, int ng,
                       int k) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ng)
        return;
    const Z* a = ao + (static_cast<long long>(k) * ng + g) * nao;
    const Z* d = dm + static_cast<long long>(k) * nao * nao;
    for (int mu = 0; mu < nao; ++mu) {
        double sr = 0.0, si = 0.0;
        for (int nu = 0; nu < nao; ++nu) {
            const Z dj = d[mu * nao + nu];
            const Z an = a[nu];
            // dm * conj(ao)
            sr += dj.r * an.r + dj.i * an.i;
            si += dj.i * an.r - dj.r * an.i;
        }
        aodm[static_cast<long long>(mu) * ng + g] = Z{sr, si};
    }
}

__global__ void k_rho1(const Z* ao1, const Z* ao2, const double* coords,
                       double kdx, double kdy, double kdz, int zero, Z* rho,
                       int nao, int ng, int k1, int k2) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ng)
        return;
    const Z* a1 = ao1 + (static_cast<long long>(k1) * ng + g) * nao;
    const Z* a2 = ao2 + (static_cast<long long>(k2) * ng + g) * nao;
    double pr = 1.0, pi = 0.0;
    if (!zero) {
        const double dot = coords[g * 3] * kdx + coords[g * 3 + 1] * kdy
                           + coords[g * 3 + 2] * kdz;
        pr = cos(-dot);
        pi = sin(-dot);
    }
    for (int i = 0; i < nao; ++i)
        for (int j = 0; j < nao; ++j) {
            // conj(ao1[i]) * phase * ao2[j]
            const Z u = a1[i];
            const Z v = a2[j];
            const double cr = u.r * pr + u.i * pi;
            const double ci = u.r * pi - u.i * pr;
            const double sr = cr * v.r - ci * v.i;
            const double si = cr * v.i + ci * v.r;
            rho[(static_cast<long long>(i * nao + j) * ng) + g] = Z{sr, si};
        }
}

__global__ void k_vrdm(const Z* rho, const Z* aodm, const double* coords,
                       double kdx, double kdy, double kdz, int zero, Z* vrdm,
                       int nao, int ng) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ng)
        return;
    double pr = 1.0, pi = 0.0;
    if (!zero) {
        const double dot = coords[g * 3] * kdx + coords[g * 3 + 1] * kdy
                           + coords[g * 3 + 2] * kdz;
        pr = cos(dot);
        pi = sin(dot);
    }
    for (int i = 0; i < nao; ++i) {
        double sr = 0.0, si = 0.0;
        for (int j = 0; j < nao; ++j) {
            const Z r = rho[(static_cast<long long>(i * nao + j) * ng) + g];
            const Z a = aodm[static_cast<long long>(j) * ng + g];
            sr += r.r * a.r - r.i * a.i;
            si += r.r * a.i + r.i * a.r;
        }
        const double tr = sr * pr - si * pi;
        const double ti = sr * pi + si * pr;
        vrdm[static_cast<long long>(i) * ng + g] = Z{tr, ti};
    }
}

__global__ void k_k_accum(const Z* vrdm, const Z* ao, Z* vk, int nao, int ng,
                          int k1, double weight) {
    const int job = blockIdx.x * blockDim.x + threadIdx.x;
    if (job >= nao * nao)
        return;
    const int p = job % nao;
    const int i = job / nao;
    double sr = 0.0, si = 0.0;
    for (int g = 0; g < ng; ++g) {
        const Z v = vrdm[static_cast<long long>(i) * ng + g];
        const Z a = ao[(static_cast<long long>(k1) * ng + g) * nao + p];
        sr += v.r * a.r - v.i * a.i;
        si += v.r * a.i + v.i * a.r;
    }
    Z* dst = vk + static_cast<long long>(k1) * nao * nao + i * nao + p;
    dst->r += weight * sr;
    dst->i += weight * si;
}

__global__ void k_zero(Z* a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        a[i] = Z{0.0, 0.0};
}

__global__ void k_xform(const Z* X, const Z* F, Z* Hp, int nao, int nmo,
                        int k) {
    // Hp = X^H F X, one block.
    const Z* x = X + static_cast<long long>(k) * nao * nao;
    const Z* f = F + static_cast<long long>(k) * nao * nao;
    for (int a = threadIdx.x; a < nmo; a += blockDim.x) {
        for (int b = 0; b < nmo; ++b) {
            double sr = 0.0, si = 0.0;
            for (int p = 0; p < nao; ++p) {
                double fr = 0.0, fi = 0.0;
                for (int q = 0; q < nao; ++q) {
                    const Z fq = f[p * nao + q];
                    const Z xq = x[q * nao + b];
                    fr += fq.r * xq.r - fq.i * xq.i;
                    fi += fq.r * xq.i + fq.i * xq.r;
                }
                const Z xp = x[p * nao + a];
                // conj(X[p,a]) * (F X)[p,b]
                sr += xp.r * fr + xp.i * fi;
                si += xp.r * fi - xp.i * fr;
            }
            Hp[a * nmo + b] = Z{sr, si};
        }
    }
}

__global__ void k_row_to_col(const Z* row, cuDoubleComplex* col, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * n)
        return;
    const int r = i / n;
    const int c = i - r * n;
    col[c * n + r] = make_cuDoubleComplex(row[r * n + c].r, row[r * n + c].i);
}

__global__ void k_apply_x(const Z* X, const cuDoubleComplex* Cp, Z* C, int nao,
                          int nmo, int k) {
    // C_ao[:, j] = X * Cp[:, j], Cp column-major.
    const int j = blockIdx.x;
    if (j >= nmo)
        return;
    const Z* x = X + static_cast<long long>(k) * nao * nao;
    Z* c = C + static_cast<long long>(k) * nao * nao;
    for (int i = threadIdx.x; i < nao; i += blockDim.x) {
        double sr = 0.0, si = 0.0;
        for (int a = 0; a < nmo; ++a) {
            const Z xa = x[i * nao + a];
            const cuDoubleComplex cp = Cp[a + j * nmo];
            sr += xa.r * cuCreal(cp) - xa.i * cuCimag(cp);
            si += xa.r * cuCimag(cp) + xa.i * cuCreal(cp);
        }
        c[i * nao + j] = Z{sr, si};
    }
}

__global__ void k_dm_from_c(const Z* C, const double* occ, Z* dm, int nao,
                            int k) {
    const Z* c = C + static_cast<long long>(k) * nao * nao;
    Z* d = dm + static_cast<long long>(k) * nao * nao;
    for (int i = threadIdx.x; i < nao; i += blockDim.x) {
        for (int j = 0; j < nao; ++j) {
            double sr = 0.0, si = 0.0;
            for (int a = 0; a < nao; ++a) {
                if (!(occ[a] > 0.0))
                    continue;
                const Z ci = c[i * nao + a];
                const Z cj = c[j * nao + a];
                sr += occ[a] * (ci.r * cj.r + ci.i * cj.i);
                si += occ[a] * (ci.i * cj.r - ci.r * cj.i);
            }
            d[i * nao + j] = Z{sr, si};
        }
    }
}

int div_up(int n, int b) { return (n + b - 1) / b; }

struct Engine {
    cudaStream_t stream = nullptr;
    cusolverDnHandle_t sol = nullptr;
    int nk = 0;
    int nao = 0;
    int ng = 0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int nocc = 0;
    int lwork = 0;
    double enuc = 0.0;
    double jw = 0.0;
    double kw = 0.0;
    double e_ha = 0.0;
    double previous = 0.0;
    bool have_previous = false;
    std::vector<int> nmo;
    std::vector<std::array<double, 3>> kpts;

    Z* d_ao = nullptr;
    Z* d_H = nullptr;
    Z* d_X = nullptr;
    Z* d_dm = nullptr;
    Z* d_vj = nullptr;
    Z* d_vk = nullptr;
    Z* d_F = nullptr;
    Z* d_vhf = nullptr;
    Z* d_rho = nullptr;
    Z* d_tmp = nullptr;
    Z* d_rho1 = nullptr;
    Z* d_aodm = nullptr;
    Z* d_vrdm = nullptr;
    Z* d_Hp = nullptr;
    Z* d_C = nullptr;
    double* d_coul_j = nullptr;
    double* d_coul_k = nullptr;
    double* d_coords = nullptr;
    double* d_e = nullptr;
    double* d_eig = nullptr;
    double* d_occ = nullptr;
    double* d_eps = nullptr;
    cuDoubleComplex* d_col = nullptr;
    cuDoubleComplex* d_work = nullptr;
    int* d_idx = nullptr;
    int* d_infos = nullptr;

    ~Engine() { release(); }

    void release() {
        if (stream)
            cudaStreamSynchronize(stream);
        if (sol) {
            cusolverDnDestroy(sol);
            sol = nullptr;
        }
        auto freep = [](auto*& p) {
            if (p) {
                cudaFree(p);
                p = nullptr;
            }
        };
        freep(d_ao);
        freep(d_H);
        freep(d_X);
        freep(d_dm);
        freep(d_vj);
        freep(d_vk);
        freep(d_F);
        freep(d_vhf);
        freep(d_rho);
        freep(d_tmp);
        freep(d_rho1);
        freep(d_aodm);
        freep(d_vrdm);
        freep(d_Hp);
        freep(d_C);
        freep(d_coul_j);
        freep(d_coul_k);
        freep(d_coords);
        freep(d_e);
        freep(d_eig);
        freep(d_occ);
        freep(d_eps);
        freep(d_col);
        freep(d_work);
        freep(d_idx);
        freep(d_infos);
        stream = nullptr;
    }

    void open(const cpu::Frame& frame, cpu::scf::BasisSet basis,
              const std::array<int, 3>& k_mesh, int charge, cudaStream_t s) {
        if (!s)
            throw std::runtime_error("cuda::scf::KRHF: no CUDA stream");
        release();
        stream = s;
        const int mesh_k[3] = {k_mesh[0], k_mesh[1], k_mesh[2]};
        const cpu::scf::internal::KrhfStatic sys =
            cpu::scf::internal::krhf_assemble(frame, basis, mesh_k, charge);
        nk = sys.nk;
        nao = sys.nao;
        ng = sys.ng;
        nx = sys.mesh[0];
        ny = sys.mesh[1];
        nz = sys.mesh[2];
        nocc = sys.nocc;
        enuc = sys.enuc;
        nmo = sys.nmo;
        kpts = sys.kpts;
        if (nk < 1 || nao < 1 || ng != nx * ny * nz)
            throw std::runtime_error("cuda::scf::KRHF: empty k-point system");
        if (nocc < 0
            || static_cast<long long>(nocc) > static_cast<long long>(nk) * nao)
            throw std::runtime_error(
                "cuda::scf::KRHF: occupied count exceeds the basis "
                "(pyscf.pbc.scf.khf.get_occ)");

        auto hz = [](const std::complex<double>& z) {
            return Z{z.real(), z.imag()};
        };
        std::vector<Z> h_ao(sys.ao.size()), h_H(sys.H.size()),
            h_X(sys.X.size());
        for (std::size_t i = 0; i < sys.ao.size(); ++i)
            h_ao[i] = hz(sys.ao[i]);
        for (std::size_t i = 0; i < sys.H.size(); ++i)
            h_H[i] = hz(sys.H[i]);
        for (std::size_t i = 0; i < sys.X.size(); ++i)
            h_X[i] = hz(sys.X[i]);

        const std::size_t nmat = static_cast<std::size_t>(nk) * nao * nao;
        try {
            check_cuda(cudaMalloc(&d_ao, h_ao.size() * sizeof(Z)), "ao");
            check_cuda(cudaMalloc(&d_H, nmat * sizeof(Z)), "H");
            check_cuda(cudaMalloc(&d_X, nmat * sizeof(Z)), "X");
            check_cuda(cudaMalloc(&d_dm, nmat * sizeof(Z)), "dm");
            check_cuda(cudaMalloc(&d_vj, nmat * sizeof(Z)), "vj");
            check_cuda(cudaMalloc(&d_vk, nmat * sizeof(Z)), "vk");
            check_cuda(cudaMalloc(&d_F, nmat * sizeof(Z)), "F");
            check_cuda(cudaMalloc(&d_vhf, nmat * sizeof(Z)), "vhf");
            check_cuda(
                cudaMalloc(&d_rho, static_cast<std::size_t>(ng) * sizeof(Z)),
                "rho");
            check_cuda(
                cudaMalloc(&d_tmp, static_cast<std::size_t>(ng) * sizeof(Z)),
                "tmp");
            check_cuda(
                cudaMalloc(&d_rho1, static_cast<std::size_t>(nao * nao * ng)
                                        * sizeof(Z)),
                "rho1");
            check_cuda(cudaMalloc(&d_aodm, static_cast<std::size_t>(nao * ng)
                                               * sizeof(Z)),
                       "aodm");
            check_cuda(cudaMalloc(&d_vrdm, static_cast<std::size_t>(nao * ng)
                                               * sizeof(Z)),
                       "vrdm");
            check_cuda(cudaMalloc(&d_Hp, static_cast<std::size_t>(nao * nao)
                                             * sizeof(Z)),
                       "Hp");
            check_cuda(cudaMalloc(&d_C, nmat * sizeof(Z)), "C");
            const std::size_t norb = static_cast<std::size_t>(nk) * nao;
            check_cuda(cudaMalloc(&d_eps, norb * sizeof(double)), "eps");
            check_cuda(cudaMalloc(&d_idx, norb * sizeof(int)), "idx");
            check_cuda(cudaMalloc(&d_infos,
                                  static_cast<std::size_t>(nk) * sizeof(int)),
                       "infos");
            check_cuda(cudaMalloc(&d_occ, norb * sizeof(double)), "occ");
            check_cuda(cudaMalloc(&d_coul_j, static_cast<std::size_t>(ng)
                                                 * sizeof(double)),
                       "coul_j");
            check_cuda(
                cudaMalloc(&d_coul_k, sys.coul_k.size() * sizeof(double)),
                "coul_k");
            check_cuda(
                cudaMalloc(&d_coords, sys.coords.size() * sizeof(double)),
                "coords");
            check_cuda(cudaMalloc(&d_e, sizeof(double)), "e");
            check_cuda(cudaMalloc(&d_col, static_cast<std::size_t>(nao * nao)
                                              * sizeof(cuDoubleComplex)),
                       "col");
            check_cuda(cudaMalloc(&d_eig, static_cast<std::size_t>(nao)
                                              * sizeof(double)),
                       "eig");
            check_cuda(cudaMemcpyAsync(d_ao, h_ao.data(),
                                       h_ao.size() * sizeof(Z),
                                       cudaMemcpyHostToDevice, stream),
                       "H2D ao");
            check_cuda(cudaMemcpyAsync(d_H, h_H.data(), nmat * sizeof(Z),
                                       cudaMemcpyHostToDevice, stream),
                       "H2D H");
            check_cuda(cudaMemcpyAsync(d_X, h_X.data(), nmat * sizeof(Z),
                                       cudaMemcpyHostToDevice, stream),
                       "H2D X");
            check_cuda(
                cudaMemcpyAsync(d_coul_j, sys.coul_j.data(),
                                static_cast<std::size_t>(ng) * sizeof(double),
                                cudaMemcpyHostToDevice, stream),
                "H2D coul");
            check_cuda(cudaMemcpyAsync(d_coul_k, sys.coul_k.data(),
                                       sys.coul_k.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, stream),
                       "H2D coulk");
            check_cuda(cudaMemcpyAsync(d_coords, sys.coords.data(),
                                       sys.coords.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, stream),
                       "H2D coords");

            check_sol(cusolverDnCreate(&sol), "create");
            check_sol(cusolverDnSetStream(sol, stream), "set stream");
            check_sol(cusolverDnZheevd_bufferSize(sol, CUSOLVER_EIG_MODE_VECTOR,
                                                  CUBLAS_FILL_MODE_LOWER, nao,
                                                  d_col, nao, d_eig, &lwork),
                      "buffer");
            check_cuda(cudaMalloc(&d_work, static_cast<std::size_t>(lwork)
                                               * sizeof(cuDoubleComplex)),
                       "work");

            check_cuda(cudaMemsetAsync(d_C, 0, nmat * sizeof(Z), stream),
                       "zero C");
            k_fill_d<<<(static_cast<int>(norb) + 127) / 128, 128, 0, stream>>>(
                d_eps, static_cast<int>(norb), 1.0e8);
            // init_guess_by_1e: occupy the core Hamiltonian (hf.kernel dm0).
            diagonalize_fock(d_H);
            density();
        } catch (...) {
            release();
            throw;
        }
        jw = sys.vol / static_cast<double>(ng);
        kw = (1.0 / static_cast<double>(nk))
             * (sys.vol / static_cast<double>(ng));
    }

    void diagonalize_fock(const Z* fock) {
        for (int k = 0; k < nk; ++k) {
            const int m = nmo[static_cast<std::size_t>(k)];
            k_xform<<<1, 64, 0, stream>>>(d_X, fock, d_Hp, nao, m, k);
            k_row_to_col<<<(m * m + 127) / 128, 128, 0, stream>>>(d_Hp, d_col,
                                                                  m);
            check_sol(cusolverDnZheevd(sol, CUSOLVER_EIG_MODE_VECTOR,
                                       CUBLAS_FILL_MODE_LOWER, m, d_col, m,
                                       d_eig, d_work, lwork, d_infos + k),
                      "zheevd");
            check_cuda(cudaMemcpyAsync(
                           d_eps + static_cast<std::size_t>(k) * nao, d_eig,
                           sizeof(double) * static_cast<std::size_t>(m),
                           cudaMemcpyDeviceToDevice, stream),
                       "D2D eig");
            k_apply_x<<<m, 32, 0, stream>>>(d_X, d_col, d_C, nao, m, k);
        }
        std::vector<int> info(static_cast<std::size_t>(nk));
        check_cuda(cudaMemcpyAsync(info.data(), d_infos,
                                   sizeof(int) * static_cast<std::size_t>(nk),
                                   cudaMemcpyDeviceToHost, stream),
                   "D2H info");
        check_cuda(cudaStreamSynchronize(stream), "sync zheevd");
        for (int k = 0; k < nk; ++k)
            if (info[static_cast<std::size_t>(k)] != 0)
                throw std::runtime_error(
                    "cuda::scf::KRHF: zheevd info="
                    + std::to_string(info[static_cast<std::size_t>(k)]));
    }

    void density() {
        const std::size_t norb = static_cast<std::size_t>(nk) * nao;
        k_occupy<<<1, 1, 0, stream>>>(d_eps, d_idx, d_occ,
                                      static_cast<int>(norb), nocc);
        for (int k = 0; k < nk; ++k)
            k_dm_from_c<<<1, 32, 0, stream>>>(
                d_C, d_occ + static_cast<std::size_t>(k) * nao, d_dm, nao, k);
        k_hermitize<<<nk, 32, 0, stream>>>(d_dm, nk, nao);
    }

    void fock() {
        const int blk = 128;
        const std::size_t nmat = static_cast<std::size_t>(nk) * nao * nao;
        k_rho<<<div_up(ng, blk), blk, 0, stream>>>(d_ao, d_dm, d_rho, nk, nao,
                                                   ng);
        fft3_dev(d_rho, d_tmp, nx, ny, nz, false, stream);
        k_scale<<<div_up(ng, blk), blk, 0, stream>>>(d_rho, d_coul_j, ng);
        fft3_dev(d_rho, d_tmp, nx, ny, nz, true, stream);
        k_j_contract<<<div_up(nk * nao * nao, blk), blk, 0, stream>>>(
            d_ao, d_rho, d_vj, nk, nao, ng, jw);
        k_hermitize<<<nk, 32, 0, stream>>>(d_vj, nk, nao);

        k_zero<<<div_up(static_cast<int>(nmat), blk), blk, 0, stream>>>(
            d_vk, static_cast<int>(nmat));
        for (int k2 = 0; k2 < nk; ++k2) {
            k_aodm<<<div_up(ng, blk), blk, 0, stream>>>(d_ao, d_dm, d_aodm, nao,
                                                        ng, k2);
            for (int k1 = 0; k1 < nk; ++k1) {
                const double kd[3] = {kpts[k2][0] - kpts[k1][0],
                                      kpts[k2][1] - kpts[k1][1],
                                      kpts[k2][2] - kpts[k1][2]};
                const int zero =
                    (std::abs(kd[0]) + std::abs(kd[1]) + std::abs(kd[2]) < 1e-9)
                        ? 1
                        : 0;
                k_rho1<<<div_up(ng, blk), blk, 0, stream>>>(
                    d_ao, d_ao, d_coords, kd[0], kd[1], kd[2], zero, d_rho1,
                    nao, ng, k1, k2);
                const double* coul =
                    d_coul_k + static_cast<std::size_t>((k2 * nk + k1) * ng);
                for (int ij = 0; ij < nao * nao; ++ij) {
                    Z* slice = d_rho1 + static_cast<std::size_t>(ij) * ng;
                    fft3_dev(slice, d_tmp, nx, ny, nz, false, stream);
                    k_scale<<<div_up(ng, blk), blk, 0, stream>>>(slice, coul,
                                                                 ng);
                    fft3_dev(slice, d_tmp, nx, ny, nz, true, stream);
                }
                k_vrdm<<<div_up(ng, blk), blk, 0, stream>>>(
                    d_rho1, d_aodm, d_coords, kd[0], kd[1], kd[2], zero, d_vrdm,
                    nao, ng);
                k_k_accum<<<div_up(nao * nao, blk), blk, 0, stream>>>(
                    d_vrdm, d_ao, d_vk, nao, ng, k1, kw);
            }
        }
        k_hermitize<<<nk, 32, 0, stream>>>(d_vk, nk, nao);
        k_vhf<<<div_up(static_cast<int>(nmat), blk), blk, 0, stream>>>(
            d_vj, d_vk, d_vhf, static_cast<int>(nmat));
        k_trace_energy<<<1, 1, 0, stream>>>(d_dm, d_H, d_vhf, nk, nao, d_e);
        check_cuda(cudaMemcpyAsync(&e_ha, d_e, sizeof(double),
                                   cudaMemcpyDeviceToHost, stream),
                   "D2H e");
        check_cuda(cudaStreamSynchronize(stream), "sync e");
        e_ha += enuc;
        // KSCF.get_fock, diis=None: F = H + vhf. Built after the energy so
        // the trace still sees the density that entered this iteration.
        k_fock<<<div_up(static_cast<int>(nmat), blk), blk, 0, stream>>>(
            d_H, d_vhf, d_F, static_cast<int>(nmat));
    }

    void diagonalize() { diagonalize_fock(d_F); }

    double energy() const { return e_ha; }

    double delta() {
        const double d = have_previous
                             ? std::abs(e_ha - previous)
                             : std::numeric_limits<double>::infinity();
        previous = e_ha;
        have_previous = true;
        return d;
    }
};

Engine* require_engine(Engine** slot) {
    if (!slot || !*slot)
        throw std::runtime_error("cuda::scf::KRHF: SCF engine is not open");
    return *slot;
}

class FockStage : public cuda::Component {
  public:
    explicit FockStage(Engine** slot) : slot_(slot) {}
    std::string name() const override { return "krhf_fock"; }
    void launch(const cuda::LaunchContext& ctx) override {
        if (!ctx.at(ATV::scf::Phase::fock))
            return;
        require_engine(slot_)->fock();
    }

  private:
    Engine** slot_ = nullptr;
};

class MixStage : public cuda::Component {
  public:
    explicit MixStage(Engine**) {}
    std::string name() const override { return "krhf_mix"; }
    void launch(const cuda::LaunchContext& ctx) override {
        if (!ctx.at(ATV::scf::Phase::mix))
            return;
    }
};

class DiagonalizeStage : public cuda::Component {
  public:
    explicit DiagonalizeStage(Engine** slot) : slot_(slot) {}
    std::string name() const override { return "krhf_diagonalize"; }
    void launch(const cuda::LaunchContext& ctx) override {
        if (!ctx.at(ATV::scf::Phase::diagonalize))
            return;
        require_engine(slot_)->diagonalize();
    }

  private:
    Engine** slot_ = nullptr;
};

class DensityStage : public cuda::Component {
  public:
    explicit DensityStage(Engine** slot) : slot_(slot) {}
    std::string name() const override { return "krhf_density"; }
    void launch(const cuda::LaunchContext& ctx) override {
        if (!ctx.at(ATV::scf::Phase::density))
            return;
        require_engine(slot_)->density();
    }

  private:
    Engine** slot_ = nullptr;
};

class CheckStage : public cuda::Component {
  public:
    explicit CheckStage(Engine** slot) : slot_(slot) {}
    std::string name() const override { return "krhf_check"; }
    void launch(const cuda::LaunchContext& ctx) override {
        if (!ctx.at(ATV::scf::Phase::check))
            return;
        if (!ctx.runtime.loop.metrics)
            throw std::runtime_error(
                "cuda::scf::KRHF: check stage has no ConvergenceMetrics");
        Engine& eng = *require_engine(slot_);
        const double ha = eng.energy();
        const double dha = eng.delta();
        ctx.runtime.loop.metrics->energy = ha * units::kHartreeInEv;
        ctx.runtime.loop.metrics->energy_change = dha * units::kHartreeInEv;
    }

  private:
    Engine** slot_ = nullptr;
};

class EnergyDelta : public cuda::Criterion {
  public:
    explicit EnergyDelta(double tol_ha)
        : tol_ev_(tol_ha * units::kHartreeInEv) {}
    std::string name() const override { return "krhf_energy_delta"; }
    bool contributes_to_convergence() const override { return true; }
    ATV::Verdict verdict(const cuda::LoopState& s) override {
        if (s.metrics.energy_change < tol_ev_)
            return ATV::Verdict::Converged;
        return ATV::Verdict::Continue;
    }

  private:
    double tol_ev_ = 0.0;
};

template <typename Span>
void pull_position(Span col, const pos_t* primary, const pos_t* corr, int n,
                   cudaStream_t stream) {
    std::vector<pos_t> hi(static_cast<std::size_t>(n));
    std::vector<pos_t> lo(static_cast<std::size_t>(n), pos_t(0));
    check_cuda(cudaMemcpyAsync(hi.data(), primary,
                               sizeof(pos_t) * static_cast<std::size_t>(n),
                               cudaMemcpyDeviceToHost, stream),
               "D2H position");
    if constexpr (kPositionCorrection)
        check_cuda(cudaMemcpyAsync(lo.data(), corr,
                                   sizeof(pos_t) * static_cast<std::size_t>(n),
                                   cudaMemcpyDeviceToHost, stream),
                   "D2H position correction");
    check_cuda(cudaStreamSynchronize(stream), "sync position");
    for (int i = 0; i < n; ++i)
        col[i] = static_cast<std::decay_t<decltype(col[i])>>(
            load_position(hi.data(), lo.data(), i));
}

cpu::Frame host_geometry(const cuda::Frame& dev, cudaStream_t stream) {
    const int n = dev.n_atoms();
    if (n <= 0)
        throw std::runtime_error("cuda::scf::KRHF: device frame has no atoms");
    if (!dev.atoms.x || !dev.atoms.y || !dev.atoms.z)
        throw std::runtime_error(
            "cuda::scf::KRHF: device frame has no positions");
    if (!dev.atoms.type)
        throw std::runtime_error(
            "cuda::scf::KRHF: device frame has no atom types");
    if (!dev.box)
        throw std::runtime_error("cuda::scf::KRHF: device frame has no cell");
    if constexpr (kPositionCorrection) {
        if (!dev.atoms.x_corr || !dev.atoms.y_corr || !dev.atoms.z_corr)
            throw std::runtime_error(
                "cuda::scf::KRHF: device frame has no position correction");
    }
    cpu::Frame host(n);
    pull_position(host.span(kAtom_x), dev.atoms.x, dev.atoms.x_corr, n, stream);
    pull_position(host.span(kAtom_y), dev.atoms.y, dev.atoms.y_corr, n, stream);
    pull_position(host.span(kAtom_z), dev.atoms.z, dev.atoms.z_corr, n, stream);
    std::vector<int> types(static_cast<std::size_t>(n));
    check_cuda(cudaMemcpyAsync(types.data(), dev.atoms.type,
                               sizeof(int) * static_cast<std::size_t>(n),
                               cudaMemcpyDeviceToHost, stream),
               "D2H type");
    geom_t box[9];
    check_cuda(cudaMemcpyAsync(box, dev.box, 9 * sizeof(geom_t),
                               cudaMemcpyDeviceToHost, stream),
               "D2H cell");
    check_cuda(cudaStreamSynchronize(stream), "sync geometry");
    auto ty = host.span(kAtom_type);
    for (int i = 0; i < n; ++i)
        ty[i] = types[static_cast<std::size_t>(i)];
    Mat3 cell;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cell(i, j) = box[i * 3 + j];
    host.box.set_cell(cell);
    return host;
}

void accumulate_energy(cuda::Frame& frame, double e_ev, cudaStream_t stream) {
    if (!frame.scalars)
        throw std::runtime_error(
            "cuda::scf::KRHF: device frame has no energy scalar");
    double e = 0.0;
    check_cuda(cudaMemcpyAsync(&e, &frame.scalars->energy, sizeof(double),
                               cudaMemcpyDeviceToHost, stream),
               "D2H energy");
    check_cuda(cudaStreamSynchronize(stream), "sync energy");
    e += e_ev;
    check_cuda(cudaMemcpyAsync(&frame.scalars->energy, &e, sizeof(double),
                               cudaMemcpyHostToDevice, stream),
               "H2D energy");
    check_cuda(cudaStreamSynchronize(stream), "sync energy write");
}

} // namespace

struct KrhfLoop {
    Engine* eng = nullptr;
    std::unique_ptr<Engine> owned;
    FockStage fock;
    MixStage mix;
    DiagonalizeStage diag;
    DensityStage density;
    CheckStage check;
    EnergyDelta criterion;

    explicit KrhfLoop(double tol_ha)
        : fock(&eng), mix(&eng), diag(&eng), density(&eng), check(&eng),
          criterion(tol_ha) {}

    void release_engine() {
        owned.reset();
        eng = nullptr;
    }
};

KRHF::KRHF(cpu::scf::BasisSet basis_in, std::array<int, 3> k_mesh_in,
           int charge_in, double tol_in, int max_iter_in)
    : basis(basis_in), k_mesh(k_mesh_in), charge(charge_in), tol(tol_in),
      max_iter(max_iter_in), loop_(std::make_unique<KrhfLoop>(tol)) {
    wire_stages_();
}

KRHF::~KRHF() { release_backend_(); }

KRHF::KRHF(KRHF&& other) noexcept
    : cuda::Driver<ATV::scf::Phase>(std::move(other)),
      cuda::DriverNode(std::move(other)), cuda::HF(std::move(other)),
      basis(other.basis), k_mesh(other.k_mesh), charge(other.charge),
      tol(other.tol), max_iter(other.max_iter), loop_(std::move(other.loop_)),
      energy_ha_(other.energy_ha_), iterations_(other.iterations_) {}

void KRHF::wire_stages_() {
    this->on(ATV::scf::Phase::fock).add_ref(loop_->fock);
    this->on(ATV::scf::Phase::mix).add_ref(loop_->mix);
    this->on(ATV::scf::Phase::diagonalize).add_ref(loop_->diag);
    this->on(ATV::scf::Phase::density).add_ref(loop_->density);
    this->on(ATV::scf::Phase::check).add_ref(loop_->check);
    this->add_criterion_ref(loop_->criterion);
    this->build();
}

void KRHF::build_on(cuda::Context&, cuda::Frame&) {
    if (this->plan().slots.empty())
        this->build();
}

void KRHF::destroy_on(cuda::Context&, cuda::Frame&) {
    if (loop_)
        loop_->release_engine();
}

void KRHF::release_backend_() {
    if (loop_)
        loop_->release_engine();
    if (!this->backend_)
        return;
    if (this->backend_->context().stream)
        this->backend_->sync();
    this->backend_->destroy_components();
    this->backend_->shutdown();
    this->backend_.reset();
    this->built_plugins_.clear();
}

void KRHF::step_stages_(cuda::Context& ctx, cuda::Frame& frame, int step) {
    struct Guard {
        bool& flag;
        explicit Guard(bool& f) : flag(f) { flag = true; }
        ~Guard() { flag = false; }
    } guard(this->launching_);
    for (const auto& plan_slot : this->plan().slots) {
        if (plan_slot.ops.empty())
            continue;
        for (const auto& op : plan_slot.ops) {
            if (!op.component)
                continue;
            cuda::LaunchContext lc{ctx, frame, plan_slot.phase, step};
            op.component->launch(lc);
        }
    }
}

int KRHF::drive(cuda::Context& ctx, cpu::Frame& host, cuda::Frame* device,
                int budget, Write write) {
    if (!(tol > 0.0) || max_iter < 1)
        throw std::runtime_error(
            "cuda::scf::KRHF: tol must be positive and max_iter >= 1");
    auto* stream = static_cast<cudaStream_t>(ctx.stream);
    if (!stream)
        throw std::runtime_error("cuda::scf::KRHF: no CUDA stream");
    loop_->owned = std::make_unique<Engine>();
    loop_->eng = loop_->owned.get();
    struct Release {
        KrhfLoop* loop = nullptr;
        ~Release() {
            if (loop)
                loop->release_engine();
        }
    } release{loop_.get()};
    loop_->owned->open(host, basis, k_mesh, charge, stream);

    const cuda::LoopServices saved = ctx.loop;
    ctx.loop.status = &this->status_;
    ctx.loop.metrics = &this->metrics_;
    struct Restore {
        cuda::Context& ctx;
        cuda::LoopServices saved;
        ~Restore() { ctx.loop = saved; }
    } restore{ctx, saved};
    this->status_.reset();
    this->metrics_ = ConvergenceMetrics{};

    cuda::Frame* stage_frame = device;
    cuda::Frame local;
    if (!stage_frame)
        stage_frame = &local;

    const int cap = budget > 0 ? budget : max_iter;
    int executed = 0;
    bool ok = false;
    for (int it = 0; it < cap; ++it) {
        this->mutable_outer_status().reset();
        step_stages_(ctx, *stage_frame, it);
        ++executed;
        if (this->poll_criteria_(it, *stage_frame, ctx)
            != ATV::Verdict::Continue) {
            ok = this->converged();
            break;
        }
    }
    if (!ok)
        throw std::runtime_error("cuda::scf::KRHF: SCF did not converge in "
                                 + std::to_string(cap) + " iterations");
    energy_ha_ = loop_->eng->energy();
    iterations_ = executed;
    if (write == Write::Accumulate) {
        if (!device)
            throw std::runtime_error(
                "cuda::scf::KRHF: nested run has no device frame");
        accumulate_energy(*device, energy_ha_ * units::kHartreeInEv, stream);
    } else if (write == Write::Overwrite) {
        const double e_ev = energy_ha_ * units::kHartreeInEv;
        host.meta.get(kEnergy) = e_ev;
        host.meta.get(kEnergyElectronic) = e_ev;
    }
    return executed;
}

void KRHF::run(const cpu::Frame& frame) {
    cpu::Frame scratch(frame);
    this->setup_pass(scratch);
    try {
        drive(this->backend_->context(), scratch, nullptr, max_iter,
              Write::None);
        release_backend_();
    } catch (...) {
        release_backend_();
        throw;
    }
}

void KRHF::run(cpu::Frame& frame) {
    this->setup_pass(frame);
    try {
        drive(this->backend_->context(), frame, nullptr, max_iter,
              Write::Overwrite);
        release_backend_();
    } catch (...) {
        release_backend_();
        throw;
    }
}

int KRHF::run_nested_erased(cuda::Context& ctx, cuda::Frame& frame,
                            int budget) {
    auto* stream = static_cast<cudaStream_t>(ctx.stream);
    cpu::Frame host = host_geometry(frame, stream);
    return drive(ctx, host, &frame, budget, Write::Accumulate);
}

cpu::scf::GradientResult KRHF::forces(cpu::Frame& /*frame*/) {
    throw std::runtime_error(
        "cuda::scf::KRHF::forces: analytic k-point nuclear forces are not "
        "implemented (missing kernel krhf_nuclear_forces)");
}

} // namespace scf
} // namespace cuda
} // namespace ATV
