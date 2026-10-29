#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include "atomiverse/cpu/planewave/internal/xc_quadrature.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/planewave/internal/xc_frozen_virial.cuh"
#include <algorithm>
#include <array>
#include <complex>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
namespace {
using namespace ATV;
namespace host = cpu::pw::internal;
namespace dev = cuda::pw::internal;
using C = std::complex<double>;
void require(bool x, const char* s) {
    if (!x)
        throw std::runtime_error(s);
}
void close(double a, double b, const char* s) {
    if (!std::isfinite(a) || std::abs(a - b) > 1e-8 + 1e-7 * std::abs(b)) {
        std::printf(
            "FAIL %s actual=%.17g expected=%.17g error=%.17g gate=%.17g\n", s,
            a, b, std::abs(a - b), 1e-8 + 1e-7 * std::abs(b));
        throw std::runtime_error(s);
    }
}
struct Resources {
    cudaStream_t stream = nullptr;
    std::vector<void*> p;
    std::size_t bytes = 0;
    Resources() {
        ATV_CUDA_CHECK(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~Resources() {
        if (stream)
            cudaStreamSynchronize(stream);
        for (auto* q : p)
            cudaFree(q);
        if (stream)
            cudaStreamDestroy(stream);
    }
    template <class T>
    T* alloc(std::size_t n) {
        T* q = nullptr;
        ATV_CUDA_CHECK(cudaMalloc(&q, n * sizeof(T)));
        p.push_back(q);
        bytes += n * sizeof(T);
        return q;
    }
    template <class T>
    T* upload(const std::vector<T>& v) {
        auto* q = alloc<T>(v.size());
        ATV_CUDA_CHECK(cudaMemcpyAsync(q, v.data(), v.size() * sizeof(T),
                                       cudaMemcpyHostToDevice, stream));
        return q;
    }
};
struct Fence {
    cudaStream_t stream;
    ~Fence() { cudaStreamSynchronize(stream); }
};
Mat3 identity() {
    Mat3 F;
    for (int i = 0; i < 3; ++i)
        F(i, i) = 1;
    return F;
}
struct Point {
    double rho;
    Vec3 gradient;
};
struct Primal {
    double energy, gdotflux, minrho;
};
Primal primal(cpu::xc::XCType type, const std::vector<Point>& points,
              const Mat3& F, double volume) {
    const double determinant = F.det();
    const auto map = F.inverse().transpose();
    long double sum = 0, gp = 0;
    double minrho = 1e9;
    for (const auto& p : points) {
        const double rho = p.rho / determinant;
        const auto g = (map * p.gradient) / determinant;
        const double sigma = g.norm_squared();
        const auto xc = cpu::xc::evaluate_xc(type, rho / 2, rho / 2, sigma / 4,
                                             sigma / 4, sigma / 4);
        sum += xc.energy_density;
        gp += (xc.vsigma_aa + .5 * xc.vsigma_ab) * sigma;
        minrho = std::min(minrho, rho);
    }
    return {double(sum * volume * determinant / points.size()),
            double(gp * volume * determinant / points.size()), minrho};
}
std::array<double, 7> cpu_tensor(cpu::xc::XCType type,
                                 const host::GVectorList& g,
                                 const std::vector<C>& rho, double volume) {
    host::PWProblem problem;
    problem.gv0 = g;
    auto& w = problem.ws;
    w.gvec = &problem.gv0;
    w.n_g = g.count();
    w.omega = volume;
    w.fft_nx = g.nx;
    w.fft_ny = g.ny;
    w.fft_nz = g.nz;
    w.fft = std::make_unique<host::FFTPlan3D>(g.nx, g.ny, g.nz);
    w.nr = w.fft->nr();
    w.nc = w.fft->nc();
    w.rho_G.resize({rho.size()});
    std::copy(rho.begin(), rho.end(), w.rho_G.begin());
    host::XCBuilder xc;
    xc.prepare(w);
    xc.step(w, type);
    const auto& W = xc.virial_ha();
    return {w.E_xc, W[0], W[1], W[2], W[3], W[4], W[5]};
}
} // namespace
TEST(CudaFrozenPbeVirial, FourFamiliesMatchIndependentAffineEnergyDerivative) {
    Mat3 cell;
    cell(0, 0) = 8;
    cell(1, 0) = .4;
    cell(2, 0) = -.2;
    cell(0, 1) = .7;
    cell(1, 1) = 7;
    cell(2, 1) = .3;
    cell(0, 2) = -.4;
    cell(1, 2) = .5;
    cell(2, 2) = 6;
    const double volume = cell.det();
    const auto reciprocal = cell.inverse().transpose();
    host::GVectorList g;
    g.nx = 8;
    g.ny = 6;
    g.nz = 10;
    std::vector<C> rho;
    const std::array<std::array<int, 3>, 3> modes = {
        {{1, 0, 0}, {0, 1, 1}, {1, -1, 1}}};
    const std::array<C, 3> weights = {C{.0012, -.0009}, C{-.001, .00075},
                                      C{.0006, .00045}};
    for (int l = -2; l <= 2; ++l)
        for (int k = -2; k <= 2; ++k)
            for (int h = -2; h <= 2; ++h) {
                g.h.push_back(h);
                g.k.push_back(k);
                g.l.push_back(l);
                g.g.push_back(reciprocal
                              * Vec3{2 * M_PI * h, 2 * M_PI * k, 2 * M_PI * l});
                g.ekin.push_back(.5 * g.g.back().norm_squared());
                C z = (h == 0 && k == 0 && l == 0) ? C{.02, 0} : C{};
                for (int j = 0; j < 3; ++j) {
                    const auto& m = modes[j];
                    if (h == m[0] && k == m[1] && l == m[2])
                        z = weights[j];
                    if (h == -m[0] && k == -m[1] && l == -m[2])
                        z = std::conj(weights[j]);
                }
                rho.push_back(z);
            }
    const int qx = host::xc_quadrature_dimension(g.nx, true),
              qy = host::xc_quadrature_dimension(g.ny, true),
              qz = host::xc_quadrature_dimension(g.nz, true);
    std::vector<Point> points;
    points.reserve(qx * qy * qz);
    for (int z = 0; z < qz; ++z)
        for (int y = 0; y < qy; ++y)
            for (int x = 0; x < qx; ++x) {
                C value{};
                std::array<C, 3> gradient{};
                for (int i = 0; i < g.count(); ++i)
                    if (rho[i] != C{}) {
                        const auto term =
                            rho[i]
                            * std::polar(1., 2 * M_PI
                                                 * (double(g.h[i]) * x / qx
                                                    + double(g.k[i]) * y / qy
                                                    + double(g.l[i]) * z / qz));
                        value += term;
                        for (int d = 0; d < 3; ++d)
                            gradient[d] += C{0, g.g[i][d]} * term;
                    }
                require(std::abs(value.imag()) < 1e-16, "Hermitian density");
                points.push_back({value.real(),
                                  {gradient[0].real(), gradient[1].real(),
                                   gradient[2].real()}});
            }
    std::vector<cufftDoubleComplex> device_rho;
    for (auto z : rho)
        device_rho.push_back({z.real(), z.imag()});
    std::array<std::vector<double>, 3> physical;
    for (auto& v : physical)
        v.resize(g.count());
    dev::FrozenOperatorVirial result{};
    Resources gpu;
    dev::FFTPlan3D fft(qx, qy, qz, gpu.stream);
    Fence fence{gpu.stream};
    const auto* h = gpu.upload(g.h);
    const auto* k = gpu.upload(g.k);
    const auto* l = gpu.upload(g.l);
    const auto* density = gpu.upload(device_rho);
    std::array<double*, 3> G;
    for (auto& v : G)
        v = gpu.alloc<double>(g.count());
    dev::FrozenPbeWorkspaceView w{};
    w.scaled_density = gpu.alloc<cufftDoubleComplex>(g.count());
    w.derivative = gpu.alloc<cufftDoubleComplex>(g.count());
    w.half = gpu.alloc<cufftDoubleComplex>(fft.nc());
    w.rho = gpu.alloc<double>(fft.nr());
    w.v = gpu.alloc<double>(fft.nr());
    w.epsilon = gpu.alloc<double>(fft.nr());
    for (int d = 0; d < 3; ++d) {
        w.raw_gradient[d] = gpu.alloc<double>(fft.nr());
        w.flux[d] = gpu.alloc<double>(fft.nr());
    }
    w.energy = gpu.alloc<double>(1);
    w.energy_partials = gpu.alloc<double>(256);
    w.virial_partials = gpu.alloc<double>(9 * 256);
    w.output = gpu.alloc<dev::FrozenOperatorVirial>(1);
    auto setup = [&](const Mat3& F) {
        const auto map = F.inverse().transpose();
        for (int i = 0; i < g.count(); ++i) {
            const auto v = map * g.g[i];
            for (int d = 0; d < 3; ++d)
                physical[d][i] = v[d];
        }
        for (int d = 0; d < 3; ++d)
            ATV_CUDA_CHECK(cudaMemcpyAsync(G[d], physical[d].data(),
                                           g.count() * sizeof(double),
                                           cudaMemcpyHostToDevice, gpu.stream));
    };
    auto run = [&](cpu::xc::XCType type, const Mat3& F) {
        setup(F);
        dev::evaluate_pbe_frozen_virial(type, fft, h, k, l, G[0], G[1], G[2],
                                        g.count(), density, 1 / F.det(),
                                        volume * F.det(), w, gpu.stream);
        ATV_CUDA_CHECK(cudaMemcpyAsync(&result, w.output, sizeof(result),
                                       cudaMemcpyDeviceToHost, gpu.stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
        return result;
    };
    const cpu::xc::XCType types[] = {
        cpu::xc::XCType::PBE, cpu::xc::XCType::PBEsol, cpu::xc::XCType::RevPBE,
        cpu::xc::XCType::RPBE};
    for (auto type : types) {
        const auto ref = run(type, identity());
        const auto direct = primal(type, points, identity(), volume);
        const auto cpu = cpu_tensor(type, g, rho, volume);
        close(ref.energy_ha, direct.energy,
              "independent Fourier-point primal E");
        close(ref.energy_ha, cpu[0], "CPU padded XC E");
        require(direct.minrho > .01, "strict positive density domain");
        require(std::abs(direct.gdotflux) > 1e-6,
                "isotropic gradient term distinguishable");
        const int a[6] = {0, 1, 2, 0, 0, 1}, b[6] = {0, 1, 2, 1, 2, 2};
        double cpu_error = 0, fd_error = 0, gpu_fd_error = 0, maxshear = 0;
        for (int j = 0; j < 6; ++j) {
            cpu_error =
                std::max(cpu_error,
                         std::abs(ref.virial_ha[3 * a[j] + b[j]] - cpu[j + 1]));
            close(ref.virial_ha[3 * a[j] + b[j]], cpu[j + 1], "CPU tensor");
        }
        for (int ab = 0; ab < 9; ++ab)
            if (ab / 3 != ab % 3)
                maxshear = std::max(maxshear, std::abs(ref.virial_ha[ab]));
        require(maxshear > 1e-7, "nontrivial shear");
        for (double step : {1e-5, 3e-6}) {
            double local = 0, local_gpu = 0;
            for (int ab = 0; ab < 9; ++ab) {
                double e[2], eg[2];
                for (int sign = 0; sign < 2; ++sign) {
                    auto F = identity();
                    F(ab / 3, ab % 3) += (sign == 0 ? step : -step);
                    e[sign] = primal(type, points, F, volume).energy;
                    eg[sign] = run(type, F).energy_ha;
                }
                const double fd = -(e[0] - e[1]) / (2 * step),
                             fdg = -(eg[0] - eg[1]) / (2 * step);
                close(fd, ref.virial_ha[ab],
                      "independent primal all9 affine FD");
                close(fdg, ref.virial_ha[ab],
                      "device production energy all9 affine FD");
                local = std::max(local, std::abs(fd - ref.virial_ha[ab]));
                local_gpu =
                    std::max(local_gpu, std::abs(fdg - ref.virial_ha[ab]));
            }
            fd_error = std::max(fd_error, local);
            gpu_fd_error = std::max(gpu_fd_error, local_gpu);
            std::printf("PBE_XC_FD model=%d h=%.17g "
                        "independent_max_Ha=%.17g device_max_Ha=%.17g\n",
                        int(type), step, local, local_gpu);
        }
        auto rotation = identity();
        rotation(0, 0) = std::cos(.37);
        rotation(0, 1) = -std::sin(.37);
        rotation(1, 0) = std::sin(.37);
        rotation(1, 1) = std::cos(.37);
        const auto rotated = run(type, rotation);
        Mat3 W;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                W(a, b) = ref.virial_ha[3 * a + b];
        const auto target = rotation * W * rotation.transpose();
        close(rotated.energy_ha, ref.energy_ha, "rotation E");
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                close(rotated.virial_ha[3 * a + b], target(a, b),
                      "rotation W9");
        std::printf(
            "PASS CUDA_PW_PBE_XC model=%d E_Ha=%.17g "
            "CPU_tensor_error_Ha=%.17g independent_9FD_Ha=%.17g "
            "device_9FD_Ha=%.17g isotropic_gradflux_Ha=%.17g minrho=%.17g "
            "maxshear_Ha=%.17g base=%dx%dx%d quadrature=%dx%dx%d "
            "explicit_bytes=%zu scf=0 bands=0\n",
            int(type), ref.energy_ha, cpu_error, fd_error, gpu_fd_error,
            direct.gdotflux, direct.minrho, maxshear, g.nx, g.ny, g.nz, qx, qy,
            qz, gpu.bytes);
    }
}
