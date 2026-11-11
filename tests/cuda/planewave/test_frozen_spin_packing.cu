#include "../../cpu/planewave/nc_gauge_fixture.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/planewave/internal/frozen_operator_virial.h"
#include "atomiverse/cpu/planewave/internal/poisson.h"
#include "atomiverse/cpu/planewave/internal/pw_radial_derivatives.h"
#include "atomiverse/cpu/planewave/internal/pw_subspace.h"
#include "atomiverse/cuda/planewave/internal/frozen_operator_virial.cuh"

#include <algorithm>
#include <cstdio>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
using namespace ATV;
using namespace ATV::cpu::pw::internal;
namespace {
using C = std::complex<AccReal>;
using internal_data = NcPotentialData;
using GList = GVectorList;
using Orbitals = FrozenBandView;
struct Term {
    double energy = 0;
    std::array<double, 9> W{};
};
Term host_term(const FrozenOperatorVirial& source) {
    return {source.energy_ha, source.virial_ha};
}
Term kinetic(const GList& g, Orbitals o) {
    return host_term(kinetic_frozen_virial(g, o));
}
Term hartree(const GList& g, std::span<const C> rho, double v) {
    return host_term(hartree_frozen_virial(g, rho, v));
}
Term local(const internal_data& d, const GList& g, std::span<const Vec3> r,
           std::span<const C> rho, double v) {
    return host_term(local_potential_frozen_virial(d, g, r, rho, v));
}
Term nonlocal(const internal_data& d, const GList& g, std::span<const Vec3> r,
              Orbitals o, double v) {
    return host_term(nonlocal_frozen_virial(d, g, r, o, v));
}
void require(bool yes, const char* why) {
    if (!yes)
        throw std::runtime_error(why);
}
void close(double a, double b, const char* why) {
    if (!std::isfinite(a) || !std::isfinite(b)
        || std::abs(a - b) > 1e-8 + 1e-7 * std::abs(b)) {
        std::printf("FAIL %s %.17g %.17g\n", why, a, b);
        throw std::runtime_error(why);
    }
}
internal_data data(double scale) { return nc_gauge_test::potential(scale); }
GList modes() { return nc_gauge_test::modes(false); }
std::vector<C> wave(const GList& g) { return nc_gauge_test::wave(g); }
std::vector<C> density() {
    return {{.019, 0},      {.001, .002},    {.001, -.002},
            {-.002, .0007}, {-.002, -.0007}, {.0005, -.001},
            {.0005, .001},  {7e-28, 1e-28},  {7e-28, -1e-28}};
}
Mat3 identity() {
    Mat3 f;
    for (int a = 0; a < 3; ++a)
        f(a, a) = 1;
    return f;
}
GList transform(const GList& g, const Mat3& f) {
    auto out = g;
    const auto reciprocal = f.inverse().transpose();
    for (auto& q : out.g)
        q = reciprocal * q;
    out.kvec = reciprocal * g.kvec;
    for (int i = 0; i < out.count(); ++i)
        out.ekin[i] = .5 * (out.g[i] + out.kvec).norm_squared();
    return out;
}
std::vector<Vec3> transform(std::span<const Vec3> r, const Mat3& f) {
    std::vector<Vec3> out;
    for (const auto& q : r)
        out.push_back(f * q);
    return out;
}
double value_local(const internal_data& d, const GList& g,
                   std::span<const Vec3> r, std::span<const C> rho,
                   double volume) {
    const auto vg = nc_vloc_to_gspace(d, g, volume);
    double e = 0;
    for (int i = 0; i < g.count(); ++i) {
        C structure{};
        for (const auto& a : r)
            structure += std::polar(1., -g.g[i].dot(a));
        e += volume * std::real(std::conj(rho[i]) * structure) * vg[i];
    }
    return e;
}
double value_nl(const internal_data& d, const GList& g, std::span<const Vec3> r,
                Orbitals o, double volume) {
    std::vector<Vec3> atoms(r.begin(), r.end());
    std::vector<std::vector<KBProjector>> proj;
    setup_kb_projectors(proj, g, d, atoms, volume);
    double e = 0;
    for (std::size_t b = 0; b < o.occupations.size(); ++b)
        for (const auto& site : proj)
            for (const auto& p : site)
                for (int m = 0; m < 2 * p.l + 1; ++m) {
                    C overlap{};
                    for (int i = 0; i < g.count(); ++i)
                        overlap += std::conj(p.beta_G(m, i))
                                   * o.coefficients[b * g.count() + i];
                    e += o.k_weight * o.occupations[b] * p.d
                         * std::norm(overlap);
                }
    return e;
}
double value_h(const GList& g, std::span<const C> rho, double volume) {
    PWWorkspace ws;
    ws.gvec = &g;
    ws.n_g = g.count();
    ws.rho_G.resize({rho.size()});
    ws.V_H_G.resize({rho.size()});
    std::copy(rho.begin(), rho.end(), ws.rho_G.begin());
    PoissonSolver poisson;
    poisson.step(ws);
    double e = 0;
    for (int i = 0; i < g.count(); ++i)
        e += .5 * volume * std::real(std::conj(rho[i]) * ws.V_H_G[i]);
    return e;
}
double value_t(const GList& g, Orbitals o) {
    double e = 0;
    for (std::size_t b = 0; b < o.occupations.size(); ++b)
        for (int i = 0; i < g.count(); ++i)
            e += o.k_weight * o.occupations[b]
                 * std::norm(o.coefficients[b * g.count() + i]) * .5
                 * (g.g[i] + g.kvec).norm_squared();
    return e;
}
struct DeviceResources {
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    std::vector<void*> storage;
    DeviceResources() {
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
            != cudaSuccess)
            throw std::runtime_error("create frozen operator stream");
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
            cudaStreamDestroy(stream);
            throw std::runtime_error("create frozen operator BLAS");
        }
        if (cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS
            || cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST)
                   != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(blas);
            cudaStreamDestroy(stream);
            throw std::runtime_error("configure frozen operator BLAS");
        }
    }
    ~DeviceResources() {
        cudaStreamSynchronize(stream);
        for (auto* p : storage)
            cudaFree(p);
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
    }
    template <class T>
    T* allocate(std::size_t n) {
        T* p = nullptr;
        if (n && cudaMalloc(&p, n * sizeof(T)) != cudaSuccess)
            throw std::runtime_error("frozen operator allocate");
        if (p)
            storage.push_back(p);
        return p;
    }
    template <class T>
    T* upload(const std::vector<T>& source) {
        auto* p = allocate<T>(source.size());
        if (p
            && cudaMemcpyAsync(p, source.data(), source.size() * sizeof(T),
                               cudaMemcpyHostToDevice, stream)
                   != cudaSuccess)
            throw std::runtime_error("frozen operator cold upload");
        return p;
    }
};
struct PackedOutput {
    std::array<Term, 4> terms;
    std::vector<double> force;
};
PackedOutput run_gpu(const internal_data& d, const GList& g,
                     std::span<const Vec3> atoms, std::span<const C> rho,
                     Orbitals state, double volume, int repeats = 1) {
    using namespace ATV::cuda::pw::internal;
    d.validate();
    require(g.count() > 0
                && state.coefficients.size()
                       == state.occupations.size() * g.count(),
            "cold shape validation");
    // All cold upload sources precede the stream-owning resource object so an
    // exceptional stream drain occurs before the source vectors are destroyed.
    std::vector<double> Gx, Gy, Gz, qx, qy, qz, R, slope, local_value,
        local_slope, positions,
        occupation(state.occupations.begin(), state.occupations.end());
    std::vector<cufftDoubleComplex> coefficients, density_values, structure;
    std::vector<FrozenProjectorColumn> columns;
    const std::vector<FrozenProjectorSpecies> species{{0, 0, d.n_proj()}};
    for (const auto& r : atoms)
        for (int k = 0; k < 3; ++k)
            positions.push_back(r[k]);
    for (const auto& c : state.coefficients)
        coefficients.push_back({c.real(), c.imag()});
    for (const auto& c : rho)
        density_values.push_back({c.real(), c.imag()});
    for (int i = 0; i < g.count(); ++i) {
        Gx.push_back(g.g[i][0]);
        Gy.push_back(g.g[i][1]);
        Gz.push_back(g.g[i][2]);
        const auto q = g.g[i] + g.kvec;
        qx.push_back(q[0]);
        qy.push_back(q[1]);
        qz.push_back(q[2]);
        const auto local_jet = nc_vloc_fourier_jet(d, g.g[i].norm(), volume);
        local_value.push_back(local_jet.value);
        local_slope.push_back(local_jet.slope);
        C s{};
        for (const auto& r : atoms)
            s += std::polar(1., -g.g[i].dot(r));
        structure.push_back({s.real(), s.imag()});
    }
    for (int p = 0; p < d.n_proj(); ++p)
        for (int i = 0; i < g.count(); ++i) {
            const auto radial =
                nc_projector_radial_jet(d, p, (g.g[i] + g.kvec).norm());
            R.push_back(radial.value);
            slope.push_back(radial.slope_over_q);
        }
    for (std::size_t atom = 0; atom < atoms.size(); ++atom)
        for (int l = 0; l <= 2; ++l)
            for (int m = 0; m < 2 * l + 1; ++m) {
                const int start = columns.size();
                std::vector<int> radial;
                for (int p = 0; p < d.n_proj(); ++p)
                    if (d.kb[p].l == l)
                        radial.push_back(p);
                for (int p : radial)
                    columns.push_back(
                        {int(atom), l, m, p, start, int(radial.size()), 0});
            }
    std::array<ATV::cuda::pw::internal::FrozenOperatorVirial, 4> downloaded;
    std::vector<double> downloaded_force(3 * atoms.size());
    DeviceResources device;
    const FrozenReciprocalView G{device.upload(Gx), device.upload(Gy),
                                 device.upload(Gz), g.count()},
        q{device.upload(qx), device.upload(qy), device.upload(qz), g.count()};
    const FrozenBandDeviceView orbital{device.upload(coefficients),
                                       device.upload(occupation),
                                       int(occupation.size()), state.k_weight};
    const FrozenLocalDeviceView lv{device.upload(local_value),
                                   device.upload(local_slope),
                                   device.upload(structure), 1};
    const auto* density = device.upload(density_values);
    auto* output =
        device.allocate<ATV::cuda::pw::internal::FrozenOperatorVirial>(4);
    if (cudaMemsetAsync(
            output, 0,
            4 * sizeof(ATV::cuda::pw::internal::FrozenOperatorVirial),
            device.stream)
        != cudaSuccess)
        throw std::runtime_error("zero frozen operator terms");
    FrozenNonlocalDeviceGeometry nl{q,
                                    device.upload(R),
                                    device.upload(slope),
                                    device.upload(d.d_ij),
                                    device.upload(positions),
                                    device.upload(columns),
                                    device.upload(species),
                                    1,
                                    int(columns.size()),
                                    4 * M_PI / std::sqrt(volume)};
    const std::size_t bc = columns.size() * occupation.size();
    FrozenNonlocalDeviceScratch scratch{
        device.allocate<cufftDoubleComplex>(columns.size() * g.count()),
        device.allocate<cufftDoubleComplex>(bc),
        device.allocate<cufftDoubleComplex>(bc),
        device.allocate<cufftDoubleComplex>(bc)};
    auto* force = device.allocate<double>(3 * atoms.size());
    auto* qC = device.allocate<cufftDoubleComplex>(std::size_t(g.count())
                                                   * occupation.size());
    if (cudaMemsetAsync(force, 0, 3 * atoms.size() * sizeof(double),
                        device.stream)
        != cudaSuccess)
        throw std::runtime_error("zero force");
    for (int repeat = 0; repeat < repeats; ++repeat) {
        ATV::cuda::pw::internal::kinetic_frozen_virial(q, orbital, output,
                                                       device.stream);
        ATV::cuda::pw::internal::hartree_frozen_virial(
            G, density, volume, output + 1, device.stream);
        ATV::cuda::pw::internal::local_potential_frozen_virial(
            G, density, volume, lv, output + 2, device.stream);
        // Force consumes the value projection B; the subsequent W call reuses B
        // for derivative columns, so this order is part of the shared contract.
        ATV::cuda::pw::internal::project_frozen_nonlocal(
            nl, orbital, scratch, device.blas, device.stream);
        ATV::cuda::pw::internal::accumulate_projected_nonlocal_forces(
            nl, orbital, scratch, qC, device.blas, force, device.stream);
        ATV::cuda::pw::internal::add_projected_nonlocal_virial(
            nl, orbital, scratch, device.blas, output + 3, device.stream);
    }
    if (cudaMemcpyAsync(downloaded_force.data(), force,
                        downloaded_force.size() * sizeof(double),
                        cudaMemcpyDeviceToHost, device.stream)
        != cudaSuccess)
        throw std::runtime_error("download force");
    if (cudaMemcpyAsync(downloaded.data(), output, sizeof(downloaded),
                        cudaMemcpyDeviceToHost, device.stream)
            != cudaSuccess
        || cudaStreamSynchronize(device.stream) != cudaSuccess)
        throw std::runtime_error("download frozen operator component boundary");
    std::array<Term, 4> result;
    for (int t = 0; t < 4; ++t) {
        result[t].energy = downloaded[t].energy_ha;
        for (int ab = 0; ab < 9; ++ab)
            result[t].W[ab] = downloaded[t].virial_ha[ab];
    }
    return {result, downloaded_force};
}

void orthonormal(std::vector<C>& c, int ng) {
    for (int b = 0; b < 3; ++b) {
        for (int j = 0; j < b; ++j) {
            C dot{};
            for (int i = 0; i < ng; ++i)
                dot += std::conj(c[j * ng + i]) * c[b * ng + i];
            for (int i = 0; i < ng; ++i)
                c[b * ng + i] -= dot * c[j * ng + i];
        }
        double norm = 0;
        for (int i = 0; i < ng; ++i)
            norm += std::norm(c[b * ng + i]);
        require(norm > 1e-8, "independent orbital rows");
        for (int i = 0; i < ng; ++i)
            c[b * ng + i] /= std::sqrt(norm);
    }
}
void packing_case(bool shifted, double amplitude, int kind,
                  bool small = false) {
    auto d = nc_gauge_test::potential(amplitude, true);
    auto g = nc_gauge_test::modes(shifted);
    if (small) {
        g.g.resize(4);
        g.h.resize(4);
        g.k.resize(4);
        g.l.resize(4);
        g.ekin.resize(4);
    }
    int ng = g.count();
    auto alpha = wave(g);
    orthonormal(alpha, ng);
    auto beta = alpha;
    for (int b = 0; b < 3; ++b)
        for (int i = 0; i < ng; ++i)
            beta[b * ng + i] = std::polar(1., .13 * (i + 1) * (b + 2))
                               * (.63 * alpha[((b + 1) % 3) * ng + i]
                                  + C{0, .37} * alpha[((b + 2) % 3) * ng + i]);
    orthonormal(beta, ng);
    std::vector<double> fa = {1, 1, 1}, fb = {1, 1, 0};
    double scale = 1;
    if (kind == 1) {
        fa = {1.1, -.3, 5e-15};
        fb = {.7, -1e-16, 0};
    }
    if (kind == 2) {
        fa = {0, 0, 5e-15};
        fb = {0, -1e-16, 0};
        scale = 1e15;
    }
    if (kind == 3) {
        beta = alpha;
        fa = {1, 1, 0};
        fb = fa;
    }
    std::vector<C> packed = alpha;
    packed.insert(packed.end(), beta.begin(), beta.end());
    std::vector<double> f = fa;
    f.insert(f.end(), fb.begin(), fb.end());
    double crossgram = 0;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            C x{};
            for (int i = 0; i < ng; ++i)
                x += std::conj(alpha[a * ng + i]) * beta[b * ng + i];
            crossgram = std::max(crossgram, std::abs(x));
        }
    require(crossgram > 1e-3, "cross-spin Gram intentionally nonzero");
    std::vector<Vec3> atoms = {{.2, .4, .7}, {1.1, -.3, .5}};
    std::vector<C> rho(ng);
    double volume = 125, weight = .37;
    Orbitals oa{alpha, fa, weight}, ob{beta, fb, weight}, op{packed, f, weight};
    auto a = run_gpu(d, g, atoms, rho, oa, volume),
         b = run_gpu(d, g, atoms, rho, ob, volume),
         both = run_gpu(d, g, atoms, rho, op, volume);
    double error = 0;
    for (int t : {0, 3}) {
        close(scale * both.terms[t].energy,
              scale * (a.terms[t].energy + b.terms[t].energy), "packed E sum");
        for (int ab = 0; ab < 9; ++ab) {
            double delta = std::abs(
                scale
                * (both.terms[t].W[ab] - a.terms[t].W[ab] - b.terms[t].W[ab]));
            error = std::max(error, delta);
            close(scale * both.terms[t].W[ab],
                  scale * (a.terms[t].W[ab] + b.terms[t].W[ab]),
                  "packed W9 sum");
        }
    }
    for (std::size_t i = 0; i < both.force.size(); ++i) {
        error = std::max(
            error, std::abs(scale * (both.force[i] - a.force[i] - b.force[i])));
        close(scale * both.force[i], scale * (a.force[i] + b.force[i]),
              "packed full NL forces sum");
    }
    auto ct = kinetic(g, op), cn = nonlocal(d, g, atoms, op, volume);
    close(scale * both.terms[0].energy, scale * ct.energy, "raw CPU T");
    close(scale * both.terms[3].energy, scale * cn.energy, "raw fullD CPU NL");
    for (int ab = 0; ab < 9; ++ab) {
        close(scale * both.terms[0].W[ab], scale * ct.W[ab], "CPU T W9");
        close(scale * both.terms[3].W[ab], scale * cn.W[ab], "CPU NL W9");
    }
    double fd_error = 0, force_error = 0;
    for (double h : {1e-5, 3e-6}) {
        for (int ab = 0; ab < 9; ++ab) {
            double et[2], en[2];
            for (int sign = 0; sign < 2; ++sign) {
                auto F = identity();
                F(ab / 3, ab % 3) += (sign == 0 ? h : -h);
                auto gf = transform(g, F);
                auto rf = transform(atoms, F);
                et[sign] = value_t(gf, op);
                en[sign] = value_nl(d, gf, rf, op, volume * F.det());
            }
            double dt = -(et[0] - et[1]) / (2 * h),
                   dn = -(en[0] - en[1]) / (2 * h);
            close(scale * both.terms[0].W[ab], scale * dt,
                  "T production value all9FD");
            close(scale * both.terms[3].W[ab], scale * dn,
                  "NL production value all9FD");
            fd_error =
                std::max({fd_error, scale * std::abs(dt - both.terms[0].W[ab]),
                          scale * std::abs(dn - both.terms[3].W[ab])});
        }
        for (int atom = 0; atom < 2; ++atom)
            for (int axis = 0; axis < 3; ++axis) {
                double e[2];
                for (int sign = 0; sign < 2; ++sign) {
                    auto r = atoms;
                    r[atom][axis] += (sign == 0 ? h : -h);
                    e[sign] = value_nl(d, g, r, op, volume);
                }
                double fd = -(e[0] - e[1]) / (2 * h);
                close(scale * both.force[3 * atom + axis], scale * fd,
                      "NL production fullF FD");
                force_error = std::max(
                    force_error,
                    scale * std::abs(fd - both.force[3 * atom + axis]));
            }
    }
    double wrong_last =
        scale * std::abs(both.terms[0].energy - b.terms[0].energy);
    require(wrong_last > 1e-6, "beta-only counterfactual distinguishable");
    std::vector<C> coherent = alpha;
    for (std::size_t i = 0; i < coherent.size(); ++i)
        coherent[i] += beta[i];
    double wrong_coherent =
        scale
        * std::abs(value_t(g, {coherent, fa, weight}) - both.terms[0].energy);
    require(wrong_coherent > 1e-6,
            "coherent spin sum counterfactual distinguishable");
    if (kind == 3) {
        std::vector<double> doubled = fa;
        for (auto& x : doubled)
            x *= 2;
        auto rks = run_gpu(d, g, atoms, rho, {alpha, doubled, weight}, volume);
        for (int t : {0, 3}) {
            close(rks.terms[t].energy, both.terms[t].energy,
                  "collapsed RKS energy");
            for (int ab = 0; ab < 9; ++ab)
                close(rks.terms[t].W[ab], both.terms[t].W[ab],
                      "collapsed RKS W9");
        }
        for (std::size_t i = 0; i < both.force.size(); ++i)
            close(rks.force[i], both.force[i], "collapsed RKS F");
    }
    std::printf("PASS PACKED_SPIN_OPERATORS shifted=%d gauge=%.17g kind=%d "
                "ng=%d live_bands=3 packed_bands=6 crossGram=%.17g "
                "sum_error_scaled=%.17g all9FD_scaled=%.17g "
                "fullF_FD_scaled=%.17g beta_only_error_scaled=%.17g "
                "coherent_error_scaled=%.17g comparison_scale=%.17g SCF=0\n",
                shifted, amplitude, kind, ng, crossgram, error, fd_error,
                force_error, wrong_last, wrong_coherent, scale);
}
} // namespace
TEST(CudaPackedSpinOperators, IndependentSpinSumAndCounterfactuals) {
    for (bool shifted : {false, true}) {
        for (double gauge : {1., std::ldexp(1., 30), std::ldexp(1., -30)})
            for (int kind : {0, 1})
                packing_case(shifted, gauge, kind);
        packing_case(shifted, 1., 2);
        packing_case(shifted, 1., 3);
    }
    packing_case(false, 1., 0, true);
}
