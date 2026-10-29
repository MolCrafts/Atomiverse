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
    if (!std::isfinite(a) || std::abs(a - b) > 1e-8 + 1e-7 * std::abs(b)) {
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
    std::vector<std::vector<PWWorkspace::KBProjG>> proj;
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
        if (cudaStreamCreate(&stream) != cudaSuccess)
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
std::array<Term, 4> run_gpu(const internal_data& d, const GList& g,
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
                        {int(atom), l, m, p, start, int(radial.size())});
            }
    std::array<ATV::cuda::pw::internal::FrozenOperatorVirial, 4> downloaded;
    DeviceResources device;
    const FrozenReciprocalView G{device.upload(Gx), device.upload(Gy),
                                 device.upload(Gz), g.count()},
        q{device.upload(qx), device.upload(qy), device.upload(qz), g.count()};
    const FrozenBandDeviceView orbital{device.upload(coefficients),
                                       device.upload(occupation),
                                       int(occupation.size()), state.k_weight};
    const FrozenLocalDeviceView lv{device.upload(local_value),
                                   device.upload(local_slope),
                                   device.upload(structure)};
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
                                    d.n_proj(),
                                    int(columns.size()),
                                    4 * M_PI / std::sqrt(volume)};
    const std::size_t bc = columns.size() * occupation.size();
    FrozenNonlocalDeviceScratch scratch{
        device.allocate<cufftDoubleComplex>(columns.size() * g.count()),
        device.allocate<cufftDoubleComplex>(bc),
        device.allocate<cufftDoubleComplex>(bc),
        device.allocate<cufftDoubleComplex>(bc)};
    for (int repeat = 0; repeat < repeats; ++repeat) {
        ATV::cuda::pw::internal::kinetic_frozen_virial(q, orbital, output,
                                                       device.stream);
        ATV::cuda::pw::internal::hartree_frozen_virial(
            G, density, volume, output + 1, device.stream);
        ATV::cuda::pw::internal::local_potential_frozen_virial(
            G, density, volume, lv, output + 2, device.stream);
        ATV::cuda::pw::internal::nonlocal_frozen_virial(
            nl, orbital, scratch, device.blas, output + 3, device.stream);
    }
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
    return result;
}

void run(bool unequal, bool shifted_k, double amplitude = 1) {
    auto d = data(amplitude);
    if (unequal) {
        d.kb[1].radial_points = 578;
        d.kb[3].radial_points = 733;
        d.kb[5].radial_points = 888;
    }
    auto g = modes();
    if (shifted_k)
        g.kvec = {.11, -.07, .03};
    auto c = wave(g);
    auto rho = density();
    const std::vector<Vec3> r = {{.2, .4, .7}, {1.1, -.3, .5}};
    const double volume = 125;
    const std::array<double, 3> occ = {2., -.17, .43};
    Orbitals o{c, occ, .73};
    const auto terms = run_gpu(d, g, r, rho, o, volume);
    const std::array<Term, 4> cpu = {kinetic(g, o), hartree(g, rho, volume),
                                     local(d, g, r, rho, volume),
                                     nonlocal(d, g, r, o, volume)};
    for (int t = 0; t < 4; ++t) {
        close(terms[t].energy, cpu[t].energy, "device vs CPU raw-fullD energy");
        for (int ab = 0; ab < 9; ++ab)
            close(terms[t].W[ab], cpu[t].W[ab], "device vs CPU full9 tensor");
    }
    const auto negative_k =
        run_gpu(d, g, r, rho, Orbitals{c, occ, -.23}, volume);
    for (int t : {0, 3}) {
        close(negative_k[t].energy, terms[t].energy * (-.23 / .73),
              "signed k weight energy");
        for (int ab = 0; ab < 9; ++ab)
            close(negative_k[t].W[ab], terms[t].W[ab] * (-.23 / .73),
                  "signed k weight W");
    }
    if (!unequal && !shifted_k && amplitude == 1) {
        const auto repeated = run_gpu(d, g, r, rho, o, volume, 2);
        for (int t = 0; t < 4; ++t) {
            close(repeated[t].energy, 2 * terms[t].energy,
                  "caller-owned accumulation energy");
            for (int ab = 0; ab < 9; ++ab)
                close(repeated[t].W[ab], 2 * terms[t].W[ab],
                      "caller-owned accumulation W9");
        }
    }
    const char* names[] = {"kinetic", "hartree", "local", "nonlocal"};
    const std::array<double, 4> values = {
        value_t(g, o), value_h(g, rho, volume),
        value_local(d, g, r, rho, volume), value_nl(d, g, r, o, volume)};
    for (int t = 0; t < 4; ++t) {
        close(terms[t].energy, values[t], "independent operator energy");
        double antisym = 0;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                antisym = std::max(antisym, std::abs(terms[t].W[3 * a + b]
                                                     - terms[t].W[3 * b + a]));
        close(antisym, 0, "unprojected antisymmetric tensor");
        std::printf("TERM %s unequal=%d k=%d E_Ha=%.12g W9=", names[t], unequal,
                    shifted_k, terms[t].energy);
        for (double w : terms[t].W)
            std::printf(" %.12g", w);
        std::printf("\n");
    }
    for (double h : {3e-4, 1e-4, 3e-5, 1e-5, 3e-6}) {
        std::array<double, 4> errors{};
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                std::array<std::array<double, 4>, 2> e{};
                for (int sign = 0; sign < 2; ++sign) {
                    auto f = identity();
                    f(a, b) += sign == 0 ? h : -h;
                    auto gs = transform(g, f);
                    auto rs = transform(r, f);
                    auto rh = rho;
                    for (auto& v : rh)
                        v /= f.det();
                    const double vs = volume * f.det();
                    e[sign] = {value_t(gs, o), value_h(gs, rh, vs),
                               value_local(d, gs, rs, rh, vs),
                               value_nl(d, gs, rs, o, vs)};
                    for (int ig = 0; ig < g.count(); ++ig)
                        for (std::size_t site = 0; site < r.size(); ++site)
                            close((gs.g[ig] + gs.kvec).dot(rs[site]),
                                  (g.g[ig] + g.kvec).dot(r[site]),
                                  "affine atom phase cancellation");
                }
                for (int t = 0; t < 4; ++t) {
                    const double fd = -(e[0][t] - e[1][t]) / (2 * h);
                    errors[t] = std::max(errors[t],
                                         std::abs(fd - terms[t].W[3 * a + b]));
                    if (h <= 1e-5)
                        close(fd, terms[t].W[3 * a + b],
                              "all-nine frozen operator strain derivative");
                }
            }
        std::printf(
            "SWEEP unequal=%d k=%d h=%.4g max_dW_Ha=%.6g %.6g %.6g %.6g\n",
            unequal, shifted_k, h, errors[0], errors[1], errors[2], errors[3]);
    }
    // Rigid Cartesian rotation must leave energies invariant and rotate the full tensor.
    auto rot = identity();
    const double angle = .31;
    rot(0, 0) = rot(1, 1) = std::cos(angle);
    rot(0, 1) = -std::sin(angle);
    rot(1, 0) = std::sin(angle);
    auto gr = transform(g, rot);
    auto rr = transform(r, rot);
    const auto rotated = run_gpu(d, gr, rr, rho, o, volume);
    for (int t = 0; t < 4; ++t) {
        close(rotated[t].energy, terms[t].energy, "rotated energy");
        Mat3 tensor;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                tensor(a, b) = terms[t].W[3 * a + b];
        const auto want = rot * tensor * rot.transpose();
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                close(rotated[t].W[3 * a + b], want(a, b),
                      "rotated unprojected tensor");
    }
    auto dropped = d;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            if (i != j)
                dropped.d_ij[6 * i + j] = 0;
    require(
        std::abs(nonlocal(dropped, g, r, o, volume).energy - terms[3].energy)
            > 1e-5,
        "nonlocal off-diagonal D negative control must be nontrivial");
    auto clipped = occ;
    for (auto& f : clipped)
        f = std::max(f, 0.);
    require(
        std::abs(kinetic(g, Orbitals{c, clipped, .73}).energy - terms[0].energy)
            > 1e-4,
        "signed occupation negative control must be nontrivial");
    std::printf("PASS CUDA_PW_OPERATOR_COMPONENTS unequal=%d k=%d\n", unequal,
                shifted_k);
}
} // namespace

TEST(CudaPWFrozenOperatorVirial, GammaFullDAllNineStrainDerivatives) {
    run(false, false);
}
TEST(CudaPWFrozenOperatorVirial, GammaUnequalRadialPrefixes) {
    run(true, false);
}
TEST(CudaPWFrozenOperatorVirial, NonzeroKFullDAllNineStrainDerivatives) {
    run(false, true);
}
TEST(CudaPWFrozenOperatorVirial, NonzeroKUnequalRadialPrefixes) {
    run(true, true);
}
TEST(CudaPWFrozenOperatorVirial,
     SmallDProjectorGaugeRetainsValueAndAllNineDerivatives) {
    run(true, true, 1e9);
}
TEST(CudaPWFrozenOperatorVirial,
     TinyRadialProjectorGaugeRetainsValueAndAllNineDerivatives) {
    run(false, false, 1e-35);
}
TEST(CudaPWFrozenOperatorVirial, EmptyInputsAndInvalidShapesHaveNoDeviceWork) {
    namespace dev = ATV::cuda::pw::internal;
    EXPECT_NO_THROW(dev::kinetic_frozen_virial({}, {}, nullptr, nullptr));
    EXPECT_NO_THROW(
        dev::hartree_frozen_virial({}, nullptr, 1, nullptr, nullptr));
    EXPECT_NO_THROW(dev::local_potential_frozen_virial({}, nullptr, 1, {},
                                                       nullptr, nullptr));
    EXPECT_NO_THROW(
        dev::nonlocal_frozen_virial({}, {}, {}, nullptr, nullptr, nullptr));
    dev::FrozenReciprocalView invalid{};
    invalid.count = -1;
    EXPECT_THROW(dev::kinetic_frozen_virial(invalid, {}, nullptr, nullptr),
                 std::invalid_argument);
    EXPECT_THROW(dev::hartree_frozen_virial({}, nullptr, 0, nullptr, nullptr),
                 std::invalid_argument);
    dev::FrozenBandDeviceView bands{};
    bands.bands = -1;
    EXPECT_THROW(dev::kinetic_frozen_virial({}, bands, nullptr, nullptr),
                 std::invalid_argument);
}

TEST(CudaPWFrozenOperatorVirial,
     BorrowedBlasMetadataMismatchPreservesOutputAndHandle) {
    namespace dev = ATV::cuda::pw::internal;
    // Host sources outlive the resources' exception-path stream drain.
    std::vector<double> scalar{1}, position{0, 0, 0};
    std::vector<cufftDoubleComplex> coefficient{{1, 0}};
    std::vector<dev::FrozenProjectorColumn> columns{{0, 0, 0, 0, 0, 1}};
    std::vector<dev::FrozenOperatorVirial> sentinel(1);
    sentinel[0].energy_ha = 11;
    for (int i = 0; i < 9; ++i)
        sentinel[0].virial_ha[i] = 12 + i;
    dev::FrozenOperatorVirial result{};
    DeviceResources device;
    DeviceResources other;
    auto* one = device.upload(scalar);
    dev::FrozenReciprocalView q{one, one, one, 1};
    dev::FrozenBandDeviceView state{device.upload(coefficient), one, 1, 1};
    dev::FrozenNonlocalDeviceGeometry geometry{
        q, one, one, one, device.upload(position), device.upload(columns),
        1, 1,   1};
    dev::FrozenNonlocalDeviceScratch scratch{
        device.allocate<cufftDoubleComplex>(1),
        device.allocate<cufftDoubleComplex>(1),
        device.allocate<cufftDoubleComplex>(1),
        device.allocate<cufftDoubleComplex>(1)};
    auto* output = device.upload(sentinel);
    EXPECT_THROW(dev::nonlocal_frozen_virial(geometry, state, scratch,
                                             device.blas, output, other.stream),
                 std::invalid_argument);
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cublasGetStream(device.blas, &stream), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(stream, device.stream);
    ASSERT_EQ(cublasSetPointerMode(device.blas, CUBLAS_POINTER_MODE_DEVICE),
              CUBLAS_STATUS_SUCCESS);
    EXPECT_THROW(dev::nonlocal_frozen_virial(geometry, state, scratch,
                                             device.blas, output,
                                             device.stream),
                 std::invalid_argument);
    cublasPointerMode_t mode;
    ASSERT_EQ(cublasGetPointerMode(device.blas, &mode), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(mode, CUBLAS_POINTER_MODE_DEVICE);
    ASSERT_EQ(cudaMemcpyAsync(&result, output, sizeof(result),
                              cudaMemcpyDeviceToHost, device.stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(device.stream), cudaSuccess);
    EXPECT_EQ(result.energy_ha, sentinel[0].energy_ha);
    for (int i = 0; i < 9; ++i)
        EXPECT_EQ(result.virial_ha[i], sentinel[0].virial_ha[i]);
}
