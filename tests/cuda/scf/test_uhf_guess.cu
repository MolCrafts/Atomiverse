/// @file cuda/scf/test_uhf_guess.cu
/// @brief Matrix-level MINAO projection checks; no SCF or convergence loop.
#ifdef ATV_HAS_HF

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/minao.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/scf/internal/uhf_guess.cuh"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using namespace ATV;

cpu::Frame geometry(const std::vector<int>& z,
                    const std::vector<double>& bohr) {
    cpu::Frame frame(z.size());
    for (std::size_t a = 0; a < z.size(); ++a) {
        frame.span(kAtom_type)[a] = z[a];
        frame.span(kAtom_x)[a] = bohr[a * 3] / units::kAngstromInBohr;
        frame.span(kAtom_y)[a] = bohr[a * 3 + 1] / units::kAngstromInBohr;
        frame.span(kAtom_z)[a] = bohr[a * 3 + 2] / units::kAngstromInBohr;
    }
    frame.box = cpu::Box::cubic(100);
    return frame;
}

struct DeviceInput {
    cuda::scf::SCFWorkspace ws;
    double* beta = nullptr;
    cudaStream_t stream = nullptr;
    std::vector<void*> allocations;

    explicit DeviceInput(const cpu::scf::internal::IntegralWorkspace& cpu) {
        ATV_CUDA_CHECK(cudaStreamCreate(&stream));
        ws.n = cpu.n_basis;
        ws.natm = static_cast<int>(cpu.atoms.size());
        ws.n_prim = 3;
        const int n = ws.n;
        std::vector<double> alpha(n * 3), coefficient(n * 3);
        std::vector<int> angular(n), ao(n), cart(n);
        for (int i = 0; i < n; ++i) {
            const auto& shell = cpu.shells[i];
            if (shell.n_prim != 3)
                throw std::logic_error("MINAO unit fixture needs STO-3G");
            angular[i] = shell.l;
            ao[i] = i;
            cart[i] = shell.cart;
            for (int p = 0; p < 3; ++p) {
                alpha[i * 3 + p] = shell.alpha[p];
                coefficient[i * 3 + p] = shell.coeff[p];
            }
        }
        upload(ws.d_shell_alpha, alpha);
        upload(ws.d_shell_coeff, coefficient);
        upload(ws.d_shell_L, angular);
        upload(ws.d_ao_shell, ao);
        upload(ws.d_ao_cart, cart);
        allocate(ws.S, n * n);
        allocate(ws.P, n * n);
        allocate(beta, n * n);
        allocate(ws.d_shell_center, n * 3);
        allocate(ws.d_atom_pos, cpu.atoms.size() * 3);
        refresh(cpu);
    }

    ~DeviceInput() {
        cudaStreamSynchronize(stream);
        for (void* ptr : allocations)
            cudaFree(ptr);
        cudaStreamDestroy(stream);
    }

    template <class T>
    void allocate(T*& dst, std::size_t count) {
        ATV_CUDA_CHECK(cudaMalloc(&dst, count * sizeof(T)));
        allocations.push_back(dst);
    }

    template <class T>
    void upload(T*& dst, const std::vector<T>& values) {
        allocate(dst, values.size());
        ATV_CUDA_CHECK(cudaMemcpyAsync(dst, values.data(),
                                       values.size() * sizeof(T),
                                       cudaMemcpyHostToDevice, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void refresh(const cpu::scf::internal::IntegralWorkspace& cpu) {
        const int n = ws.n;
        std::vector<double> S(n * n), centers(n * 3), pos(cpu.atoms.size() * 3);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                S[i + j * n] = cpu.S(i, j);
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < 3; ++d)
                centers[i * 3 + d] = cpu.shells[i].center[d];
        for (std::size_t a = 0; a < cpu.atoms.size(); ++a)
            for (int d = 0; d < 3; ++d)
                pos[a * 3 + d] = cpu.atoms[a].pos[d];
        for (const auto& item : {std::make_pair(ws.S, &S),
                                 std::make_pair(ws.d_shell_center, &centers),
                                 std::make_pair(ws.d_atom_pos, &pos)})
            ATV_CUDA_CHECK(cudaMemcpyAsync(item.first, item.second->data(),
                                           item.second->size() * sizeof(double),
                                           cudaMemcpyHostToDevice, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    std::vector<double> read(const double* ptr) const {
        std::vector<double> values(ws.n * ws.n);
        ATV_CUDA_CHECK(cudaMemcpyAsync(values.data(), ptr,
                                       values.size() * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        return values;
    }
};

void expect_projection(DeviceInput& input,
                       cuda::scf::internal::UhfMinaoGuess& guess,
                       const cpu::scf::internal::IntegralWorkspace& cpu) {
    const auto reference = cpu::scf::build_minao_density(cpu);
    guess.seed(input.ws, input.beta, input.stream);
    const auto alpha = input.read(input.ws.P), beta = input.read(input.beta);
    ASSERT_EQ(alpha, beta);
    for (int j = 0; j < input.ws.n; ++j)
        for (int i = 0; i < input.ws.n; ++i)
            EXPECT_NEAR(alpha[i + j * input.ws.n], 0.5 * reference(i, j), 2e-12)
                << "row=" << i << " col=" << j;
}

TEST(CudaUhfGuess, SupportedMinaoElementsMatchCpuProjection) {
    // The target STO-3G tables and MINAO tables intersect at H/C/N/O.
    // MINAO also provides F, but the target basis has no F entry.
    for (int z : {1, 6, 7, 8}) {
        const auto frame = geometry({z}, {0.23, -0.71, 0.49});
        const cpu::scf::internal::IntegralWorkspace cpu{
            frame, cpu::scf::BasisSet::STO3G};
        DeviceInput input(cpu);
        cuda::scf::internal::UhfMinaoGuess guess(&z, 1);
        guess.build(input.ws, input.stream);
        expect_projection(input, guess, cpu);
    }
}

TEST(CudaUhfGuess, ReusesSetupForChangedGeometryAndNeutralAtomicOccupations) {
    for (const std::vector<int> z : {std::vector<int>{8, 1}, {8, 8}}) {
        auto frame = geometry(z, {0.13, -0.21, 0.37, 1.62, 0.49, -1.13});
        cpu::scf::internal::IntegralWorkspace cpu{
            frame, cpu::scf::BasisSet::STO3G, /*charge=*/1};
        DeviceInput input(cpu);
        cuda::scf::internal::UhfMinaoGuess guess(z.data(), z.size());
        guess.build(input.ws, input.stream);
        expect_projection(input, guess, cpu);
        const auto first = input.read(input.ws.P);
        // A common translation changes device coordinates, but no seed entry.
        for (std::size_t a = 0; a < z.size(); ++a) {
            frame.span(kAtom_x)[a] += 0.37;
            frame.span(kAtom_y)[a] -= 0.19;
        }
        cpu = cpu::scf::internal::IntegralWorkspace{
            frame, cpu::scf::BasisSet::STO3G, 1};
        input.refresh(cpu);
        expect_projection(input, guess, cpu);
        const auto translated = input.read(input.ws.P);
        for (std::size_t i = 0; i < first.size(); ++i)
            EXPECT_NEAR(first[i], translated[i], 2e-12);
        // A bond displacement must recompute the geometry-dependent projector.
        frame.span(kAtom_z)[1] += 0.11;
        cpu = cpu::scf::internal::IntegralWorkspace{
            frame, cpu::scf::BasisSet::STO3G, 1};
        input.refresh(cpu);
        expect_projection(input, guess, cpu);
        const auto displaced = input.read(input.ws.P);
        double difference = 0;
        for (std::size_t i = 0; i < first.size(); ++i)
            difference += std::abs(first[i] - displaced[i]);
        EXPECT_GT(difference, 1e-5);
    }
}

TEST(CudaUhfGuess, SingularOverlapRejectsWithoutPublishingDensity) {
    const int z = 1;
    const auto frame = geometry({z}, {0, 0, 0});
    const cpu::scf::internal::IntegralWorkspace cpu{frame,
                                                    cpu::scf::BasisSet::STO3G};
    DeviceInput input(cpu);
    cuda::scf::internal::UhfMinaoGuess guess(&z, 1);
    guess.build(input.ws, input.stream);
    ATV_CUDA_CHECK(
        cudaMemsetAsync(input.ws.S, 0, sizeof(double), input.stream));
    ATV_CUDA_CHECK(
        cudaMemsetAsync(input.ws.P, 0, sizeof(double), input.stream));
    ATV_CUDA_CHECK(
        cudaMemsetAsync(input.beta, 0, sizeof(double), input.stream));
    EXPECT_THROW(guess.seed(input.ws, input.beta, input.stream),
                 std::runtime_error);
    EXPECT_EQ(input.read(input.ws.P)[0], 0);
    EXPECT_EQ(input.read(input.beta)[0], 0);
    input.refresh(cpu);
    EXPECT_NO_THROW(guess.seed(input.ws, input.beta, input.stream));
    expect_projection(input, guess, cpu);
}

TEST(CudaUhfGuess, UnsupportedElementKeepsCoreHamiltonianStart) {
    const int z = 2;
    const auto frame = geometry({z}, {0, 0, 0});
    const cpu::scf::internal::IntegralWorkspace cpu{frame,
                                                    cpu::scf::BasisSet::STO3G};
    DeviceInput input(cpu);
    cuda::scf::internal::UhfMinaoGuess guess(&z, 1);
    guess.build(input.ws, input.stream);
    ATV_CUDA_CHECK(
        cudaMemsetAsync(input.ws.P, 0, sizeof(double), input.stream));
    ATV_CUDA_CHECK(
        cudaMemsetAsync(input.beta, 0, sizeof(double), input.stream));
    EXPECT_NO_THROW(guess.seed(input.ws, input.beta, input.stream));
    EXPECT_EQ(input.read(input.ws.P)[0], 0);
    EXPECT_EQ(input.read(input.beta)[0], 0);
}

} // namespace
#endif
