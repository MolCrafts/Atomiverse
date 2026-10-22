/// @file tests/cuda/longrange/test_kspace_virial.cu
/// @brief cuda::KSpaceEwald and cuda::KSpacePME under a changing cell:
///        reciprocal virial against central energy differences at fixed
///        fractional coordinates, and the refreshed device tables against
///        independent rebuilds.
///
/// Virial convention: W_ab = −∂E/∂ε_ab for the symmetric strain
/// r' = (I + ε) r, cell' = (I + ε) cell (lattice vectors are the cell
/// columns), α and the mode set (Ewald) / mesh and order (PME) fixed.
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/system_scalars.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/longrange/ewald.cuh"
#ifdef ATV_HAS_KSPACE
#include "atomiverse/cuda/longrange/pme.cuh"
#endif

using namespace ATV;

namespace {

constexpr double kAlpha = 0.45;
constexpr int kOrder = 5;
constexpr int kAtoms = 12;
constexpr bool kFloatMesh = std::is_same_v<cuda::force_t, float>;

struct Result {
    double energy = 0;
    double virial[6] = {};
    std::vector<double> force; ///< x, y, z planes.
};

// Ewald accuracy tight enough that a rebuild on a strained cell, which may
// pick a slightly different mode set, agrees with the followed cell.
cuda::KSpaceEwald make_solver(cuda::KSpaceEwald*) {
    return cuda::KSpaceEwald{kAlpha, 1.0e-12};
}
#ifdef ATV_HAS_KSPACE
cuda::KSpacePME make_solver(cuda::KSpacePME*) {
    return cuda::KSpacePME{kAlpha, 1.0, kOrder};
}
#endif

/// @brief One reciprocal solver driven directly, with a device-resident Box
///        the test publishes like the backend does (cell + revision).
template <class Solver>
class Harness {
  public:
    Harness(const std::array<double, 9>& cell, bool neutral) {
        host_.set_nlocal(kAtoms);
        host_.box.set_cell(to_mat(cell));
        // Fixed fractional coordinates; deterministic, well separated.
        for (int i = 0; i < kAtoms; ++i) {
            frac_[i] = {std::fmod(0.137 + 0.618034 * i, 1.0),
                        std::fmod(0.291 + 0.414214 * i, 1.0),
                        std::fmod(0.533 + 0.732051 * i, 1.0)};
            host_.span(kAtom_charge)[i] =
                (i % 2 ? -0.8 : 0.8) + (neutral ? 0.0 : 0.05 * (i % 3));
            host_.span(kAtom_mass)[i] = 1;
            host_.span(kAtom_tag)[i] = i;
            host_.span(kAtom_type)[i] = 0;
        }
        host_.rebuild_tag_index(0, kAtoms);
        place(cell);
        device_ = cpu::to_cuda(host_);
        EXPECT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&resident_, sizeof(cuda::Box)), cudaSuccess);
        ctx_.stream = stream_;
        publish(cell);
        solver_.build({ctx_, device_});
    }
    ~Harness() {
        solver_.destroy({ctx_, device_});
        cudaFree(resident_);
        cudaStreamDestroy(stream_);
        cuda::destroy(device_);
    }

    /// @brief Change the cell keeping fractional coordinates (no rebuild).
    void set_cell(const std::array<double, 9>& cell) {
        place(cell);
        std::vector<cuda::pos_t> hi(kAtoms), lo(kAtoms);
        const ColKey<StorageReal>* keys[3] = {&kAtom_x, &kAtom_y, &kAtom_z};
        cuda::pos_t* primary[3] = {device_.atoms.x, device_.atoms.y,
                                   device_.atoms.z};
        cuda::pos_t* correction[3] = {
            device_.atoms.x_corr, device_.atoms.y_corr, device_.atoms.z_corr};
        for (int a = 0; a < 3; ++a) {
            for (int i = 0; i < kAtoms; ++i)
                cuda::split_position(host_.span(*keys[a])[i], hi[i], lo[i]);
            ASSERT_EQ(cudaMemcpy(primary[a], hi.data(),
                                 kAtoms * sizeof(cuda::pos_t),
                                 cudaMemcpyHostToDevice),
                      cudaSuccess);
            if (correction[a])
                ASSERT_EQ(cudaMemcpy(correction[a], lo.data(),
                                     kAtoms * sizeof(cuda::pos_t),
                                     cudaMemcpyHostToDevice),
                          cudaSuccess);
        }
        publish(cell);
    }

    /// @brief Replay the captured reciprocal pipeline once.
    Result evaluate() {
        EXPECT_EQ(cudaMemset(device_.scalars, 0, sizeof(SystemScalars)),
                  cudaSuccess);
        for (auto* f : {device_.atoms.fx, device_.atoms.fy, device_.atoms.fz})
            EXPECT_EQ(cudaMemset(f, 0, kAtoms * sizeof(cuda::fixed_t)),
                      cudaSuccess);
        solver_.launch({ctx_, device_, 0, 0});
        EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        SystemScalars scalars{};
        EXPECT_EQ(cudaMemcpy(&scalars, device_.scalars, sizeof(scalars),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        Result r;
        r.energy = scalars.energy;
        for (int k = 0; k < 6; ++k)
            r.virial[k] = scalars.stress[k];
        std::vector<cuda::fixed_t> words(kAtoms);
        for (auto* f : {device_.atoms.fx, device_.atoms.fy, device_.atoms.fz}) {
            EXPECT_EQ(cudaMemcpy(words.data(), f,
                                 kAtoms * sizeof(cuda::fixed_t),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            for (auto w : words)
                r.force.push_back(cuda::decode_force<double>(w));
        }
        return r;
    }

    /// @brief Build-time discretisation that a rebuild must reproduce.
    std::array<int, 3> mesh() const {
        if constexpr (std::is_same_v<Solver, cuda::KSpaceEwald>)
            return {0, 0, 0};
        else
            return solver_.mesh();
    }

  private:
    static Mat3 to_mat(const std::array<double, 9>& c) {
        Mat3 m;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m(i, j) = c[3 * i + j];
        return m;
    }
    void place(const std::array<double, 9>& cell) {
        for (int i = 0; i < kAtoms; ++i)
            for (int a = 0; a < 3; ++a) {
                double r = 0;
                for (int k = 0; k < 3; ++k)
                    r += cell[3 * a + k] * frac_[i][k];
                const ColKey<StorageReal>& key =
                    a == 0 ? kAtom_x : (a == 1 ? kAtom_y : kAtom_z);
                host_.span(key)[i] = r;
            }
    }
    void publish(const std::array<double, 9>& cell) {
        cuda::Box box{};
        box.assign_cell(cell.data());
        box.revision = ++revision_;
        ASSERT_EQ(
            cudaMemcpy(resident_, &box, sizeof(box), cudaMemcpyHostToDevice),
            cudaSuccess);
        box.resident = resident_;
        ctx_.host_box = box;
    }

    std::array<std::array<double, 3>, kAtoms> frac_{};
    cpu::Frame host_{kAtoms};
    cuda::Frame device_{};
    cudaStream_t stream_ = nullptr;
    cuda::Box* resident_ = nullptr;
    unsigned long long revision_ = 0;
    cuda::Context ctx_{};
    Solver solver_ = make_solver(static_cast<Solver*>(nullptr));
};

std::array<double, 9> strained(const std::array<double, 9>& cell, int a, int b,
                               double h) {
    double eps[3][3] = {};
    eps[a][b] += h;
    if (a != b)
        eps[b][a] += h;
    std::array<double, 9> out{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            out[3 * i + j] = cell[3 * i + j];
            for (int k = 0; k < 3; ++k)
                out[3 * i + j] += eps[i][k] * cell[3 * k + j];
        }
    return out;
}

template <class Solver>
void check_virial(const std::array<double, 9>& cell, bool neutral,
                  const char* label) {
    Harness<Solver> solver(cell, neutral);
    const Result base = solver.evaluate();
    // Fourth-order central difference. In FP32 modes the stored primary
    // coordinates are rounded independently at every strained cell, which
    // perturbs the fractional coordinates by ~1e-7 relative and the energy by
    // ~1e-5 eV; a wide step keeps the difference above that while the O(h⁴)
    // truncation stays negligible. FP64 checks the virial to 1e-7.
    const double h = kFloatMesh ? 1.0e-2 : 1.0e-4;
    constexpr int pairs[6][2] = {{0, 0}, {1, 1}, {2, 2},
                                 {0, 1}, {0, 2}, {1, 2}};
    double scale = 0;
    for (double w : base.virial)
        scale = std::max(scale, std::abs(w));
    const double tolerance = (kFloatMesh ? 1.0e-3 : 1.0e-7) * scale;
    for (int k = 0; k < 6; ++k) {
        const int a = pairs[k][0], b = pairs[k][1];
        auto energy = [&](double step) {
            solver.set_cell(strained(cell, a, b, step));
            return solver.evaluate().energy;
        };
        const double d1 = energy(h) - energy(-h);
        const double d2 = energy(2 * h) - energy(-2 * h);
        const double fd = -(8 * d1 - d2) / (12 * h) / (a == b ? 1.0 : 2.0);
        std::printf("%s mesh=%dx%dx%d W[%d]=%.12g fd=%.12g\n", label,
                    solver.mesh()[0], solver.mesh()[1], solver.mesh()[2], k,
                    base.virial[k], fd);
        EXPECT_NEAR(base.virial[k], fd, tolerance)
            << label << " component " << k;
    }
}

const std::array<double, 9> kEvenOrtho = {11.5, 0, 0, 0, 11.5, 0, 0, 0, 15.5};
const std::array<double, 9> kOddTriclinic = {14.5, 0,    0,    1.25, 12.5,
                                             0,    0.75, -1.0, 14.5};
const std::array<double, 9> kEvenTriclinic = {11.5, 0,    0,   1.5, 11.5,
                                              0,    -0.5, 1.0, 15.5};

template <class Solver>
void check_changed_cell() {
    Harness<Solver> continuous(kEvenTriclinic, false);
    const Result first = continuous.evaluate();
    const auto other =
        strained(strained(kEvenTriclinic, 0, 0, 0.01), 1, 2, -0.004);
    continuous.set_cell(other);
    const Result changed = continuous.evaluate();
    Harness<Solver> rebuilt(other, false);
    ASSERT_EQ(rebuilt.mesh(), continuous.mesh());
    const Result fresh = rebuilt.evaluate();
    const double tol = kFloatMesh ? 1.0e-5 : 1.0e-10;
    auto expect_same = [&](const Result& a, const Result& b, const char* what) {
        EXPECT_NEAR(a.energy, b.energy, tol * std::abs(b.energy)) << what;
        for (int k = 0; k < 6; ++k)
            EXPECT_NEAR(a.virial[k], b.virial[k],
                        tol * (std::abs(b.virial[k]) + std::abs(b.energy)))
                << what << " virial " << k;
        for (std::size_t i = 0; i < a.force.size(); ++i)
            EXPECT_NEAR(a.force[i], b.force[i], tol * 10 + 1.0e-9)
                << what << " force " << i;
    };
    expect_same(changed, fresh, "changed cell vs rebuild");
    EXPECT_GT(std::abs(changed.energy - first.energy), 1.0e-6);
    continuous.set_cell(kEvenTriclinic); // restore: tables refresh back
    expect_same(continuous.evaluate(), first, "restored cell");
    // Replaying with the same revision reuses the tables unchanged.
    expect_same(continuous.evaluate(), first, "graph replay");
}

} // namespace

TEST(CudaEwaldVirial, OrthorhombicNeutral) {
    check_virial<cuda::KSpaceEwald>(kEvenOrtho, true, "ewald-ortho-neutral");
}

TEST(CudaEwaldVirial, TriclinicCharged) {
    check_virial<cuda::KSpaceEwald>(kOddTriclinic, false,
                                    "ewald-triclinic-charged");
}

TEST(CudaEwaldVirial, ChangedCellMatchesIndependentBuildAndRestores) {
    check_changed_cell<cuda::KSpaceEwald>();
}

#ifdef ATV_HAS_KSPACE
TEST(CudaPmeVirial, EvenOrthorhombicNeutral) {
    check_virial<cuda::KSpacePME>(kEvenOrtho, true, "even-ortho-neutral");
}

TEST(CudaPmeVirial, OddTriclinicCharged) {
    check_virial<cuda::KSpacePME>(kOddTriclinic, false,
                                  "odd-triclinic-charged");
}

TEST(CudaPmeVirial, EvenTriclinicNyquistCharged) {
    check_virial<cuda::KSpacePME>(kEvenTriclinic, false,
                                  "even-triclinic-charged");
}

TEST(CudaPmeVirial, ChangedCellMatchesIndependentBuildAndRestores) {
    check_changed_cell<cuda::KSpacePME>();
}
#endif
