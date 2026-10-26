/// @file test_eigen_solver.cu
/// @brief Single-method Jacobi eigensolver tests against analytic spectra.
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/scf/internal/eigen_solver.cuh"

namespace {
class CudaScfEigenSolver : public ::testing::Test {
  protected:
    int n = 33;
    ATV::cuda::Frame frame;
    ATV::cuda::scf::SCFWorkspace ws;
    ATV::cuda::scf::internal::EigenSolver solver;
    cudaStream_t stream = nullptr;

    void SetUp() override {
        ATV_CUDA_CHECK(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ws.n = n;
        ATV_CUDA_CHECK(cudaMalloc(&ws.H, n * n * sizeof(double)));
        ATV_CUDA_CHECK(cudaMalloc(&ws.C, n * n * sizeof(double)));
        ATV_CUDA_CHECK(cudaMalloc(&ws.eig, n * sizeof(double)));
        solver.build({stream, frame, 0}, ws);
        std::printf("CUDA eigen setup n=%d device_workspace_bytes=%zu "
                    "host_workspace_bytes=%zu\n",
                    n, solver.device_workspace_bytes(),
                    solver.host_workspace_bytes());
    }
    void TearDown() override {
        cudaStreamSynchronize(stream);
        solver.destroy({stream, frame, 0}, ws);
        cudaFree(ws.H);
        cudaFree(ws.C);
        cudaFree(ws.eig);
        cudaStreamDestroy(stream);
    }

    void check(const std::vector<double>& matrix,
               const std::vector<double>& expected, double input_scale = 1) {
        std::vector<double> scaled(matrix);
        for (double& value : scaled)
            value *= input_scale;
        ATV_CUDA_CHECK(cudaMemcpyAsync(ws.H, scaled.data(),
                                       n * n * sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
        solver.step({stream, frame, 0}, ws);
        std::vector<double> eig(n), vectors(n * n);
        ATV_CUDA_CHECK(cudaMemcpyAsync(eig.data(), ws.eig, n * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaMemcpyAsync(vectors.data(), ws.C,
                                       n * n * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
        ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int j = 0; j < n; ++j) {
            const double lambda = eig[j] / input_scale;
            EXPECT_NEAR(lambda, expected[j], 5e-13) << "eigenvalue " << j;
            for (int i = 0; i < n; ++i) {
                double av = 0, overlap = 0;
                for (int k = 0; k < n; ++k) {
                    av += matrix[i + k * n] * vectors[k + j * n];
                    overlap += vectors[k + i * n] * vectors[k + j * n];
                }
                EXPECT_NEAR(av, lambda * vectors[i + j * n], 5e-13)
                    << "residual " << i << ',' << j;
                EXPECT_NEAR(overlap, i == j ? 1.0 : 0.0, 5e-13)
                    << "orthogonality " << i << ',' << j;
            }
        }
    }
    void check_core_valence_spectrum() {
        // An analytic orthogonal similarity makes a dense Fock-like matrix:
        // core levels ~-30 Ha, valence ~-1 Ha, repeated virtual levels >0.
        // Q=I-2vv^T/(v^Tv), A=Q diag(expected) Q^T, so the exact spectrum
        // is prescribed without another eigensolver or a saved ATV result.
        std::vector<double> matrix(n * n, 0), expected(n), v(n);
        double ss = 0, weighted_ss = 0;
        for (int i = 0; i < n; ++i) {
            expected[i] =
                i < n / 5   ? -30.0 + 0.5 * (i / 2)
                : i < n / 2 ? -1.0 + (n > 64 ? 0.01 : 0.05) * ((i - n / 5) / 2)
                            : 0.2 + 0.01 * ((i - n / 2) / 3);
            v[i] = std::sin(double(i + 1));
            ss += v[i] * v[i];
            weighted_ss += expected[i] * v[i] * v[i];
        }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                matrix[i + j * n] =
                    (i == j ? expected[i] : 0)
                    - 2 * v[i] * v[j] * (expected[i] + expected[j]) / ss
                    + 4 * weighted_ss * v[i] * v[j] / (ss * ss);
        check(matrix, expected);
    }
    void check_tridiagonal(double input_scale = 1) {
        std::vector<double> matrix(n * n, 0), expected(n);
        const double pi = std::acos(-1.0);
        for (int i = 0; i < n; ++i) {
            matrix[i + i * n] = 2;
            if (i + 1 < n) {
                matrix[i + (i + 1) * n] = -1;
                matrix[i + 1 + i * n] = -1;
            }
            expected[i] = 2 - 2 * std::cos((i + 1) * pi / (n + 1));
        }
        check(matrix, expected, input_scale);
    }
};

TEST_F(CudaScfEigenSolver, TridiagonalAnalyticSpectrumAndEigenvectors) {
    check_tridiagonal();
}

class CudaScfEigenSolverLimit : public CudaScfEigenSolver {
  public:
    CudaScfEigenSolverLimit() { n = 215; }
};

class CudaScfEigenSolverWater16 : public CudaScfEigenSolver {
  public:
    CudaScfEigenSolverWater16() { n = 112; }
};

TEST_F(CudaScfEigenSolverWater16, DenseCoreValenceAndRepeatedSpectrum) {
    check_core_valence_spectrum();
}

TEST_F(CudaScfEigenSolverWater16, HugeAndTinyFiniteScale) {
    check_tridiagonal(1e150);
    check_tridiagonal(1e-150);
}

class CudaScfEigenSolverSharedEdge : public CudaScfEigenSolver {
  public:
    CudaScfEigenSolverSharedEdge() { n = 116; }
};

TEST_F(CudaScfEigenSolverSharedEdge, AnalyticSpectrumNearSharedCapacity) {
    check_tridiagonal();
}

class CudaScfEigenSolverSharedNext : public CudaScfEigenSolver {
  public:
    CudaScfEigenSolverSharedNext() { n = 117; }
};

TEST_F(CudaScfEigenSolverSharedNext, AnalyticSpectrumNearSharedCapacity) {
    check_tridiagonal();
}

TEST_F(CudaScfEigenSolverLimit, DenseHfIndexLimitSpectrumAndEigenvectors) {
    check_tridiagonal();
}

class CudaScfEigenSolverSyntheticLegacy : public CudaScfEigenSolver {
  public:
    CudaScfEigenSolverSyntheticLegacy() { n = 216; }
};

TEST_F(CudaScfEigenSolverSyntheticLegacy,
       GenericOrderBeyondMolecularDenseLimit) {
    // Molecular HF rejects n>215 before this helper. The independent generic
    // synthetic SCF helper retains its larger-matrix library contract; this
    // test makes no CUDA-only residency claim for that legacy branch.
    check_tridiagonal();
}

class CudaScfEigenSolverUhfScale : public CudaScfEigenSolver {
  public:
    CudaScfEigenSolverUhfScale() { n = 10; }
};

TEST_F(CudaScfEigenSolverUhfScale, DenseCoreValenceAndRepeatedVirtualSpectrum) {
    check_core_valence_spectrum();
}

class CudaScfEigenSolverRhfScale : public CudaScfEigenSolver {
  public:
    CudaScfEigenSolverRhfScale() { n = 43; }
};

TEST_F(CudaScfEigenSolverRhfScale, DenseCoreValenceAndRepeatedVirtualSpectrum) {
    check_core_valence_spectrum();
}

class CudaScfEigenSolverLibraryBoundary : public CudaScfEigenSolver {
  public:
    CudaScfEigenSolverLibraryBoundary() { n = 64; }
};

TEST_F(CudaScfEigenSolverLibraryBoundary, AnalyticSpectrumAtLibraryBoundary) {
    check_tridiagonal();
}

class CudaScfEigenSolverDeviceBoundary : public CudaScfEigenSolver {
  public:
    CudaScfEigenSolverDeviceBoundary() { n = 65; }
};

TEST_F(CudaScfEigenSolverDeviceBoundary, AnalyticSpectrumAtDeviceBoundary) {
    check_tridiagonal();
}

TEST_F(CudaScfEigenSolverDeviceBoundary, DenseCoreValenceAndRepeatedSpectrum) {
    check_core_valence_spectrum();
}

TEST_F(CudaScfEigenSolverLimit, DenseCoreValenceAndRepeatedSpectrum) {
    check_core_valence_spectrum();
}

TEST_F(CudaScfEigenSolverDeviceBoundary, HugeAndTinyFiniteScale) {
    // Compare the same normalized analytic eigenpair gates after restoring
    // units. The scaled inputs exercise overflow/underflow-safe norm control.
    check_tridiagonal(1e150);
    check_tridiagonal(1e-150);
}

TEST_F(CudaScfEigenSolverDeviceBoundary, RejectsNonfiniteInput) {
    std::vector<double> matrix(n * n, 0);
    matrix[0] = std::numeric_limits<double>::quiet_NaN();
    ATV_CUDA_CHECK(cudaMemcpyAsync(ws.H, matrix.data(), n * n * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    EXPECT_THROW(solver.step({stream, frame, 0}, ws), std::runtime_error);
}

TEST_F(CudaScfEigenSolver, RepeatedEigenvaluesRemainOrthonormal) {
    std::vector<double> matrix(n * n, 0), expected(n);
    for (int i = 0; i < n; ++i) {
        expected[i] = i / 3;
        matrix[i + i * n] = expected[i];
    }
    check(matrix, expected);
    // A second solve reuses the same configured library state and scratch.
    check(matrix, expected);
}

TEST_F(CudaScfEigenSolver, RejectsChangedWorkspaceBeforeSolverDispatch) {
    ws.n = n + 1;
    EXPECT_THROW(solver.step({stream, frame, 0}, ws), std::logic_error);
    ws.n = n;
}
} // namespace
