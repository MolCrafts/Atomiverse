/// @file examples/01_scf.cpp
/// @brief Standalone SCF introduction example.
///
/// Shows the complete SCF algorithm in xtensor code:
///   1. Build a deterministic Hamiltonian H0
///   2. Iterate: H = H0 + alpha*P, diagonalize, rebuild P, check convergence
///   3. Print per-iteration energy and residual
///
/// Build (via CMake):
///   cmake --build build --target 01_scf
///
/// Run:
///   ./build/examples/01_scf

#include <cmath>
#include <iomanip>
#include <iostream>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/reducers/xnorm.hpp>
#include <xtensor/views/xview.hpp>

int main() {
    // ── Parameters ────────────────────────────────────────────────────
    int n = 16;         // matrix dimension
    int n_occ = 4;      // occupied orbitals
    double alpha = 0.5; // density coupling
    double tol = 1e-10; // convergence tolerance
    int max_iter = 100;

    std::cout << "=== SCF Introduction ===\n"
              << "n=" << n << ", n_occ=" << n_occ << ", alpha=" << std::fixed
              << std::setprecision(2) << alpha << ", tol=" << std::scientific
              << std::setprecision(1) << tol << "\n\n";

    // ── Build H0: deterministic symmetric positive-definite ───────────
    //
    //   H0[i,i] = n + 1 - i          (diagonal, ensures positive-definite)
    //   H0[i,j] = 1 / (1 + |i-j|)   (off-diagonal, decaying coupling)
    //
    // Same construction as the CUDA Hamiltonian component.
    xt::xtensor<double, 2> H0 = xt::zeros<double>({n, n});
    for (int i = 0; i < n; ++i) {
        H0(i, i) = double(n + 1 - i);
        for (int j = i + 1; j < n; ++j) {
            double val = 1.0 / (1.0 + (j - i));
            H0(i, j) = val;
            H0(j, i) = val;
        }
    }

    // ── SCF loop ──────────────────────────────────────────────────────
    //
    // Each iteration mirrors one "step" through the CUDA SCF phases:
    //   build_hamiltonian -> diagonalize -> build_density -> convergence_check
    //
    xt::xtensor<double, 2> P = xt::zeros<double>({n, n});
    xt::xtensor<double, 2> P_old = xt::zeros<double>({n, n});

    std::cout << std::setw(5) << "Iter" << std::setw(17) << "Energy"
              << std::setw(13) << "Residual" << "\n";

    bool converged = false;
    double final_energy = 0;
    int final_iter = 0;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Phase: build_hamiltonian
        //   H = H0 + alpha * P
        xt::xtensor<double, 2> H = H0 + alpha * P;

        // Phase: diagonalize
        //   xt::linalg::eigh() calls LAPACK dsyev under the hood.
        //   Returns eigenvalues (ascending) and eigenvectors (columns).
        auto [eig, C] = xt::linalg::eigh(H);

        // Phase: build_density
        //   P = C_occ * C_occ^T  (first n_occ columns of C)
        P_old = P;
        auto C_occ = xt::view(C, xt::all(), xt::range(0, n_occ));
        P = xt::linalg::dot(C_occ, xt::transpose(C_occ));

        // Phase: convergence_check
        double residual = xt::norm_l2(P - P_old)();
        double energy = xt::sum(H * P)();

        std::cout << std::setw(5) << iter << std::fixed << std::setprecision(10)
                  << std::setw(17) << energy << std::scientific
                  << std::setprecision(2) << std::setw(13) << residual << "\n";

        final_energy = energy;
        final_iter = iter;

        if (iter > 0 && residual < tol) {
            converged = true;
            std::cout << "\nConverged at iteration " << iter << "\n";

            // Print occupied eigenvalues
            std::cout << "Occupied eigenvalues:";
            for (int k = 0; k < n_occ; ++k)
                std::cout << " " << std::fixed << std::setprecision(6)
                          << eig(k);
            std::cout << "\n";
            break;
        }
    }

    if (!converged)
        std::cout << "\nDid not converge in " << max_iter << " iterations.\n";

    std::cout << std::fixed << std::setprecision(10)
              << "Final energy: " << final_energy << "\n";

    // ── Density matrix sanity checks ──────────────────────────────────
    //   Tr(P) should equal n_occ (number of occupied states).
    //   ||P^2 - P||_F should be small (idempotency, exact for converged SCF).
    double trace_P = xt::sum(xt::diagonal(P))();
    auto P2 = xt::linalg::dot(P, P);
    double idempotency = xt::norm_l2(P2 - P)();

    std::cout << "\nDensity checks:\n"
              << "  Tr(P)         = " << std::fixed << std::setprecision(6)
              << trace_P << "  (expected: " << n_occ << ")\n"
              << "  ||P^2 - P||_F = " << std::scientific << std::setprecision(2)
              << idempotency << "  (idempotency)\n";

    // ── Reference: alpha=0 ────────────────────────────────────────────
    std::cout << "\n--- alpha=0 reference (no coupling, 2 iterations) ---\n";
    {
        xt::xtensor<double, 2> P0 = xt::zeros<double>({n, n});
        auto [eig0, C0] = xt::linalg::eigh(H0);
        auto C0_occ = xt::view(C0, xt::all(), xt::range(0, n_occ));
        P0 = xt::linalg::dot(C0_occ, xt::transpose(C0_occ));
        double E0 = xt::sum(H0 * P0)();
        std::cout << std::fixed << std::setprecision(10)
                  << "  Energy (alpha=0): " << E0 << "\n"
                  << "  Energy shift:     " << (final_energy - E0) << "\n";
    }

    std::cout << "\n=== Done ===\n";
    return 0;
}
