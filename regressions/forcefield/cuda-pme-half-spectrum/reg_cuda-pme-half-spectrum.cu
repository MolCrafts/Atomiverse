/// @file regressions/forcefield/cuda-pme-half-spectrum/reg_cuda-pme-half-spectrum.cu
/// @brief PME representation against an independent full discrete Fourier sum.
/// Smooth PME: Essmann et al., JCP 103, 8577 (1995), doi:10.1063/1.470117.
/// The oracle uses the cardinal spline's closed polynomial, not the device
/// recurrence, and explicitly sums every signed Fourier mode and its gradient.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/longrange/pme.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/coul_long.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <type_traits>
#include <vector>

using namespace ATV;
namespace {
using Complex = std::complex<double>;
constexpr int order = 6;
constexpr double alpha = 0.9;
constexpr double pi = units::pi;

// Mn(t) = sum_k (-1)^k C(n,k) max(t-k,0)^(n-1) / (n-1)!.
// Its derivative supplies the analytic derivative of the discrete energy.
double cardinal(double t, int n, bool derivative = false) {
    const int degree = n - 1 - int(derivative);
    int choose = 1;
    double factorial = 1;
    for (int k = 2; k <= degree; ++k)
        factorial *= k;
    double result = 0;
    for (int k = 0; k <= n; ++k) {
        if (t > k)
            result += (k % 2 ? -1 : 1) * choose * std::pow(t - k, degree);
        choose = choose * (n - k) / (k + 1);
    }
    return result / factorial;
}

struct Result {
    double energy = 0;
    double force[2][3] = {};
};

Result full_fourier_oracle(const cpu::Frame& f, const std::array<int, 3>& K,
                           int n) {
    const auto& inverse = f.box.inv_cell();
    double inv[3][3];
    for (int j = 0; j < 3; ++j)
        for (int d = 0; d < 3; ++d)
            inv[j][d] = ATV::cuda::motion_t(inverse(j, d));
    double u[2][3], charge[2];
    for (int i = 0; i < 2; ++i) {
        const double r[] = {f.span(kAtom_x)[i], f.span(kAtom_y)[i],
                            f.span(kAtom_z)[i]};
        charge[i] = f.span(kAtom_charge)[i];
        for (int d = 0; d < 3; ++d) {
            u[i][d] = 0;
            for (int j = 0; j < 3; ++j)
                u[i][d] += inv[j][d] * r[j] * K[d];
        }
    }
    std::array<std::vector<double>, 3> moduli;
    for (int d = 0; d < 3; ++d) {
        moduli[d].resize(K[d]);
        for (int m = 0; m < K[d]; ++m) {
            Complex b{};
            for (int a = 0; a < n; ++a)
                b += cardinal(n - 1 - a, n)
                     * std::polar(1.0, 2 * pi * m * a / K[d]);
            moduli[d][m] = 1 / std::norm(b);
        }
    }
    Result result;
    const double escale = units::coulomb / (2 * pi * f.box.volume());
    for (int m0 = 0; m0 < K[0]; ++m0)
        for (int m1 = 0; m1 < K[1]; ++m1)
            for (int m2 = 0; m2 < K[2]; ++m2) {
                if (m0 == 0 && m1 == 0 && m2 == 0)
                    continue;
                const int m[] = {m0, m1, m2};
                double k2 = 0;
                for (int j = 0; j < 3; ++j) {
                    double kj = 0;
                    for (int d = 0; d < 3; ++d)
                        kj +=
                            (m[d] <= K[d] / 2 ? m[d] : m[d] - K[d]) * inv[j][d];
                    k2 += kj * kj;
                }
                const double H = std::exp(-pi * pi * k2 / (alpha * alpha)) / k2
                                 * moduli[0][m0] * moduli[1][m1]
                                 * moduli[2][m2];
                Complex Q{}, dQ[2][3]{};
                for (int i = 0; i < 2; ++i) {
                    Complex factor[3]{}, derivative[3]{};
                    for (int d = 0; d < 3; ++d) {
                        const int base = int(std::floor(u[i][d]));
                        for (int a = 0; a < n; ++a) {
                            const int grid = base - n + 1 + a;
                            const auto wave =
                                std::polar(1.0, -2 * pi * m[d] * grid / K[d]);
                            const double t = u[i][d] - grid;
                            factor[d] += cardinal(t, n) * wave;
                            derivative[d] += cardinal(t, n, true) * wave;
                        }
                    }
                    Q += charge[i] * factor[0] * factor[1] * factor[2];
                    for (int j = 0; j < 3; ++j)
                        for (int d = 0; d < 3; ++d) {
                            Complex term = charge[i] * derivative[d]
                                           * double(K[d]) * inv[j][d];
                            for (int e = 0; e < 3; ++e)
                                if (e != d)
                                    term *= factor[e];
                            dQ[i][j] += term;
                        }
                }
                result.energy += escale * H * std::norm(Q);
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 3; ++j)
                        result.force[i][j] -=
                            2 * escale * H * std::real(std::conj(Q) * dQ[i][j]);
            }
    for (double q : charge)
        result.energy -= units::coulomb * alpha / std::sqrt(pi) * q * q;
    return result;
}

bool check(int nz, bool triclinic, int n = order) {
    cpu::Frame f(2);
    Mat3 cell = Mat3::diagonal(9, 10, nz);
    if (triclinic) {
        cell(0, 1) = 1.25;
        cell(0, 2) = 0.5;
        cell(1, 2) = 1;
    }
    f.box.set_cell(cell);
    constexpr double xyz[2][3] = {{1.25, 2.5, 3.125}, {5.75, 6.375, 7.5}};
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_x)[i] = xyz[i][0];
        f.span(kAtom_y)[i] = xyz[i][1];
        f.span(kAtom_z)[i] = xyz[i][2];
        f.span(kAtom_charge)[i] = i == 0 ? 1 : -1;
        f.span(kAtom_mass)[i] = 1;
        f.span(kAtom_tag)[i] = i;
        f.span(kAtom_type)[i] = 0;
    }
    f.rebuild_tag_index(0, 2);
    const auto expected = full_fourier_oracle(f, {9, 10, nz}, n);
    ATV::cuda::Evaluate eval;
    // No real-space pair lies inside 0.1 A: the result is the reciprocal
    // functional plus self energy, isolating the Fourier representation.
    eval.set_neighbor(ATV::cuda::NeighborList{0.1, 0.1});
    eval.set_evaluator(ATV::cuda::CoulLong{alpha, 0.1});
    eval.on(evaluate::Phase::evaluate)
        .add(ATV::cuda::KSpacePME{alpha, 1, n}, 100);
    eval.build();
    eval.run(f);
    const double de = std::abs(f.meta.get(kEnergy) - expected.energy);
    double df = 0;
    for (int i = 0; i < 2; ++i) {
        const double actual[] = {f.span(kAtom_fx)[i], f.span(kAtom_fy)[i],
                                 f.span(kAtom_fz)[i]};
        for (int j = 0; j < 3; ++j) {
            if (!std::isfinite(actual[j]))
                return false;
            df = std::max(df, std::abs(actual[j] - expected.force[i][j]));
        }
    }
    constexpr double tolerance =
        std::is_same_v<ATV::cuda::work_t, float> ? 5e-5 : 5e-8;
    std::printf(
        "PME full DFT oracle: order=%d nz=%d triclinic=%d dE=%.6g dF=%.6g\n", n,
        nz, triclinic, de, df);
    return std::isfinite(de) && de < tolerance && df < tolerance;
}
} // namespace

int main() try {
    return check(12, false) && check(15, false) && check(12, true)
                   && check(15, true) && check(12, true, 4)
               ? 0
               : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "cuda-pme-half-spectrum: %s\n", e.what());
    return 1;
}
