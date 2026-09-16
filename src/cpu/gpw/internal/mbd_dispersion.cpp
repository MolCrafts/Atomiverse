/// @file cpu/gpw/internal/mbd_dispersion.cpp
/// @brief MBD@rsSCS free-atom / periodic energy + ionic forces (libMBD pure-Python).

#include "atomiverse/cpu/gpw/internal/mbd_dispersion.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV {
namespace cpu {
namespace gpw {
namespace {

using linalg::RealMatrix;
using Cplx = std::complex<double>;

// Gauss–Legendre n = 15 on [-1,1] (numpy.polynomial.legendre.leggauss).
constexpr int kNLeg = 15;
constexpr double kLegX[kNLeg] = {-0.9879925180204854,  -0.9372733924007058,
                                 -0.8482065834104272,  -0.7244177313601701,
                                 -0.5709721726085388,  -0.3941513470775634,
                                 -0.20119409399743451, 0.0,
                                 0.20119409399743451,  0.3941513470775634,
                                 0.5709721726085388,   0.7244177313601701,
                                 0.8482065834104272,   0.9372733924007058,
                                 0.9879925180204854};
constexpr double kLegW[kNLeg] = {
    0.030753241996117203, 0.0703660474881084,  0.10715922046717141,
    0.13957067792615444,  0.16626920581699398, 0.1861610000155622,
    0.1984314853271116,   0.2025782419255613,  0.1984314853271116,
    0.1861610000155622,   0.16626920581699398, 0.13957067792615444,
    0.10715922046717141,  0.0703660474881084,  0.030753241996117203};

// libMBD freq_grid(n=15, L=0.6): prepend u=0 with weight 0, reverse-mapped
// Gauss–Legendre half-line.
constexpr int kNFreq = 16; // 1 + n

void free_atom_params(int Z, double& alpha, double& c6, double& rvdw) {
    if (!ts_supports_element(Z))
        throw std::runtime_error(
            "GPW MBD: element Z=" + std::to_string(Z)
            + " is not in the in-tree free-atom table (no fallback)");
    const double* row = kTsFreeAtom[Z - 1];
    alpha = row[0];
    c6 = row[1];
    rvdw = row[2];
}

/// @brief Fermi damping f = 1 / (1 + exp(−a (R/S − 1))).
double damping_fermi(double r, double s_vdw, double a) {
    if (!(s_vdw > 0.0))
        return 0.0;
    const double x = -a * (r / s_vdw - 1.0);
    if (x > 60.0)
        return 0.0;
    if (x < -60.0)
        return 1.0;
    return 1.0 / (1.0 + std::exp(x));
}

/// @brief Bare dipole tensor (libMBD T_bare).
void T_bare(double Rx, double Ry, double Rz, double T[3][3]) {
    const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
    if (r2 < 1e-30) {
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                T[a][b] = 0.0;
        return;
    }
    const double r = std::sqrt(r2);
    const double r5 = r2 * r2 * r;
    const double inv_r5 = 1.0 / r5;
    const double R[3] = {Rx, Ry, Rz};
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            T[a][b] = (-3.0 * R[a] * R[b] + (a == b ? r2 : 0.0)) * inv_r5;
        }
    }
}

/// @brief Gaussian-damped Coulomb dipole tensor (libMBD T_erf_coulomb).
void T_erf_coulomb(double Rx, double Ry, double Rz, double sigma,
                   double T[3][3]) {
    double bare[3][3];
    T_bare(Rx, Ry, Rz, bare);
    const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
    if (r2 < 1e-30 || !(sigma > 0.0)) {
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                T[a][b] = 0.0;
        return;
    }
    const double r = std::sqrt(r2);
    const double r5 = r2 * r2 * r;
    const double inv_r5 = 1.0 / r5;
    const double zeta = r / sigma;
    const double theta = 2.0 * zeta / std::sqrt(M_PI) * std::exp(-zeta * zeta);
    const double erf_theta = std::erf(zeta) - theta;
    const double R[3] = {Rx, Ry, Rz};
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            const double rr_r5 = R[a] * R[b] * inv_r5;
            T[a][b] =
                erf_theta * bare[a][b] + (2.0 * zeta * zeta * theta) * rr_r5;
        }
    }
}

/// @brief erfc-screened Coulomb dipole tensor (libMBD T_erfc).
void T_erfc(double Rx, double Ry, double Rz, double alpha, double T[3][3]) {
    const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
    if (r2 < 1e-30) {
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                T[a][b] = 0.0;
        return;
    }
    const double r = std::sqrt(r2);
    const double r3 = r2 * r;
    const double r5 = r2 * r2 * r;
    const double ar = alpha * r;
    const double expa = std::exp(-(ar * ar));
    const double B = (std::erfc(ar) + (2.0 * ar / std::sqrt(M_PI)) * expa) / r3;
    const double C =
        (3.0 * std::erfc(ar)
         + (2.0 * ar / std::sqrt(M_PI)) * (3.0 + 2.0 * ar * ar) * expa)
        / r5;
    const double R[3] = {Rx, Ry, Rz};
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            T[a][b] = -C * R[a] * R[b] + (a == b ? B : 0.0);
        }
    }
}

/// @brief libMBD supercell_circum for orthorhombic cell (vectors on diagonal).
void supercell_circum(const double cell[3], double radius, int range[3]) {
    for (int d = 0; d < 3; ++d) {
        const double L = cell[d];
        if (!(L > 0.0))
            throw std::runtime_error("GPW MBD: cell edge must be positive");
        range[d] = static_cast<int>(std::ceil(radius / L + 0.5));
    }
}

void freq_grid(std::vector<double>& freq, std::vector<double>& freq_w) {
    constexpr double L = 0.6;
    freq.resize(kNFreq);
    freq_w.resize(kNFreq);
    freq[0] = 0.0;
    freq_w[0] = 0.0;
    std::array<double, kNLeg> xmap{}, wmap{};
    for (int i = 0; i < kNLeg; ++i) {
        const double x = kLegX[i];
        const double w = kLegW[i];
        wmap[static_cast<std::size_t>(i)] =
            2.0 * L / ((1.0 - x) * (1.0 - x)) * w;
        xmap[static_cast<std::size_t>(i)] = L * (1.0 + x) / (1.0 - x);
    }
    for (int i = 0; i < kNLeg; ++i) {
        freq[static_cast<std::size_t>(1 + i)] =
            xmap[static_cast<std::size_t>(kNLeg - 1 - i)];
        freq_w[static_cast<std::size_t>(1 + i)] =
            wmap[static_cast<std::size_t>(kNLeg - 1 - i)];
    }
}

/// @brief α_nlc = (α0^{-1} I + T)^{-1}  (3N×3N real).
RealMatrix screening_matrix(const std::vector<double>& alpha_0,
                            const RealMatrix& dip) {
    const int n = static_cast<int>(alpha_0.size());
    const int n3 = 3 * n;
    RealMatrix A = dip;
    for (int i = 0; i < n; ++i) {
        const double inv_a = 1.0 / alpha_0[static_cast<std::size_t>(i)];
        for (int a = 0; a < 3; ++a)
            A(3 * i + a, 3 * i + a) += inv_a;
    }
    return xt::linalg::inv(A);
}

std::vector<double> contract_atomic_alpha(const RealMatrix& a_nlc, int n) {
    std::vector<double> a(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j)
            for (int ax = 0; ax < 3; ++ax)
                sum += a_nlc(3 * i + ax, 3 * j + ax);
        a[static_cast<std::size_t>(i)] = sum / 3.0;
    }
    return a;
}

/// @brief Real dipole matrix (isolated or Γ-only lattice sum; no k-phase).
///
/// damping gg: short-range (1−f)×T_erf for rsSCS (no Ewald).
/// damping !gg: long-range f×T_bare; when periodic, Ewald split + reciprocal.
RealMatrix dipole_matrix_real(const std::vector<std::array<double, 3>>& pos,
                              const std::vector<double>& R_vdw,
                              const std::vector<double>& sigma, double beta,
                              double a_fermi, bool gg, bool periodic,
                              const double cell[3]) {
    const int n = static_cast<int>(pos.size());
    const int n3 = 3 * n;
    RealMatrix dip = xt::zeros<double>({n3, n3});

    int range_cell[3] = {0, 0, 0};
    double ewald_alpha = 0.0;
    double real_space_cutoff = 0.0;
    const bool do_ewald = periodic && !gg;
    if (periodic) {
        const double volume = cell[0] * cell[1] * cell[2];
        const double vol = std::max(std::abs(volume), 0.2);
        ewald_alpha = 2.5 / std::cbrt(vol);
        real_space_cutoff = 6.0 / ewald_alpha;
        supercell_circum(cell, real_space_cutoff, range_cell);
    }

    for (int nx = -range_cell[0]; nx <= range_cell[0]; ++nx) {
        for (int ny = -range_cell[1]; ny <= range_cell[1]; ++ny) {
            for (int nz = -range_cell[2]; nz <= range_cell[2]; ++nz) {
                const double Rcx = periodic ? cell[0] * nx : 0.0;
                const double Rcy = periodic ? cell[1] * ny : 0.0;
                const double Rcz = periodic ? cell[2] * nz : 0.0;
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        const double Rx = pos[static_cast<std::size_t>(i)][0]
                                          - pos[static_cast<std::size_t>(j)][0]
                                          + Rcx;
                        const double Ry = pos[static_cast<std::size_t>(i)][1]
                                          - pos[static_cast<std::size_t>(j)][1]
                                          + Rcy;
                        const double Rz = pos[static_cast<std::size_t>(i)][2]
                                          - pos[static_cast<std::size_t>(j)][2]
                                          + Rcz;
                        const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
                        const double r = std::sqrt(r2);
                        // pure-Python pymbd: no real-space cutoff skip inside
                        // the supercell image loop (range_cell already bounds it).
                        if (!periodic && i == j)
                            continue;
                        if (!periodic && r < 1e-15)
                            continue;
                        const double s_vdw =
                            beta
                            * (R_vdw[static_cast<std::size_t>(i)]
                               + R_vdw[static_cast<std::size_t>(j)]);
                        const double f = damping_fermi(r, s_vdw, a_fermi);
                        double T[3][3];
                        if (gg) {
                            const double sig = std::sqrt(
                                sigma[static_cast<std::size_t>(i)]
                                    * sigma[static_cast<std::size_t>(i)]
                                + sigma[static_cast<std::size_t>(j)]
                                      * sigma[static_cast<std::size_t>(j)]);
                            T_erf_coulomb(Rx, Ry, Rz, sig, T);
                            const double pref = 1.0 - f;
                            for (int a = 0; a < 3; ++a)
                                for (int b = 0; b < 3; ++b)
                                    T[a][b] *= pref;
                        } else {
                            T_bare(Rx, Ry, Rz, T);
                            for (int a = 0; a < 3; ++a)
                                for (int b = 0; b < 3; ++b)
                                    T[a][b] *= f;
                            if (do_ewald) {
                                double Te[3][3], Tb[3][3];
                                T_erfc(Rx, Ry, Rz, ewald_alpha, Te);
                                T_bare(Rx, Ry, Rz, Tb);
                                for (int a = 0; a < 3; ++a)
                                    for (int b = 0; b < 3; ++b)
                                        T[a][b] += Te[a][b] - Tb[a][b];
                            }
                        }
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b)
                                dip(3 * i + a, 3 * j + b) += T[a][b];
                    }
                }
            }
        }
    }

    if (do_ewald) {
        // Reciprocal-space Ewald at k = 0 (real) + self-energy.
        const double volume = std::abs(cell[0] * cell[1] * cell[2]);
        const double rec_space_cutoff = 10.0 * ewald_alpha;
        const double b1 = 2.0 * M_PI / cell[0];
        const double b2 = 2.0 * M_PI / cell[1];
        const double b3 = 2.0 * M_PI / cell[2];
        const double rec_cell[3] = {b1, b2, b3};
        int range_G[3];
        supercell_circum(rec_cell, rec_space_cutoff, range_G);

        const double self = 4.0 * ewald_alpha * ewald_alpha * ewald_alpha
                            / (3.0 * std::sqrt(M_PI));
        for (int i = 0; i < n; ++i)
            for (int a = 0; a < 3; ++a)
                dip(3 * i + a, 3 * i + a) -= self;

        // G = 0 singular contribution (libMBD): 4π/(3V) on each 3×3 diagonal
        // block (all atom pairs including i≠j? pure-python:
        // dipmat += diag(repeat(4π/(3V),3)) applied as
        // np.diag(np.repeat(...)) which is 3x3 eye * scalar added via
        // broadcasting to ALL pairs? Looking again:
        //   dipmat += np.diag(np.repeat(4*pi/(3*volume), 3))
        // but dipmat is (n,n,3,3) and this is wrong shape...
        // Actually:
        //   else:
        //     dipmat += np.diag(np.repeat(4 * np.pi / (3 * volume), 3))
        // Wait, in dipole_matrix_ewald return is after reshape of dipmat
        // (n,n,3,3). Looking at pure python again carefully:

        // dipmat += -np.eye(len(Rs))[:,:,None,None] * np.diag(np.repeat(...))
        // → only diagonal atom blocks, -self * I_3
        // Then k_sq == 0:
        // dipmat += np.diag(np.repeat(4 * np.pi / (3 * volume), 3))
        // That's a (3,3) matrix added to (n,n,3,3) — numpy would broadcast
        // if shapes align... Actually this might be a bug that works because
        // they reshape. Let me check pure python behaviour:
        // np.diag(repeat(x,3)) is 3x3. Adding to (n,n,3,3) fails in numpy
        // unless... they use it after reshape?
        // Looking at the code again from the file:
        //
        // dipmat += -np.eye(len(Rs))[:, :, None, None] * np.diag(
        //     np.repeat(4 * alpha**3 / (3 * np.sqrt(np.pi)), 3)
        // )
        // ...
        // else:
        //     dipmat += np.diag(np.repeat(4 * np.pi / (3 * volume), 3))
        //
        // The second line would error on shape. Unless older numpy broadcasts
        // diag onto last dims? In numpy, (n,n,3,3) + (3,3) broadcasts to
        // (n,n,3,3) — YES! So 4π/(3V) I_3 is added to EVERY atom-pair block.

        const double g0 = 4.0 * M_PI / (3.0 * volume);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int a = 0; a < 3; ++a)
                    dip(3 * i + a, 3 * j + a) += g0;

        for (int gx = -range_G[0]; gx <= range_G[0]; ++gx) {
            for (int gy = -range_G[1]; gy <= range_G[1]; ++gy) {
                for (int gz = -range_G[2]; gz <= range_G[2]; ++gz) {
                    if (gx == 0 && gy == 0 && gz == 0)
                        continue;
                    const double Gx = b1 * gx;
                    const double Gy = b2 * gy;
                    const double Gz = b3 * gz;
                    const double g2 = Gx * Gx + Gy * Gy + Gz * Gz;
                    const double gnorm = std::sqrt(g2);
                    if (gnorm > rec_space_cutoff)
                        continue;
                    const double pref =
                        4.0 * M_PI / volume
                        * std::exp(-g2 / (4.0 * ewald_alpha * ewald_alpha))
                        / g2;
                    const double G[3] = {Gx, Gy, Gz};
                    for (int i = 0; i < n; ++i) {
                        for (int j = 0; j < n; ++j) {
                            const double dx =
                                pos[static_cast<std::size_t>(i)][0]
                                - pos[static_cast<std::size_t>(j)][0];
                            const double dy =
                                pos[static_cast<std::size_t>(i)][1]
                                - pos[static_cast<std::size_t>(j)][1];
                            const double dz =
                                pos[static_cast<std::size_t>(i)][2]
                                - pos[static_cast<std::size_t>(j)][2];
                            // k_point is None → cos(G·R)
                            const double phase =
                                std::cos(Gx * dx + Gy * dy + Gz * dz);
                            for (int a = 0; a < 3; ++a)
                                for (int b = 0; b < 3; ++b)
                                    dip(3 * i + a, 3 * j + b) +=
                                        pref * G[a] * G[b] * phase;
                        }
                    }
                }
            }
        }
    }
    return dip;
}

/// @brief Complex dipole matrix at k-point (periodic long-range, fermi,dip).
std::vector<Cplx> dipole_matrix_k(const std::vector<std::array<double, 3>>& pos,
                                  const std::vector<double>& R_vdw, double beta,
                                  double a_fermi, const double cell[3],
                                  const double k_point[3]) {
    const int n = static_cast<int>(pos.size());
    const int n3 = 3 * n;
    std::vector<Cplx> dip(static_cast<std::size_t>(n3 * n3), Cplx{0.0, 0.0});
    auto at = [&](int row, int col) -> Cplx& {
        // column-major for LAPACK packing later: store row-major then convert,
        // or store column-major from the start: idx = row + col*n3
        return dip[static_cast<std::size_t>(row + col * n3)];
    };

    const double volume = std::abs(cell[0] * cell[1] * cell[2]);
    const double vol = std::max(volume, 0.2);
    const double ewald_alpha = 2.5 / std::cbrt(vol);
    const double real_space_cutoff = 6.0 / ewald_alpha;
    int range_cell[3];
    supercell_circum(cell, real_space_cutoff, range_cell);

    for (int nx = -range_cell[0]; nx <= range_cell[0]; ++nx) {
        for (int ny = -range_cell[1]; ny <= range_cell[1]; ++ny) {
            for (int nz = -range_cell[2]; nz <= range_cell[2]; ++nz) {
                const double Rcx = cell[0] * nx;
                const double Rcy = cell[1] * ny;
                const double Rcz = cell[2] * nz;
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        const double Rx = pos[static_cast<std::size_t>(i)][0]
                                          - pos[static_cast<std::size_t>(j)][0]
                                          + Rcx;
                        const double Ry = pos[static_cast<std::size_t>(i)][1]
                                          - pos[static_cast<std::size_t>(j)][1]
                                          + Rcy;
                        const double Rz = pos[static_cast<std::size_t>(i)][2]
                                          - pos[static_cast<std::size_t>(j)][2]
                                          + Rcz;
                        const double r = std::sqrt(Rx * Rx + Ry * Ry + Rz * Rz);
                        if (r > real_space_cutoff)
                            continue;
                        const double s_vdw =
                            beta
                            * (R_vdw[static_cast<std::size_t>(i)]
                               + R_vdw[static_cast<std::size_t>(j)]);
                        const double f = damping_fermi(r, s_vdw, a_fermi);
                        double Tb[3][3], Te[3][3];
                        T_bare(Rx, Ry, Rz, Tb);
                        T_erfc(Rx, Ry, Rz, ewald_alpha, Te);
                        // T = f * T_bare + (T_erfc - T_bare)
                        const double phase_arg =
                            -(k_point[0] * Rx + k_point[1] * Ry
                              + k_point[2] * Rz);
                        const Cplx kpref{std::cos(phase_arg),
                                         std::sin(phase_arg)};
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                const double t =
                                    f * Tb[a][b] + (Te[a][b] - Tb[a][b]);
                                at(3 * i + a, 3 * j + b) += kpref * t;
                            }
                        }
                    }
                }
            }
        }
    }

    // Reciprocal Ewald
    const double rec_space_cutoff = 10.0 * ewald_alpha;
    const double b1 = 2.0 * M_PI / cell[0];
    const double b2 = 2.0 * M_PI / cell[1];
    const double b3 = 2.0 * M_PI / cell[2];
    const double rec_cell[3] = {b1, b2, b3};
    int range_G[3];
    supercell_circum(rec_cell, rec_space_cutoff, range_G);

    const double self =
        4.0 * ewald_alpha * ewald_alpha * ewald_alpha / (3.0 * std::sqrt(M_PI));
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < 3; ++a)
            at(3 * i + a, 3 * i + a) -= self;

    const double ksq0 = k_point[0] * k_point[0] + k_point[1] * k_point[1]
                        + k_point[2] * k_point[2];
    if (std::sqrt(ksq0) > 1e-15) {
        const double pref =
            4.0 * M_PI / volume
            * std::exp(-ksq0 / (4.0 * ewald_alpha * ewald_alpha)) / ksq0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        at(3 * i + a, 3 * j + b) +=
                            pref * k_point[a] * k_point[b];
    } else {
        const double g0 = 4.0 * M_PI / (3.0 * volume);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int a = 0; a < 3; ++a)
                    at(3 * i + a, 3 * j + a) += g0;
    }

    for (int gx = -range_G[0]; gx <= range_G[0]; ++gx) {
        for (int gy = -range_G[1]; gy <= range_G[1]; ++gy) {
            for (int gz = -range_G[2]; gz <= range_G[2]; ++gz) {
                if (gx == 0 && gy == 0 && gz == 0)
                    continue;
                const double Gx = b1 * gx;
                const double Gy = b2 * gy;
                const double Gz = b3 * gz;
                const double kx = k_point[0] + Gx;
                const double ky = k_point[1] + Gy;
                const double kz = k_point[2] + Gz;
                const double k2 = kx * kx + ky * ky + kz * kz;
                const double knorm = std::sqrt(k2);
                if (knorm > rec_space_cutoff)
                    continue;
                const double pref =
                    4.0 * M_PI / volume
                    * std::exp(-k2 / (4.0 * ewald_alpha * ewald_alpha)) / k2;
                const double K[3] = {kx, ky, kz};
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        const double dx = pos[static_cast<std::size_t>(i)][0]
                                          - pos[static_cast<std::size_t>(j)][0];
                        const double dy = pos[static_cast<std::size_t>(i)][1]
                                          - pos[static_cast<std::size_t>(j)][1];
                        const double dz = pos[static_cast<std::size_t>(i)][2]
                                          - pos[static_cast<std::size_t>(j)][2];
                        const double parg = Gx * dx + Gy * dy + Gz * dz;
                        const Cplx fourier{std::cos(parg), std::sin(parg)};
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b)
                                at(3 * i + a, 3 * j + b) +=
                                    pref * K[a] * K[b] * fourier;
                    }
                }
            }
        }
    }
    return dip;
}

/// @brief Sum of sqrt of Hermitian eigenvalues (ascending, column-major A).
double sum_sqrt_eigvals_hermitian(std::vector<Cplx>& A_colmaj, int n3) {
    // JOBZ='N': eigenvalues only. Documented min: LWORK=n+1, LRWORK=n, LIWORK=1
    // Use comfortable workspace.
    char jobz = 'N';
    char uplo = 'U';
    int n = n3;
    int lda = n3;
    std::vector<double> w(static_cast<std::size_t>(n3));
    int lwork = std::max(2 * n3 + 1, 1);
    int lrwork = std::max(n3, 1);
    int liwork = 1;
    // Complex workspace; LAPACK sees interleaved re/im doubles (see
    // linalg/eigensolve.h zheevd_ binding).
    std::vector<Cplx> work(static_cast<std::size_t>(lwork));
    std::vector<double> rwork(static_cast<std::size_t>(lrwork));
    std::vector<int> iwork(static_cast<std::size_t>(liwork));
    int info = 0;
    zheevd_(&jobz, &uplo, &n, reinterpret_cast<double*>(A_colmaj.data()), &lda,
            w.data(), reinterpret_cast<double*>(work.data()), &lwork,
            rwork.data(), &lrwork, iwork.data(), &liwork, &info);
    if (info != 0)
        throw std::runtime_error("GPW MBD: zheevd failed with info="
                                 + std::to_string(info));
    double sum = 0.0;
    for (int k = 0; k < n3; ++k) {
        if (w[static_cast<std::size_t>(k)] < -1e-10)
            throw std::runtime_error(
                "GPW MBD: negative eigenvalue of dynamical matrix ("
                + std::to_string(w[static_cast<std::size_t>(k)]) + ")");
        sum += std::sqrt(std::max(w[static_cast<std::size_t>(k)], 0.0));
    }
    return sum;
}

/// @brief libMBD get_kpts: shifted mesh (shift=0.5), fold into (−0.5,0.5].
std::vector<std::array<double, 3>> get_kpts(const double cell[3],
                                            const int k_mesh[3]) {
    const double b1 = 2.0 * M_PI / cell[0];
    const double b2 = 2.0 * M_PI / cell[1];
    const double b3 = 2.0 * M_PI / cell[2];
    std::vector<std::array<double, 3>> pts;
    pts.reserve(static_cast<std::size_t>(k_mesh[0] * k_mesh[1] * k_mesh[2]));
    for (int i = 0; i < k_mesh[0]; ++i) {
        for (int j = 0; j < k_mesh[1]; ++j) {
            for (int k = 0; k < k_mesh[2]; ++k) {
                double fx = (static_cast<double>(i) + 0.5)
                            / static_cast<double>(k_mesh[0]);
                double fy = (static_cast<double>(j) + 0.5)
                            / static_cast<double>(k_mesh[1]);
                double fz = (static_cast<double>(k) + 0.5)
                            / static_cast<double>(k_mesh[2]);
                if (fx > 0.5)
                    fx -= 1.0;
                if (fy > 0.5)
                    fy -= 1.0;
                if (fz > 0.5)
                    fz -= 1.0;
                pts.push_back({fx * b1, fy * b2, fz * b3});
            }
        }
    }
    return pts;
}

/// @brief Energy-only MBD@rsSCS (no forces). Shared by public API + force FD.
double mbd_rsscs_energy(const std::vector<int>& Z,
                        const std::vector<std::array<double, 3>>& pos_bohr,
                        const double cell_bohr[3], bool periodic, double beta,
                        double a_fermi, const std::vector<double>& vol_ratio,
                        const int k_mesh[3]) {
    const std::size_t n = Z.size();
    if (n == 0)
        return 0.0;

    std::vector<double> alpha0(n), c6(n), rvdw(n);
    for (std::size_t i = 0; i < n; ++i) {
        free_atom_params(Z[i], alpha0[i], c6[i], rvdw[i]);
        const double v = vol_ratio.empty() ? 1.0 : vol_ratio[i];
        if (!(v > 0.0) || !std::isfinite(v))
            throw std::runtime_error(
                "GPW MBD: volume ratio must be positive finite");
        alpha0[i] *= v;
        c6[i] *= v * v;
        rvdw[i] *= std::cbrt(v);
    }

    std::vector<double> omega(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double a2 = alpha0[i] * alpha0[i];
        if (!(a2 > 0.0))
            throw std::runtime_error("GPW MBD: alpha_0 must be positive");
        omega[i] = (4.0 / 3.0) * c6[i] / a2;
    }

    std::vector<double> freq, freq_w;
    freq_grid(freq, freq_w);
    const int nf = static_cast<int>(freq.size());
    std::vector<std::vector<double>> alpha_dyn_rsscs(
        static_cast<std::size_t>(nf), std::vector<double>(n, 0.0));

    // rsSCS screening: lattice images when periodic; NO Ewald (fermi,dip,gg).
    for (int iu = 0; iu < nf; ++iu) {
        const double u = freq[static_cast<std::size_t>(iu)];
        std::vector<double> a_dyn(n);
        std::vector<double> sigma(n);
        for (std::size_t i = 0; i < n; ++i) {
            a_dyn[i] = alpha0[i] / (1.0 + (u / omega[i]) * (u / omega[i]));
            sigma[i] = std::cbrt(std::sqrt(2.0 / M_PI) * a_dyn[i] / 3.0);
        }
        const RealMatrix dip =
            dipole_matrix_real(pos_bohr, rvdw, sigma, beta, a_fermi,
                               /*gg=*/true, periodic, cell_bohr);
        const RealMatrix a_nlc = screening_matrix(a_dyn, dip);
        alpha_dyn_rsscs[static_cast<std::size_t>(iu)] =
            contract_atomic_alpha(a_nlc, static_cast<int>(n));
    }

    std::vector<double> alpha_rsscs(n), c6_rsscs(n), rvdw_rsscs(n);
    for (std::size_t i = 0; i < n; ++i) {
        alpha_rsscs[i] = alpha_dyn_rsscs[0][i];
        double s = 0.0;
        for (int iu = 0; iu < nf; ++iu) {
            const double au = alpha_dyn_rsscs[static_cast<std::size_t>(iu)][i];
            s += freq_w[static_cast<std::size_t>(iu)] * au * au;
        }
        c6_rsscs[i] = 3.0 / M_PI * s;
        const double ratio =
            (alpha0[i] > 0.0) ? (alpha_rsscs[i] / alpha0[i]) : 1.0;
        if (!(ratio > 0.0))
            throw std::runtime_error("GPW MBD: screened alpha non-positive");
        rvdw_rsscs[i] = rvdw[i] * std::cbrt(ratio);
    }

    std::vector<double> omega_rsscs(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double a2 = alpha_rsscs[i] * alpha_rsscs[i];
        if (!(a2 > 0.0))
            throw std::runtime_error(
                "GPW MBD: screened alpha_0 must be positive");
        omega_rsscs[i] = (4.0 / 3.0) * c6_rsscs[i] / a2;
    }

    const int n3 = 3 * static_cast<int>(n);
    std::vector<double> pre(static_cast<std::size_t>(n3));
    for (std::size_t i = 0; i < n; ++i) {
        const double p = omega_rsscs[i] * std::sqrt(alpha_rsscs[i]);
        for (int a = 0; a < 3; ++a)
            pre[3 * i + static_cast<std::size_t>(a)] = p;
    }

    double sum_sqrt_avg = 0.0;
    if (!periodic) {
        std::vector<double> sigma_dummy(n, 0.0);
        const RealMatrix Tlr = dipole_matrix_real(
            pos_bohr, rvdw_rsscs, sigma_dummy, beta, a_fermi, /*gg=*/false,
            /*periodic=*/false, cell_bohr);
        RealMatrix C = xt::zeros<double>({n3, n3});
        for (int p = 0; p < n3; ++p) {
            C(p, p) += omega_rsscs[static_cast<std::size_t>(p / 3)]
                       * omega_rsscs[static_cast<std::size_t>(p / 3)];
            for (int q = 0; q < n3; ++q)
                C(p, q) += pre[static_cast<std::size_t>(p)]
                           * pre[static_cast<std::size_t>(q)] * Tlr(p, q);
        }
        auto eigs = xt::linalg::eigvalsh(C);
        double sum_sqrt = 0.0;
        for (int k = 0; k < n3; ++k) {
            const double lam = eigs(k);
            if (lam < -1e-10)
                throw std::runtime_error(
                    "GPW MBD: negative eigenvalue of dynamical matrix ("
                    + std::to_string(lam) + ")");
            sum_sqrt += std::sqrt(std::max(lam, 0.0));
        }
        sum_sqrt_avg = sum_sqrt;
    } else {
        const auto kpts = get_kpts(cell_bohr, k_mesh);
        const int nk = static_cast<int>(kpts.size());
        double sum_all = 0.0;
        for (int ik = 0; ik < nk; ++ik) {
            const double kp[3] = {kpts[static_cast<std::size_t>(ik)][0],
                                  kpts[static_cast<std::size_t>(ik)][1],
                                  kpts[static_cast<std::size_t>(ik)][2]};
            std::vector<Cplx> Tlr = dipole_matrix_k(pos_bohr, rvdw_rsscs, beta,
                                                    a_fermi, cell_bohr, kp);
            // C = diag(ω²) + outer(pre,pre) * T   (Hermitian)
            std::vector<Cplx> Cmat(static_cast<std::size_t>(n3 * n3),
                                   Cplx{0.0, 0.0});
            for (int p = 0; p < n3; ++p) {
                const int ip = p / 3;
                Cmat[static_cast<std::size_t>(p + p * n3)] +=
                    omega_rsscs[static_cast<std::size_t>(ip)]
                    * omega_rsscs[static_cast<std::size_t>(ip)];
                for (int q = 0; q < n3; ++q) {
                    Cmat[static_cast<std::size_t>(p + q * n3)] +=
                        pre[static_cast<std::size_t>(p)]
                        * pre[static_cast<std::size_t>(q)]
                        * Tlr[static_cast<std::size_t>(p + q * n3)];
                }
            }
            // Hermitize numerically (kill round-off anti-Hermitian noise)
            for (int p = 0; p < n3; ++p) {
                for (int q = p; q < n3; ++q) {
                    const Cplx& a = Cmat[static_cast<std::size_t>(p + q * n3)];
                    const Cplx& b = Cmat[static_cast<std::size_t>(q + p * n3)];
                    const Cplx h = 0.5 * (a + std::conj(b));
                    Cmat[static_cast<std::size_t>(p + q * n3)] = h;
                    Cmat[static_cast<std::size_t>(q + p * n3)] = std::conj(h);
                }
            }
            sum_all += sum_sqrt_eigvals_hermitian(Cmat, n3);
        }
        sum_sqrt_avg = sum_all / static_cast<double>(nk);
    }

    double sum_omega = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        sum_omega += omega_rsscs[i];

    return 0.5 * sum_sqrt_avg - 1.5 * sum_omega;
}

} // namespace

MbdResult mbd_rsscs_pbe(const std::vector<int>& Z,
                        const std::vector<std::array<double, 3>>& pos_bohr,
                        const double cell_bohr[3], bool periodic, double beta,
                        double a_fermi, const std::vector<double>& vol_ratio,
                        const int* k_mesh_in) {
    const std::size_t n = Z.size();
    if (pos_bohr.size() != n)
        throw std::runtime_error("GPW MBD: Z and pos size mismatch");
    if (!vol_ratio.empty() && vol_ratio.size() != n)
        throw std::runtime_error("GPW MBD: vol_ratio size mismatch");
    if (!(beta > 0.0) || !(a_fermi > 0.0))
        throw std::runtime_error("GPW MBD: beta and a_fermi must be positive");

    int k_mesh[3] = {2, 2, 2};
    if (k_mesh_in) {
        k_mesh[0] = k_mesh_in[0];
        k_mesh[1] = k_mesh_in[1];
        k_mesh[2] = k_mesh_in[2];
    }
    if (periodic) {
        if (!(cell_bohr[0] > 0.0) || !(cell_bohr[1] > 0.0)
            || !(cell_bohr[2] > 0.0))
            throw std::runtime_error(
                "GPW MBD: periodic requires positive orthorhombic cell edges");
        for (int d = 0; d < 3; ++d) {
            if (k_mesh[d] < 1)
                throw std::runtime_error(
                    "GPW MBD: periodic k_mesh components must be ≥ 1");
        }
    }

    MbdResult res;
    res.forces.assign(n, {0.0, 0.0, 0.0});
    if (n == 0)
        return res;

    res.energy = mbd_rsscs_energy(Z, pos_bohr, cell_bohr, periodic, beta,
                                  a_fermi, vol_ratio, k_mesh);

    // Product ionic forces: central FD of the geometry-only energy that matches
    // libMBD. Step 1e-4 Bohr is a stable compromise for the 3N×3N eigensolve
    // (unit tests re-check vs an independent h=1e-5 FD).
    constexpr double kH = 1.0e-4;
    auto pos = pos_bohr;
    for (std::size_t i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) {
            const double save = pos[i][d];
            pos[i][d] = save + kH;
            const double ep = mbd_rsscs_energy(
                Z, pos, cell_bohr, periodic, beta, a_fermi, vol_ratio, k_mesh);
            pos[i][d] = save - kH;
            const double em = mbd_rsscs_energy(
                Z, pos, cell_bohr, periodic, beta, a_fermi, vol_ratio, k_mesh);
            pos[i][d] = save;
            res.forces[i][d] = -(ep - em) / (2.0 * kH);
        }
    }
    return res;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
