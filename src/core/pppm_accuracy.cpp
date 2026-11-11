/// @file src/core/pppm_accuracy.cpp
/// @brief LAMMPS `PPPM::init` accuracy → (α, mesh) for ik PPPM.
///
/// Port of LAMMPS `src/KSPACE/pppm.cpp` (`acons`, `init`,
/// `set_grid_global`, `factorable`, `compute_df_kspace`,
/// `estimate_ik_error`, `adjust_gewald`, `newton_raphson_f`, `derivf`,
/// `final_accuracy`) and `src/kspace.cpp` (`two_charge`, `lamda2xT`,
/// `x2lamdaT`), checkout e8c4c278a8. LAMMPS: Copyright (2003) Sandia
/// Corporation. GPL-2.0-only. Scope and deviations: core/pppm_accuracy.h.
#include "atomiverse/core/pppm_accuracy.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/core/pppm_ik.h"
#include "atomiverse/core/units.h"

namespace ATV::longrange {
namespace {

constexpr int kLarge = 10000;      // LAMMPS LARGE (adjust_gewald iterations)
constexpr double kSmall = 0.00001; // LAMMPS SMALL (|f| convergence) [eV/Å]
constexpr int kOffset = 16384;     // LAMMPS OFFSET (grid size bound)

// LAMMPS PPPM::PPPM acons[order][m], JCP 109, 7698 (1998).
double acons(int order, int m) {
    static const double a[8][7] = {
        {0, 0, 0, 0, 0, 0, 0},
        {2.0 / 3.0, 0, 0, 0, 0, 0, 0},
        {1.0 / 50.0, 5.0 / 294.0, 0, 0, 0, 0, 0},
        {1.0 / 588.0, 7.0 / 1440.0, 21.0 / 3872.0, 0, 0, 0, 0},
        {1.0 / 4320.0, 3.0 / 1936.0, 7601.0 / 2271360.0, 143.0 / 28800.0, 0, 0,
         0},
        {1.0 / 23232.0, 7601.0 / 13628160.0, 143.0 / 69120.0,
         517231.0 / 106536960.0, 106640677.0 / 11737571328.0, 0, 0},
        {691.0 / 68140800.0, 13.0 / 57600.0, 47021.0 / 35512320.0,
         9694607.0 / 2095994880.0, 733191589.0 / 59609088000.0,
         326190917.0 / 11700633600.0, 0},
        {1.0 / 345600.0, 3617.0 / 35512320.0, 745739.0 / 838397952.0,
         56399353.0 / 12773376000.0, 25091609.0 / 1560084480.0,
         1755948832039.0 / 36229939200000.0, 4887769399.0 / 37838389248.0},
    };
    return a[order][m];
}

// LAMMPS PPPM::factorable with factors {2, 3, 5}.
bool factorable(int n) {
    while (n > 1) {
        int i = 0;
        for (const int f : {2, 3, 5}) {
            if (n % f == 0) {
                n /= f;
                break;
            }
            ++i;
        }
        if (i == 3)
            return false;
    }
    return true;
}

// The LAMMPS restricted-triclinic box of a cell (core/pppm_ik.h
// pppm_lammps_box) with Domain::h_inv and the KSpace transposed maps.
struct LammpsBox {
    double h[6] = {};
    double h_inv[6] = {};
    bool triclinic = false;
    double prd(int d) const { return h[d]; }
    // KSpace::lamda2xT: v = H^T λ with the tilts as |tilt|.
    void lamda2xT(const double* lamda, double* v) const {
        const double l0 = lamda[0], l1 = lamda[1], l2 = lamda[2];
        v[0] = h[0] * l0;
        v[1] = std::fabs(h[5]) * l0 + h[1] * l1;
        v[2] = std::fabs(h[4]) * l0 + std::fabs(h[3]) * l1 + h[2] * l2;
    }
    // KSpace::x2lamdaT: λ = H^{-T} v.
    void x2lamdaT(const double* v, double* lamda) const {
        const double v0 = v[0], v1 = v[1], v2 = v[2];
        lamda[0] = h_inv[0] * v0;
        lamda[1] = h_inv[5] * v0 + h_inv[1] * v1;
        lamda[2] = h_inv[4] * v0 + h_inv[3] * v1 + h_inv[2] * v2;
    }
};

LammpsBox lammps_box(const double* cell) {
    LammpsBox box;
    double* h = box.h;
    pppm_lammps_box(cell, h);
    if (!(h[0] > 0) || !(h[1] > 0) || !(h[2] > 0))
        throw std::invalid_argument(
            "pppm_from_accuracy: cell must have a positive volume");
    // Domain::set_global_box h_inv.
    double* hi = box.h_inv;
    hi[0] = 1.0 / h[0];
    hi[1] = 1.0 / h[1];
    hi[2] = 1.0 / h[2];
    hi[3] = -h[3] / (h[1] * h[2]);
    hi[4] = (h[3] * h[5] - h[1] * h[4]) / (h[0] * h[1] * h[2]);
    hi[5] = -h[5] / (h[0] * h[1]);
    box.triclinic = h[3] != 0.0 || h[4] != 0.0 || h[5] != 0.0;
    return box;
}

// The state LAMMPS PPPM carries between set_grid_global, adjust_gewald and
// final_accuracy.
struct Solver {
    LammpsBox box;
    int order = 6;
    long long natoms = 0;
    double q2 = 0.0;       // qsqsum · qqrd2e [eV·Å]
    double cutoff = 0.0;   // [Å]
    double accuracy = 0.0; // absolute [eV/Å]
    double g_ewald = 0.0;  // [1/Å]
    int n[3] = {0, 0, 0};
    double h[3] = {0, 0, 0};

    double volume() const { return box.prd(0) * box.prd(1) * box.prd(2); }

    double estimate(double hd, double prd) const {
        return pppm_estimate_ik_error(hd, prd, natoms, q2, g_ewald, order);
    }

    // PPPM::compute_df_kspace (ik branch).
    double df_kspace() const {
        const double lprx = estimate(h[0], box.prd(0));
        const double lpry = estimate(h[1], box.prd(1));
        const double lprz = estimate(h[2], box.prd(2));
        return std::sqrt(lprx * lprx + lpry * lpry + lprz * lprz)
               / std::sqrt(3.0);
    }

    double df_rspace() const {
        return 2.0 * q2 * std::exp(-g_ewald * g_ewald * cutoff * cutoff)
               / std::sqrt(static_cast<double>(natoms) * cutoff * volume());
    }

    // PPPM::newton_raphson_f.
    double newton_raphson_f() const { return df_rspace() - df_kspace(); }

    // PPPM::derivf: forward difference with step 1e-6.
    double derivf() {
        const double step = 0.000001;
        const double f1 = newton_raphson_f();
        const double g_old = g_ewald;
        g_ewald += step;
        const double f2 = newton_raphson_f();
        g_ewald = g_old;
        return (f2 - f1) / step;
    }

    // PPPM::adjust_gewald.
    void adjust_gewald() {
        for (int i = 0; i < kLarge; ++i) {
            const double dfx = derivf();
            if (dfx == 0.0 || dfx != dfx)
                break;
            double dx = newton_raphson_f() / dfx;
            while (g_ewald - dx <= 0.0)
                dx *= 0.5;
            g_ewald -= dx;
            if (std::fabs(newton_raphson_f()) < kSmall)
                return;
        }
        throw std::runtime_error(
            "pppm_from_accuracy: could not compute g_ewald (LAMMPS "
            "adjust_gewald)");
    }

    // PPPM::set_grid_global, ik branch, slab_volfactor = 1.
    void set_grid_global(bool gridflag) {
        const double xprd = box.prd(0), yprd = box.prd(1), zprd = box.prd(2);
        if (!gridflag) {
            h[0] = h[1] = h[2] = 1.0 / g_ewald;
            n[0] = static_cast<int>(xprd / h[0]) + 1;
            n[1] = static_cast<int>(yprd / h[1]) + 1;
            n[2] = static_cast<int>(zprd / h[2]) + 1;
            // The error is evaluated before the mesh grows, exactly as in
            // LAMMPS (the loop exits one point past the first passing h).
            const double prd[3] = {xprd, yprd, zprd};
            for (int d = 0; d < 3; ++d) {
                double err = estimate(h[d], prd[d]);
                while (err > accuracy) {
                    err = estimate(h[d], prd[d]);
                    ++n[d];
                    h[d] = prd[d] / n[d];
                    if (n[d] >= kOffset)
                        throw std::invalid_argument(
                            "pppm_from_accuracy: PPPM grid is too large");
                }
            }
            if (box.triclinic) {
                double tmp[3] = {n[0] / xprd, n[1] / yprd, n[2] / zprd};
                box.lamda2xT(tmp, tmp);
                for (int d = 0; d < 3; ++d)
                    n[d] = static_cast<int>(tmp[d]) + 1;
            }
        }
        for (int d = 0; d < 3; ++d) {
            while (!factorable(n[d]))
                ++n[d];
            // Atomiverse stencil bound (core/pppm_accuracy.h).
            if (n[d] < order + 1)
                n[d] = pppm_fft_size(order + 1);
        }
        if (!box.triclinic) {
            h[0] = xprd / n[0];
            h[1] = yprd / n[1];
            h[2] = zprd / n[2];
        } else {
            double tmp[3] = {double(n[0]), double(n[1]), double(n[2])};
            box.x2lamdaT(tmp, tmp);
            for (int d = 0; d < 3; ++d)
                h[d] = 1.0 / tmp[d];
        }
        for (int d = 0; d < 3; ++d)
            if (n[d] >= kOffset)
                throw std::invalid_argument(
                    "pppm_from_accuracy: PPPM grid is too large");
    }
};

} // namespace

double pppm_estimate_ik_error(double h, double prd, long long natoms, double q2,
                              double alpha, int order) {
    if (order < 1 || order > kPppmAccuracyMaxOrder)
        throw std::invalid_argument(
            "pppm_estimate_ik_error: order must be in [1, 7] (LAMMPS acons)");
    if (natoms == 0)
        return 0.0;
    double sum = 0.0;
    for (int m = 0; m < order; ++m)
        sum += acons(order, m) * std::pow(h * alpha, 2.0 * m);
    return q2 * std::pow(h * alpha, static_cast<double>(order))
           * std::sqrt(alpha * prd * std::sqrt(2.0 * units::pi) * sum
                       / static_cast<double>(natoms))
           / (prd * prd);
}

PppmAccuracyResult pppm_from_accuracy(const PppmAccuracyRequest& req) {
    if (!(req.accuracy > 0.0) || !(req.accuracy <= 1.0))
        throw std::invalid_argument(
            "pppm_from_accuracy: accuracy must be in (0, 1] (relative, LAMMPS "
            "kspace_style pppm)");
    if (!(req.cutoff > 0.0) || !std::isfinite(req.cutoff))
        throw std::invalid_argument(
            "pppm_from_accuracy: cutoff must be finite and > 0");
    if (req.order < 3 || req.order > kPppmAccuracyMaxOrder)
        throw std::invalid_argument(
            "pppm_from_accuracy: order must be in [3, 7], got "
            + std::to_string(req.order));
    if (req.natoms <= 0)
        throw std::invalid_argument("pppm_from_accuracy: natoms must be > 0");
    if (!(req.qsqsum >= 0.0) || !std::isfinite(req.qsqsum))
        throw std::invalid_argument(
            "pppm_from_accuracy: qsqsum must be finite and >= 0");
    const bool any_mesh = req.mesh[0] > 0 || req.mesh[1] > 0 || req.mesh[2] > 0;
    const bool gridflag = req.mesh[0] > 0 && req.mesh[1] > 0 && req.mesh[2] > 0;
    if ((any_mesh && !gridflag) || req.mesh[0] < 0 || req.mesh[1] < 0
        || req.mesh[2] < 0)
        throw std::invalid_argument(
            "pppm_from_accuracy: mesh must be all zero (derive) or all > 0");
    const bool gewaldflag = req.alpha > 0.0;

    Solver s;
    s.box = lammps_box(req.cell);
    s.order = req.order;
    s.natoms = req.natoms;
    s.cutoff = req.cutoff;
    // KSpace::two_charge and PPPM::qsum_qsq for the request's units row.
    const units::UnitSystem& us = req.unit_system;
    const double two_charge_force = us.two_charge_force();
    if (!(two_charge_force > 0.0) || !(us.qqrd2e() > 0.0))
        throw std::invalid_argument(
            "pppm_from_accuracy: unit system two_charge_force and qqrd2e "
            "must be > 0");
    s.q2 = req.qsqsum * us.qqrd2e();
    s.accuracy = req.accuracy * two_charge_force;

    // PPPM::init: initial α estimate.
    if (!gewaldflag) {
        if (s.q2 == 0.0)
            throw std::invalid_argument(
                "pppm_from_accuracy: an uncharged system needs a fixed alpha "
                "(LAMMPS kspace_modify gewald)");
        double g =
            s.accuracy
            * std::sqrt(static_cast<double>(s.natoms) * s.cutoff * s.volume())
            / (2.0 * s.q2);
        if (g >= 1.0)
            g = (1.35 - 0.15 * std::log(s.accuracy)) / s.cutoff;
        else
            g = std::sqrt(-std::log(g)) / s.cutoff;
        s.g_ewald = g;
    } else {
        s.g_ewald = req.alpha;
    }
    if (gridflag)
        for (int d = 0; d < 3; ++d)
            s.n[d] = req.mesh[d];

    s.set_grid_global(gridflag);
    if (!gewaldflag)
        s.adjust_gewald();

    PppmAccuracyResult out;
    out.alpha = s.g_ewald;
    for (int d = 0; d < 3; ++d)
        out.mesh[d] = s.n[d];
    out.order = s.order;
    out.df_kspace = s.df_kspace();
    out.df_real = s.df_rspace();
    // PPPM::final_accuracy without the table term (pair_modify table 0).
    out.accuracy_absolute =
        std::sqrt(out.df_kspace * out.df_kspace + out.df_real * out.df_real);
    out.accuracy_relative = out.accuracy_absolute / two_charge_force;
    return out;
}

} // namespace ATV::longrange
