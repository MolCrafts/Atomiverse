/// @file tests/cpu/dynamics/lammps_nhc_reference.h
/// @brief Literal transcription of LAMMPS `FixNH::setup` (chain part) and
///        `FixNH::nhc_temp_integrate` for test oracles.
///
/// LAMMPS `src/fix_nh.cpp`, SHA e8c4c278a8599165a0321ebfcb4e4046ef79ad79,
/// with nc_tchain = 1 and drag = 0, in energy units (`kecurrent` = 2 KE,
/// `ke_target` = N_f kT). Copyright (2003) Sandia Corporation. Under the
/// terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S.
/// Government retains certain rights in this software. GPL-2.0-only.
#pragma once

#include <cmath>
#include <vector>

namespace atv_test {

struct LammpsNhc {
    int mtchain;
    double tdof, kT, t_freq, dt;
    std::vector<double> eta, eta_dot, eta_dotdot, eta_mass;

    LammpsNhc(int m, double dof, double kt, double tau, double step)
        : mtchain(m), tdof(dof), kT(kt), t_freq(1.0 / tau), dt(step),
          eta(m, 0.0), eta_dot(m + 1, 0.0), eta_dotdot(m, 0.0),
          eta_mass(m, 0.0) {
        eta_mass[0] = tdof * kT / (t_freq * t_freq);
        for (int ich = 1; ich < mtchain; ich++)
            eta_mass[ich] = kT / (t_freq * t_freq);
        for (int ich = 1; ich < mtchain; ich++)
            eta_dotdot[ich] =
                (eta_mass[ich - 1] * eta_dot[ich - 1] * eta_dot[ich - 1] - kT)
                / eta_mass[ich];
    }

    /// One half-step. `kinetic` [eV] leaves scaled (`t_current *=
    /// factor_eta^2`). @return `factor_eta`.
    double integrate(double& kinetic) {
        const double dthalf = 0.5 * dt, dt4 = 0.25 * dt, dt8 = 0.125 * dt;
        const double ke_target = tdof * kT;
        double kecurrent = 2.0 * kinetic;
        eta_dotdot[0] = (kecurrent - ke_target) / eta_mass[0];
        double expfac;
        for (int ich = mtchain - 1; ich > 0; ich--) {
            expfac = std::exp(-dt8 * eta_dot[ich + 1]);
            eta_dot[ich] *= expfac;
            eta_dot[ich] += eta_dotdot[ich] * dt4;
            eta_dot[ich] *= expfac;
        }
        expfac = std::exp(-dt8 * eta_dot[1]);
        eta_dot[0] *= expfac;
        eta_dot[0] += eta_dotdot[0] * dt4;
        eta_dot[0] *= expfac;
        const double factor_eta = std::exp(-dthalf * eta_dot[0]);
        kinetic *= factor_eta * factor_eta;
        kecurrent = 2.0 * kinetic;
        eta_dotdot[0] = (kecurrent - ke_target) / eta_mass[0];
        for (int ich = 0; ich < mtchain; ich++)
            eta[ich] += dthalf * eta_dot[ich];
        eta_dot[0] *= expfac;
        eta_dot[0] += eta_dotdot[0] * dt4;
        eta_dot[0] *= expfac;
        for (int ich = 1; ich < mtchain; ich++) {
            expfac = std::exp(-dt8 * eta_dot[ich + 1]);
            eta_dot[ich] *= expfac;
            eta_dotdot[ich] =
                (eta_mass[ich - 1] * eta_dot[ich - 1] * eta_dot[ich - 1] - kT)
                / eta_mass[ich];
            eta_dot[ich] += eta_dotdot[ich] * dt4;
            eta_dot[ich] *= expfac;
        }
        return factor_eta;
    }
};

} // namespace atv_test
