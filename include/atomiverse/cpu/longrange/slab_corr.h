/// @file cpu/longrange/slab_corr.h
/// @brief Yeh–Berkowitz slab (\(k=0\)) correction for quasi-2D electrostatics.
///
/// Used with 3D Ewald/PME on a z-extended cell to approximate non-periodic
/// boundaries in z (LAMMPS `kspace_modify slab <volfactor>`). Citation:
/// Yeh & Berkowitz, *J. Chem. Phys.* **111**, 3155 (1999); non-neutral
/// extension Ballenegger et al., *J. Chem. Phys.* **131**, 094107 (2009).
/// Formulae match LAMMPS `Ewald::slabcorr` (metal units via `units::coulomb`).
///
/// Units: positions [Å], charges [e], volume [Å³], energy [eV], force [eV/Å].
#pragma once

#include <cmath>
#include <cstddef>

#include "atomiverse/core/precision.h"
#include "atomiverse/core/units.h"

namespace ATV {
namespace longrange {

/// @brief Yeh–Berkowitz energy [eV] for the global dipole (and optional net Q).
/// @param mz      Global dipole \(M_z = \sum q_i z_i\) [e·Å].
/// @param q_total Net charge \(\sum q_i\) [e].
/// @param mz_r2   \(\sum q_i z_i^2\) [e·Å²]; used only when \(|Q|>0\).
/// @param volume  Extended k-space volume \(L_x L_y (L_z f)\) [Å³].
/// @param zprd_slab Extended z period \(L_z f\) [Å].
/// @return \(E_{\mathrm{slab}}\) [eV].
inline double slab_corr_energy(double mz, double q_total, double mz_r2,
                               double volume, double zprd_slab) {
    const double ke = ATV::units::coulomb;
    const double two_pi = 2.0 * ATV::units::pi;
    // LAMMPS: e_slabcorr = 2π (Mz² − Q·Σ q z² − Q² Lz'²/12) / V'
    const double raw = two_pi
                       * (mz * mz - q_total * mz_r2
                          - q_total * q_total * zprd_slab * zprd_slab / 12.0)
                       / volume;
    return ke * raw;
}

/// @brief Add slab-correction z-forces on @p n owned atoms.
/// @param q       Charges [e], length @p n.
/// @param z       z coordinates [Å], length @p n.
/// @param n       Atom count.
/// @param mz      Global \(M_z\) [e·Å].
/// @param q_total Global net charge [e].
/// @param volume  Extended k-space volume [Å³].
/// @param[in,out] fz Force z accumulators [eV/Å], length @p n.
inline void slab_corr_add_forces(const StorageReal* q, const StorageReal* z,
                                 int n, double mz, double q_total,
                                 double volume, AccReal* fz) {
    // ffact = ke * (−4π / V');  Fz_i += ffact * q_i * (Mz − Q z_i)
    const double ffact = ATV::units::coulomb * (-4.0 * ATV::units::pi / volume);
    for (int i = 0; i < n; ++i) {
        const double qi = static_cast<double>(q[i]);
        if (qi == 0.0)
            continue;
        fz[i] += ffact * qi * (mz - q_total * static_cast<double>(z[i]));
    }
}

/// @brief Accumulate local \(M_z\), \(Q\), and \(\sum q z^2\) over @p n atoms.
inline void slab_dipole_partial(const StorageReal* q, const StorageReal* z,
                                int n, double& mz, double& q_sum,
                                double& mz_r2) {
    mz = 0.0;
    q_sum = 0.0;
    mz_r2 = 0.0;
    for (int i = 0; i < n; ++i) {
        const double qi = static_cast<double>(q[i]);
        const double zi = static_cast<double>(z[i]);
        mz += qi * zi;
        q_sum += qi;
        mz_r2 += qi * zi * zi;
    }
}

} // namespace longrange
} // namespace ATV
