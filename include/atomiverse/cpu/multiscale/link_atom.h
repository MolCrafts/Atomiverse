/// @file cpu/multiscale/link_atom.h
/// @brief H link-atom geometry and force projection for covalent QM/MM cuts.
///
/// Placement (Field / Senn–Thiel style scale factor):
/// \f$ \mathbf{r}_{\mathrm{H}} = \mathbf{r}_{\mathrm{Q}}
///     + g\,(\mathbf{r}_{\mathrm{M}}-\mathbf{r}_{\mathrm{Q}}) \f$
/// with dimensionless scale \f$g\f$ (typical ~0.709 for a C–C cut so that
/// \f$|\mathbf{r}_{\mathrm{QH}}|\approx r_{\mathrm{C-H}}^0\f$).
///
/// Force chain rule for \f$\mathbf{r}_{\mathrm{H}}=(1-g)\mathbf{r}_{\mathrm{Q}}
/// + g\,\mathbf{r}_{\mathrm{M}}\f$:
/// \f$ \mathbf{F}_{\mathrm{Q}}+=(1-g)\mathbf{F}_{\mathrm{H}},\;
///     \mathbf{F}_{\mathrm{M}}+=g\,\mathbf{F}_{\mathrm{H}} \f$.
///
/// Units: positions [Å], forces [eV/Å]. Link H exists only in the private QM
/// geometry — it is never written into the user Frame topology.
#pragma once

#include "atomiverse/core/precision.h"

namespace ATV {
namespace cpu {

/// @brief One covalent QM–MM cut (global tags).
struct LinkBond {
    TagInt qm_tag = 0; ///< QM-side heavy atom (host of the H cap).
    TagInt mm_tag = 0; ///< MM-side heavy atom (replaced by H in the QM view).
};

/// @brief Place a link hydrogen on the QM–MM bond.
/// @param qx,qy,qz QM host position [Å].
/// @param mx,my,mz MM partner position [Å].
/// @param scale Dimensionless scale g (>0).
/// @param[out] hx,hy,hz Link-H position [Å].
inline void place_link_hydrogen(double qx, double qy, double qz, double mx,
                                double my, double mz, double scale, double& hx,
                                double& hy, double& hz) {
    hx = qx + scale * (mx - qx);
    hy = qy + scale * (my - qy);
    hz = qz + scale * (mz - qz);
}

/// @brief Project a force on the link H onto the real QM host and MM partner.
/// @param scale Dimensionless scale g used in placement.
/// @param fhx,fhy,fhz Force on the link H from the QM calculation [eV/Å].
/// @param[in,out] fqx,fqy,fqz Accumulated force on the QM host [eV/Å].
/// @param[in,out] fmx,fmy,fmz Accumulated force on the MM partner [eV/Å].
inline void project_link_force(double scale, double fhx, double fhy, double fhz,
                               double& fqx, double& fqy, double& fqz,
                               double& fmx, double& fmy, double& fmz) {
    const double omt = 1.0 - scale;
    fqx += omt * fhx;
    fqy += omt * fhy;
    fqz += omt * fhz;
    fmx += scale * fhx;
    fmy += scale * fhy;
    fmz += scale * fhz;
}

} // namespace cpu
} // namespace ATV
