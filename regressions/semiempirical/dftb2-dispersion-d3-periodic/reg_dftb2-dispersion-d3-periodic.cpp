/// @file regressions/semiempirical/dftb2-dispersion-d3-periodic/reg_dftb2-dispersion-d3-periodic.cpp
/// @brief Γ-PBC lattice-sum D3(BJ) for DFTB2 — E_disp vs offline s-dftd3 pin.
///
/// Oracle: Simple-DFTD3 1.2.1, periodic `.gen` input (see REFERENCE.md).
/// The periodic-specific content is the *image* contribution: the cluster sum
/// is pinned too, and the gate is on their difference as well as on the
/// absolute periodic value — so this is not a second molecular pin.
/// Public API only. Exit non-zero on failure. ≤ 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

/// Cubic cell edge [Å] — small enough that images are well inside the D3 cutoff.
constexpr double kBoxAng = 12.0;

// Offline s-dftd3 1.2.1 `--bj dftb3`, energies in Hartree (see REFERENCE.md).
constexpr double kEDispPbcHa = -1.6959579071183e-03; // dimer_pbc12.gen
constexpr double kEDispMolHa = -1.6892886213669e-03; // dimer_mol.xyz

/// simple-dftd3's default two-body real-space cutoff [Bohr]. The lattice-sum
/// tail beyond it is discarded, so the oracle's cutoff is part of the pinned
/// quantity and ATV must be told to truncate at the same radius.
constexpr double kOracleCutoffBohr = 60.0;

constexpr double kTolEv = 1e-6;      // absolute pin tolerance
constexpr double kTolImageEv = 1e-6; // image-term tolerance

cpu::Frame make_dimer() {
    cpu::Frame f(6);
    const double xyz[6][3] = {
        {0.000, 0.000, 0.000}, {0.757, 0.000, 0.586}, {-0.757, 0.000, 0.586},
        {0.000, 0.000, 2.900}, {0.757, 0.000, 3.486}, {-0.757, 0.000, 3.486},
    };
    const int Z[6] = {8, 1, 1, 8, 1, 1};
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_type)[i] = Z[i];
        f.span(kAtom_x)[i] = static_cast<StorageReal>(xyz[i][0]);
        f.span(kAtom_y)[i] = static_cast<StorageReal>(xyz[i][1]);
        f.span(kAtom_z)[i] = static_cast<StorageReal>(xyz[i][2]);
        f.span(kAtom_mass)[i] = (Z[i] == 8) ? 15.999 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(kBoxAng);
    f.meta.get(kEnergy) = 0;
    return f;
}

/// Total energy [eV] and E_disp [eV] for one D3 setting.
void run_case(bool d3, bool d3_periodic, double& e_tot, double& e_disp) {
    auto f = make_dimer();
    cpu::Evaluate eval;
    cpu::semiempirical::DFTB2 m;
    m.d3 = d3;
    m.d3_periodic = d3_periodic;
    m.d3_cutoff = kOracleCutoffBohr * units::kBohrInAngstrom;
    m.with_forces = false;
    m.scc_tol = 1e-10;
    eval.on(evaluate::Phase::evaluate).add_ref(m);
    eval.build();
    eval.run(f);
    e_tot = f.meta.get(kEnergy);
    e_disp = m.last_e_disp();
}

} // namespace

int main() {
    const double pin_pbc = kEDispPbcHa * units::kHartreeInEv;
    const double pin_mol = kEDispMolHa * units::kHartreeInEv;
    const double pin_image = pin_pbc - pin_mol;

    double e_off = 0.0, d_off = 0.0;
    run_case(/*d3=*/false, /*d3_periodic=*/false, e_off, d_off);

    double e_mol = 0.0, disp_mol = 0.0;
    run_case(/*d3=*/true, /*d3_periodic=*/false, e_mol, disp_mol);

    double e_pbc = 0.0, disp_pbc = 0.0;
    run_case(/*d3=*/true, /*d3_periodic=*/true, e_pbc, disp_pbc);

    const double image = disp_pbc - disp_mol;
    std::printf("DFTB2 water dimer, cubic %.1f A cell\n", kBoxAng);
    std::printf("  E_disp cluster  = %.12f eV   pin = %.12f eV  |d|=%.3e\n",
                disp_mol, pin_mol, std::fabs(disp_mol - pin_mol));
    std::printf("  E_disp lattice  = %.12f eV   pin = %.12f eV  |d|=%.3e\n",
                disp_pbc, pin_pbc, std::fabs(disp_pbc - pin_pbc));
    std::printf("  image term      = %.12f eV   pin = %.12f eV  |d|=%.3e\n",
                image, pin_image, std::fabs(image - pin_image));

    if (!std::isfinite(e_off) || !std::isfinite(e_mol) || !std::isfinite(e_pbc)
        || !std::isfinite(disp_pbc)) {
        std::printf("FAIL non-finite energy\n");
        return 1;
    }
    if (disp_pbc >= 0.0) {
        std::printf("FAIL lattice E_disp not attractive\n");
        return 1;
    }
    if (!(disp_pbc < disp_mol)) {
        std::printf("FAIL lattice images must lower E_disp (%.12e vs %.12e)\n",
                    disp_pbc, disp_mol);
        return 1;
    }
    if (std::fabs(disp_mol - pin_mol) > kTolEv) {
        std::printf("FAIL cluster E_disp vs s-dftd3 pin (tol %.1e eV)\n",
                    kTolEv);
        return 1;
    }
    if (std::fabs(disp_pbc - pin_pbc) > kTolEv) {
        std::printf("FAIL lattice E_disp vs s-dftd3 pin (tol %.1e eV)\n",
                    kTolEv);
        return 1;
    }
    // The discriminating gate: the *image* contribution alone, which a
    // cluster-only sum would report as exactly zero.
    if (std::fabs(image - pin_image) > kTolImageEv) {
        std::printf("FAIL image term vs s-dftd3 (tol %.1e eV)\n", kTolImageEv);
        return 1;
    }
    const double de = e_pbc - e_off;
    if (std::fabs(de - disp_pbc) > 1e-8) {
        std::printf("FAIL E_tot(d3 pbc)-E_tot(off)=%.12e != E_disp=%.12e\n", de,
                    disp_pbc);
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
