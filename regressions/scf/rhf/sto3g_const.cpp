/// @file regressions/scf/rhf/sto3g_const.cpp
/// @brief RHF run(const Frame&) on H2/STO-3G and H2O/STO-3G — CPU path.
///
/// Same SCF as run(Frame&) (PySCF scf.hf.kernel / energy_tot). The const
/// entry writes nothing: kEnergy, kEnergyElectronic, positions, and forces
/// keep the caller's values. energy() is the total [Hartree].
/// When ATV_BUILD_CUDA=ON, CMake builds sto3g_const.cu instead
/// (--device=cpu|cuda|both). Public headers only. See REFERENCE.md.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "harness/registry.h"

namespace atv_reg_scf_sto3g_const {

namespace {

using ATV::cpu::Frame;

Frame make_h2() {
    using namespace ATV;
    Frame frame(2);
    frame.span(kAtom_type)[0] = 1;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.74;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

Frame make_h2o() {
    using namespace ATV;
    Frame frame(3);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_x)[2] = 0.0;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

struct Stamp {
    double energy = 0.0;
    double electronic = 0.0;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> fx;
    std::vector<double> fy;
    std::vector<double> fz;
};

Stamp stamp_frame(Frame& frame) {
    using namespace ATV;
    Stamp s;
    s.energy = 42.0;
    s.electronic = -7.0;
    frame.meta.get(kEnergy) = s.energy;
    frame.meta.get(kEnergyElectronic) = s.electronic;
    const int n = static_cast<int>(frame.nlocal());
    s.x.resize(static_cast<std::size_t>(n));
    s.y.resize(static_cast<std::size_t>(n));
    s.z.resize(static_cast<std::size_t>(n));
    s.fx.resize(static_cast<std::size_t>(n));
    s.fy.resize(static_cast<std::size_t>(n));
    s.fz.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const auto k = static_cast<std::size_t>(i);
        s.x[k] = static_cast<double>(frame.span(kAtom_x)[i]);
        s.y[k] = static_cast<double>(frame.span(kAtom_y)[i]);
        s.z[k] = static_cast<double>(frame.span(kAtom_z)[i]);
        frame.span(kAtom_fx)[i] = 1.5 + i;
        frame.span(kAtom_fy)[i] = -2.5 - i;
        frame.span(kAtom_fz)[i] = 3.5 + i;
        s.fx[k] = static_cast<double>(frame.span(kAtom_fx)[i]);
        s.fy[k] = static_cast<double>(frame.span(kAtom_fy)[i]);
        s.fz[k] = static_cast<double>(frame.span(kAtom_fz)[i]);
    }
    return s;
}

int frame_unchanged(const char* label, const Frame& frame, const Stamp& s) {
    using namespace ATV;
    const double e = static_cast<double>(frame.meta.get(kEnergy));
    const double ee = static_cast<double>(frame.meta.get(kEnergyElectronic));
    if (e != s.energy || ee != s.electronic) {
        std::fprintf(stderr,
                     "FAIL [cpu]: %s const wrote meta "
                     "kEnergy got=%.12f want=%.12f "
                     "kEnergyElectronic got=%.12f want=%.12f\n",
                     label, e, s.energy, ee, s.electronic);
        return 1;
    }
    const int n = static_cast<int>(frame.nlocal());
    for (int i = 0; i < n; ++i) {
        const auto k = static_cast<std::size_t>(i);
        const double x = static_cast<double>(frame.span(kAtom_x)[i]);
        const double y = static_cast<double>(frame.span(kAtom_y)[i]);
        const double z = static_cast<double>(frame.span(kAtom_z)[i]);
        const double fx = static_cast<double>(frame.span(kAtom_fx)[i]);
        const double fy = static_cast<double>(frame.span(kAtom_fy)[i]);
        const double fz = static_cast<double>(frame.span(kAtom_fz)[i]);
        if (x != s.x[k] || y != s.y[k] || z != s.z[k] || fx != s.fx[k]
            || fy != s.fy[k] || fz != s.fz[k]) {
            std::fprintf(stderr,
                         "FAIL [cpu]: %s const wrote atom %d "
                         "x got=%.12f want=%.12f "
                         "fx got=%.12f want=%.12f "
                         "fy got=%.12f want=%.12f "
                         "fz got=%.12f want=%.12f\n",
                         label, i, x, s.x[k], fx, s.fx[k], fy, s.fy[k], fz,
                         s.fz[k]);
            return 1;
        }
    }
    return 0;
}

int check_cpu(const char* label, Frame frame, double e_want) {
    using namespace ATV;
    constexpr double kTol = 1e-8;
    const Stamp s = stamp_frame(frame);
    const Frame& geometry = frame;
    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/0,
                      /*tol=*/1e-12, /*max_iter=*/200};
    rhf.run(geometry);
    const double got = rhf.energy();
    int rc = 0;
    if (!rhf.converged() || !std::isfinite(got)
        || std::abs(got - e_want) > kTol) {
        std::fprintf(stderr,
                     "FAIL [cpu]: %s const E got=%.12f want=%.12f "
                     "tol=%.0e Ha conv=%d\n",
                     label, got, e_want, kTol, rhf.converged() ? 1 : 0);
        rc = 1;
    }
    rc |= frame_unchanged(label, frame, s);
    if (rc == 0) {
        std::printf("PASS [cpu]: %s const E=%.12f Ha (want %.12f +/- %.0e) "
                    "frame unchanged [PySCF 2.14.0]\n",
                    label, got, e_want, kTol);
    }
    return rc;
}

} // namespace

int run() {
    int rc = 0;
    rc |= check_cpu("H2/STO-3G", make_h2(), -1.1167593073964255);
    rc |= check_cpu("H2O/STO-3G", make_h2o(), -74.96302313846289);
    return rc;
}

} // namespace atv_reg_scf_sto3g_const

ATV_REGRESSION("scf/rhf", "sto3g_const", atv_reg_scf_sto3g_const::run);
