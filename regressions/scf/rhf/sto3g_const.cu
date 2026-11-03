/// @file regressions/scf/rhf/sto3g_const.cu
/// @brief RHF run(const Frame&) on H2/STO-3G and H2O/STO-3G.
///
/// --device=cpu|cuda|both (default both). Both systems are STO-3G l <= 1,
/// so the CUDA leg is a device SCF (launch_*_sp) that must not assemble
/// launch_rhf_forces_sp. energy() [Hartree] matches the PySCF pin; the
/// caller's Frame is unchanged. See REFERENCE.md.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cuda/scf/rhf.cuh"
#include "harness/registry.h"

namespace atv_reg_scf_sto3g_const {

namespace {

using ATV::cpu::Frame;

std::string parse_device_arg(int argc, char** argv) {
    std::string want = "both";
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strncmp(a, "--device=", 9) == 0) {
            want = a + 9;
        } else if (std::strcmp(a, "--device") == 0 && i + 1 < argc) {
            want = argv[++i];
        } else if (std::strcmp(a, "cpu") == 0 || std::strcmp(a, "cuda") == 0
                   || std::strcmp(a, "both") == 0) {
            want = a;
        }
    }
    return want;
}

Frame make_h2() {
    using namespace ATV;
    Frame frame(2);
    frame.span(kAtom_atomic_number)[0] = 1;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_atomic_number)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = 0.74;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

Frame make_h2o() {
    using namespace ATV;
    Frame frame(3);
    frame.span(kAtom_atomic_number)[0] = 8;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_atomic_number)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_atomic_number)[2] = 1;
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

int frame_unchanged(const char* device, const char* label, const Frame& frame,
                    const Stamp& s) {
    using namespace ATV;
    const double e = static_cast<double>(frame.meta.get(kEnergy));
    const double ee = static_cast<double>(frame.meta.get(kEnergyElectronic));
    if (e != s.energy || ee != s.electronic) {
        std::fprintf(stderr,
                     "FAIL [%s]: %s const wrote meta "
                     "kEnergy got=%.12f want=%.12f "
                     "kEnergyElectronic got=%.12f want=%.12f\n",
                     device, label, e, s.energy, ee, s.electronic);
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
                         "FAIL [%s]: %s const wrote atom %d "
                         "x got=%.12f want=%.12f y got=%.12f want=%.12f "
                         "z got=%.12f want=%.12f "
                         "fx got=%.12f want=%.12f "
                         "fy got=%.12f want=%.12f "
                         "fz got=%.12f want=%.12f\n",
                         device, label, i, x, s.x[k], y, s.y[k], z, s.z[k], fx,
                         s.fx[k], fy, s.fy[k], fz, s.fz[k]);
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
    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, 1e-12, 200};
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
    rc |= frame_unchanged("cpu", label, frame, s);
    if (rc == 0) {
        std::printf("PASS [cpu]: %s const E=%.12f Ha (want %.12f +/- %.0e) "
                    "frame unchanged [PySCF 2.14.0]\n",
                    label, got, e_want, kTol);
    }
    return rc;
}

int check_cuda(const char* label, Frame frame, double e_want) {
    using namespace ATV;
    constexpr double kTol = 1e-6;
    const Stamp s = stamp_frame(frame);
    const Frame& geometry = frame;
    cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, 1e-12, 200};
    if (!rhf.compute_force) {
        std::fprintf(stderr, "FAIL [cuda]: %s compute_force default is false\n",
                     label);
        return 1;
    }
    rhf.run(geometry);
    const double got = rhf.energy();
    int rc = 0;
    if (!rhf.last_scf_on_device() || rhf.last_force_on_device()
        || !rhf.compute_force) {
        std::fprintf(stderr,
                     "FAIL [cuda]: %s const path scf_on_device=%d "
                     "force_on_device=%d compute_force=%d\n",
                     label, rhf.last_scf_on_device() ? 1 : 0,
                     rhf.last_force_on_device() ? 1 : 0,
                     rhf.compute_force ? 1 : 0);
        rc = 1;
    }
    if (!rhf.converged() || !std::isfinite(got)
        || std::abs(got - e_want) > kTol) {
        std::fprintf(stderr,
                     "FAIL [cuda]: %s const E got=%.12f want=%.12f "
                     "tol=%.0e Ha conv=%d\n",
                     label, got, e_want, kTol, rhf.converged() ? 1 : 0);
        rc = 1;
    }
    rc |= frame_unchanged("cuda", label, frame, s);
    if (rc == 0) {
        std::printf("PASS [cuda]: %s const E=%.12f Ha (want %.12f +/- %.0e) "
                    "frame unchanged path=device [PySCF 2.14.0]\n",
                    label, got, e_want, kTol);
    }
    return rc;
}

} // namespace

int run(int argc, char** argv) {
    const std::string want = parse_device_arg(argc, argv);
    if (want != "cpu" && want != "cuda" && want != "both") {
        std::fprintf(stderr, "FAIL: unknown --device=%s (want cpu|cuda|both)\n",
                     want.c_str());
        return 1;
    }
    int rc = 0;
    if (want == "cpu" || want == "both") {
        rc |= check_cpu("H2/STO-3G", make_h2(), -1.1167593073964255);
        rc |= check_cpu("H2O/STO-3G", make_h2o(), -74.96302313846289);
    }
    if (want == "cuda" || want == "both") {
        rc |= check_cuda("H2/STO-3G", make_h2(), -1.1167593073964255);
        rc |= check_cuda("H2O/STO-3G", make_h2o(), -74.96302313846289);
    }
    return rc;
}

} // namespace atv_reg_scf_sto3g_const

ATV_REGRESSION("scf/rhf", "sto3g_const", atv_reg_scf_sto3g_const::run);
