/// @file regressions/scf/h2o_rhf_sto3g_grad/reg_h2o_rhf_sto3g_grad.cu
/// @brief H2O / STO-3G RHF analytic gradient vs PySCF — **one case**,
///        `--device=cpu|cuda|both` (default both). See REFERENCE.md.
///
/// Public RHF + Frame only. The CUDA leg is the delivered DEVICE s/p analytic
/// force path (`launch_rhf_forces_sp`) and is **fail-closed**: it asserts
/// `last_scf_on_device()` and `last_force_on_device()`, so a silent host
/// fallback (which would grade `cpu::scf::RHF::forces` twice and prove
/// nothing about CUDA) fails loudly instead of passing. On top of the shared
/// PySCF pin the CUDA leg grades same-settings CPU parity — the sanctioned
/// CUDA-ES correctness oracle.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cuda/scf/rhf.cuh"

namespace {

enum class Device { Cpu, Cuda };
const char* device_name(Device d) { return d == Device::Cuda ? "cuda" : "cpu"; }

std::string parse_device_arg(int argc, char** argv) {
    std::string want = "both";
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strncmp(a, "--device=", 9) == 0)
            want = a + 9;
        else if (std::strcmp(a, "--device") == 0 && i + 1 < argc)
            want = argv[++i];
        else if (std::strcmp(a, "cpu") == 0 || std::strcmp(a, "cuda") == 0
                 || std::strcmp(a, "both") == 0)
            want = a;
    }
    return want;
}

ATV::cpu::Frame make_frame() {
    using namespace ATV;
    cpu::Frame frame(3);
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

/// @brief dE/dR [Ha/Bohr] from a GradientResult force entry [eV/Å].
double grad_ha_bohr(const ATV::cpu::scf::GradientResult& res, int A, int d) {
    using namespace ATV;
    return -units::convert(res.forces(A, d), units::Force::eVPerAngstrom,
                           units::Force::HartreePerBohr);
}

int run_on_device(Device dev) {
    using namespace ATV;
    cpu::Frame frame = make_frame();
    cpu::scf::GradientResult res;
    bool converged = false;

    if (dev == Device::Cpu) {
        cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, 1e-12, 200};
        res = rhf.forces(frame);
        converged = rhf.converged();
    } else {
        cuda::scf::RHF rhf{cpu::scf::BasisSet::STO3G, 0, 1e-12, 200};
        res = rhf.forces(frame);
        converged = rhf.converged();

        // Fail-closed honesty gate: H2O/STO-3G is inside the device s/p
        // envelope (`device_sp_sto3g_ok`), so SCF *and* force assembly must
        // report the device kernels. A host fallback here would grade
        // cpu::scf::RHF::forces under a "cuda" banner — fail loudly instead.
        if (!rhf.last_scf_on_device() || !rhf.last_force_on_device()) {
            std::fprintf(stderr,
                         "FAIL [cuda]: expected device s/p path "
                         "(scf_on_device=%d force_on_device=%d) — a silent "
                         "host fallback proves nothing about CUDA\n",
                         rhf.last_scf_on_device() ? 1 : 0,
                         rhf.last_force_on_device() ? 1 : 0);
            return 1;
        }
        std::printf("[cuda] path=device (s/p STO-3G SCF + analytic forces)\n");
    }

    if (!converged) {
        std::fprintf(stderr, "FAIL [%s]: RHF forces did not converge\n",
                     device_name(dev));
        return 1;
    }

    // PySCF 2.14.0 pin (REFERENCE.md): E_RHF + analytic gradient [Ha/Bohr].
    const double e_want = -74.963023138463; // Ha
    const double want[3][3] = {
        {0.0, 0.0, -6.142775911808e-02},                // O
        {0.0, -2.364133873480e-02, 3.071387955904e-02}, // H (+y)
        {0.0, 2.364133873480e-02, 3.071387955904e-02}   // H (−y)
    };
    // Stated tolerances (REFERENCE.md): the CPU leg is the analytic twin of
    // the PySCF gradient; the CUDA leg adds device accumulation-order noise.
    const double e_tol = (dev == Device::Cuda) ? 1e-6 : 1e-8;   // Ha
    const double tol = (dev == Device::Cuda) ? 5e-6 : 1e-6;     // Ha/Bohr
    const double net_tol = (dev == Device::Cuda) ? 1e-7 : 1e-8; // Ha/Bohr

    const double e_ha =
        units::convert(res.energy, units::Energy::eV, units::Energy::Hartree);
    if (!(std::fabs(e_ha - e_want) < e_tol)) {
        std::printf("FAIL [%s]: E got=%.12f want=%.12f tol=%.0e Ha\n",
                    device_name(dev), e_ha, e_want, e_tol);
        return 1;
    }

    double max_err = 0.0;
    double net[3] = {0.0, 0.0, 0.0};
    for (int A = 0; A < 3; ++A) {
        for (int d = 0; d < 3; ++d) {
            const double g = grad_ha_bohr(res, A, d);
            net[d] += g;
            max_err = std::max(max_err, std::fabs(g - want[A][d]));
        }
    }
    const double net_max =
        std::max({std::fabs(net[0]), std::fabs(net[1]), std::fabs(net[2])});

    std::printf("[%s] max_elementwise_err=%.3e tol=%.0e  |Σg|_max=%.3e\n",
                device_name(dev), max_err, tol, net_max);

    if (!(max_err < tol)) {
        std::printf("FAIL [%s]: gradient off PySCF oracle\n", device_name(dev));
        return 1;
    }
    if (!(net_max < net_tol)) {
        std::printf("FAIL [%s]: not translationally invariant\n",
                    device_name(dev));
        return 1;
    }

    // CUDA↔CPU same-settings parity — the sanctioned CUDA-ES correctness
    // oracle on top of the external pin. 2e-6 Ha/Bohr ≈ the 1e-4 eV/Å twin
    // tolerance stated in tests/cuda/scf/test_rhf.cu.
    if (dev == Device::Cuda) {
        cpu::Frame frame_cpu = make_frame();
        cpu::scf::RHF cpu_rhf{cpu::scf::BasisSet::STO3G, 0, 1e-12, 200};
        const cpu::scf::GradientResult res_cpu = cpu_rhf.forces(frame_cpu);
        if (!cpu_rhf.converged()) {
            std::fprintf(stderr,
                         "FAIL [cuda]: CPU parity oracle did not converge\n");
            return 1;
        }
        double max_par = 0.0;
        for (int A = 0; A < 3; ++A)
            for (int d = 0; d < 3; ++d)
                max_par =
                    std::max(max_par, std::fabs(grad_ha_bohr(res, A, d)
                                                - grad_ha_bohr(res_cpu, A, d)));
        std::printf("[cuda] cpu_parity_max_err=%.3e tol=2e-06 Ha/Bohr\n",
                    max_par);
        if (!(max_par < 2e-6)) {
            std::printf("FAIL [cuda]: device gradient off same-settings CPU\n");
            return 1;
        }
    }

    std::printf("PASS [%s]\n", device_name(dev));
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::string want = parse_device_arg(argc, argv);
    if (want != "cpu" && want != "cuda" && want != "both") {
        std::fprintf(stderr, "FAIL: unknown --device=%s\n", want.c_str());
        return 1;
    }
    int rc = 0;
    if (want == "cpu" || want == "both")
        rc |= run_on_device(Device::Cpu);
    if (want == "cuda" || want == "both")
        rc |= run_on_device(Device::Cuda);
    return rc;
}
