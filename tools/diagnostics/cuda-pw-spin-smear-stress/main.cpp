#include "../../qe_oracle/inputs/pw-spin-smear-stress/pins.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cuda/planewave/pw_solver.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ATV;
namespace {
using Stress = std::array<double, 6>;
constexpr double step = 0.002;
constexpr double twin_tol = 5e-4;
constexpr double fd_tol = 2e-3;
constexpr double rotation_tol = 2e-3;
double selected_width = .02;
constexpr int axis_a[6] = {0, 1, 2, 0, 0, 1};
constexpr int axis_b[6] = {0, 1, 2, 1, 2, 2};
void require(bool condition, const std::string& why) {
    if (!condition)
        throw std::runtime_error(why);
}
void near(double a, double b, double tol, const std::string& why) {
    require(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) < tol,
            why + " got=" + std::to_string(a)
                + " expected=" + std::to_string(b));
}
cpu::Frame geometry() {
    cpu::Frame f(3);
    Mat3 cell;
    const double h[3][3] = {{4.2, .3, .2}, {0, 4.0, .1}, {0, 0, 3.8}};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cell(i, j) = h[i][j];
    f.box.set_cell(cell);
    const double xyz[3][3] = {
        {1.2, 1.1, 1.0}, {2.1, 1.6, 1.3}, {1.6, 2.25, 1.8}};
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_mass)[i] = 1.008;
        f.span(kAtom_x)[i] = xyz[i][0];
        f.span(kAtom_y)[i] = xyz[i][1];
        f.span(kAtom_z)[i] = xyz[i][2];
    }
    return f;
}
// General deformation, deliberately independent from the production symmetric
// strain helper. A simple shear changes only F_ab, so its derivative is the
// physical sigma_ab, with no factor of two to share with the implementation.
cpu::Frame transform(const cpu::Frame& source, const Mat3& deformation) {
    auto out = source;
    Mat3 cell;
    const auto old = source.box.cell();
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            cell(a, b) = 0;
            for (int k = 0; k < 3; ++k)
                cell(a, b) += deformation(a, k) * old(k, b);
        }
    out.box.set_cell(cell);
    for (int i = 0; i < out.nlocal(); ++i) {
        const double r[3] = {source.span(kAtom_x)[i], source.span(kAtom_y)[i],
                             source.span(kAtom_z)[i]};
        double moved[3] = {0, 0, 0};
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                moved[a] += deformation(a, b) * r[b];
        out.span(kAtom_x)[i] = moved[0];
        out.span(kAtom_y)[i] = moved[1];
        out.span(kAtom_z)[i] = moved[2];
    }
    return out;
}
Mat3 identity() {
    Mat3 out;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out(i, j) = i == j ? 1.0 : 0.0;
    return out;
}
template <class Method>
void controls(Method& m, cpu::pw::Smearing smear) {
    m.spin_polarized = true;
    m.multiplicity = 2;
    m.tol = 1e-9;
    m.max_iter = 250;
    m.mix_beta = .35;
    m.smearing = smear;
    m.degauss = smear == cpu::pw::Smearing::fermi_dirac ? selected_width
                                                        : 2 * selected_width;
    m.extra_bands = 3;
    m.mp_order = 1;
}
struct FDEvaluation {
    Stress stress{};
    bool converged = true;
    std::vector<std::string> failures;
};
template <class Method>
FDEvaluation simple_fd(const cpu::Frame& original, Method& method,
                       double fd_step = step) {
    FDEvaluation out;
    for (int v = 0; v < 6; ++v) {
        auto plus = identity(), minus = identity();
        plus(axis_a[v], axis_b[v]) += fd_step;
        minus(axis_a[v], axis_b[v]) -= fd_step;
        auto fp = transform(original, plus), fm = transform(original, minus);
        const auto rp = method.run(fp), rm = method.run(fm);
        std::printf(
            "SIMPLE_SAMPLE v=%d Eplus=%.14g Eminus=%.14g Ry "
            "Mplus=%.10g Mminus=%.10g iterations=%d,%d converged=%d,%d\n",
            v, rp.energy, rm.energy, rp.magnetization, rm.magnetization,
            rp.iterations, rm.iterations, rp.converged, rm.converged);
        for (const auto& sample : {std::pair{'+', rp}, std::pair{'-', rm}})
            if (!sample.second.converged) {
                out.converged = false;
                out.failures.push_back(
                    "v=" + std::to_string(v) + " sign=" + sample.first
                    + " iterations="
                    + std::to_string(sample.second.iterations));
            }
        out.stress[v] = -(rp.energy - rm.energy)
                        * cpu::pw::qe_units::rydberg_to_ev
                        / (2 * fd_step * original.box.volume());
    }
    return out;
}
double difference(const Stress& a, const Stress& b) {
    double max = 0;
    for (int i = 0; i < 6; ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite stress");
        max = std::max(max, std::abs(a[i] - b[i]));
    }
    return max;
}
void metadata(const cpu::pw::PWResult& got, const cpu::pw::PWResult& ref) {
    require(got.converged == ref.converged && got.iterations == ref.iterations,
            "stress reference convergence metadata");
    require(got.n_kpts == ref.n_kpts && got.n_kpts_local == ref.n_kpts_local,
            "stress reference k-point metadata");
    near(got.energy, ref.energy, 1e-8, "stress reference energy");
    near(got.magnetization, ref.magnetization, 1e-8,
         "stress reference magnetization");
    near(got.e_fermi, ref.e_fermi, 1e-8, "stress reference chemical potential");
    near(got.e_entropy, ref.e_entropy, 1e-8, "stress reference entropy");
    near(got.n_elec_occ, ref.n_elec_occ, 1e-8, "stress reference occupations");
}
Mat3 rotation() {
    // R_z(0.37) R_y(0.23), a proper orthogonal matrix.
    auto q = identity();
    const double c = std::cos(.37), s = std::sin(.37), a = std::cos(.23),
                 b = std::sin(.23);
    q(0, 0) = c * a;
    q(0, 1) = -s;
    q(0, 2) = c * b;
    q(1, 0) = s * a;
    q(1, 1) = c;
    q(1, 2) = s * b;
    q(2, 0) = -b;
    q(2, 1) = 0;
    q(2, 2) = a;
    return q;
}
Stress rotate_stress(const Stress& old, const Mat3& q) {
    double tensor[3][3] = {{old[0], old[3], old[4]},
                           {old[3], old[1], old[5]},
                           {old[4], old[5], old[2]}};
    Stress result{};
    for (int v = 0; v < 6; ++v)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result[v] += q(axis_a[v], i) * tensor[i][j] * q(axis_b[v], j);
    return result;
}
void json_array(std::ostream& out, const Stress& data) {
    out << '[';
    for (int i = 0; i < 6; ++i) {
        if (i)
            out << ',';
        out << data[i];
    }
    out << ']';
}
const qe_spin_stress::Pin& expected_pin(cpu::scf::XCType xc,
                                        cpu::pw::Smearing smear, double width) {
    const std::string functional =
        xc == cpu::scf::XCType::LSDA ? "LSDA" : "PBE";
    const std::string kind = smear == cpu::pw::Smearing::fermi_dirac ? "fd"
                             : smear == cpu::pw::Smearing::gaussian ? "gaussian"
                                                                    : "mp1";
    for (const auto& pin : qe_spin_stress::pins)
        if (functional == pin.xc && kind == pin.smearing
            && std::abs(width - pin.degauss_ry) < 1e-14)
            return pin;
    throw std::runtime_error(
        "no actual QE pin for this input; refusing an unmatched comparison");
}
void qe_pin(const cpu::pw::PWResult& reference, const Stress& stress,
            cpu::scf::XCType xc, cpu::pw::Smearing smear, double width) {
    const auto& pin = expected_pin(xc, smear, width);
    near(reference.energy_ha, pin.energy_ha, 1e-5,
         "actual QE7.6 H3 absolute free energy");
    for (int v = 0; v < 6; ++v)
        near(stress[v], pin.stress_ev_ang3[v], 2e-3,
             "actual QE7.6 physical stress component " + std::to_string(v));
    std::printf(
        "PASS actual QE7.6 xc=%d smear=%d width=%.9g energy error=%.12g Ha "
        "stress max error=%.12g eV/A3\n",
        static_cast<int>(xc), static_cast<int>(smear), width,
        std::abs(reference.energy_ha - pin.energy_ha),
        difference(stress, pin.stress_ev_ang3));
}

bool run_case(cpu::scf::XCType xc, cpu::pw::Smearing smear, std::ostream& out,
              bool rotate, int mesh = 1) {
    auto f = geometry();
    if (mesh > 1) {
        auto cell = f.box.cell();
        cell(0, 0) = 6.2;
        cell(1, 1) = 6.;
        cell(2, 2) = 5.8;
        f.box.set_cell(cell);
    }
    auto pp = std::make_shared<cpu::pw::GthPseudopotential>(molrs::Element::H,
                                                            "GTH-PBE-q1");
    cuda::pw::PW gpu{pp, 6., mesh, xc}, fd_gpu{pp, 6., mesh, xc};
    cpu::pw::PW cpu{pp, 6., mesh, xc};
    controls(gpu, smear);
    controls(fd_gpu, smear);
    controls(cpu, smear);
    std::vector<std::string> failures;
    const auto check = [&](bool good, const std::string& reason) {
        if (!good)
            failures.push_back(reason);
    };
    const auto attempt = [&](const std::string& name, auto operation) {
        try {
            operation();
            return true;
        } catch (const std::exception& e) {
            failures.push_back(name + ": " + e.what());
            return false;
        }
    };
    const auto ref = gpu.run(f);
    std::printf("REFERENCE xc=%d smear=%d mesh=%d width=%.9g M=%.12g "
                "entropy=%.12g Ry E=%.12g Ry iterations=%d converged=%d\n",
                static_cast<int>(xc), static_cast<int>(smear), mesh,
                gpu.degauss, ref.magnetization, ref.e_entropy, ref.energy,
                ref.iterations, ref.converged);
    check(ref.converged, "reference SCF did not converge");
    check(std::abs(ref.magnetization) > 1e-3,
          "reference magnetization vacuous");
    check(std::abs(ref.e_entropy) > 1e-5, "thermal correction vacuous");
    attempt("electrons",
            [&] { near(ref.n_elec_occ, 3., 1e-8, "electron count"); });
    const auto force_result = gpu.forces(f);
    check(force_result.converged, "force reference SCF did not converge");
    double amplitude = 0;
    for (const auto key : {kAtom_fx, kAtom_fy, kAtom_fz})
        for (int i = 0; i < f.nlocal(); ++i)
            amplitude = std::max(amplitude,
                                 std::abs(static_cast<double>(f.span(key)[i])));
    check(amplitude > .01, "forces vacuous");
    const auto sg = gpu.stress(f);
    attempt("metadata", [&] { metadata(gpu.last_result(), ref); });
    attempt("Frame energy", [&] {
        near(f.meta.get(kEnergy), ref.energy * cpu::pw::qe_units::rydberg_to_ev,
             1e-9, "Frame stress energy");
    });
    for (int v = 0; v < 6; ++v)
        attempt("Frame virial", [&] {
            near(f.meta.get(kStress)[v], sg[v] * f.box.volume(), 1e-10,
                 "Frame stress virial");
        });
    check(std::max({std::abs(sg[3]), std::abs(sg[4]), std::abs(sg[5])}) > 1e-4,
          "shear stress vacuous");
    const bool qe_ok = mesh == 1 && attempt("QE", [&] {
                           qe_pin(ref, sg, xc, smear, gpu.degauss);
                       });
    const auto sc = cpu.stress(f);
    check(cpu.last_result().converged,
          "CPU stress reference SCF did not converge");
    const double fd_step = mesh > 1 ? 1e-4 : step;
    const auto cg = simple_fd(f, cpu, fd_step),
               gg = simple_fd(f, fd_gpu, fd_step);
    check(cg.converged,
          "CPU independent fullSCF FD contains unconverged samples");
    check(gg.converged,
          "CUDA independent fullSCF FD contains unconverged samples");
    const auto twin_error = difference(sg, sc),
               gpu_fd_error = difference(sg, gg.stress),
               cpu_fd_error = difference(sc, cg.stress);
    std::printf("xc=%d smear=%d M=%.9g correction=%.9g Ry |F|max=%.9g "
                "stressTwin=%.9g gpuSimpleFD=%.9g cpuSimpleFD=%.9g eV/A3 "
                "CPU_FD_converged=%d CUDA_FD_converged=%d\n",
                static_cast<int>(xc), static_cast<int>(smear),
                ref.magnetization, ref.e_entropy, amplitude, twin_error,
                gpu_fd_error, cpu_fd_error, cg.converged, gg.converged);
    for (const auto& entry :
         {std::pair{"CUDA_NSCF", sg}, std::pair{"CPU_NSCF", sc},
          std::pair{"CUDA_SIMPLE_FD", gg.stress},
          std::pair{"CPU_SIMPLE_FD", cg.stress}}) {
        std::printf("%s", entry.first);
        for (auto value : entry.second)
            std::printf(" %.12g", value);
        std::printf("\n");
    }
    check(twin_error < twin_tol,
          "CPU/CUDA NSCF stress twin exceeds original gate");
    check(gpu_fd_error < fd_tol,
          "CUDA NSCF/fullSCF stress difference exceeds original gate");
    check(cpu_fd_error < fd_tol,
          "CPU NSCF/fullSCF stress difference exceeds original gate");
    double rotation_error = 0;
    if (rotate) {
        auto q = rotation();
        auto rf = transform(f, q);
        const auto sr = gpu.stress(rf);
        check(gpu.last_result().converged,
              "rotated reference SCF did not converge");
        rotation_error = difference(sr, rotate_stress(sg, q));
        std::printf("rotation max tensor error=%.12g eV/A3\n", rotation_error);
        check(rotation_error < rotation_tol,
              "rotation covariance exceeds original gate");
    }
    gpu.max_iter = 1;
    const auto short_ref = gpu.run(f);
    gpu.stress(f);
    check(!short_ref.converged,
          "one-iteration reference unexpectedly converged");
    attempt("truncated reuse", [&] { metadata(gpu.last_result(), short_ref); });
    const bool passed = failures.empty();
    out << "{\"xc\":" << static_cast<int>(xc)
        << ",\"smearing\":" << static_cast<int>(smear) << ",\"mesh\":" << mesh
        << ",\"width_ry\":" << gpu.degauss
        << ",\"passed\":" << (passed ? "true" : "false")
        << ",\"external_qe_checked\":" << (mesh == 1 ? "true" : "false")
        << ",\"external_qe_passed\":" << (qe_ok ? "true" : "false")
        << ",\"reference_converged\":" << (ref.converged ? "true" : "false")
        << ",\"energy_ha\":" << ref.energy_ha
        << ",\"iterations\":" << ref.iterations
        << ",\"independent_fd_step\":" << fd_step
        << ",\"cpu_fd_converged\":" << (cg.converged ? "true" : "false")
        << ",\"cuda_fd_converged\":" << (gg.converged ? "true" : "false")
        << ",\"magnetization\":" << ref.magnetization
        << ",\"entropy_ry\":" << ref.e_entropy
        << ",\"max_force_ev_ang\":" << amplitude << ",\"cuda_stress_ev_ang3\":";
    json_array(out, sg);
    out << ",\"cpu_stress_ev_ang3\":";
    json_array(out, sc);
    out << ",\"cuda_simple_fd_stress_ev_ang3\":";
    json_array(out, gg.stress);
    out << ",\"cpu_full_fd_stress_ev_ang3\":";
    json_array(out, cg.stress);
    out << ",\"twin_error\":" << twin_error
        << ",\"cuda_simple_fd_error\":" << gpu_fd_error
        << ",\"cpu_full_fd_error\":" << cpu_fd_error
        << ",\"rotation_error\":" << rotation_error;
    const auto list = [&](const char* name, const auto& strings) {
        out << ",\"" << name << "\":[";
        bool first = true;
        for (const auto& entry : strings) {
            if (!first)
                out << ',';
            first = false;
            out << std::quoted(entry);
        }
        out << ']';
    };
    list("cpu_fd_failures", cg.failures);
    list("cuda_fd_failures", gg.failures);
    list("failures", failures);
    out << '}';
    out.flush();
    std::printf("%s xc=%d smear=%d reference_QE=%d CPU_FD=%d CUDA_FD=%d\n",
                passed ? "PASS" : "FAIL", static_cast<int>(xc),
                static_cast<int>(smear), qe_ok, cg.converged, gg.converged);
    return passed;
}

} // namespace
int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--stress-smoke") {
        try {
            auto pp = std::make_shared<cpu::pw::GthPseudopotential>(
                molrs::Element::H, "GTH-PBE-q1");
            cuda::pw::PW gpu{pp, 6.0, 1, cpu::scf::XCType::PBE};
            controls(gpu, cpu::pw::Smearing::fermi_dirac);
            auto frame = geometry();
            const auto ref = gpu.run(frame);
            require(ref.converged, "stress smoke reference convergence");
            gpu.forces(frame);
            const auto stress = gpu.stress(frame);
            metadata(gpu.last_result(), ref);
            qe_pin(ref, stress, gpu.xc, gpu.smearing, gpu.degauss);
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FAIL: %s\n", e.what());
            return 1;
        }
    }
    if (argc > 1 && std::string(argv[1]) == "--probe") {
        auto pp = std::make_shared<cpu::pw::GthPseudopotential>(
            molrs::Element::H, "GTH-PBE-q1");
        for (auto xc : {cpu::scf::XCType::LSDA, cpu::scf::XCType::PBE})
            for (double width : {.12, .04, .02}) {
                auto frame = geometry();
                cuda::pw::PW gpu{pp, 6.0, 1, xc};
                controls(gpu, cpu::pw::Smearing::fermi_dirac);
                gpu.degauss = width;
                const auto result = gpu.run(frame);
                std::printf("PROBE xc=%d width=%.9g M=%.12g entropy=%.12g "
                            "E=%.12g Ry N=%.12g converged=%d iterations=%d\n",
                            static_cast<int>(xc), width, result.magnetization,
                            result.e_entropy, result.energy, result.n_elec_occ,
                            result.converged, result.iterations);
            }
        return 0;
    }
    const std::string path = argc > 1 ? argv[1] : "spin-smear-stress.json";
    if (argc > 2)
        selected_width = std::stod(argv[2]);
    std::ofstream out(path);
    out << std::setprecision(17);
    out << "{\"qualification\":\"CPU twin and independent fullSCF simple "
           "shear; "
           "actual QE7.6 LSDA/PBE x FD/Gaussian/MP1 Gamma stress and energy "
           "pins\",\"width_ry\":"
        << selected_width << ",\"twin_tolerance\":" << twin_tol
        << ",\"fd_tolerance\":" << fd_tol
        << ",\"rotation_tolerance\":" << rotation_tol << ",\"cases\":[";
    bool first = true, all_passed = true;
    auto append = [&](cpu::scf::XCType xc, cpu::pw::Smearing smear, int mesh,
                      bool rotate) {
        std::ostringstream row;
        row << std::setprecision(17);
        bool passed = false;
        try {
            passed = run_case(xc, smear, row, rotate, mesh);
        } catch (const std::exception& e) {
            row.str("");
            row.clear();
            row << "{\"xc\":" << static_cast<int>(xc)
                << ",\"smearing\":" << static_cast<int>(smear)
                << ",\"mesh\":" << mesh
                << ",\"passed\":false,\"exception\":" << std::quoted(e.what())
                << '}';
        }
        all_passed = all_passed && passed;
        if (!first)
            out << ',';
        first = false;
        out << row.str();
        out.flush();
    };
    try {
        if (argc > 3 && std::string(argv[3]) == "multi-k") {
            append(cpu::scf::XCType::PBE, cpu::pw::Smearing::fermi_dirac, 2,
                   false);
        } else {
            for (const auto xc :
                 {cpu::scf::XCType::LSDA, cpu::scf::XCType::PBE})
                for (const auto smear : {cpu::pw::Smearing::fermi_dirac,
                                         cpu::pw::Smearing::gaussian,
                                         cpu::pw::Smearing::methfessel_paxton})
                    append(xc, smear, 1,
                           smear == cpu::pw::Smearing::fermi_dirac);
        }
        out << "],\"passed\":" << (all_passed ? "true" : "false") << "}\n";
        return all_passed ? 0 : 1;
    } catch (const std::exception& e) {
        out << "],\"passed\":false,\"failure\":" << std::quoted(e.what())
            << "}\n";
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
