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
#include <stdexcept>
#include <string>

using namespace ATV;
namespace {
using Stress = std::array<double, 6>;
constexpr double step = 0.002;
constexpr double twin_tol = 5e-4;
constexpr double fd_tol = 2e-3;
constexpr double rotation_tol = 2e-3;
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
    m.degauss = .12;
    m.extra_bands = 3;
    m.mp_order = 1;
}
template <class Method>
Stress simple_fd(const cpu::Frame& original, Method& method) {
    Stress stress{};
    for (int v = 0; v < 6; ++v) {
        auto plus = identity(), minus = identity();
        plus(axis_a[v], axis_b[v]) += step;
        minus(axis_a[v], axis_b[v]) -= step;
        auto fp = transform(original, plus), fm = transform(original, minus);
        const auto rp = method.run(fp), rm = method.run(fm);
        require(rp.converged && rm.converged,
                "simple-shear fullSCF failed convergence");
        stress[v] = -(rp.energy - rm.energy) * cpu::pw::qe_units::rydberg_to_ev
                    / (2 * step * original.box.volume());
    }
    return stress;
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
void run_case(cpu::scf::XCType xc, cpu::pw::Smearing smear, std::ostream& out,
              bool rotate) {
    auto f = geometry();
    auto pp = std::make_shared<cpu::pw::GthPseudopotential>(molrs::Element::H,
                                                            "GTH-PBE-q1");
    cuda::pw::PW gpu{pp, 6.0, 1, xc}, fd_gpu{pp, 6.0, 1, xc};
    cpu::pw::PW cpu{pp, 6.0, 1, xc};
    controls(gpu, smear);
    controls(fd_gpu, smear);
    controls(cpu, smear);
    const auto ref = gpu.run(f);
    require(ref.converged, "reference SCF did not converge");
    require(std::abs(ref.magnetization) > 1e-3,
            "reference magnetization vacuous");
    require(std::abs(ref.e_entropy) > 1e-5, "thermal entropy vacuous");
    near(ref.n_elec_occ, 3.0, 1e-8, "electron count");
    gpu.forces(f);
    double amplitude = 0;
    for (const auto key : {kAtom_fx, kAtom_fy, kAtom_fz})
        for (int i = 0; i < f.nlocal(); ++i)
            amplitude = std::max(amplitude,
                                 std::abs(static_cast<double>(f.span(key)[i])));
    require(amplitude > .01, "forces vacuous");
    const auto sg = gpu.stress(f);
    metadata(gpu.last_result(), ref);
    near(f.meta.get(kEnergy), ref.energy * cpu::pw::qe_units::rydberg_to_ev,
         1e-9, "Frame stress energy");
    for (int v = 0; v < 6; ++v)
        near(f.meta.get(kStress)[v], sg[v] * f.box.volume(), 1e-10,
             "Frame stress virial");
    require(std::max({std::abs(sg[3]), std::abs(sg[4]), std::abs(sg[5])})
                > 1e-4,
            "shear stress vacuous");
    const auto sc = cpu.stress(f);
    require(cpu.last_result().converged,
            "CPU stress reference did not converge");
    const auto cg = cpu::pw::pw_stress_fd(f, cpu, step);
    const auto gg = simple_fd(f, fd_gpu);
    const auto twin_error = difference(sg, sc),
               gpu_fd_error = difference(sg, gg),
               cpu_fd_error = difference(sc, cg);
    std::printf("xc=%d smear=%d M=%.9g entropy=%.9g Ry |F|max=%.9g "
                "stressTwin=%.9g gpuSimpleFD=%.9g cpuFullFD=%.9g eV/A3\n",
                static_cast<int>(xc), static_cast<int>(smear),
                ref.magnetization, ref.e_entropy, amplitude, twin_error,
                gpu_fd_error, cpu_fd_error);
    require(twin_error < twin_tol, "CPU/CUDA stress twin exceeded gate");
    require(gpu_fd_error < fd_tol,
            "CUDA stress vs simple-shear fullSCF exceeded gate");
    require(cpu_fd_error < fd_tol, "CPU NSCF vs fullSCF exceeded gate");
    double rotation_error = 0;
    if (rotate) {
        const auto q = rotation();
        auto rf = transform(f, q);
        const auto sr = gpu.stress(rf), expected = rotate_stress(sg, q);
        require(gpu.last_result().converged, "rotated SCF did not converge");
        rotation_error = difference(sr, expected);
        std::printf("rotation max tensor error=%.9g eV/A3\n", rotation_error);
        require(rotation_error < rotation_tol,
                "rotated stress tensor covariance exceeded gate");
    }
    out << "{\"xc\":" << static_cast<int>(xc)
        << ",\"smearing\":" << static_cast<int>(smear)
        << ",\"magnetization\":" << ref.magnetization
        << ",\"entropy_ry\":" << ref.e_entropy
        << ",\"max_force_ev_ang\":" << amplitude << ",\"cuda_stress_ev_ang3\":";
    json_array(out, sg);
    out << ",\"cpu_stress_ev_ang3\":";
    json_array(out, sc);
    out << ",\"cuda_simple_fd_stress_ev_ang3\":";
    json_array(out, gg);
    out << ",\"cpu_full_fd_stress_ev_ang3\":";
    json_array(out, cg);
    out << ",\"twin_error\":" << twin_error
        << ",\"cuda_simple_fd_error\":" << gpu_fd_error
        << ",\"cpu_full_fd_error\":" << cpu_fd_error
        << ",\"rotation_error\":" << rotation_error << "}";
    out.flush();
    // Reuse must publish the current truncated reference SCF, not the last
    // converged result from either the base or rotated geometry.
    gpu.max_iter = 1;
    const auto short_ref = gpu.run(f);
    require(!short_ref.converged,
            "one-iteration reference unexpectedly converged");
    gpu.stress(f);
    metadata(gpu.last_result(), short_ref);
    std::printf("PASS xc=%d smear=%d physical stress+metadata+reuse\n",
                static_cast<int>(xc), static_cast<int>(smear));
}
} // namespace
int main(int argc, char** argv) {
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
    std::ofstream out(path);
    out << std::setprecision(17);
    try {
        out << "{\"qualification\":\"CPU twin and internal fullSCF "
               "consistency; no external stress oracle\",\"twin_tolerance\":"
            << twin_tol << ",\"fd_tolerance\":" << fd_tol
            << ",\"rotation_tolerance\":" << rotation_tol << ",\"cases\":[";
        bool first = true;
        for (const auto xc : {cpu::scf::XCType::LSDA, cpu::scf::XCType::PBE})
            for (const auto smear :
                 {cpu::pw::Smearing::fermi_dirac, cpu::pw::Smearing::gaussian,
                  cpu::pw::Smearing::methfessel_paxton}) {
                if (!first)
                    out << ',';
                first = false;
                run_case(xc, smear, out,
                         xc == cpu::scf::XCType::PBE
                             && smear == cpu::pw::Smearing::fermi_dirac);
            }
        out << "],\"passed\":true}\n";
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
