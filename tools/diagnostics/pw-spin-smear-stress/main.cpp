// Public CPU PW stress checks. This program is an opt-in scientific diagnostic,
// not a CTest unit and not an independent external electronic-structure oracle.
#include "../../qe_oracle/inputs/pw-spin-smear-stress/pins.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef ATV_PW_STRESS_BEFORE
namespace ATV::cpu::pw::internal {
std::array<double, 6>
pw_stress_analytic_impl_before(const ATV::cpu::Frame&, const ATV::cpu::pw::PW&,
                               ATV::cpu::pw::PWResult*,
                               ATV::cpu::pw::internal::PWProblem*);
}
#endif
namespace {
using namespace ATV;
using PW = cpu::pw::PW;
using Result = cpu::pw::PWResult;
using XC = cpu::xc::XCType;
using Six = std::array<double, 6>;
constexpr int ia[6] = {0, 1, 2, 0, 0, 1}, ib[6] = {0, 1, 2, 1, 2, 2};
constexpr double h = 2e-3;
constexpr double normalization_gate = 5e-5;
constexpr double nscf_gate = 2e-3;
constexpr double rotation_gate = 2e-3;
int rank = 0, ranks = 1;
double width_ry = .02;
double mp_width_override = .04;
int scf_max_iter = 250;
double scf_mix_beta = .35;
bool case_failures = false;

void require(bool value, const std::string& message) {
    if (!value)
        throw std::runtime_error(message);
}

double error(const Six& a, const Six& b) {
    double result = 0;
    for (int i = 0; i < 6; ++i)
        result = std::max(result, std::abs(a[i] - b[i]));
    return result;
}

std::string vector_json(const Six& a) {
    std::ostringstream s;
    s << std::setprecision(17) << '[';
    for (int i = 0; i < 6; ++i) {
        if (i)
            s << ',';
        s << a[i];
    }
    return s.str() + ']';
}

cpu::Frame frame() {
    cpu::Frame f(3);
    constexpr double xyz[3][3] = {
        {1.2, 1.1, 1.0}, {2.1, 1.6, 1.3}, {1.6, 2.25, 1.8}};
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_mass)[i] = 1.008;
        f.span(kAtom_x)[i] = xyz[i][0];
        f.span(kAtom_y)[i] = xyz[i][1];
        f.span(kAtom_z)[i] = xyz[i][2];
    }
    Mat3 c{};
    c(0, 0) = 4.2;
    c(0, 1) = .3;
    c(0, 2) = .2;
    c(1, 1) = 4.0;
    c(1, 2) = .1;
    c(2, 2) = 3.8;
    f.box.set_cell(c);
    return f;
}

PW method(XC xc, ATV::electronic::Smearing smear, int kmesh = 1,
          bool spin = true) {
    PW m{std::make_shared<cpu::pw::GthPseudopotential>(molrs::Element::H,
                                                       "GTH-PBE-q1"),
         6., kmesh, xc};
    m.spin_polarized = spin;
    m.multiplicity = spin ? 2 : 1;
    m.smearing = smear;
    m.degauss = width_ry;
    m.extra_bands = 3;
    m.tol = 1e-9;
    m.max_iter = scf_max_iter;
    m.mix_beta = scf_mix_beta;
    return m;
}

cpu::Frame transform(const cpu::Frame& input, const Mat3& F) {
    cpu::Frame out = input;
    const Mat3 c = input.box.cell();
    Mat3 d{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                d(i, j) += F(i, k) * c(k, j);
    out.box.set_cell(d);
    for (int n = 0; n < input.nlocal(); ++n) {
        const double xyz[3] = {input.span(kAtom_x)[n], input.span(kAtom_y)[n],
                               input.span(kAtom_z)[n]};
        double v[3] = {};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                v[i] += F(i, j) * xyz[j];
        out.span(kAtom_x)[n] = v[0];
        out.span(kAtom_y)[n] = v[1];
        out.span(kAtom_z)[n] = v[2];
    }
    return out;
}

// Independent deformation: change ONLY F_ab. For a!=b this includes an
// infinitesimal rigid rotation, which cannot contribute to dE. There is no
// symmetric-shear multiplicity or production stress helper in this path.
Six simple_fd(const cpu::Frame& f, XC xc, ATV::electronic::Smearing smear,
              int mesh = 1) {
    Six sigma{};
    for (int v = 0; v < 6; ++v) {
        double energy[2] = {};
        for (int sign = 0; sign < 2; ++sign) {
            Mat3 F{};
            for (int i = 0; i < 3; ++i)
                F(i, i) = 1;
            F(ia[v], ib[v]) += (sign ? -h : h);
            auto strained = transform(f, F);
            auto m = method(xc, smear, mesh);
            if (rank == 0)
                std::cout << "STRAIN_SAMPLE xc="
                          << (xc == XC::LSDA ? "LSDA" : "PBE")
                          << " smear=" << int(smear) << " axis=" << v
                          << " sign=" << (sign ? -1 : 1) << std::endl;
            const auto result = m.run(strained);
            if (rank == 0)
                std::cout << std::setprecision(17)
                          << "STRAIN_RESULT converged=" << result.converged
                          << " iterations=" << result.iterations
                          << " E_Ha=" << result.energy_ha
                          << " mu_Ry=" << result.e_fermi
                          << " M=" << result.magnetization
                          << " entropy_Ry=" << result.e_entropy << std::endl;
            require(result.converged,
                    "full-SCF strain sample did not converge: xc="
                        + std::string(xc == XC::LSDA ? "LSDA" : "PBE")
                        + " smear=" + std::to_string(int(smear)) + " axis="
                        + std::to_string(v) + " sign=" + (sign ? "-1" : "+1"));
            energy[sign] = result.energy_ha;
        }
        sigma[v] = -(energy[0] - energy[1]) * units::kHartreeInEv
                   / (2 * h * f.box.volume());
    }
    return sigma;
}

Mat3 rotation() {
    // Rodrigues rotation around a generic normalized axis.
    const double angle = .37, c = std::cos(angle), s = std::sin(angle);
    const double a[3] = {1 / std::sqrt(14.), 2 / std::sqrt(14.),
                         3 / std::sqrt(14.)};
    Mat3 R{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R(i, j) = (i == j ? c : 0) + (1 - c) * a[i] * a[j];
    R(0, 1) -= s * a[2];
    R(1, 0) += s * a[2];
    R(0, 2) += s * a[1];
    R(2, 0) -= s * a[1];
    R(1, 2) -= s * a[0];
    R(2, 1) += s * a[0];
    return R;
}

Six rotate_stress(const Six& sigma, const Mat3& R) {
    Mat3 S{}, T{};
    for (int v = 0; v < 6; ++v)
        S(ia[v], ib[v]) = S(ib[v], ia[v]) = sigma[v];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    T(i, j) += R(i, a) * S(a, b) * R(j, b);
    Six result{};
    for (int v = 0; v < 6; ++v)
        result[v] = T(ia[v], ib[v]);
    return result;
}

void metadata(const Result& run, const Result& stress) {
    require(run.converged && stress.converged,
            "reference SCF did not converge");
    require(run.iterations == stress.iterations,
            "reference iterations changed");
    require(run.n_kpts == stress.n_kpts
                && run.n_kpts_local == stress.n_kpts_local,
            "reference k point metadata changed");
    for (auto [x, y] : {std::pair{run.energy, stress.energy},
                        std::pair{run.energy_ha, stress.energy_ha},
                        std::pair{run.e_fermi, stress.e_fermi},
                        std::pair{run.e_entropy, stress.e_entropy},
                        std::pair{run.magnetization, stress.magnetization},
                        std::pair{run.n_elec_occ, stress.n_elec_occ}})
        require(std::abs(x - y) < 1e-10,
                "reference energy/occupation metadata changed");
}

std::string check_case(XC xc, ATV::electronic::Smearing smear, bool rotate) {
    auto f = frame();
    auto run_m = method(xc, smear);
    const auto reference = run_m.run(f);
    run_m.forces(f);
    require(run_m.last_result().converged, "force SCF did not converge");
    double max_force = 0;
    for (int n = 0; n < f.nlocal(); ++n)
        max_force = std::max({max_force, std::abs(f.span(kAtom_fx)[n]),
                              std::abs(f.span(kAtom_fy)[n]),
                              std::abs(f.span(kAtom_fz)[n])});
    require(max_force > .01, "spin-smear case has trivial forces");
    auto stress_m = method(xc, smear);
    const Six sigma = stress_m.stress(f);
    metadata(reference, stress_m.last_result());
    require(std::abs(reference.n_elec_occ - 3) < 1e-7,
            "fractional electron count wrong");
    require(std::abs(reference.magnetization) > 1e-3,
            "spin case lost magnetization");
    require(std::abs(reference.e_entropy) > 1e-5,
            "smearing case has trivial entropy");
    const auto virial = f.meta.get(kStress);
    for (int v = 0; v < 6; ++v)
        require(std::abs(virial[v] - sigma[v] * f.box.volume()) < 1e-10,
                "frame virial mismatch");
    for (int v = 3; v < 6; ++v)
        require(std::abs(sigma[v]) > 1e-4, "trivial shear component");
    // Actual QE 7.6 observations. A model has to match its own XC, smear
    // kind and width; another smearing's energy is never substituted.
    const std::string smear_name =
        smear == ATV::electronic::Smearing::fermi_dirac ? "fd"
        : smear == ATV::electronic::Smearing::gaussian  ? "gaussian"
                                                        : "mp1";
    const qe_spin_stress::Pin* pin = nullptr;
    for (const auto& candidate : qe_spin_stress::pins)
        if (std::string(candidate.xc) == (xc == XC::PBE ? "PBE" : "LSDA")
            && candidate.smearing == smear_name
            && std::abs(candidate.degauss_ry - width_ry) < 1e-14)
            pin = &candidate;
    require(pin != nullptr, "no matched independent QE pin for this model");
    const bool qe_case = true;
    const double qe_stress_error = error(sigma, pin->stress_ev_ang3);
    const double qe_energy_error =
        std::abs(reference.energy_ha - pin->energy_ha);
    require(qe_stress_error < 2e-3,
            "physical stress differs from actual QE tensor");
    require(qe_energy_error < 1e-5, "Mermin energy differs from actual QE");
    Six single{};
    double ns_error = 0;
    std::string fd_failure;
    try {
        single = simple_fd(f, xc, smear);
        ns_error = error(sigma, single);
        require(ns_error < nscf_gate,
                "frozen-density stress differs from full-SCF derivative");
    } catch (const std::exception& exception) {
        fd_failure = exception.what();
        case_failures = true;
    }
    if (rank == 0)
        std::cout << std::setprecision(17) << "STRESS_OBSERVATION xc="
                  << (xc == XC::PBE ? "PBE" : "LSDA") << " smear=" << int(smear)
                  << " width=" << width_ry << " sigma=" << vector_json(sigma)
                  << " simple_fd=" << vector_json(single)
                  << " fd_failure=" << std::quoted(fd_failure)
                  << " nscf_error=" << ns_error << std::endl;
    double rot_error = 0;
    if (rotate) {
        const Mat3 R = rotation();
        auto rotated = transform(f, R);
        auto m = method(xc, smear);
        const Six actual = m.stress(rotated),
                  expected = rotate_stress(sigma, R);
        require(m.last_result().converged, "rotated SCF did not converge");
        rot_error = error(actual, expected);
        require(rot_error < rotation_gate,
                "stress is not a rotated physical tensor");
    }
    std::ostringstream out;
    out << std::setprecision(17) << "{\"xc\":\""
        << (xc == XC::PBE ? "PBE" : "LSDA") << "\",\"smearing\":" << int(smear)
        << ",\"stress_ev_ang3\":" << vector_json(sigma)
        << ",\"simple_full_scf_fd_ev_ang3\":"
        << (fd_failure.empty() ? vector_json(single) : "null")
        << ",\"nscf_error_ev_ang3\":"
        << (fd_failure.empty() ? ([&]() {
               std::ostringstream value;
               value << std::setprecision(17) << ns_error;
               return value.str();
           })()
                               : "null")
        << ",\"full_scf_fd_error\":" << std::quoted(fd_failure)
        << ",\"external_qe_applied\":" << (qe_case ? "true" : "false")
        << ",\"qe_stress_error_ev_ang3\":" << qe_stress_error
        << ",\"qe_energy_error_ha\":" << qe_energy_error
        << ",\"rotation_error_ev_ang3\":" << rot_error
        << ",\"rotation_applied\":" << (rotate ? "true" : "false")
        << ",\"energy_ha\":" << reference.energy_ha
        << ",\"degauss_ry\":" << width_ry
        << ",\"max_force_ev_ang\":" << max_force
        << ",\"magnetization\":" << reference.magnetization
        << ",\"electron_count\":" << reference.n_elec_occ
        << ",\"entropy_ry\":" << reference.e_entropy
        << ",\"iterations\":" << reference.iterations
        << ",\"passed\":" << (fd_failure.empty() ? "true" : "false") << '}';
    return out.str();
}

std::string checked_case(XC xc, ATV::electronic::Smearing smear, bool rotate) {
    try {
        return check_case(xc, smear, rotate);
    } catch (const std::exception& exception) {
        case_failures = true;
        std::ostringstream row;
        row << "{\"xc\":\"" << (xc == XC::PBE ? "PBE" : "LSDA")
            << "\",\"smearing\":" << int(smear)
            << ",\"error\":" << std::quoted(exception.what())
            << ",\"passed\":false}";
        return row.str();
    }
}

using GVectorList = cpu::pw::internal::GVectorList;
auto mode_set(const GVectorList& gv) {
    std::vector<std::array<int, 3>> labels;
    for (int i = 0; i < gv.count(); ++i)
        labels.push_back({gv.h[i], gv.k[i], gv.l[i]});
    std::sort(labels.begin(), labels.end());
    return labels;
}

void cutoff_survey(std::vector<std::string>& records) {
    auto f = frame();
    auto m = method(XC::LSDA, ATV::electronic::Smearing::fermi_dirac);
    cpu::pw::internal::PWProblem base;
    cpu::pw::internal::setup_pw_problem(f, *m.potential, 6., 1, base);
    for (int v = 0; v < 6; ++v)
        for (bool sym : {false, true})
            for (int sign : {1, -1}) {
                Mat3 F{};
                for (int i = 0; i < 3; ++i)
                    F(i, i) = 1;
                F(ia[v], ib[v]) += sign * h;
                if (sym && ia[v] != ib[v])
                    F(ib[v], ia[v]) += sign * h;
                auto strained = transform(f, F);
                cpu::pw::internal::PWProblem changed;
                cpu::pw::internal::setup_pw_problem(strained, *m.potential, 6.,
                                                    1, changed);
                std::ostringstream out;
                out << "{\"mode\":\"cutoff-shapes\",\"voigt\":" << v
                    << ",\"sign\":" << sign
                    << ",\"symmetric\":" << (sym ? "true" : "false")
                    << ",\"base_wave_count\":" << base.gv_k[0].count()
                    << ",\"base_density_count\":" << base.gv0.count()
                    << ",\"wave_count\":" << changed.gv_k[0].count()
                    << ",\"density_count\":" << changed.gv0.count()
                    << ",\"wave_modes_changed\":"
                    << (mode_set(base.gv_k[0]) != mode_set(changed.gv_k[0])
                            ? "true"
                            : "false")
                    << ",\"density_modes_changed\":"
                    << (mode_set(base.gv0) != mode_set(changed.gv0) ? "true"
                                                                    : "false")
                    << ",\"fft\":[" << changed.ws.fft_nx << ','
                    << changed.ws.fft_ny << ',' << changed.ws.fft_nz << "]}";
                records.push_back(out.str());
            }
    const Six symmetric = cpu::pw::pw_stress_fd(f, m, h);
    const Six simple =
        simple_fd(f, XC::LSDA, ATV::electronic::Smearing::fermi_dirac);
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\"mode\":\"invalid-hard-cutoff-reference\",\"symmetric_full_scf_"
           "fd_ev_ang3\":"
        << vector_json(symmetric)
        << ",\"simple_full_scf_fd_ev_ang3\":" << vector_json(simple)
        << ",\"normalization_error_ev_ang3\":" << error(symmetric, simple)
        << ",\"physical_oracle\":false}";
    records.push_back(out.str());
    require(
        error(symmetric, simple) < normalization_gate,
        "hard-cutoff recut full-SCF strains are not a physical stress oracle");
}

std::string check_mpi() {
    auto f = frame();
    auto m = method(XC::PBE, ATV::electronic::Smearing::fermi_dirac, 2);
    const Six sigma = m.stress(f);
    const auto result = m.last_result();
    require(result.converged, "MPI spin-smear reference failed");
    require(std::abs(result.n_elec_occ - 3) < 1e-7, "MPI electron count wrong");
    std::ostringstream out;
    out << std::setprecision(17) << "{\"mode\":\"mpi\",\"ranks\":" << ranks
        << ",\"k_mesh\":2,\"stress_ev_ang3\":" << vector_json(sigma)
        << ",\"energy_ha\":" << result.energy_ha
        << ",\"electron_count\":" << result.n_elec_occ
        << ",\"magnetization\":" << result.magnetization << ",\"passed\":true}";
    return out.str();
}

std::string metadata_baseline(bool spin) {
    auto f = frame();
    const auto smear = spin ? ATV::electronic::Smearing::none
                            : ATV::electronic::Smearing::fermi_dirac;
    auto a = method(XC::PBE, smear, 1, spin);
    const Result reference = a.run(f);
    auto b = method(XC::PBE, smear, 1, spin);
    b.stress(f);
    metadata(reference, b.last_result());
    return std::string("{\"mode\":\"metadata-baseline\",\"spin\":")
           + (spin ? "true" : "false") + ",\"passed\":true}";
}

std::string probe(XC xc, ATV::electronic::Smearing smear) {
    auto f = frame();
    auto m = method(xc, smear);
    const auto r = m.run(f);
    std::ostringstream out;
    out << std::setprecision(17) << "{\"xc\":\""
        << (xc == XC::PBE ? "PBE" : "LSDA") << "\",\"smearing\":" << int(smear)
        << ",\"energy_ha\":" << r.energy_ha
        << ",\"magnetization\":" << r.magnetization
        << ",\"electron_count\":" << r.n_elec_occ
        << ",\"entropy_ry\":" << r.e_entropy
        << ",\"converged\":" << (r.converged ? "true" : "false") << '}';
    return out.str();
}
} // namespace

int main(int argc, char** argv) {
#ifdef ATV_HAS_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
#endif
    int status = 0;
    std::string output, mode = "matrix", failure;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--output" && i + 1 < argc)
            output = argv[++i];
        else if (std::string(argv[i]) == "--mode" && i + 1 < argc)
            mode = argv[++i];
        else if (std::string(argv[i]) == "--width" && i + 1 < argc)
            width_ry = std::stod(argv[++i]);
        else if (std::string(argv[i]) == "--mp-width" && i + 1 < argc)
            mp_width_override = std::stod(argv[++i]);
        else if (std::string(argv[i]) == "--max-iter" && i + 1 < argc)
            scf_max_iter = std::stoi(argv[++i]);
        else if (std::string(argv[i]) == "--mix" && i + 1 < argc)
            scf_mix_beta = std::stod(argv[++i]);
    }
    std::vector<std::string> records;
    try {
        if (mode == "cutoff")
            cutoff_survey(records);
        else if (mode == "mpi")
            records.push_back(check_mpi());
        else if (mode == "probe")
            for (XC xc : {XC::LSDA, XC::PBE})
                for (ATV::electronic::Smearing smear :
                     {ATV::electronic::Smearing::fermi_dirac,
                      ATV::electronic::Smearing::gaussian,
                      ATV::electronic::Smearing::methfessel_paxton})
                    records.push_back(probe(xc, smear));
        else if (mode == "mp-strains") {
            width_ry = mp_width_override;
            for (XC xc : {XC::LSDA, XC::PBE})
                records.push_back(checked_case(
                    xc, ATV::electronic::Smearing::methfessel_paxton, false));
        } else if (mode == "matrix") {
            records.push_back(metadata_baseline(true));
            records.push_back(metadata_baseline(false));
            for (XC xc : {XC::LSDA, XC::PBE})
                for (ATV::electronic::Smearing smear :
                     {ATV::electronic::Smearing::fermi_dirac,
                      ATV::electronic::Smearing::gaussian,
                      ATV::electronic::Smearing::methfessel_paxton}) {
                    const double base_width = width_ry;
                    if (smear == ATV::electronic::Smearing::gaussian)
                        width_ry *=
                            2; // .04 Ry, selected by the recorded probe.
                    else if (smear
                                 == ATV::electronic::Smearing::methfessel_paxton
                             && mp_width_override > 0)
                        width_ry = mp_width_override;
                    records.push_back(checked_case(
                        xc, smear,
                        smear == ATV::electronic::Smearing::fermi_dirac));
                    width_ry = base_width;
                }
        }
#ifdef ATV_PW_STRESS_BEFORE
        else if (mode == "negative") {
            auto f = frame();
            auto m = method(XC::PBE, ATV::electronic::Smearing::none);
            const Six old = cpu::pw::internal::pw_stress_analytic_impl_before(
                f, m, nullptr, nullptr);
            const Six reference =
                simple_fd(f, XC::PBE, ATV::electronic::Smearing::none);
            std::ostringstream record;
            record << std::setprecision(17)
                   << "{\"mode\":\"negative\",\"original_stress_ev_ang3\":"
                   << vector_json(old)
                   << ",\"simple_fd_ev_ang3\":" << vector_json(reference)
                   << ",\"max_error_ev_ang3\":" << error(old, reference)
                   << ",\"passed\":false}";
            records.push_back(record.str());
            require(
                error(old, reference) < nscf_gate,
                "original stress violates physical simple-shear derivative");
        }
#endif
        else
            throw std::runtime_error("unknown diagnostic mode");
    } catch (const std::exception& e) {
        status = 1;
        failure = e.what();
        if (rank == 0)
            std::cerr << "DIAGNOSTIC FAILURE: " << e.what() << '\n';
    }
    if (case_failures) {
        status = 1;
        if (failure.empty())
            failure = "one or more case checks failed";
    }
#ifdef ATV_HAS_MPI
    int combined = 0;
    MPI_Allreduce(&status, &combined, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    status = combined;
#endif
    if (rank == 0) {
        std::ostringstream json;
        json << "{\"passed\":" << (status ? "false" : "true")
             << ",\"ranks\":" << ranks << ",\"mode\":\"" << mode
             << "\",\"error\":" << std::quoted(failure) << ",\"compiler\":\""
             << __VERSION__
             << "\",\"ecut_ry\":6,\"k_mesh\":" << (mode == "mpi" ? 2 : 1)
             << ",\"degauss_ry\":" << width_ry
             << ",\"mp_width_override_ry\":" << mp_width_override
             << ",\"tol\":1e-9,\"max_iter\":" << scf_max_iter
             << ",\"mix_beta\":" << scf_mix_beta
             << ",\"potential\":\"GTH-PBE-q1\",\"multiplicity\":2,\"extra_"
                "bands\":3"
             << ",\"cell_angstrom_columns\":[[4.2,0,0],[0.3,4,0],[0.2,0.1,3.8]]"
             << ",\"positions_angstrom\":[[1.2,1.1,1.0],[2.1,1.6,1.3],[1.6,2."
                "25,1.8]]"
             << ",\"atomic_numbers\":[1,1,1]"
#ifdef _OPENMP
             << ",\"openmp_threads\":" << omp_get_max_threads()
#endif
             << ",\"gates\":{\"nscf\":2e-3,\"rotation\":2e-3,\"qe_energy_ha\":"
                "1e-5,\"qe_stress_ev_ang3\":2e-3,\"cutoff_normalization\":5e-5}"
                ",\"records\":[";
        for (std::size_t i = 0; i < records.size(); ++i) {
            if (i)
                json << ',';
            json << records[i];
        }
        json << "]}\n";
        if (!output.empty()) {
            std::ofstream file(output);
            file << json.str();
        }
        std::cout << "RESULT_JSON " << json.str();
    }
#ifdef ATV_HAS_MPI
    MPI_Finalize();
#endif
    return status;
}
