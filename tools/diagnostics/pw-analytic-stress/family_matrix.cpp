#include "family_fixture.h"
#include "general_trace.h"
#include <array>
#include <cmath>
#include <cstdio>
using namespace ATV;
int main(int argc, char** argv) {
    try {
        if (argc != 3)
            throw std::runtime_error("index baseline|shadow");
        int idx = std::stoi(argv[1]);
        bool shadow = std::string(argv[2]) == "shadow";
        constexpr cpu::xc::XCType types[] = {
            cpu::xc::XCType::PBE, cpu::xc::XCType::PBEsol,
            cpu::xc::XCType::RevPBE, cpu::xc::XCType::RPBE};
        int model = idx / 4;
        bool spin = (idx % 4) / 2, fd = idx % 2;
        auto f = family_fixture::hydrogen(spin || fd);
        cpu::pw::PW m{std::make_shared<cpu::pw::GthPseudopotential>(
                          molrs::Element::H, "GTH-PBE-q1"),
                      6., 1, types[model]};
        family_fixture::configure(m, spin,
                                  fd ? electronic::Smearing::fermi_dirac
                                     : electronic::Smearing::none);
        trace::Counter counter(m);
        m.on(pw::Phase::check).add_ref(counter, 10);
        trace::reset();
        auto sigma = m.stress(f);
        auto r = m.last_result();
        if (!r.converged || counter.iterations != r.iterations
            || counter.builds != 1 || counter.destroys != 1)
            throw std::runtime_error("matrix actual convergence/lifecycle");
        if (shadow)
            trace::closed_budget(counter, spin ? 2 : 1);
        else if (trace::counts.legacy_stress != 1
                 || trace::counts.assembly != 0)
            throw std::runtime_error("matrix baseline numerical dispatch");
        if (fd) {
            if (std::abs(r.n_elec_occ - f.nlocal()) > 1e-8
                || std::abs(r.e_entropy) <= 1e-8)
                throw std::runtime_error("original FD fixture count/entropy");
        }
        std::printf("PASS PW_ANALYTIC_GENERAL_MATRIX index=%d model=%d spin=%d "
                    "FD=%d E_Ha=%.17g iters=%d Davidson=%d extract=%d "
                    "assembly=%d legacy=%d entropy_Ry=%.17g sigma_eV_A3=",
                    idx, model, spin, fd, r.energy_ha, r.iterations,
                    trace::counts.davidson, trace::counts.extract,
                    trace::counts.assembly, trace::counts.legacy_stress,
                    r.e_entropy);
        for (auto s : sigma) {
            if (!std::isfinite(s))
                throw std::runtime_error("matrix finite stress");
            std::printf(" %.17g", s);
        }
        std::printf("\n");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL PW_ANALYTIC_GENERAL_MATRIX %s\n", e.what());
        return 1;
    }
}
