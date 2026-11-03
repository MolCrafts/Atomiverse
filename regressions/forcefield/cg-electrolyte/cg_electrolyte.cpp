/// @file regressions/forcefield/cg-electrolyte/cg_electrolyte.cpp
/// @brief CPU single point of the coarse-grained polymer-electrolyte case,
///        term by term, for the LAMMPS comparison in REFERENCE.md.
///
/// Archived evidence, no runner. Usage:
///   cg_electrolyte <system.txt> <out_dir>
/// writes <out_dir>/cpu.<term>.txt for term in fene angle lj r4 coul kspace
/// total
/// (energies, six-component virial, per-tag forces).
#include <cstdio>
#include <exception>
#include <string>

#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/longrange/pppm.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/angle_cosine.h"
#include "atomiverse/cpu/potentials/bond_fene.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_coul_long.h"
#include "cg_electrolyte_system.h"

using namespace ATV;
namespace cg = cg_electrolyte;

namespace {

constexpr double kSkin = 0.5;
/// Coulomb cutoff [A] of the k-space-only term (below every distance).
constexpr double kKspaceOnlyCut = 0.05;

void run(const cg::System& s, const std::string& term, const std::string& out) {
    cpu::Frame f = cg::frame(s);
    cpu::Evaluate eval;
    eval.enable_timing(false);
    eval.set_neighbor(cpu::md::NeighborList<>{s.rc_coul, kSkin, 1, 0, true});
    const double spacing = s.box / s.mesh;
    auto fene = [&] { return cpu::BondFene{{s.fene_k * s.eps0}, {s.fene_r0}}; };
    auto cosine = [&] { return cpu::AngleCosine{{s.cos_k * s.eps0}}; };
    if (term == "fene") {
        eval.set_evaluator(fene());
    } else if (term == "angle") {
        eval.set_evaluator(cosine());
    } else if (term == "lj") {
        eval.set_evaluator(cpu::PairLJCut{cg::lj_rows(s), s.rc_coul});
    } else if (term == "r4") {
        eval.set_evaluator(
            cpu::PairLJCut{cg::lj_rows(s, 0.0), s.rc_coul}.set_r4_attraction(
                {cg::r4_row(s)}));
    } else if (term == "coul") {
        eval.set_evaluator(
            cpu::PairLJCutCoulLong{cg::lj_rows(s, 0.0), s.alpha, s.rc_coul}
                .set_special_bonds(1, 1, 1, 1, 1, 1));
        eval.set_evaluator(cpu::KSpacePPPM{s.alpha, spacing, s.order});
    } else if (term == "kspace") {
        // Reciprocal part alone: the matched real-space partner keeps the
        // same alpha but a Coulomb cutoff (kKspaceOnlyCut) below every
        // interatomic distance, so it contributes nothing.
        eval.set_evaluator(
            cpu::PairLJCutCoulLong{cg::lj_rows(s, 0.0), s.alpha, kKspaceOnlyCut}
                .set_special_bonds(1, 1, 1, 1, 1, 1));
        eval.set_evaluator(cpu::KSpacePPPM{s.alpha, spacing, s.order});
    } else if (term == "total") {
        eval.set_evaluator(
            cpu::PairLJCutCoulLong{cg::lj_rows(s), s.alpha, s.rc_coul}
                .set_special_bonds(1, 1, 1, 1, 1, 1)
                .set_r4_attraction({cg::r4_row(s)}));
        eval.set_evaluator(fene());
        eval.set_evaluator(cosine());
        eval.set_evaluator(cpu::KSpacePPPM{s.alpha, spacing, s.order});
    } else {
        throw std::invalid_argument("unknown term " + term);
    }
    eval.build();
    eval.run(f);
    cg::write(out + "/cpu." + term + ".txt", f);
    std::printf("cpu %-6s E = %.17g eV\n", term.c_str(), f.meta.get(kEnergy));
}

} // namespace

int main(int argc, char** argv) try {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <system.txt> <out_dir>\n", argv[0]);
        return 2;
    }
    const cg::System s = cg::read(argv[1]);
    for (const char* term :
         {"fene", "angle", "lj", "r4", "coul", "kspace", "total"})
        run(s, term, argv[2]);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "cg_electrolyte: %s\n", e.what());
    return 1;
}
