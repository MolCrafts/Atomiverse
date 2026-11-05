/// @file regressions/forcefield/cg-electrolyte/cuda-cg-electrolyte.cu
/// @brief CUDA single point of the coarse-grained polymer-electrolyte case,
///        term by term, for the LAMMPS comparison in REFERENCE.md.
///
/// Archived evidence, no runner. Usage:
///   cuda-cg-electrolyte <system.txt> <out_dir>
/// writes <out_dir>/cuda.<term>.txt for term in fene angle lj r4 coul kspace
/// total
/// (energies, six-component virial, per-tag forces). Same Frame, rows and
/// parameters as `cg_electrolyte.cpp` (shared `cg_electrolyte_system.h`).
#include <cstdio>
#include <exception>
#include <string>

#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/longrange/pme.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/angle_cosine.cuh"
#include "atomiverse/cuda/potentials/bond_fene.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut_coul_long.cuh"
#include "cg_electrolyte_system.h"

using namespace ATV;
namespace cg = cg_electrolyte;

namespace {

constexpr double kSkin = 0.5;
/// Coulomb cutoff [A] of the k-space-only term (below every distance).
constexpr double kKspaceOnlyCut = 0.05;

void run(const cg::System& s, const std::string& term, const std::string& out) {
    cpu::Frame f = cg::frame(s);
    {
        cuda::Evaluate eval;
        eval.set_neighbor(cuda::NeighborList{s.rc_coul, kSkin});
        const double spacing = s.box / s.mesh;
        auto fene = [&] {
            return cuda::BondFene{{s.fene_k * s.eps0}, {s.fene_r0}};
        };
        auto cosine = [&] { return cuda::AngleCosine{{s.cos_k * s.eps0}}; };
        auto pppm = [&] {
            eval.on(evaluate::Phase::evaluate)
                .add(cuda::KSpacePPPM{s.alpha, spacing, s.order}, 100);
        };
        if (term == "fene") {
            eval.set_evaluator(fene());
        } else if (term == "angle") {
            eval.set_evaluator(cosine());
        } else if (term == "lj") {
            eval.set_evaluator(cuda::PairLJCut{cg::lj_rows(s), s.rc_coul});
        } else if (term == "r4") {
            eval.set_evaluator(cuda::PairLJCut{cg::lj_rows(s, 0.0), s.rc_coul}
                                   .set_r4_attraction({cg::r4_row(s)}));
        } else if (term == "coul") {
            eval.set_evaluator(
                cuda::PairLJCutCoulLong{cg::lj_rows(s, 0.0), s.alpha, s.rc_coul}
                    .set_special_bonds(1, 1, 1, 1, 1, 1));
            pppm();
        } else if (term == "kspace") {
            // Reciprocal part alone: same alpha, a real-space Coulomb cutoff
            // (kKspaceOnlyCut) below every interatomic distance.
            eval.set_evaluator(cuda::PairLJCutCoulLong{cg::lj_rows(s, 0.0),
                                                       s.alpha, kKspaceOnlyCut}
                                   .set_special_bonds(1, 1, 1, 1, 1, 1));
            pppm();
        } else if (term == "total") {
            eval.set_evaluator(
                cuda::PairLJCutCoulLong{cg::lj_rows(s), s.alpha, s.rc_coul}
                    .set_special_bonds(1, 1, 1, 1, 1, 1)
                    .set_r4_attraction({cg::r4_row(s)}));
            eval.set_evaluator(fene());
            eval.set_evaluator(cosine());
            pppm();
        } else {
            throw std::invalid_argument("unknown term " + term);
        }
        eval.build();
        eval.run(f);
    }
    cg::write(out + "/cuda." + term + ".txt", f);
    std::printf("cuda %-6s E = %.17g eV\n", term.c_str(), f.meta.get(kEnergy));
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
    std::fprintf(stderr, "cuda-cg-electrolyte: %s\n", e.what());
    return 1;
}
