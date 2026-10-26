/// Public C++ HF diagnostic; deliberately outside CTest. Each sample starts
/// with a fresh Frame and method. JSON lines include energy and optional forces.
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "atomiverse/cpu/scf/methods.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

using namespace ATV;
using namespace ATV::cpu::scf;

namespace {
cpu::Frame geometry(const std::string& system, int waters) {
    const bool water = system == "water";
    const int n = water ? 3 * waters : (system == "h3" ? 3 : 2);
    cpu::Frame f(n);
    f.box = cpu::Box::cubic(1000);
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_x)[i] =
            water ? 4.0 * (i / 3) : i * (system == "o2" ? 1.207 : 0.74);
        f.span(kAtom_y)[i] = water ? (i % 3 == 1   ? 0.7572
                                      : i % 3 == 2 ? -0.7572
                                                   : 0)
                                   : 0;
        f.span(kAtom_z)[i] = water ? (i % 3 == 0 ? 0.1173 : -0.4692) : 0;
        f.span(kAtom_type)[i] =
            water ? (i % 3 == 0 ? 8 : 1) : (system == "o2" ? 8 : 1);
    }
    if (system == "h3") {
        f.span(kAtom_x)[0] = 0;
        f.span(kAtom_x)[1] = 1;
        f.span(kAtom_x)[2] = 0.5;
        f.span(kAtom_y)[2] = std::sqrt(3.0) / 2;
    }
    return f;
}

template <typename Method>
void sample(Method& mf, cpu::Frame& frame, int threads, double memory,
            bool forces, int repeat, std::ostream& out) {
    mf.max_memory_mb = memory;
    mf.mpi = std::getenv("ATV_BENCH_MPI") != nullptr;
    mf.set_threads(threads);
    mf.init_guess = InitGuess::Hcore;
    mf.tol = 1e-10;
    mf.conv_tol_grad = 1e-8;
    mf.max_iter = 200;
    if constexpr (std::is_same_v<Method, GHF>) {
        mf.guess_spin_rotation = 2 * std::acos(-1.0) / 3;
    } else {
        mf.damping_factor = 0;
        mf.diis_max_vectors = 8;
        mf.diis_start_iter = 1;
    }
    GradientResult gradient;
    if (forces)
        gradient = mf.forces(frame);
    else
        mf.run(frame);
    if (!mf.converged())
        throw std::runtime_error("SCF did not converge");
    out << "{\"repeat\":" << repeat << ",\"energy_hartree\":" << mf.energy()
        << ",\"iterations\":" << mf.iterations();
    if (forces) {
        out << ",\"forces_hartree_per_bohr\":[";
        for (std::size_t i = 0; i < frame.nlocal(); ++i) {
            if (i)
                out << ',';
            out << '['
                << units::convert(gradient.forces(i, 0),
                                  units::Force::eVPerAngstrom,
                                  units::Force::HartreePerBohr)
                << ','
                << units::convert(gradient.forces(i, 1),
                                  units::Force::eVPerAngstrom,
                                  units::Force::HartreePerBohr)
                << ','
                << units::convert(gradient.forces(i, 2),
                                  units::Force::eVPerAngstrom,
                                  units::Force::HartreePerBohr)
                << ']';
        }
        out << ']';
    }
}
} // namespace

int main(int argc, char** argv) {
#ifdef ATV_HAS_MPI
    // This executable owns MPI lifetime; the library never initializes it.
    struct MpiLifetime {
        MpiLifetime(int& argc, char**& argv) {
            int provided = 0;
            MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
            if (provided < MPI_THREAD_FUNNELED)
                MPI_Abort(MPI_COMM_WORLD, 1);
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank != 0)
                std::cout.setstate(std::ios_base::failbit);
        }
        ~MpiLifetime() { MPI_Finalize(); }
    } mpi_lifetime(argc, argv);
#endif
    try {
        if (argc != 9)
            throw std::invalid_argument(
                "usage: bench_hf_cpu METHOD BASIS SYSTEM WATERS THREADS "
                "REPEATS CACHE_MB FORCES");
        const std::string method = argv[1], basis_name = argv[2],
                          system = argv[3];
        const BasisSet basis = basis_name == "sto3g"     ? BasisSet::STO3G
                               : basis_name == "def2svp" ? BasisSet::Def2SVP
                                                         : BasisSet::STO3G;
        if (basis_name != "sto3g" && basis_name != "def2svp")
            throw std::invalid_argument("unknown basis");
        if (system != "water" && system != "h2" && system != "o2"
            && system != "h3")
            throw std::invalid_argument("unknown system");
        const int waters = std::stoi(argv[4]), threads = std::stoi(argv[5]);
        const int repeats = std::stoi(argv[6]);
        const double memory = std::stod(argv[7]);
        const bool forces = std::stoi(argv[8]) != 0;
        if (waters < 1 || repeats < 1 || threads < 0)
            throw std::invalid_argument("invalid sample controls");
        std::cout << std::setprecision(17);
        for (int r = 0; r < repeats; ++r) {
            std::ostringstream record;
            record << std::setprecision(17);
            const auto start = std::chrono::steady_clock::now();
            {
                auto frame = geometry(system, waters);
                if (method == "RHF") {
                    RHF mf(basis);
                    sample(mf, frame, threads, memory, forces, r, record);
                } else if (method == "UHF") {
                    const int na = system == "o2" ? 9 : 1,
                              nb = system == "o2" ? 7 : 1;
                    UHF mf(basis, na, nb);
                    sample(mf, frame, threads, memory, forces, r, record);
                } else if (method == "ROHF") {
                    const int na = system == "o2" ? 9 : 1,
                              nb = system == "o2" ? 7 : 1;
                    ROHF mf(basis, na, nb);
                    sample(mf, frame, threads, memory, forces, r, record);
                } else if (method == "GHF") {
                    GHF mf(basis, system == "h3" ? 3 : 2);
                    sample(mf, frame, threads, memory, forces, r, record);
                } else
                    throw std::invalid_argument("unknown method");
            }
            const double seconds = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - start)
                                       .count();
            int nt = 1;
#ifdef _OPENMP
            nt = omp_get_max_threads();
#endif
            std::cout << record.str() << ",\"threads\":" << nt
                      << ",\"seconds\":" << seconds << "}\n";
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
