#include <backend/cpu/solver_cpu.h>
#include <potential.h>
#include "determinism.h"
#include "testsupport/generator.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <map>
#include <string>
#include <vector>

struct BenchConfig {
    int dim;
    double L;
    double sigma;
    std::vector<std::vector<int>> rd_list;
};

struct Options {
    std::string size = "small";
    int threads = 1;
    int iters = 1;
    bool deterministic = true;
};

static void print_usage(const char* name)
{
    std::cout << "Usage: " << name << " [--size small|medium] [--threads N] [--iters N] [--deterministic|--no-deterministic]\n";
}

static Options parse_args(int argc, char** argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--size" && i + 1 < argc) {
            opt.size = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            opt.threads = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            opt.iters = std::stoi(argv[++i]);
        } else if (arg == "--deterministic") {
            opt.deterministic = true;
        } else if (arg == "--no-deterministic") {
            opt.deterministic = false;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    return opt;
}

static BenchConfig configForSize(const std::string& size)
{
    if (size == "medium") {
        return BenchConfig{
            48,
            16.0,
            0.7,
            {{0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}, {1,1,0}, {1,0,1}, {0,1,1}}
        };
    }
    return BenchConfig{32, 12.0, 0.6, {{0,0,0}, {1,0,0}, {0,1,0}}};
}

static void buildWannierMaps(const BenchConfig& cfg,
                             std::map<int, WannierFunction>& vWannMap,
                             std::map<int, WannierFunction>& cWannMap)
{
    std::vector<int> dim{cfg.dim, cfg.dim, cfg.dim};
    std::vector<double> origin{0.0, 0.0, 0.0};
    std::vector<std::vector<double>> lattice(3);
    lattice[0] = std::vector<double>{cfg.L, 0.0, 0.0};
    lattice[1] = std::vector<double>{0.0, cfg.L, 0.0};
    lattice[2] = std::vector<double>{0.0, 0.0, cfg.L};

    auto meshgrid = std::make_shared<RealMeshgrid>(dim, lattice, origin);

    const double sigma = cfg.sigma;

    // Conduction WFs
    cWannMap.insert({0, generateGauss(0, meshgrid, lattice, sigma, cfg.L * 0.20, cfg.L * 0.20, cfg.L * 0.20, 1.0)});
    cWannMap.insert({1, generateGauss(1, meshgrid, lattice, sigma, cfg.L * 0.35, cfg.L * 0.20, cfg.L * 0.20, 1.0)});

    // Valence WFs
    vWannMap.insert({0, generateGauss(0, meshgrid, lattice, sigma, cfg.L * 0.80, cfg.L * 0.80, cfg.L * 0.80, 1.0)});
    vWannMap.insert({1, generateGauss(1, meshgrid, lattice, sigma, cfg.L * 0.65, cfg.L * 0.80, cfg.L * 0.80, 1.0)});
}

static std::vector<Integral> buildIntegrals(const std::vector<std::vector<int>>& rd_list)
{
    std::vector<Integral> integrals;
    const std::vector<int> zero{0, 0, 0};
    const std::vector<int> ids{0, 1};

    for (int c1 : ids) {
        for (int c2 : ids) {
            for (int v1 : ids) {
                for (int v2 : ids) {
                    for (const auto& rd : rd_list) {
                        integrals.emplace_back(c1, c2, v1, v2, rd, zero, zero);
                    }
                }
            }
        }
    }

    return integrals;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Options opt = parse_args(argc, argv);
    const BenchConfig cfg = configForSize(opt.size);

    if (opt.deterministic) {
        wo_determinism::applyDeterministicOpenMP(true);
    }
    omp_set_num_threads(opt.threads);

    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }

    std::map<int, WannierFunction> vWannMap;
    std::map<int, WannierFunction> cWannMap;
    buildWannierMaps(cfg, vWannMap, cWannMap);

    CoulombSolver solver(vWannMap, cWannMap);
    auto integrals = buildIntegrals(cfg.rd_list);

    for (int iter = 0; iter < opt.iters; ++iter) {
        solver.calculate(integrals, false, 1, static_cast<unsigned int>(opt.threads));
    }

    double checksum = 0.0;
    bool has_nan = false;
    for (const auto& integral : integrals) {
        if (!std::isfinite(integral.value)) {
            has_nan = true;
        } else {
            checksum += integral.value;
        }
    }

    std::cout.setf(std::ios::fixed);
    std::cout.precision(12);
    std::cout << "BENCH_SIZE=" << opt.size << "\n";
    std::cout << "BENCH_THREADS=" << opt.threads << "\n";
    std::cout << "BENCH_ITERS=" << opt.iters << "\n";
    std::cout << "BENCH_INTEGRALS=" << integrals.size() << "\n";
    std::cout << "CHECKSUM=" << checksum << "\n";

    if (has_nan) {
        std::cerr << "ERROR: NaN/Inf detected in benchmark results. Check input size or solver stability.\n";
        MPI_Finalize();
        return 2;
    }

    MPI_Finalize();
    return 0;
}
