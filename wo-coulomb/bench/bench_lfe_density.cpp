#include "backend/density_metrics.h"
#include "density.h"
#include "potential.h"
#include "scheduler.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <mpi.h>

namespace {

struct Options
{
    int dim = 48;
    int supercell = 5;
    int n_conduction = 10;
    int n_valence = 10;
    double cell_length = 6.0;
    double sigma = 1.2;
    double abscharge_threshold = 0.0;
};

Options parse_args(int argc, char** argv)
{
    Options opt{};

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(2);
            }
        };

        if (arg == "--dim") {
            require_value("--dim");
            opt.dim = std::stoi(argv[++i]);
        } else if (arg == "--supercell") {
            require_value("--supercell");
            opt.supercell = std::stoi(argv[++i]);
        } else if (arg == "--nc") {
            require_value("--nc");
            opt.n_conduction = std::stoi(argv[++i]);
        } else if (arg == "--nv") {
            require_value("--nv");
            opt.n_valence = std::stoi(argv[++i]);
        } else if (arg == "--cell-length") {
            require_value("--cell-length");
            opt.cell_length = std::stod(argv[++i]);
        } else if (arg == "--sigma") {
            require_value("--sigma");
            opt.sigma = std::stod(argv[++i]);
        } else if (arg == "--threshold") {
            require_value("--threshold");
            opt.abscharge_threshold = std::stod(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: bench_lfe_density [--dim N] [--supercell N] [--nc N] [--nv N] "
                         "[--cell-length A] [--sigma A] [--threshold Q]\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::exit(2);
        }
    }

    if (opt.dim <= 0 || opt.supercell <= 0 || opt.n_conduction <= 0 || opt.n_valence <= 0 ||
        opt.cell_length <= 0.0 || opt.sigma <= 0.0) {
        std::cerr << "All numeric arguments must be positive.\n";
        std::exit(2);
    }

    return opt;
}

WannierFunction make_gaussian_wannier(
    int id,
    std::shared_ptr<RealMeshgrid> const& mesh,
    std::vector<std::vector<double>> const& unitcell,
    double sigma,
    double x0,
    double y0,
    double z0)
{
    const int n_points = mesh->getNumDataPoints();
    std::unique_ptr<double[], free_deleter> data{(double*) malloc(sizeof(double) * n_points)};

    const double inv_norm = 1.0 / (std::pow(2.0 * M_PI, 1.5) * std::pow(sigma, 3.0));

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    for (int i = 0; i < n_points; ++i) {
        mesh->xyz(i, x, y, z);
        const double dx = x - x0;
        const double dy = y - y0;
        const double dz = z - z0;
        data[i] = inv_norm * std::exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma * sigma));
    }

    WannierFunction wf{id, mesh, std::move(data), unitcell};
    wf.normalize(1.0);
    return wf;
}

std::vector<double> fractional_triplet(int idx)
{
    // Deterministic spread in unit-cell coordinates to avoid accidental symmetry artifacts.
    const int a = (idx * 17 + 3) % 97;
    const int b = (idx * 31 + 7) % 89;
    const int c = (idx * 43 + 11) % 83;
    return {
        0.1 + 0.8 * (static_cast<double>(a) / 97.0),
        0.1 + 0.8 * (static_cast<double>(b) / 89.0),
        0.1 + 0.8 * (static_cast<double>(c) / 83.0),
    };
}

}  // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const Options opt = parse_args(argc, argv);

    const std::vector<int> dim{opt.dim, opt.dim, opt.dim};
    const std::vector<double> origin{0.0, 0.0, 0.0};

    // Build a supercell mesh while keeping a smaller primitive unit cell.
    // This creates non-trivial shift shells without inflating the real-space grid.
    std::vector<std::vector<double>> unitcell(3, std::vector<double>(3, 0.0));
    std::vector<std::vector<double>> lattice(3, std::vector<double>(3, 0.0));
    for (int i = 0; i < 3; ++i) {
        unitcell[i][i] = opt.cell_length;
        lattice[i][i] = opt.cell_length * static_cast<double>(opt.supercell);
    }

    auto mesh = std::make_shared<RealMeshgrid>(dim, lattice, origin);
    mesh->createMeshgridArrays();

    std::map<int, WannierFunction> cWannMap{};
    std::map<int, WannierFunction> vWannMap{};

    for (int i = 0; i < opt.n_conduction; ++i) {
        const std::vector<double> frac = fractional_triplet(i + 1);
        cWannMap.emplace(
            i + 1,
            make_gaussian_wannier(
                i + 1,
                mesh,
                unitcell,
                opt.sigma,
                frac[0] * opt.cell_length,
                frac[1] * opt.cell_length,
                frac[2] * opt.cell_length));
    }
    for (int i = 0; i < opt.n_valence; ++i) {
        const std::vector<double> frac = fractional_triplet(i + 101);
        vWannMap.emplace(
            i + 1,
            make_gaussian_wannier(
                i + 1,
                mesh,
                unitcell,
                opt.sigma,
                frac[0] * opt.cell_length + 0.15 * opt.cell_length,
                frac[1] * opt.cell_length + 0.10 * opt.cell_length,
                frac[2] * opt.cell_length + 0.05 * opt.cell_length));
    }

    const std::vector<double> origin_wf = cWannMap.begin()->second.getOriginInUnitcellBasis();
    const std::vector<double> supercell_wf = cWannMap.begin()->second.getLatticeInUnitcellBasis();
    const auto shells = createShells(
        origin_wf,
        supercell_wf,
        cWannMap.begin()->second.getUnitcell(),
        std::vector<double>{0.0, 0.0, 0.0},
        std::vector<double>{0.0, 0.0, 0.0},
        std::vector<bool>{true, true, true});

    size_t n_shifts = 0;
    for (const auto& shell : shells) {
        n_shifts += shell.size();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();
    const auto indicators = calcLFE_estimates_parallel(cWannMap, vWannMap, shells, opt.abscharge_threshold);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto t1 = std::chrono::steady_clock::now();

    if (rank == 0) {
        double checksum = 0.0;
        for (const auto& kv : indicators) {
            checksum += kv.second.absCharge;
        }

        const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
        const bool gpu_metrics = density_metrics_gpu_available();
        const size_t expected_specs = static_cast<size_t>(opt.n_conduction) * static_cast<size_t>(opt.n_valence) * n_shifts;

        std::cout.setf(std::ios::fixed);
        std::cout.precision(6);
        std::cout << "BENCH=lfe_density_metrics\n";
        std::cout << "GPU_METRICS=" << (gpu_metrics ? 1 : 0) << "\n";
        std::cout << "DIM=" << opt.dim << "\n";
        std::cout << "SUPERCELL=" << opt.supercell << "\n";
        std::cout << "NC=" << opt.n_conduction << "\n";
        std::cout << "NV=" << opt.n_valence << "\n";
        std::cout << "NSHELLS=" << shells.size() << "\n";
        std::cout << "NSHIFTS=" << n_shifts << "\n";
        std::cout << "EXPECTED_SPECS=" << expected_specs << "\n";
        std::cout << "INDICATORS=" << indicators.size() << "\n";
        std::cout << "THRESHOLD=" << opt.abscharge_threshold << "\n";
        std::cout << "ELAPSED_SECONDS=" << elapsed_s << "\n";
        std::cout.precision(12);
        std::cout << "CHECKSUM=" << checksum << "\n";
    }

    MPI_Finalize();
    return 0;
}
