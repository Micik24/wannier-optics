#include "backend/pre_fft_density.h"
#include "testsupport/generator.h"

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Options
{
    int dim = 48;
    int supercell = 5;
    int n_conduction = 12;
    int n_valence = 12;
    int max_shift = 1;
    double cell_length = 6.0;
    double sigma = 1.2;
    double q_threshold = 1.0e-6;
    double a0_min = 1.0e-16;
    bool wrap_aux = true;
    std::string kind = "overlap";
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
        } else if (arg == "--max-shift") {
            require_value("--max-shift");
            opt.max_shift = std::stoi(argv[++i]);
        } else if (arg == "--cell-length") {
            require_value("--cell-length");
            opt.cell_length = std::stod(argv[++i]);
        } else if (arg == "--sigma") {
            require_value("--sigma");
            opt.sigma = std::stod(argv[++i]);
        } else if (arg == "--q-threshold") {
            require_value("--q-threshold");
            opt.q_threshold = std::stod(argv[++i]);
        } else if (arg == "--a0-min") {
            require_value("--a0-min");
            opt.a0_min = std::stod(argv[++i]);
        } else if (arg == "--kind") {
            require_value("--kind");
            opt.kind = argv[++i];
        } else if (arg == "--no-wrap") {
            opt.wrap_aux = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: bench_pre_fft_gpu [--dim N] [--supercell N] [--nc N] [--nv N] [--max-shift N] "
                         "[--cell-length A] [--sigma A] [--q-threshold Q] [--a0-min A0] "
                         "[--kind overlap|transition] [--no-wrap]\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::exit(2);
        }
    }

    if (opt.dim <= 0 || opt.supercell <= 0 || opt.n_conduction <= 0 || opt.n_valence <= 0 ||
        opt.max_shift < 0 || opt.cell_length <= 0.0 || opt.sigma <= 0.0) {
        std::cerr << "Invalid arguments: expected positive dimensions and sigma/cell-length.\n";
        std::exit(2);
    }
    if (opt.kind != "overlap" && opt.kind != "transition") {
        std::cerr << "Invalid --kind. Use overlap or transition.\n";
        std::exit(2);
    }

    return opt;
}

std::vector<double> fractional_triplet(int idx)
{
    const int a = (idx * 17 + 3) % 97;
    const int b = (idx * 31 + 7) % 89;
    const int c = (idx * 43 + 11) % 83;
    return {
        0.1 + 0.8 * (static_cast<double>(a) / 97.0),
        0.1 + 0.8 * (static_cast<double>(b) / 89.0),
        0.1 + 0.8 * (static_cast<double>(c) / 83.0),
    };
}

double percent(double value, double total)
{
    if (!(total > 0.0)) return 0.0;
    return 100.0 * value / total;
}

}  // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const Options opt = parse_args(argc, argv);

    if (!pre_fft_density_gpu_available()) {
        if (rank == 0) {
            std::cerr << "No CUDA device available for pre-FFT GPU benchmark.\n";
        }
        MPI_Finalize();
        return 2;
    }

    const std::vector<int> dim{opt.dim, opt.dim, opt.dim};
    const std::vector<double> origin{0.0, 0.0, 0.0};

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
            generateGauss(
                i + 1,
                mesh,
                unitcell,
                opt.sigma,
                frac[0] * opt.cell_length,
                frac[1] * opt.cell_length,
                frac[2] * opt.cell_length,
                1.0));
    }

    for (int i = 0; i < opt.n_valence; ++i) {
        const std::vector<double> frac = fractional_triplet(i + 101);
        vWannMap.emplace(
            i + 1,
            generateGauss(
                i + 1,
                mesh,
                unitcell,
                opt.sigma,
                frac[0] * opt.cell_length + 0.15 * opt.cell_length,
                frac[1] * opt.cell_length + 0.10 * opt.cell_length,
                frac[2] * opt.cell_length + 0.05 * opt.cell_length,
                1.0));
    }

    std::vector<std::array<int, 3>> shifts;
    shifts.reserve(static_cast<size_t>(2 * opt.max_shift + 1) * static_cast<size_t>(2 * opt.max_shift + 1) * static_cast<size_t>(2 * opt.max_shift + 1));
    for (int ix = -opt.max_shift; ix <= opt.max_shift; ++ix) {
        for (int iy = -opt.max_shift; iy <= opt.max_shift; ++iy) {
            for (int iz = -opt.max_shift; iz <= opt.max_shift; ++iz) {
                shifts.push_back({ix, iy, iz});
            }
        }
    }

    std::vector<PreFftDensitySpec> specs;
    specs.reserve(
        static_cast<size_t>(opt.n_conduction) *
        static_cast<size_t>(opt.kind == "overlap" ? opt.n_conduction : opt.n_valence) *
        shifts.size());

    int tag = 0;
    if (opt.kind == "overlap") {
        for (int i = 1; i <= opt.n_conduction; ++i) {
            for (int j = 1; j <= opt.n_conduction; ++j) {
                for (const auto& R : shifts) {
                    specs.push_back(PreFftDensitySpec{i, j, R, tag++});
                }
            }
        }
    } else {
        for (int i = 1; i <= opt.n_conduction; ++i) {
            for (int j = 1; j <= opt.n_valence; ++j) {
                for (const auto& R : shifts) {
                    specs.push_back(PreFftDensitySpec{i, j, R, tag++});
                }
            }
        }
    }

    PreFftAuxConfig cfg{};
    cfg.wrap_aux = opt.wrap_aux;
    cfg.q_abs_threshold = opt.q_threshold;
    cfg.a0_min = opt.a0_min;
    cfg.enable_timing = true;

    const PreFftDensityKind kind =
        (opt.kind == "overlap") ? PreFftDensityKind::SameBand : PreFftDensityKind::TransitionCv;

    MPI_Barrier(MPI_COMM_WORLD);
    GpuDensityBatch batch = build_pre_fft_density_batch_gpu(kind, cWannMap, vWannMap, specs, cfg);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        const auto& t = batch.timings_ms;
        const size_t K = batch.K;
        const size_t B = batch.B;
        const size_t S = batch.B_selected;

        const size_t bytes_rho_full = sizeof(double) * batch.ld * batch.B;
        const size_t bytes_metadata_full = sizeof(double) * batch.B * 14;
        const size_t bytes_selected_maps = sizeof(int) * (batch.B + batch.B_selected);
        const size_t bytes_rho_selected = sizeof(double) * batch.ld_selected * batch.B_selected;
        const size_t bytes_sigma_alpha_selected = sizeof(double) * batch.B_selected * 2;
        const size_t bytes_aux_temp = sizeof(double) * batch.ld_selected * batch.B_selected;
        const size_t bytes_total_est =
            bytes_rho_full +
            bytes_metadata_full +
            bytes_selected_maps +
            bytes_rho_selected +
            bytes_sigma_alpha_selected +
            bytes_aux_temp;

        const double subset_total =
            t.subset_compaction_gather +
            t.auxiliary_build +
            t.auxiliary_subtraction +
            t.scatter_back;

        std::cout.setf(std::ios::fixed);
        std::cout << std::setprecision(6);
        std::cout << "BENCH=pre_fft_gpu_density\n";
        std::cout << "KIND=" << opt.kind << "\n";
        std::cout << "DIM=" << opt.dim << "\n";
        std::cout << "SUPERCELL=" << opt.supercell << "\n";
        std::cout << "NC=" << opt.n_conduction << "\n";
        std::cout << "NV=" << opt.n_valence << "\n";
        std::cout << "MAX_SHIFT=" << opt.max_shift << "\n";
        std::cout << "K=" << K << "\n";
        std::cout << "B=" << B << "\n";
        std::cout << "B_SELECTED=" << S << "\n";
        std::cout << "SELECTED_RATIO=" << percent(static_cast<double>(S), static_cast<double>(B)) << "\n";

        std::cout << "RHO_LAYOUT=column_major_density_contiguous\n";
        std::cout << "LD=" << batch.ld << "\n";
        std::cout << "LD_SELECTED=" << batch.ld_selected << "\n";

        std::cout << "BYTES_RHO_FULL=" << bytes_rho_full << "\n";
        std::cout << "BYTES_METADATA_FULL=" << bytes_metadata_full << "\n";
        std::cout << "BYTES_SELECTED_MAPS=" << bytes_selected_maps << "\n";
        std::cout << "BYTES_RHO_SELECTED=" << bytes_rho_selected << "\n";
        std::cout << "BYTES_SIGMA_ALPHA_SELECTED=" << bytes_sigma_alpha_selected << "\n";
        std::cout << "BYTES_AUX_TEMP=" << bytes_aux_temp << "\n";
        std::cout << "BYTES_TOTAL_EST=" << bytes_total_est << "\n";

        std::cout << "TIME_MS_DENSITY_MATERIALIZATION=" << t.density_materialization << "\n";
        std::cout << "TIME_MS_METADATA_REDUCTION=" << t.metadata_reduction << "\n";
        std::cout << "TIME_MS_METADATA_COPY_GPU_TO_CPU=" << t.metadata_copy_gpu_to_cpu << "\n";
        std::cout << "TIME_MS_CPU_AUX_SELECTION_POLICY=" << t.cpu_aux_selection_policy << "\n";
        std::cout << "TIME_MS_SUBSET_COMPACTION_GATHER=" << t.subset_compaction_gather << "\n";
        std::cout << "TIME_MS_SIGMA_ALPHA=" << t.sigma_alpha << "\n";
        std::cout << "TIME_MS_AUXILIARY_BUILD=" << t.auxiliary_build << "\n";
        std::cout << "TIME_MS_AUXILIARY_SUBTRACTION=" << t.auxiliary_subtraction << "\n";
        std::cout << "TIME_MS_SCATTER_BACK=" << t.scatter_back << "\n";
        std::cout << "TIME_MS_SUBSET_TOTAL=" << subset_total << "\n";
        std::cout << "TIME_MS_TOTAL=" << t.total << "\n";

        std::cout << "PCT_DENSITY_MATERIALIZATION=" << percent(t.density_materialization, t.total) << "\n";
        std::cout << "PCT_METADATA_REDUCTION=" << percent(t.metadata_reduction, t.total) << "\n";
        std::cout << "PCT_METADATA_COPY_GPU_TO_CPU=" << percent(t.metadata_copy_gpu_to_cpu, t.total) << "\n";
        std::cout << "PCT_CPU_AUX_SELECTION_POLICY=" << percent(t.cpu_aux_selection_policy, t.total) << "\n";
        std::cout << "PCT_SUBSET_COMPACTION_GATHER=" << percent(t.subset_compaction_gather, t.total) << "\n";
        std::cout << "PCT_SIGMA_ALPHA=" << percent(t.sigma_alpha, t.total) << "\n";
        std::cout << "PCT_AUXILIARY_BUILD=" << percent(t.auxiliary_build, t.total) << "\n";
        std::cout << "PCT_AUXILIARY_SUBTRACTION=" << percent(t.auxiliary_subtraction, t.total) << "\n";
        std::cout << "PCT_SCATTER_BACK=" << percent(t.scatter_back, t.total) << "\n";
        std::cout << "PCT_SUBSET_TOTAL=" << percent(subset_total, t.total) << "\n";
    }

    MPI_Finalize();
    return 0;
}
