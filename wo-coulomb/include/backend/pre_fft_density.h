#ifndef WO_COULOMB_PRE_FFT_DENSITY_H
#define WO_COULOMB_PRE_FFT_DENSITY_H

#include <array>
#include <cstddef>
#include <map>
#include <utility>
#include <vector>

#include "wannierfunction.h"

enum class PreFftDensityKind
{
    SameBand,     // idx1/idx2 both taken from cWannMap
    TransitionCv  // idx1 from cWannMap, idx2 from vWannMap
};

struct PreFftDensitySpec
{
    int idx1 = -1;
    int idx2 = -1;
    std::array<int, 3> R{0, 0, 0};
    int user_tag = -1;
};

struct PreFftAuxConfig
{
    // If false, only materialize real-space densities on GPU and return.
    // Auxiliary Gaussian subtraction and metadata policy stages are skipped.
    bool apply_auxiliary_subtraction = true;

    // Keep compatibility with existing benchmark/timing paths.
    // Currently required when apply_auxiliary_subtraction=true because
    // host-side policy uses copied metadata.
    bool copy_metadata_to_host = true;

    bool wrap_aux = true;
    double q_abs_threshold = 0.0;
    double a0_min = 1e-16;
    double points_per_std = 2.0;
    double std_per_cell = 11.0;
    bool enable_timing = true;

    // Keep a raw (pre-aux-subtraction) density copy in device memory.
    // Useful when both raw and auxiliary-corrected spectra are needed
    // from the same staging chunk.
    bool keep_raw_density_copy = false;
};

struct PreFftStageTimingsMs
{
    double density_materialization = 0.0;
    double metadata_reduction = 0.0;
    double metadata_copy_gpu_to_cpu = 0.0;
    double cpu_aux_selection_policy = 0.0;
    double subset_compaction_gather = 0.0;
    double sigma_alpha = 0.0;
    double auxiliary_build = 0.0;
    double auxiliary_subtraction = 0.0;
    double scatter_back = 0.0;
    double total = 0.0;
};

struct GpuDensityBatch;
void release_gpu_density_batch(GpuDensityBatch& batch);

/**
 * @brief Device-resident density batch for the GPU pre-FFT stage.
 *
 * Storage contract:
 * - rho_device and rho_selected_device are dense column-major buffers.
 * - One full density is contiguous in memory.
 * - ld / ld_selected are explicit leading dimensions (in doubles).
 *
 * Metadata arrays are device-resident; *_host vectors keep small host-side copies.
 */
struct GpuDensityBatch
{
    size_t K = 0;             //!< number of grid points per density
    size_t B = 0;             //!< number of densities
    size_t ld = 0;            //!< leading dimension for rho_device
    size_t B_selected = 0;    //!< number of selected densities for aux correction
    size_t ld_selected = 0;   //!< leading dimension for rho_selected_device
    size_t ld_raw = 0;        //!< leading dimension for optional rho_raw_device copy

    std::vector<PreFftDensitySpec> specs; //!< one spec per column in rho_device
    std::vector<int> selected_columns;    //!< original column index for each selected density
    std::vector<int> selected_inverse;    //!< size B, -1 if not selected else index in selected_columns

    // Main dense density buffers (device)
    double* rho_device = nullptr;
    double* rho_raw_device = nullptr;
    double* rho_selected_device = nullptr;

    // Selection helpers (device)
    int* selected_columns_device = nullptr;
    int* selected_inverse_device = nullptr;

    // Device metadata (full batch)
    double* Q_device = nullptr;
    double* A0_device = nullptr;
    double* Ax_device = nullptr;
    double* Ay_device = nullptr;
    double* Az_device = nullptr;
    double* r0x_device = nullptr;
    double* r0y_device = nullptr;
    double* r0z_device = nullptr;
    double* Mx_device = nullptr;
    double* My_device = nullptr;
    double* Mz_device = nullptr;
    double* M2_device = nullptr;
    double* sigma_device = nullptr;
    double* alpha_device = nullptr;

    // Optional selected-subset metadata (device)
    double* sigma_selected_device = nullptr;
    double* alpha_selected_device = nullptr;

    // Small host metadata copies for policy / validation
    std::vector<double> Q_host;
    std::vector<double> A0_host;
    std::vector<double> Ax_host;
    std::vector<double> Ay_host;
    std::vector<double> Az_host;
    std::vector<double> r0x_host;
    std::vector<double> r0y_host;
    std::vector<double> r0z_host;
    std::vector<double> Mx_host;
    std::vector<double> My_host;
    std::vector<double> Mz_host;
    std::vector<double> M2_host;
    std::vector<double> sigma_host;
    std::vector<double> alpha_host;
    PreFftStageTimingsMs timings_ms{};

    GpuDensityBatch() = default;
    ~GpuDensityBatch() { release_gpu_density_batch(*this); }

    GpuDensityBatch(GpuDensityBatch const&) = delete;
    GpuDensityBatch& operator=(GpuDensityBatch const&) = delete;

    GpuDensityBatch(GpuDensityBatch&& other) noexcept
    {
        *this = std::move(other);
    }

    GpuDensityBatch& operator=(GpuDensityBatch&& other) noexcept
    {
        if (this == &other) return *this;
        release_gpu_density_batch(*this);

        K = other.K;
        B = other.B;
        ld = other.ld;
        B_selected = other.B_selected;
        ld_selected = other.ld_selected;
        ld_raw = other.ld_raw;

        specs = std::move(other.specs);
        selected_columns = std::move(other.selected_columns);
        selected_inverse = std::move(other.selected_inverse);

        rho_device = other.rho_device;
        rho_raw_device = other.rho_raw_device;
        rho_selected_device = other.rho_selected_device;
        selected_columns_device = other.selected_columns_device;
        selected_inverse_device = other.selected_inverse_device;

        Q_device = other.Q_device;
        A0_device = other.A0_device;
        Ax_device = other.Ax_device;
        Ay_device = other.Ay_device;
        Az_device = other.Az_device;
        r0x_device = other.r0x_device;
        r0y_device = other.r0y_device;
        r0z_device = other.r0z_device;
        Mx_device = other.Mx_device;
        My_device = other.My_device;
        Mz_device = other.Mz_device;
        M2_device = other.M2_device;
        sigma_device = other.sigma_device;
        alpha_device = other.alpha_device;
        sigma_selected_device = other.sigma_selected_device;
        alpha_selected_device = other.alpha_selected_device;

        Q_host = std::move(other.Q_host);
        A0_host = std::move(other.A0_host);
        Ax_host = std::move(other.Ax_host);
        Ay_host = std::move(other.Ay_host);
        Az_host = std::move(other.Az_host);
        r0x_host = std::move(other.r0x_host);
        r0y_host = std::move(other.r0y_host);
        r0z_host = std::move(other.r0z_host);
        Mx_host = std::move(other.Mx_host);
        My_host = std::move(other.My_host);
        Mz_host = std::move(other.Mz_host);
        M2_host = std::move(other.M2_host);
        sigma_host = std::move(other.sigma_host);
        alpha_host = std::move(other.alpha_host);
        timings_ms = other.timings_ms;

        other.K = 0;
        other.B = 0;
        other.ld = 0;
        other.B_selected = 0;
        other.ld_selected = 0;
        other.ld_raw = 0;
        other.rho_device = nullptr;
        other.rho_raw_device = nullptr;
        other.rho_selected_device = nullptr;
        other.selected_columns_device = nullptr;
        other.selected_inverse_device = nullptr;
        other.Q_device = nullptr;
        other.A0_device = nullptr;
        other.Ax_device = nullptr;
        other.Ay_device = nullptr;
        other.Az_device = nullptr;
        other.r0x_device = nullptr;
        other.r0y_device = nullptr;
        other.r0z_device = nullptr;
        other.Mx_device = nullptr;
        other.My_device = nullptr;
        other.Mz_device = nullptr;
        other.M2_device = nullptr;
        other.sigma_device = nullptr;
        other.alpha_device = nullptr;
        other.sigma_selected_device = nullptr;
        other.alpha_selected_device = nullptr;
        other.timings_ms = PreFftStageTimingsMs{};
        return *this;
    }
};

bool pre_fft_density_gpu_available();
void release_pre_fft_density_gpu_workspace();

/**
 * @brief Execute GPU-native pre-FFT density pipeline.
 *
 * Pipeline:
 * 1) materialize densities on GPU (packed dense layout),
 * 2) compute metadata on GPU (Q, A0/r0, Mx/My/Mz/M2),
 * 3) CPU policy on tiny metadata (selection by |Q| threshold),
 * 4) compact selected subset on GPU,
 * 5) compute sigma/alpha on GPU and apply wrapped auxiliary Gaussian subtraction,
 * 6) scatter corrected selected densities back to rho_device.
 *
 * The output batch remains device-resident and is ready for later FFT stages.
 */
GpuDensityBatch build_pre_fft_density_batch_gpu(
    PreFftDensityKind kind,
    std::map<int, WannierFunction> const& cWannMap,
    std::map<int, WannierFunction> const& vWannMap,
    std::vector<PreFftDensitySpec> const& specs,
    PreFftAuxConfig const& config = {});

#endif  // WO_COULOMB_PRE_FFT_DENSITY_H
