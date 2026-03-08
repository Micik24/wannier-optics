#include "backend/pre_fft_density.h"

#include <stdexcept>

void release_gpu_density_batch(GpuDensityBatch& batch)
{
    batch.K = 0;
    batch.B = 0;
    batch.ld = 0;
    batch.B_selected = 0;
    batch.ld_selected = 0;
    batch.rho_device = nullptr;
    batch.rho_selected_device = nullptr;
    batch.selected_columns_device = nullptr;
    batch.selected_inverse_device = nullptr;
    batch.Q_device = nullptr;
    batch.A0_device = nullptr;
    batch.Ax_device = nullptr;
    batch.Ay_device = nullptr;
    batch.Az_device = nullptr;
    batch.r0x_device = nullptr;
    batch.r0y_device = nullptr;
    batch.r0z_device = nullptr;
    batch.Mx_device = nullptr;
    batch.My_device = nullptr;
    batch.Mz_device = nullptr;
    batch.M2_device = nullptr;
    batch.sigma_device = nullptr;
    batch.alpha_device = nullptr;
    batch.sigma_selected_device = nullptr;
    batch.alpha_selected_device = nullptr;
    batch.specs.clear();
    batch.selected_columns.clear();
    batch.selected_inverse.clear();
    batch.Q_host.clear();
    batch.A0_host.clear();
    batch.Ax_host.clear();
    batch.Ay_host.clear();
    batch.Az_host.clear();
    batch.r0x_host.clear();
    batch.r0y_host.clear();
    batch.r0z_host.clear();
    batch.Mx_host.clear();
    batch.My_host.clear();
    batch.Mz_host.clear();
    batch.M2_host.clear();
    batch.sigma_host.clear();
    batch.alpha_host.clear();
}

bool pre_fft_density_gpu_available()
{
    return false;
}

void release_pre_fft_density_gpu_workspace()
{
}

GpuDensityBatch build_pre_fft_density_batch_gpu(
    PreFftDensityKind,
    std::map<int, WannierFunction> const&,
    std::map<int, WannierFunction> const&,
    std::vector<PreFftDensitySpec> const&,
    PreFftAuxConfig const&)
{
    throw std::runtime_error("GPU pre-FFT density pipeline is not available in this build.");
}
