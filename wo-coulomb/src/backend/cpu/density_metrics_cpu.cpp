#include "backend/density_metrics.h"

#include <stdexcept>

bool density_metrics_gpu_available()
{
    return false;
}

size_t density_metrics_recommended_max_specs()
{
    return 32768;
}

void release_density_metrics_gpu_workspace()
{
}

std::vector<double> compute_abs_charge_batch_gpu(
    DensityMetricKind,
    std::map<int, WannierFunction> const&,
    std::map<int, WannierFunction> const&,
    std::vector<DensityMetricSpec> const&)
{
    throw std::runtime_error("GPU density metrics are not available in this build.");
}

std::vector<DensityMoments> compute_density_moments_batch_gpu(
    DensityMetricKind,
    std::map<int, WannierFunction> const&,
    std::map<int, WannierFunction> const&,
    std::vector<DensityMetricSpec> const&)
{
    throw std::runtime_error("GPU density metrics are not available in this build.");
}
