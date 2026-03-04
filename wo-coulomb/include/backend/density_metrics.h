#ifndef WO_COULOMB_DENSITY_METRICS_H
#define WO_COULOMB_DENSITY_METRICS_H

#include <array>
#include <cstddef>
#include <map>
#include <vector>

#include "wannierfunction.h"

enum class DensityMetricKind
{
    TransitionCv
};

struct DensityMetricSpec
{
    int idx1 = -1;
    int idx2 = -1;
    std::array<int, 3> R{0, 0, 0};
};

bool density_metrics_gpu_available();
size_t density_metrics_recommended_max_specs();

std::vector<double> compute_abs_charge_batch_gpu(
    DensityMetricKind kind,
    std::map<int, WannierFunction> const& cWannMap,
    std::map<int, WannierFunction> const& vWannMap,
    std::vector<DensityMetricSpec> const& specs);

#endif  // WO_COULOMB_DENSITY_METRICS_H
