#include "backend/transition_dipole.h"

#include <stdexcept>

bool transition_dipole_gpu_available()
{
    return false;
}

std::vector<TransitionDipoleValue> compute_transition_dipoles_gpu(
    std::vector<const double*> const&,
    std::vector<const double*> const&,
    std::vector<int> const&,
    std::vector<double> const&,
    std::vector<double> const&,
    double,
    const double*,
    const double*,
    const double*,
    std::vector<std::vector<double>> const&,
    std::vector<std::vector<double>> const&,
    std::vector<std::vector<double>> const&,
    std::vector<TransitionDipoleTask> const&,
    bool)
{
    throw std::runtime_error("GPU transition dipoles are not available in this build.");
}
