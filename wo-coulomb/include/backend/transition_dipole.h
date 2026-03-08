#ifndef WO_COULOMB_TRANSITION_DIPOLE_H
#define WO_COULOMB_TRANSITION_DIPOLE_H

#include <array>
#include <vector>

struct TransitionDipoleTask
{
    int c_index = -1;
    int v_index = -1;
    std::array<int, 3> shift{0, 0, 0};  // S
};

struct TransitionDipoleValue
{
    double q = 0.0;
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
};

bool transition_dipole_gpu_available();

/**
 * @brief Compute transition dipoles for a list of (c,v,S) tasks on GPU.
 *
 * The implementation matches the CPU semantics used in calc_transition:
 * rho(x)=w_c0(x)w_v,-S(x),
 * Q=dV*sum rho,
 * d=(dV*sum rho*x, dV*sum rho*y, dV*sum rho*z),
 * optional non-orthogonality correction.
 */
std::vector<TransitionDipoleValue> compute_transition_dipoles_gpu(
    std::vector<const double*> const& conduction_values,
    std::vector<const double*> const& valence_values,
    std::vector<int> const& dim,
    std::vector<double> const& supercell_cond,
    std::vector<double> const& supercell_val,
    double dV,
    const double* XX,
    const double* YY,
    const double* ZZ,
    std::vector<std::vector<double>> const& center_c,
    std::vector<std::vector<double>> const& center_v,
    std::vector<std::vector<double>> const& unitcell_t,
    std::vector<TransitionDipoleTask> const& tasks,
    bool apply_correction);

#endif  // WO_COULOMB_TRANSITION_DIPOLE_H
