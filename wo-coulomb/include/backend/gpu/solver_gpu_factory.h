#ifndef WO_COULOMB_SOLVER_GPU_FACTORY_H
#define WO_COULOMB_SOLVER_GPU_FACTORY_H

#include <map>
#include <memory>

#include "solver.h"
#include "wannierfunction.h"

std::unique_ptr<Solver> make_coulomb_solver_gpu(
    std::map<int, WannierFunction> const& vWannMap,
    std::map<int, WannierFunction> const& cWannMap,
    bool wrap_aux = true);

std::unique_ptr<Solver> make_local_field_effects_solver_gpu(
    std::map<int, WannierFunction> const& vWannMap,
    std::map<int, WannierFunction> const& cWannMap);

std::unique_ptr<Solver> make_yukawa_solver_gpu(
    std::map<int, WannierFunction> const& vWannMap,
    std::map<int, WannierFunction> const& cWannMap,
    std::map<int, double> const& vMeanDensity,
    std::map<int, double> const& cMeanDensity,
    double relativePermittivity,
    double screeningAlpha);

#endif  // WO_COULOMB_SOLVER_GPU_FACTORY_H
