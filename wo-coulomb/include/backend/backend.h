#ifndef WO_COULOMB_BACKEND_H
#define WO_COULOMB_BACKEND_H

/**
 * @file backend.h
 * @brief Backend interface for solver creation (CPU/GPU).
 */

#include <map>
#include <memory>

#include "solver.h"
#include "wannierfunction.h"

class Backend
{
public:
    virtual ~Backend() = default;

    virtual std::unique_ptr<Solver> createCoulombSolver(
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap) = 0;

    virtual std::unique_ptr<Solver> createLocalFieldEffectsSolver(
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap) = 0;

    virtual std::unique_ptr<Solver> createYukawaSolver(
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap,
        std::map<int, double> const& vMeanDensity,
        std::map<int, double> const& cMeanDensity,
        double relativePermittivity,
        double screeningAlpha) = 0;
};

std::unique_ptr<Backend> makeBackend();

#endif  // WO_COULOMB_BACKEND_H
