#include "backend/backend.h"
#include "backend/cpu/solver_cpu.h"

#include <memory>

class CpuBackend final : public Backend
{
public:
    std::unique_ptr<Solver> createCoulombSolver(
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap) override
    {
        return std::make_unique<CoulombSolver>(vWannMap, cWannMap);
    }

    std::unique_ptr<Solver> createLocalFieldEffectsSolver(
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap) override
    {
        return std::make_unique<LocalFieldEffectsSolver>(vWannMap, cWannMap);
    }

    std::unique_ptr<Solver> createYukawaSolver(
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap,
        std::map<int, double> const& vMeanDensity,
        std::map<int, double> const& cMeanDensity,
        double relativePermittivity,
        double screeningAlpha) override
    {
        return std::make_unique<YukawaSolver>(vWannMap, cWannMap, vMeanDensity, cMeanDensity, relativePermittivity, screeningAlpha);
    }
};

std::unique_ptr<Backend> make_cpu_backend()
{
    return std::make_unique<CpuBackend>();
}
