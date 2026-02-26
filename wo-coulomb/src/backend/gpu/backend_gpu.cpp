#include "backend/backend.h"

#include <stdexcept>

namespace {
[[noreturn]] std::unique_ptr<Solver> throwNotImplemented()
{
    throw std::runtime_error("GPU backend not implemented yet.");
}
}  // namespace

class GpuBackend final : public Backend
{
public:
    std::unique_ptr<Solver> createCoulombSolver(
        std::map<int, WannierFunction> const&,
        std::map<int, WannierFunction> const&) override
    {
        return throwNotImplemented();
    }

    std::unique_ptr<Solver> createLocalFieldEffectsSolver(
        std::map<int, WannierFunction> const&,
        std::map<int, WannierFunction> const&) override
    {
        return throwNotImplemented();
    }

    std::unique_ptr<Solver> createYukawaSolver(
        std::map<int, WannierFunction> const&,
        std::map<int, WannierFunction> const&,
        std::map<int, double> const&,
        std::map<int, double> const&,
        double,
        double) override
    {
        return throwNotImplemented();
    }
};

std::unique_ptr<Backend> make_gpu_backend()
{
    return std::make_unique<GpuBackend>();
}
