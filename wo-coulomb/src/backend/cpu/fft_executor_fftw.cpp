#include "backend/fft_executor.h"

#include <fftw3.h>
#include <omp.h>

#include <stdexcept>

namespace {
class FftwExecutor final : public FftExecutor
{
public:
    FftwExecutor(const std::vector<int>& dims, FftBufferView buffer_view, FftDirection direction)
        : plan(nullptr)
    {
        if (dims.size() != 3) {
            throw std::runtime_error("FFTW executor expects 3D dims.");
        }
        if (!buffer_view.data) {
            throw std::runtime_error("FFTW executor received a null buffer.");
        }
        if (buffer_view.location != FftBufferLocation::Host) {
            throw std::runtime_error("FFTW executor requires host memory.");
        }

        auto* buffer = static_cast<fftw_complex*>(buffer_view.data);
        const int nx = dims[2];
        const int ny = dims[1];
        const int nz = dims[0];
        const int sign = (direction == FftDirection::Forward) ? FFTW_FORWARD : FFTW_BACKWARD;

        // FFTW planning is not thread-safe unless planner thread safety is enabled.
        #pragma omp critical
        {
            plan = fftw_plan_dft_3d(nx, ny, nz, buffer, buffer, sign, FFTW_ESTIMATE);
        }

        if (!plan) {
            throw std::runtime_error("FFTW plan creation failed.");
        }
    }

    ~FftwExecutor() override
    {
        if (plan) {
            fftw_destroy_plan(plan);
        }
    }

    void exec() override
    {
        fftw_execute(plan);
    }

private:
    fftw_plan plan;
};

class FftwExecutorFactory final : public FftExecutorFactory
{
public:
    std::unique_ptr<FftExecutor> create(
        const std::vector<int>& dims,
        FftBufferView buffer,
        FftDirection direction) override
    {
        return std::make_unique<FftwExecutor>(dims, buffer, direction);
    }
};
}  // namespace

std::shared_ptr<FftExecutorFactory> make_fftw_executor_factory()
{
    static std::shared_ptr<FftExecutorFactory> factory = std::make_shared<FftwExecutorFactory>();
    return factory;
}
