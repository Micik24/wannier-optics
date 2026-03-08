#include "backend/backend.h"
#include "backend/fft_executor.h"
#include "backend/gpu/solver_gpu_factory.h"

#include <cuda_runtime.h>
#include <mpi.h>

#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace {
void checkCuda(cudaError_t status, const char* what)
{
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream msg;
    msg << what << ": " << cudaGetErrorString(status);
    throw std::runtime_error(msg.str());
}

int get_local_rank()
{
    const char* envs[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "MPI_LOCALRANKID",
        "PMI_LOCAL_RANK",
    };
    for (const char* env : envs) {
        const char* value = std::getenv(env);
        if (value && *value) {
            return std::atoi(value);
        }
    }
    return -1;
}
}  // namespace

class GpuBackend final : public Backend
{
public:
    GpuBackend()
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int device_count = 0;
        checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count <= 0) {
            throw std::runtime_error("No CUDA devices available.");
        }

        const int local_rank = get_local_rank();
        const int device_index = (local_rank >= 0 ? local_rank : rank) % device_count;

        checkCuda(cudaSetDevice(device_index), "cudaSetDevice");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        fft_factory = make_cufft_executor_factory();
    }

    std::unique_ptr<Solver> createCoulombSolver(
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap) override
    {
        return make_coulomb_solver_gpu(vWannMap, cWannMap, true);
    }

    std::unique_ptr<Solver> createLocalFieldEffectsSolver(
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap) override
    {
        return make_local_field_effects_solver_gpu(vWannMap, cWannMap);
    }

    std::unique_ptr<Solver> createYukawaSolver(
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap,
        std::map<int, double> const& vMeanDensity,
        std::map<int, double> const& cMeanDensity,
        double relativePermittivity,
        double screeningAlpha) override
    {
        return make_yukawa_solver_gpu(
            vWannMap, cWannMap, vMeanDensity, cMeanDensity, relativePermittivity, screeningAlpha);
    }

private:
    std::shared_ptr<FftExecutorFactory> fft_factory;
};

std::unique_ptr<Backend> make_gpu_backend()
{
    return std::make_unique<GpuBackend>();
}
