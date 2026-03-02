#include "backend/backend.h"

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

class GpuSolverBase : public Solver
{
public:
    GpuSolverBase(
        const std::string& name,
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap)
        : Solver(name, vWannMap, cWannMap)
    {
        const RealMeshgrid* mesh = vWannMap.begin()->second.getMeshgrid();
        num_points = static_cast<size_t>(mesh->getNumDataPoints());

        if (num_points == 0) {
            throw std::runtime_error("GPU solver cannot allocate zero-sized buffers.");
        }

        checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_f1), sizeof(double2) * num_points),
            "cudaMalloc(device_f1)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_f2), sizeof(double2) * num_points),
            "cudaMalloc(device_f2)");
    }

    ~GpuSolverBase() override
    {
        if (device_f1) {
            cudaFree(device_f1);
        }
        if (device_f2) {
            cudaFree(device_f2);
        }
    }

    void calculate(
        std::vector<Integral>&,
        const bool = true,
        const unsigned int = 1,
        const unsigned int = 1) override
    {
        throw std::runtime_error("GPU solver not implemented yet.");
    }

private:
    double2* device_f1{};
    double2* device_f2{};
    size_t num_points{};
};
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
    }

    std::unique_ptr<Solver> createCoulombSolver(
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap) override
    {
        return std::make_unique<GpuSolverBase>("GpuCoulomb", vWannMap, cWannMap);
    }

    std::unique_ptr<Solver> createLocalFieldEffectsSolver(
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap) override
    {
        return std::make_unique<GpuSolverBase>("GpuLocalFieldEffects", vWannMap, cWannMap);
    }

    std::unique_ptr<Solver> createYukawaSolver(
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap,
        std::map<int, double> const&,
        std::map<int, double> const&,
        double,
        double) override
    {
        return std::make_unique<GpuSolverBase>("GpuYukawa", vWannMap, cWannMap);
    }
};

std::unique_ptr<Backend> make_gpu_backend()
{
    return std::make_unique<GpuBackend>();
}
