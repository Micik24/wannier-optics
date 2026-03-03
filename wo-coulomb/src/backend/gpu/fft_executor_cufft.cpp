#include "backend/fft_executor.h"

#include <cuda_runtime.h>
#include <cufft.h>

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

void checkCufft(cufftResult status, const char* what)
{
    if (status == CUFFT_SUCCESS) {
        return;
    }
    std::ostringstream msg;
    msg << what << ": cuFFT error code " << static_cast<int>(status);
    throw std::runtime_error(msg.str());
}

class CufftExecutor final : public FftExecutor
{
public:
    CufftExecutor(const std::vector<int>& dims, FftBufferView buffer_view, FftDirection direction)
        : host_buffer(nullptr), plan(0), device_buffer(nullptr), dir(direction), owns_device_buffer(false)
    {
        if (dims.size() != 3) {
            throw std::runtime_error("cuFFT executor expects 3D dims.");
        }
        if (!buffer_view.data) {
            throw std::runtime_error("cuFFT executor received a null buffer.");
        }
        const int nx = dims[2];
        const int ny = dims[1];
        const int nz = dims[0];

        num_points_ = static_cast<size_t>(dims[0]) * dims[1] * dims[2];
        if (num_points_ == 0) {
            throw std::runtime_error("cuFFT executor cannot allocate zero-sized buffers.");
        }

        if (buffer_view.location == FftBufferLocation::Device) {
            device_buffer = static_cast<cufftDoubleComplex*>(buffer_view.data);
            owns_device_buffer = false;
        } else {
            host_buffer = buffer_view.data;
            owns_device_buffer = true;
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_buffer),
                sizeof(cufftDoubleComplex) * num_points_), "cudaMalloc(device_buffer)");
        }

        checkCufft(cufftPlan3d(&plan, nx, ny, nz, CUFFT_Z2Z), "cufftPlan3d");
    }

    ~CufftExecutor() override
    {
        if (plan) {
            cufftDestroy(plan);
        }
        if (owns_device_buffer && device_buffer) {
            cudaFree(device_buffer);
        }
    }

    void exec() override
    {
        const size_t bytes = sizeof(cufftDoubleComplex) * num_points_;
        if (host_buffer) {
            checkCuda(cudaMemcpy(device_buffer, host_buffer, bytes, cudaMemcpyHostToDevice),
                "cudaMemcpy host->device");
        }

        const int sign = (dir == FftDirection::Forward) ? CUFFT_FORWARD : CUFFT_INVERSE;
        checkCufft(cufftExecZ2Z(plan, device_buffer, device_buffer, sign), "cufftExecZ2Z");

        if (host_buffer) {
            checkCuda(cudaMemcpy(host_buffer, device_buffer, bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy device->host");
        }
    }

private:
    void* host_buffer;
    cufftHandle plan;
    cufftDoubleComplex* device_buffer;
    FftDirection dir;
    bool owns_device_buffer;
    size_t num_points_{};
};

class CufftExecutorFactory final : public FftExecutorFactory
{
public:
    std::unique_ptr<FftExecutor> create(
        const std::vector<int>& dims,
        FftBufferView buffer,
        FftDirection direction) override
    {
        return std::make_unique<CufftExecutor>(dims, buffer, direction);
    }
};
}  // namespace

std::shared_ptr<FftExecutorFactory> make_cufft_executor_factory()
{
    static std::shared_ptr<FftExecutorFactory> factory = std::make_shared<CufftExecutorFactory>();
    return factory;
}
