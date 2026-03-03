#ifndef WO_COULOMB_FFT_EXECUTOR_H
#define WO_COULOMB_FFT_EXECUTOR_H

#include <memory>
#include <vector>

enum class FftDirection
{
    Forward,
    Inverse
};

enum class FftBufferLocation
{
    Host,
    Device
};

struct FftBufferView
{
    void* data = nullptr;
    FftBufferLocation location = FftBufferLocation::Host;
};

class FftExecutor
{
public:
    virtual ~FftExecutor() = default;
    virtual void exec() = 0;
};

class FftExecutorFactory
{
public:
    virtual ~FftExecutorFactory() = default;
    virtual std::unique_ptr<FftExecutor> create(
        const std::vector<int>& dims,
        FftBufferView buffer,
        FftDirection direction) = 0;
};

// CPU FFTW factory (always available).
std::shared_ptr<FftExecutorFactory> make_fftw_executor_factory();

// GPU cuFFT factory (defined in the GPU backend build).
std::shared_ptr<FftExecutorFactory> make_cufft_executor_factory();

#endif  // WO_COULOMB_FFT_EXECUTOR_H
