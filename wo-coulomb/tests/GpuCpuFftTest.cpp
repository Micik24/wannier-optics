#include <gtest/gtest.h>

#include "backend/fft_executor.h"

#include <fftw3.h>

#include <cmath>
#include <vector>

#if defined(WO_TEST_CUDA)
#include <cuda_runtime.h>
#endif

namespace {
#if defined(WO_TEST_CUDA)
bool hasCudaDevice()
{
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    return status == cudaSuccess && count > 0;
}
#endif

void fillDeterministic(fftw_complex* data, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        data[i][0] = std::sin(0.1 * x) + 0.01 * std::cos(0.3 * x);
        data[i][1] = std::cos(0.07 * x) - 0.02 * std::sin(0.2 * x);
    }
}

double maxAbsDiff(const fftw_complex* a, const fftw_complex* b, size_t n)
{
    double max_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double dr = std::abs(a[i][0] - b[i][0]);
        const double di = std::abs(a[i][1] - b[i][1]);
        max_diff = std::max(max_diff, dr);
        max_diff = std::max(max_diff, di);
    }
    return max_diff;
}
}  // namespace

TEST(FftExecutorGpuCpu, ForwardAndInverseMatch)
{
#if !defined(WO_TEST_CUDA)
    GTEST_SKIP() << "CUDA not enabled for tests.";
#else
    if (!hasCudaDevice()) {
        GTEST_SKIP() << "No CUDA device available.";
    }

    const std::vector<int> dims{4, 5, 6};  // z, y, x (solver convention)
    const size_t n = static_cast<size_t>(dims[0]) * dims[1] * dims[2];

    fftw_complex* cpu = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * n));
    fftw_complex* gpu = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * n));
    ASSERT_NE(cpu, nullptr);
    ASSERT_NE(gpu, nullptr);

    fillDeterministic(cpu, n);
    for (size_t i = 0; i < n; ++i) {
        gpu[i][0] = cpu[i][0];
        gpu[i][1] = cpu[i][1];
    }

    auto cpu_factory = make_fftw_executor_factory();
    auto gpu_factory = make_cufft_executor_factory();

    auto fft_cpu_fwd = cpu_factory->create(dims, FftBufferView{cpu, FftBufferLocation::Host}, FftDirection::Forward);
    auto fft_gpu_fwd = gpu_factory->create(dims, FftBufferView{gpu, FftBufferLocation::Host}, FftDirection::Forward);

    fft_cpu_fwd->exec();
    fft_gpu_fwd->exec();

    const double fwd_diff = maxAbsDiff(cpu, gpu, n);
    EXPECT_LT(fwd_diff, 1e-8);

    auto fft_cpu_inv = cpu_factory->create(dims, FftBufferView{cpu, FftBufferLocation::Host}, FftDirection::Inverse);
    auto fft_gpu_inv = gpu_factory->create(dims, FftBufferView{gpu, FftBufferLocation::Host}, FftDirection::Inverse);

    fft_cpu_inv->exec();
    fft_gpu_inv->exec();

    const double inv_diff = maxAbsDiff(cpu, gpu, n);
    EXPECT_LT(inv_diff, 1e-8);

    fftw_free(cpu);
    fftw_free(gpu);
#endif
}
