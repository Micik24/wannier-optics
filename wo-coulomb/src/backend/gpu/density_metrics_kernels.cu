#include "density_metrics_kernels.h"

#include <cmath>

namespace {

#if __CUDA_ARCH__ < 600
__device__ inline double atomicAddDouble(double* address, double val)
{
    auto* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ inline double atomicAddDouble(double* address, double val)
{
    return atomicAdd(address, val);
}
#endif

__global__ void abs_charge_kernel(
    const double* const* d_w1_ptrs,
    const double* const* d_w2_ptrs,
    const int* d_offsets_xyz,
    const int* d_valid,
    int batch_size,
    int dimx,
    int dimy,
    int dimz,
    double dV,
    double* d_abs_charge)
{
    const int spec = blockIdx.y;
    if (spec >= batch_size || d_valid[spec] == 0) {
        return;
    }

    const double* w1 = d_w1_ptrs[spec];
    const double* w2 = d_w2_ptrs[spec];
    if (w1 == nullptr || w2 == nullptr) {
        return;
    }

    const int offx = d_offsets_xyz[3 * spec + 0];
    const int offy = d_offsets_xyz[3 * spec + 1];
    const int offz = d_offsets_xyz[3 * spec + 2];

    const int N = dimx * dimy * dimz;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double value = 0.0;
    if (idx < N) {
        const int i = idx % dimx;
        const int yz = idx / dimx;
        const int j = yz % dimy;
        const int k = yz / dimy;

        const int i2 = i + offx;
        const int j2 = j + offy;
        const int k2 = k + offz;

        if (i2 >= 0 && i2 < dimx &&
            j2 >= 0 && j2 < dimy &&
            k2 >= 0 && k2 < dimz) {
            const int idx2 = i2 + dimx * (j2 + dimy * k2);
            value = fabs(w1[idx] * w2[idx2]) * dV;
        }
    }

    extern __shared__ double sdata[];
    sdata[threadIdx.x] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAddDouble(&d_abs_charge[spec], sdata[0]);
    }
}

}  // namespace

void launch_abs_charge_kernel(
    const double* const* d_w1_ptrs,
    const double* const* d_w2_ptrs,
    const int* d_offsets_xyz,
    const int* d_valid,
    int batch_size,
    int dimx,
    int dimy,
    int dimz,
    double dV,
    double* d_abs_charge,
    cudaStream_t stream)
{
    if (batch_size <= 0) {
        return;
    }

    const int threads = 256;
    const int N = dimx * dimy * dimz;
    const int blocks_x = (N + threads - 1) / threads;
    const dim3 grid(blocks_x, batch_size, 1);

    abs_charge_kernel<<<grid, threads, sizeof(double) * threads, stream>>>(
        d_w1_ptrs,
        d_w2_ptrs,
        d_offsets_xyz,
        d_valid,
        batch_size,
        dimx,
        dimy,
        dimz,
        dV,
        d_abs_charge);
}
