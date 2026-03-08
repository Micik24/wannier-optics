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

__global__ void density_moments_kernel(
    const double* const* d_w1_ptrs,
    const double* const* d_w2_ptrs,
    const int* d_offsets_xyz,
    const int* d_valid,
    int batch_size,
    int dimx,
    int dimy,
    int dimz,
    double dV,
    const double* lattice,
    const double* origin,
    double* d_abs_charge,
    double* d_mx,
    double* d_my,
    double* d_mz)
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
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
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

            const double x = i * lattice[0] / dimx + j * lattice[3] / dimy + k * lattice[6] / dimz + origin[0];
            const double y = i * lattice[1] / dimx + j * lattice[4] / dimy + k * lattice[7] / dimz + origin[1];
            const double z = i * lattice[2] / dimx + j * lattice[5] / dimy + k * lattice[8] / dimz + origin[2];
            vx = value * x;
            vy = value * y;
            vz = value * z;
        }
    }

    extern __shared__ double sdata[];
    double* s_abs = sdata;
    double* s_mx = s_abs + blockDim.x;
    double* s_my = s_mx + blockDim.x;
    double* s_mz = s_my + blockDim.x;

    s_abs[threadIdx.x] = value;
    s_mx[threadIdx.x] = vx;
    s_my[threadIdx.x] = vy;
    s_mz[threadIdx.x] = vz;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_abs[threadIdx.x] += s_abs[threadIdx.x + stride];
            s_mx[threadIdx.x] += s_mx[threadIdx.x + stride];
            s_my[threadIdx.x] += s_my[threadIdx.x + stride];
            s_mz[threadIdx.x] += s_mz[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAddDouble(&d_abs_charge[spec], s_abs[0]);
        atomicAddDouble(&d_mx[spec], s_mx[0]);
        atomicAddDouble(&d_my[spec], s_my[0]);
        atomicAddDouble(&d_mz[spec], s_mz[0]);
    }
}

}  // namespace

void launch_density_moments_kernel(
    const double* const* d_w1_ptrs,
    const double* const* d_w2_ptrs,
    const int* d_offsets_xyz,
    const int* d_valid,
    int batch_size,
    int dimx,
    int dimy,
    int dimz,
    double dV,
    const double* lattice,
    const double* origin,
    double* d_abs_charge,
    double* d_mx,
    double* d_my,
    double* d_mz,
    cudaStream_t stream)
{
    if (batch_size <= 0) {
        return;
    }

    const int threads = 256;
    const int N = dimx * dimy * dimz;
    const int blocks_x = (N + threads - 1) / threads;
    const dim3 grid(blocks_x, batch_size, 1);

    density_moments_kernel<<<grid, threads, sizeof(double) * threads * 4, stream>>>(
        d_w1_ptrs,
        d_w2_ptrs,
        d_offsets_xyz,
        d_valid,
        batch_size,
        dimx,
        dimy,
        dimz,
        dV,
        lattice,
        origin,
        d_abs_charge,
        d_mx,
        d_my,
        d_mz);
}
