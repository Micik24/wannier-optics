#ifndef WO_COULOMB_GPU_DENSITY_METRICS_KERNELS_H
#define WO_COULOMB_GPU_DENSITY_METRICS_KERNELS_H

#include <cuda_runtime.h>

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
    cudaStream_t stream);

#endif  // WO_COULOMB_GPU_DENSITY_METRICS_KERNELS_H
