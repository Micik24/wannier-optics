#include "backend/pre_fft_density.h"

#include <cuda_runtime.h>
#include <mpi.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace {

constexpr double PI_CONST = 3.141592653589793238462643383279502884;

void checkCuda(cudaError_t status, const char* what)
{
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream msg;
    msg << what << ": " << cudaGetErrorString(status);
    throw std::runtime_error(msg.str());
}

void safeCudaFree(void* ptr)
{
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

struct ScopedCudaEvents
{
    explicit ScopedCudaEvents(bool enabled)
    {
        if (enabled) {
            checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
            checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
        }
    }

    ~ScopedCudaEvents()
    {
        if (start != nullptr) cudaEventDestroy(start);
        if (stop != nullptr) cudaEventDestroy(stop);
    }

    ScopedCudaEvents(ScopedCudaEvents const&) = delete;
    ScopedCudaEvents& operator=(ScopedCudaEvents const&) = delete;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
};

template <typename T>
struct ScopedDeviceBuffer
{
    T* ptr = nullptr;

    ~ScopedDeviceBuffer() { safeCudaFree(ptr); }

    ScopedDeviceBuffer(ScopedDeviceBuffer const&) = delete;
    ScopedDeviceBuffer& operator=(ScopedDeviceBuffer const&) = delete;
    ScopedDeviceBuffer() = default;
};

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

struct DeviceWannier
{
    double* values = nullptr;
    const double* host_values = nullptr;
};

class PreFftGpuContext
{
public:
    static PreFftGpuContext& instance()
    {
        auto& ctx = globalInstance();
        if (!ctx) {
            ctx.reset(new PreFftGpuContext());
        }
        return *ctx;
    }

    static void releaseGlobal()
    {
        auto& ctx = globalInstance();
        ctx.reset();
    }

    ~PreFftGpuContext()
    {
        for (auto& [id, wf] : c_cache_) {
            safeCudaFree(wf.values);
        }
        for (auto& [id, wf] : v_cache_) {
            safeCudaFree(wf.values);
        }

        safeCudaFree(d_x_);
        safeCudaFree(d_y_);
        safeCudaFree(d_z_);
        safeCudaFree(d_wq_);
        safeCudaFree(d_wx_);
        safeCudaFree(d_wy_);
        safeCudaFree(d_wz_);
        safeCudaFree(d_w2_);
        safeCudaFree(d_shifts27_);
        safeCudaFree(d_w1_ptrs_);
        safeCudaFree(d_w2_ptrs_);
        safeCudaFree(d_offsets_xyz_);
        safeCudaFree(d_valid_);

        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    void initMesh(
        std::map<int, WannierFunction> const& cWannMap,
        std::map<int, WannierFunction> const& vWannMap)
    {
        if (cWannMap.empty()) throw std::runtime_error("cWannMap must not be empty.");
        if (vWannMap.empty()) throw std::runtime_error("vWannMap must not be empty.");

        const RealMeshgrid* c_mesh = cWannMap.begin()->second.getMeshgrid();
        const RealMeshgrid* v_mesh = vWannMap.begin()->second.getMeshgrid();
        if (!(*c_mesh == *v_mesh)) {
            throw std::runtime_error("Conduction and valence meshes are not compatible.");
        }

        const std::vector<int> dim = c_mesh->getDim();
        if (!initialized_) {
            dimx_ = dim[0];
            dimy_ = dim[1];
            dimz_ = dim[2];
            K_ = static_cast<size_t>(dimx_) * dimy_ * dimz_;
            dV_ = c_mesh->getdV();

            lattice_ = c_mesh->getLattice();
            origin_ = c_mesh->getOrigin();

            std::vector<double> hx(K_), hy(K_), hz(K_);
            std::vector<double> hwq(K_), hwx(K_), hwy(K_), hwz(K_), hw2(K_);
            for (size_t i = 0; i < K_; ++i) {
                double x = 0.0;
                double y = 0.0;
                double z = 0.0;
                c_mesh->xyz(static_cast<int>(i), x, y, z);
                hx[i] = x;
                hy[i] = y;
                hz[i] = z;
                hwq[i] = dV_;
                hwx[i] = x * dV_;
                hwy[i] = y * dV_;
                hwz[i] = z * dV_;
                hw2[i] = (x * x + y * y + z * z) * dV_;
            }

            std::vector<double> shifts27(27 * 3, 0.0);
            int l = 0;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    for (int k = -1; k <= 1; ++k) {
                        shifts27[3 * l + 0] = i * lattice_[0][0] + j * lattice_[1][0] + k * lattice_[2][0];
                        shifts27[3 * l + 1] = i * lattice_[0][1] + j * lattice_[1][1] + k * lattice_[2][1];
                        shifts27[3 * l + 2] = i * lattice_[0][2] + j * lattice_[1][2] + k * lattice_[2][2];
                        l++;
                    }
                }
            }

            checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_x_), sizeof(double) * K_), "cudaMalloc(d_x)");
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_y_), sizeof(double) * K_), "cudaMalloc(d_y)");
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_z_), sizeof(double) * K_), "cudaMalloc(d_z)");
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_wq_), sizeof(double) * K_), "cudaMalloc(d_wq)");
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_wx_), sizeof(double) * K_), "cudaMalloc(d_wx)");
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_wy_), sizeof(double) * K_), "cudaMalloc(d_wy)");
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_wz_), sizeof(double) * K_), "cudaMalloc(d_wz)");
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_w2_), sizeof(double) * K_), "cudaMalloc(d_w2)");
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_shifts27_), sizeof(double) * 27 * 3), "cudaMalloc(d_shifts27)");

            checkCuda(cudaMemcpyAsync(d_x_, hx.data(), sizeof(double) * K_, cudaMemcpyHostToDevice, stream_), "copy x");
            checkCuda(cudaMemcpyAsync(d_y_, hy.data(), sizeof(double) * K_, cudaMemcpyHostToDevice, stream_), "copy y");
            checkCuda(cudaMemcpyAsync(d_z_, hz.data(), sizeof(double) * K_, cudaMemcpyHostToDevice, stream_), "copy z");
            checkCuda(cudaMemcpyAsync(d_wq_, hwq.data(), sizeof(double) * K_, cudaMemcpyHostToDevice, stream_), "copy wq");
            checkCuda(cudaMemcpyAsync(d_wx_, hwx.data(), sizeof(double) * K_, cudaMemcpyHostToDevice, stream_), "copy wx");
            checkCuda(cudaMemcpyAsync(d_wy_, hwy.data(), sizeof(double) * K_, cudaMemcpyHostToDevice, stream_), "copy wy");
            checkCuda(cudaMemcpyAsync(d_wz_, hwz.data(), sizeof(double) * K_, cudaMemcpyHostToDevice, stream_), "copy wz");
            checkCuda(cudaMemcpyAsync(d_w2_, hw2.data(), sizeof(double) * K_, cudaMemcpyHostToDevice, stream_), "copy w2");
            checkCuda(cudaMemcpyAsync(d_shifts27_, shifts27.data(), sizeof(double) * 27 * 3, cudaMemcpyHostToDevice, stream_), "copy shifts27");
            checkCuda(cudaStreamSynchronize(stream_), "sync initMesh");

            initialized_ = true;
            return;
        }

        if (dimx_ != dim[0] || dimy_ != dim[1] || dimz_ != dim[2]) {
            throw std::runtime_error("Mesh dimensions changed during GPU pre-FFT pipeline.");
        }
    }

    void ensureCapacity(size_t B)
    {
        if (B <= capacity_) return;

        safeCudaFree(d_w1_ptrs_);
        safeCudaFree(d_w2_ptrs_);
        safeCudaFree(d_offsets_xyz_);
        safeCudaFree(d_valid_);

        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_w1_ptrs_), sizeof(double*) * B), "cudaMalloc(d_w1_ptrs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_w2_ptrs_), sizeof(double*) * B), "cudaMalloc(d_w2_ptrs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_offsets_xyz_), sizeof(int) * B * 3), "cudaMalloc(d_offsets_xyz)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_valid_), sizeof(int) * B), "cudaMalloc(d_valid)");
        capacity_ = B;
    }

    const double* ensureUploadedConduction(WannierFunction const& wf)
    {
        return ensureUploaded(wf, c_cache_);
    }

    const double* ensureUploadedValence(WannierFunction const& wf)
    {
        return ensureUploaded(wf, v_cache_);
    }

    cudaStream_t stream() const { return stream_; }
    int dimx() const { return dimx_; }
    int dimy() const { return dimy_; }
    int dimz() const { return dimz_; }
    size_t K() const { return K_; }
    std::vector<std::vector<double>> const& lattice() const { return lattice_; }
    const double* d_x() const { return d_x_; }
    const double* d_y() const { return d_y_; }
    const double* d_z() const { return d_z_; }
    const double* d_wq() const { return d_wq_; }
    const double* d_wx() const { return d_wx_; }
    const double* d_wy() const { return d_wy_; }
    const double* d_wz() const { return d_wz_; }
    const double* d_w2() const { return d_w2_; }
    const double* d_shifts27() const { return d_shifts27_; }
    const double** d_w1_ptrs() const { return d_w1_ptrs_; }
    const double** d_w2_ptrs() const { return d_w2_ptrs_; }
    int* d_offsets_xyz() const { return d_offsets_xyz_; }
    int* d_valid() const { return d_valid_; }

private:
    static std::unique_ptr<PreFftGpuContext>& globalInstance()
    {
        static std::unique_ptr<PreFftGpuContext> ctx{};
        return ctx;
    }

    PreFftGpuContext()
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int device_count = 0;
        checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count <= 0) {
            throw std::runtime_error("No CUDA device available for pre-FFT density pipeline.");
        }

        const int local_rank = get_local_rank();
        const int device_index = (local_rank >= 0 ? local_rank : rank) % device_count;
        checkCuda(cudaSetDevice(device_index), "cudaSetDevice");
        checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    const double* ensureUploaded(WannierFunction const& wf, std::unordered_map<int, DeviceWannier>& cache)
    {
        auto it = cache.find(wf.getId());
        const double* host_values = wf.getValue();
        if (it == cache.end()) {
            DeviceWannier d{};
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&d.values), sizeof(double) * K_), "cudaMalloc(wannier values)");
            checkCuda(cudaMemcpyAsync(d.values, host_values, sizeof(double) * K_, cudaMemcpyHostToDevice, stream_), "copy wannier");
            d.host_values = host_values;
            auto [new_it, ok] = cache.insert({wf.getId(), d});
            if (!ok) {
                throw std::runtime_error("Failed to cache uploaded Wannier values.");
            }
            return new_it->second.values;
        }

        if (it->second.host_values != host_values) {
            checkCuda(cudaMemcpyAsync(it->second.values, host_values, sizeof(double) * K_, cudaMemcpyHostToDevice, stream_), "refresh wannier");
            it->second.host_values = host_values;
        }
        return it->second.values;
    }

    bool initialized_ = false;
    int dimx_ = 0;
    int dimy_ = 0;
    int dimz_ = 0;
    size_t K_ = 0;
    double dV_ = 0.0;

    std::vector<std::vector<double>> lattice_{};
    std::vector<double> origin_{};

    std::unordered_map<int, DeviceWannier> c_cache_;
    std::unordered_map<int, DeviceWannier> v_cache_;

    cudaStream_t stream_ = nullptr;

    double* d_x_ = nullptr;
    double* d_y_ = nullptr;
    double* d_z_ = nullptr;
    double* d_wq_ = nullptr;
    double* d_wx_ = nullptr;
    double* d_wy_ = nullptr;
    double* d_wz_ = nullptr;
    double* d_w2_ = nullptr;
    double* d_shifts27_ = nullptr;

    size_t capacity_ = 0;
    const double** d_w1_ptrs_ = nullptr;
    const double** d_w2_ptrs_ = nullptr;
    int* d_offsets_xyz_ = nullptr;
    int* d_valid_ = nullptr;
};

template <typename T>
void allocDevice(T*& ptr, size_t n, const char* what)
{
    if (n == 0) {
        ptr = nullptr;
        return;
    }
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n), what);
}

void allocBatchBuffers(GpuDensityBatch& batch)
{
    allocDevice(batch.rho_device, batch.ld * batch.B, "cudaMalloc(rho_device)");

    allocDevice(batch.Q_device, batch.B, "cudaMalloc(Q_device)");
    allocDevice(batch.A0_device, batch.B, "cudaMalloc(A0_device)");
    allocDevice(batch.Ax_device, batch.B, "cudaMalloc(Ax_device)");
    allocDevice(batch.Ay_device, batch.B, "cudaMalloc(Ay_device)");
    allocDevice(batch.Az_device, batch.B, "cudaMalloc(Az_device)");
    allocDevice(batch.r0x_device, batch.B, "cudaMalloc(r0x_device)");
    allocDevice(batch.r0y_device, batch.B, "cudaMalloc(r0y_device)");
    allocDevice(batch.r0z_device, batch.B, "cudaMalloc(r0z_device)");
    allocDevice(batch.Mx_device, batch.B, "cudaMalloc(Mx_device)");
    allocDevice(batch.My_device, batch.B, "cudaMalloc(My_device)");
    allocDevice(batch.Mz_device, batch.B, "cudaMalloc(Mz_device)");
    allocDevice(batch.M2_device, batch.B, "cudaMalloc(M2_device)");
    allocDevice(batch.sigma_device, batch.B, "cudaMalloc(sigma_device)");
    allocDevice(batch.alpha_device, batch.B, "cudaMalloc(alpha_device)");
}

void zeroBatchMomentBuffers(GpuDensityBatch& batch, cudaStream_t stream)
{
    checkCuda(cudaMemsetAsync(batch.Q_device, 0, sizeof(double) * batch.B, stream), "memset Q");
    checkCuda(cudaMemsetAsync(batch.A0_device, 0, sizeof(double) * batch.B, stream), "memset A0");
    checkCuda(cudaMemsetAsync(batch.Ax_device, 0, sizeof(double) * batch.B, stream), "memset Ax");
    checkCuda(cudaMemsetAsync(batch.Ay_device, 0, sizeof(double) * batch.B, stream), "memset Ay");
    checkCuda(cudaMemsetAsync(batch.Az_device, 0, sizeof(double) * batch.B, stream), "memset Az");
    checkCuda(cudaMemsetAsync(batch.r0x_device, 0, sizeof(double) * batch.B, stream), "memset r0x");
    checkCuda(cudaMemsetAsync(batch.r0y_device, 0, sizeof(double) * batch.B, stream), "memset r0y");
    checkCuda(cudaMemsetAsync(batch.r0z_device, 0, sizeof(double) * batch.B, stream), "memset r0z");
    checkCuda(cudaMemsetAsync(batch.Mx_device, 0, sizeof(double) * batch.B, stream), "memset Mx");
    checkCuda(cudaMemsetAsync(batch.My_device, 0, sizeof(double) * batch.B, stream), "memset My");
    checkCuda(cudaMemsetAsync(batch.Mz_device, 0, sizeof(double) * batch.B, stream), "memset Mz");
    checkCuda(cudaMemsetAsync(batch.M2_device, 0, sizeof(double) * batch.B, stream), "memset M2");
    checkCuda(cudaMemsetAsync(batch.sigma_device, 0, sizeof(double) * batch.B, stream), "memset sigma");
    checkCuda(cudaMemsetAsync(batch.alpha_device, 0, sizeof(double) * batch.B, stream), "memset alpha");
}

__global__ void materialize_density_kernel(
    const double* const* d_w1_ptrs,
    const double* const* d_w2_ptrs,
    const int* d_offsets_xyz,
    const int* d_valid,
    int B,
    int dimx,
    int dimy,
    int dimz,
    int ld,
    double* rho)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int K = dimx * dimy * dimz;
    const size_t total = static_cast<size_t>(K) * static_cast<size_t>(B);
    if (idx >= total) return;

    const int i_point = static_cast<int>(idx % static_cast<size_t>(K));
    const int col = static_cast<int>(idx / static_cast<size_t>(K));

    double value = 0.0;
    if (d_valid[col] != 0) {
        const double* w1 = d_w1_ptrs[col];
        const double* w2 = d_w2_ptrs[col];
        if (w1 != nullptr && w2 != nullptr) {
            const int offx = d_offsets_xyz[3 * col + 0];
            const int offy = d_offsets_xyz[3 * col + 1];
            const int offz = d_offsets_xyz[3 * col + 2];

            const int ix = i_point % dimx;
            const int yz = i_point / dimx;
            const int iy = yz % dimy;
            const int iz = yz / dimy;

            const int ix2 = ix + offx;
            const int iy2 = iy + offy;
            const int iz2 = iz + offz;

            if (ix2 >= 0 && ix2 < dimx && iy2 >= 0 && iy2 < dimy && iz2 >= 0 && iz2 < dimz) {
                const int i2 = ix2 + dimx * (iy2 + dimy * iz2);
                value = w1[i_point] * w2[i2];
            }
        }
    }

    rho[static_cast<size_t>(col) * static_cast<size_t>(ld) + static_cast<size_t>(i_point)] = value;
}

__global__ void density_metadata_kernel(
    const double* rho,
    int B,
    int K,
    int ld,
    const int* d_valid,
    const double* wq,
    const double* wx,
    const double* wy,
    const double* wz,
    const double* w2,
    double* out_Q,
    double* out_A0,
    double* out_Ax,
    double* out_Ay,
    double* out_Az,
    double* out_Mx,
    double* out_My,
    double* out_Mz,
    double* out_M2)
{
    const int col = blockIdx.y;
    if (col >= B || d_valid[col] == 0) return;

    const int i_point = blockIdx.x * blockDim.x + threadIdx.x;
    double q = 0.0;
    double a0 = 0.0;
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;
    double mx = 0.0;
    double my = 0.0;
    double mz = 0.0;
    double m2 = 0.0;
    if (i_point < K) {
        const double val = rho[static_cast<size_t>(col) * static_cast<size_t>(ld) + static_cast<size_t>(i_point)];
        const double abs_val = fabs(val);
        q = val * wq[i_point];
        a0 = abs_val * wq[i_point];
        ax = abs_val * wx[i_point];
        ay = abs_val * wy[i_point];
        az = abs_val * wz[i_point];
        mx = val * wx[i_point];
        my = val * wy[i_point];
        mz = val * wz[i_point];
        m2 = val * w2[i_point];
    }

    extern __shared__ double sdata[];
    double* sQ = sdata;
    double* sA0 = sQ + blockDim.x;
    double* sAx = sA0 + blockDim.x;
    double* sAy = sAx + blockDim.x;
    double* sAz = sAy + blockDim.x;
    double* sMx = sAz + blockDim.x;
    double* sMy = sMx + blockDim.x;
    double* sMz = sMy + blockDim.x;
    double* sM2 = sMz + blockDim.x;

    sQ[threadIdx.x] = q;
    sA0[threadIdx.x] = a0;
    sAx[threadIdx.x] = ax;
    sAy[threadIdx.x] = ay;
    sAz[threadIdx.x] = az;
    sMx[threadIdx.x] = mx;
    sMy[threadIdx.x] = my;
    sMz[threadIdx.x] = mz;
    sM2[threadIdx.x] = m2;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sQ[threadIdx.x] += sQ[threadIdx.x + stride];
            sA0[threadIdx.x] += sA0[threadIdx.x + stride];
            sAx[threadIdx.x] += sAx[threadIdx.x + stride];
            sAy[threadIdx.x] += sAy[threadIdx.x + stride];
            sAz[threadIdx.x] += sAz[threadIdx.x + stride];
            sMx[threadIdx.x] += sMx[threadIdx.x + stride];
            sMy[threadIdx.x] += sMy[threadIdx.x + stride];
            sMz[threadIdx.x] += sMz[threadIdx.x + stride];
            sM2[threadIdx.x] += sM2[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAddDouble(&out_Q[col], sQ[0]);
        atomicAddDouble(&out_A0[col], sA0[0]);
        atomicAddDouble(&out_Ax[col], sAx[0]);
        atomicAddDouble(&out_Ay[col], sAy[0]);
        atomicAddDouble(&out_Az[col], sAz[0]);
        atomicAddDouble(&out_Mx[col], sMx[0]);
        atomicAddDouble(&out_My[col], sMy[0]);
        atomicAddDouble(&out_Mz[col], sMz[0]);
        atomicAddDouble(&out_M2[col], sM2[0]);
    }
}

__global__ void finalize_r0_kernel(
    int B,
    const double* A0,
    const double* Ax,
    const double* Ay,
    const double* Az,
    double a0_min,
    double* r0x,
    double* r0y,
    double* r0z)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    if (A0[i] > a0_min) {
        r0x[i] = Ax[i] / A0[i];
        r0y[i] = Ay[i] / A0[i];
        r0z[i] = Az[i] / A0[i];
    } else {
        r0x[i] = 0.0;
        r0y[i] = 0.0;
        r0z[i] = 0.0;
    }
}

__global__ void gather_selected_columns_kernel(
    const double* rho_full,
    int K,
    int ld_full,
    const int* selected_cols,
    int S,
    int ld_sel,
    double* rho_sel)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(K) * static_cast<size_t>(S);
    if (idx >= total) return;

    const int i_point = static_cast<int>(idx % static_cast<size_t>(K));
    const int s = static_cast<int>(idx / static_cast<size_t>(K));
    const int col = selected_cols[s];

    rho_sel[static_cast<size_t>(s) * static_cast<size_t>(ld_sel) + static_cast<size_t>(i_point)] =
        rho_full[static_cast<size_t>(col) * static_cast<size_t>(ld_full) + static_cast<size_t>(i_point)];
}

__global__ void scatter_selected_columns_kernel(
    const double* rho_sel,
    int K,
    int ld_sel,
    const int* selected_cols,
    int S,
    int ld_full,
    double* rho_full)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(K) * static_cast<size_t>(S);
    if (idx >= total) return;

    const int i_point = static_cast<int>(idx % static_cast<size_t>(K));
    const int s = static_cast<int>(idx / static_cast<size_t>(K));
    const int col = selected_cols[s];

    rho_full[static_cast<size_t>(col) * static_cast<size_t>(ld_full) + static_cast<size_t>(i_point)] =
        rho_sel[static_cast<size_t>(s) * static_cast<size_t>(ld_sel) + static_cast<size_t>(i_point)];
}

__global__ void sigma_alpha_selected_kernel(
    const int* selected_cols,
    int S,
    const double* Q,
    const double* r0x,
    const double* r0y,
    const double* r0z,
    const double* Mx,
    const double* My,
    const double* Mz,
    const double* M2,
    double sigma_min,
    double sigma_max,
    double* sigma_full,
    double* alpha_full,
    double* sigma_sel,
    double* alpha_sel)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;

    const int col = selected_cols[s];
    const double q = Q[col];
    const double x0 = r0x[col];
    const double y0 = r0y[col];
    const double z0 = r0z[col];
    const double m1 = x0 * Mx[col] + y0 * My[col] + z0 * Mz[col];
    const double r0_sq = x0 * x0 + y0 * y0 + z0 * z0;
    const double I = M2[col] - 2.0 * m1 + r0_sq * q;

    double sigma = sqrt(fabs(I) / 3.0);
    if (sigma < sigma_min) sigma = sigma_min;
    if (sigma > sigma_max) sigma = sigma_max;
    const double alpha = 1.0 / (2.0 * sigma * sigma);

    sigma_full[col] = sigma;
    alpha_full[col] = alpha;
    sigma_sel[s] = sigma;
    alpha_sel[s] = alpha;
}

__global__ void build_auxiliary_kernel(
    double* aux_sel,
    int K,
    int ld_sel,
    const int* selected_cols,
    int S,
    const double* Q_full,
    const double* r0x_full,
    const double* r0y_full,
    const double* r0z_full,
    const double* alpha_full,
    const double* x,
    const double* y,
    const double* z,
    bool wrap_aux,
    const double* shifts27)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(K) * static_cast<size_t>(S);
    if (idx >= total) return;

    const int i_point = static_cast<int>(idx % static_cast<size_t>(K));
    const int s = static_cast<int>(idx / static_cast<size_t>(K));
    const int col = selected_cols[s];

    const double q = Q_full[col];
    const double alpha = alpha_full[col];
    const size_t out_idx = static_cast<size_t>(s) * static_cast<size_t>(ld_sel) + static_cast<size_t>(i_point);
    if (!(alpha > 0.0)) {
        aux_sel[out_idx] = 0.0;
        return;
    }

    const double x0 = r0x_full[col];
    const double y0 = r0y_full[col];
    const double z0 = r0z_full[col];

    const double px = x[i_point];
    const double py = y[i_point];
    const double pz = z[i_point];

    double r2 = 0.0;
    if (wrap_aux) {
        double min_r2 = 1.0e300;
        for (int l = 0; l < 27; ++l) {
            const double dx = px - x0 - shifts27[3 * l + 0];
            const double dy = py - y0 - shifts27[3 * l + 1];
            const double dz = pz - z0 - shifts27[3 * l + 2];
            const double cand = dx * dx + dy * dy + dz * dz;
            min_r2 = (cand < min_r2) ? cand : min_r2;
        }
        r2 = min_r2;
    } else {
        const double dx = px - x0;
        const double dy = py - y0;
        const double dz = pz - z0;
        r2 = dx * dx + dy * dy + dz * dz;
    }

    const double norm = q * pow(alpha / PI_CONST, 1.5);
    aux_sel[out_idx] = norm * exp(-alpha * r2);
}

__global__ void subtract_auxiliary_kernel(
    double* rho_sel,
    const double* aux_sel,
    int K,
    int S,
    int ld_sel)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(K) * static_cast<size_t>(S);
    if (idx >= total) return;

    const int i_point = static_cast<int>(idx % static_cast<size_t>(K));
    const int s = static_cast<int>(idx / static_cast<size_t>(K));
    const size_t at = static_cast<size_t>(s) * static_cast<size_t>(ld_sel) + static_cast<size_t>(i_point);
    rho_sel[at] -= aux_sel[at];
}

template <typename T>
void copyDeviceVectorToHost(std::vector<T>& out, const T* device_ptr, size_t n)
{
    out.resize(n);
    if (n == 0) return;
    checkCuda(cudaMemcpy(out.data(), device_ptr, sizeof(T) * n, cudaMemcpyDeviceToHost), "copy device->host vector");
}

}  // namespace

void release_gpu_density_batch(GpuDensityBatch& batch)
{
    safeCudaFree(batch.rho_device);
    safeCudaFree(batch.rho_raw_device);
    safeCudaFree(batch.rho_selected_device);
    safeCudaFree(batch.selected_columns_device);
    safeCudaFree(batch.selected_inverse_device);
    safeCudaFree(batch.Q_device);
    safeCudaFree(batch.A0_device);
    safeCudaFree(batch.Ax_device);
    safeCudaFree(batch.Ay_device);
    safeCudaFree(batch.Az_device);
    safeCudaFree(batch.r0x_device);
    safeCudaFree(batch.r0y_device);
    safeCudaFree(batch.r0z_device);
    safeCudaFree(batch.Mx_device);
    safeCudaFree(batch.My_device);
    safeCudaFree(batch.Mz_device);
    safeCudaFree(batch.M2_device);
    safeCudaFree(batch.sigma_device);
    safeCudaFree(batch.alpha_device);
    safeCudaFree(batch.sigma_selected_device);
    safeCudaFree(batch.alpha_selected_device);

    batch.rho_device = nullptr;
    batch.rho_raw_device = nullptr;
    batch.rho_selected_device = nullptr;
    batch.selected_columns_device = nullptr;
    batch.selected_inverse_device = nullptr;
    batch.Q_device = nullptr;
    batch.A0_device = nullptr;
    batch.Ax_device = nullptr;
    batch.Ay_device = nullptr;
    batch.Az_device = nullptr;
    batch.r0x_device = nullptr;
    batch.r0y_device = nullptr;
    batch.r0z_device = nullptr;
    batch.Mx_device = nullptr;
    batch.My_device = nullptr;
    batch.Mz_device = nullptr;
    batch.M2_device = nullptr;
    batch.sigma_device = nullptr;
    batch.alpha_device = nullptr;
    batch.sigma_selected_device = nullptr;
    batch.alpha_selected_device = nullptr;

    batch.K = 0;
    batch.B = 0;
    batch.ld = 0;
    batch.ld_raw = 0;
    batch.B_selected = 0;
    batch.ld_selected = 0;
    batch.specs.clear();
    batch.selected_columns.clear();
    batch.selected_inverse.clear();
    batch.Q_host.clear();
    batch.A0_host.clear();
    batch.Ax_host.clear();
    batch.Ay_host.clear();
    batch.Az_host.clear();
    batch.r0x_host.clear();
    batch.r0y_host.clear();
    batch.r0z_host.clear();
    batch.Mx_host.clear();
    batch.My_host.clear();
    batch.Mz_host.clear();
    batch.M2_host.clear();
    batch.sigma_host.clear();
    batch.alpha_host.clear();
    batch.timings_ms = PreFftStageTimingsMs{};
}

bool pre_fft_density_gpu_available()
{
    int device_count = 0;
    const auto status = cudaGetDeviceCount(&device_count);
    return status == cudaSuccess && device_count > 0;
}

void release_pre_fft_density_gpu_workspace()
{
    PreFftGpuContext::releaseGlobal();
}

GpuDensityBatch build_pre_fft_density_batch_gpu(
    PreFftDensityKind kind,
    std::map<int, WannierFunction> const& cWannMap,
    std::map<int, WannierFunction> const& vWannMap,
    std::vector<PreFftDensitySpec> const& specs,
    PreFftAuxConfig const& config)
{
    GpuDensityBatch batch{};
    if (specs.empty()) {
        return batch;
    }

    auto& ctx = PreFftGpuContext::instance();
    ctx.initMesh(cWannMap, vWannMap);
    ctx.ensureCapacity(specs.size());
    const cudaStream_t stream = ctx.stream();
    const bool timing_enabled = config.enable_timing;
    ScopedCudaEvents stage_events(timing_enabled);

    auto time_cpu_stage = [&](double& out_ms, auto&& fn) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        out_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    auto time_gpu_stage = [&](double& out_ms, auto&& fn) {
        if (timing_enabled) {
            checkCuda(cudaEventRecord(stage_events.start, stream), "cudaEventRecord(stage start)");
            fn();
            checkCuda(cudaEventRecord(stage_events.stop, stream), "cudaEventRecord(stage stop)");
            checkCuda(cudaEventSynchronize(stage_events.stop), "cudaEventSynchronize(stage stop)");
            float ms = 0.0f;
            checkCuda(cudaEventElapsedTime(&ms, stage_events.start, stage_events.stop), "cudaEventElapsedTime(stage)");
            out_ms += static_cast<double>(ms);
            return;
        }
        fn();
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(stage)");
    };

    auto finalize_total = [&]() {
        batch.timings_ms.total =
            batch.timings_ms.density_materialization +
            batch.timings_ms.metadata_reduction +
            batch.timings_ms.metadata_copy_gpu_to_cpu +
            batch.timings_ms.cpu_aux_selection_policy +
            batch.timings_ms.subset_compaction_gather +
            batch.timings_ms.sigma_alpha +
            batch.timings_ms.auxiliary_build +
            batch.timings_ms.auxiliary_subtraction +
            batch.timings_ms.scatter_back;
    };

    batch.K = ctx.K();
    batch.B = specs.size();
    batch.ld = batch.K;
    batch.ld_raw = 0;
    batch.specs = specs;
    allocBatchBuffers(batch);

    std::vector<const double*> h_w1_ptrs(batch.B, nullptr);
    std::vector<const double*> h_w2_ptrs(batch.B, nullptr);
    std::vector<int> h_offsets_xyz(batch.B * 3, 0);
    std::vector<int> h_valid(batch.B, 1);

    const int threads = 256;
    time_cpu_stage(batch.timings_ms.density_materialization, [&]() {
        for (size_t i = 0; i < batch.B; ++i) {
            auto c1_it = cWannMap.find(specs[i].idx1);
            if (c1_it == cWannMap.end()) {
                throw std::runtime_error("pre-FFT density spec references unknown idx1.");
            }
            const WannierFunction& w1 = c1_it->second;

            const WannierFunction* w2 = nullptr;
            if (kind == PreFftDensityKind::SameBand) {
                auto c2_it = cWannMap.find(specs[i].idx2);
                if (c2_it == cWannMap.end()) {
                    throw std::runtime_error("SameBand pre-FFT spec references unknown idx2 in cWannMap.");
                }
                w2 = &c2_it->second;
            } else {
                auto v2_it = vWannMap.find(specs[i].idx2);
                if (v2_it == vWannMap.end()) {
                    throw std::runtime_error("TransitionCv pre-FFT spec references unknown idx2 in vWannMap.");
                }
                w2 = &v2_it->second;
            }

            if (!w1.isCompatible(*w2)) {
                throw std::runtime_error("Incompatible Wannier functions in pre-FFT density batch.");
            }

            std::vector<int> Rvec{specs[i].R[0], specs[i].R[1], specs[i].R[2]};
            const std::vector<double> supercell = w1.getLatticeInUnitcellBasis();
            if ((std::round(supercell[0]) <= std::abs(Rvec[0])) ||
                (std::round(supercell[1]) <= std::abs(Rvec[1])) ||
                (std::round(supercell[2]) <= std::abs(Rvec[2]))) {
                h_valid[i] = 0;
                continue;
            }

            h_w1_ptrs[i] = ctx.ensureUploadedConduction(w1);
            h_w2_ptrs[i] = (kind == PreFftDensityKind::SameBand)
                ? ctx.ensureUploadedConduction(*w2)
                : ctx.ensureUploadedValence(*w2);

            const std::vector<int> offs = w2->getMeshgrid()->getIndexOffset(Rvec, w2->getLatticeInUnitcellBasis());
            h_offsets_xyz[3 * i + 0] = offs[0];
            h_offsets_xyz[3 * i + 1] = offs[1];
            h_offsets_xyz[3 * i + 2] = offs[2];
        }
    });

    const size_t total_values = batch.K * batch.B;
    const int blocks = static_cast<int>((total_values + threads - 1) / threads);
    time_gpu_stage(batch.timings_ms.density_materialization, [&]() {
        checkCuda(cudaMemcpyAsync(ctx.d_w1_ptrs(), h_w1_ptrs.data(), sizeof(double*) * batch.B, cudaMemcpyHostToDevice, stream), "copy w1 ptrs");
        checkCuda(cudaMemcpyAsync(ctx.d_w2_ptrs(), h_w2_ptrs.data(), sizeof(double*) * batch.B, cudaMemcpyHostToDevice, stream), "copy w2 ptrs");
        checkCuda(cudaMemcpyAsync(ctx.d_offsets_xyz(), h_offsets_xyz.data(), sizeof(int) * batch.B * 3, cudaMemcpyHostToDevice, stream), "copy offsets");
        checkCuda(cudaMemcpyAsync(ctx.d_valid(), h_valid.data(), sizeof(int) * batch.B, cudaMemcpyHostToDevice, stream), "copy valid");
        materialize_density_kernel<<<blocks, threads, 0, stream>>>(
            ctx.d_w1_ptrs(),
            ctx.d_w2_ptrs(),
            ctx.d_offsets_xyz(),
            ctx.d_valid(),
            static_cast<int>(batch.B),
            ctx.dimx(),
            ctx.dimy(),
            ctx.dimz(),
            static_cast<int>(batch.ld),
            batch.rho_device);
        checkCuda(cudaGetLastError(), "materialize_density_kernel");

        if (config.keep_raw_density_copy) {
            batch.ld_raw = batch.K;
            allocDevice(batch.rho_raw_device, batch.ld_raw * batch.B, "cudaMalloc(rho_raw_device)");
            checkCuda(
                cudaMemcpyAsync(
                    batch.rho_raw_device,
                    batch.rho_device,
                    sizeof(double) * batch.ld * batch.B,
                    cudaMemcpyDeviceToDevice,
                    stream),
                "copy rho -> rho_raw");
        }
    });

    const int blocks_x = static_cast<int>((batch.K + threads - 1) / threads);
    const dim3 grid_mom(blocks_x, static_cast<unsigned int>(batch.B), 1);
    const int blocks_cols = static_cast<int>((batch.B + threads - 1) / threads);

    if (!config.apply_auxiliary_subtraction) {
        batch.B_selected = 0;
        batch.ld_selected = 0;
        if (!config.copy_metadata_to_host) {
            // Keep host vectors empty in pure device-only build mode.
            batch.Q_host.clear();
            batch.A0_host.clear();
            batch.Ax_host.clear();
            batch.Ay_host.clear();
            batch.Az_host.clear();
            batch.r0x_host.clear();
            batch.r0y_host.clear();
            batch.r0z_host.clear();
            batch.Mx_host.clear();
            batch.My_host.clear();
            batch.Mz_host.clear();
            batch.M2_host.clear();
            batch.sigma_host.clear();
            batch.alpha_host.clear();
            finalize_total();
            return batch;
        }
    }

    time_gpu_stage(batch.timings_ms.metadata_reduction, [&]() {
        zeroBatchMomentBuffers(batch, stream);
        density_metadata_kernel<<<grid_mom, threads, sizeof(double) * threads * 9, stream>>>(
            batch.rho_device,
            static_cast<int>(batch.B),
            static_cast<int>(batch.K),
            static_cast<int>(batch.ld),
            ctx.d_valid(),
            ctx.d_wq(),
            ctx.d_wx(),
            ctx.d_wy(),
            ctx.d_wz(),
            ctx.d_w2(),
            batch.Q_device,
            batch.A0_device,
            batch.Ax_device,
            batch.Ay_device,
            batch.Az_device,
            batch.Mx_device,
            batch.My_device,
            batch.Mz_device,
            batch.M2_device);
        checkCuda(cudaGetLastError(), "density_metadata_kernel");

        finalize_r0_kernel<<<blocks_cols, threads, 0, stream>>>(
            static_cast<int>(batch.B),
            batch.A0_device,
            batch.Ax_device,
            batch.Ay_device,
            batch.Az_device,
            config.a0_min,
            batch.r0x_device,
            batch.r0y_device,
            batch.r0z_device);
        checkCuda(cudaGetLastError(), "finalize_r0_kernel");
    });

    if (config.copy_metadata_to_host) {
        time_cpu_stage(batch.timings_ms.metadata_copy_gpu_to_cpu, [&]() {
            copyDeviceVectorToHost(batch.Q_host, batch.Q_device, batch.B);
            copyDeviceVectorToHost(batch.A0_host, batch.A0_device, batch.B);
            copyDeviceVectorToHost(batch.Ax_host, batch.Ax_device, batch.B);
            copyDeviceVectorToHost(batch.Ay_host, batch.Ay_device, batch.B);
            copyDeviceVectorToHost(batch.Az_host, batch.Az_device, batch.B);
            copyDeviceVectorToHost(batch.r0x_host, batch.r0x_device, batch.B);
            copyDeviceVectorToHost(batch.r0y_host, batch.r0y_device, batch.B);
            copyDeviceVectorToHost(batch.r0z_host, batch.r0z_device, batch.B);
            copyDeviceVectorToHost(batch.Mx_host, batch.Mx_device, batch.B);
            copyDeviceVectorToHost(batch.My_host, batch.My_device, batch.B);
            copyDeviceVectorToHost(batch.Mz_host, batch.Mz_device, batch.B);
            copyDeviceVectorToHost(batch.M2_host, batch.M2_device, batch.B);
        });
    }

    if (!config.apply_auxiliary_subtraction) {
        batch.B_selected = 0;
        batch.ld_selected = 0;
        batch.selected_inverse.clear();
        batch.selected_columns.clear();
        batch.sigma_host.assign(batch.B, 0.0);
        batch.alpha_host.assign(batch.B, 0.0);
        finalize_total();
        return batch;
    }

    if (!config.copy_metadata_to_host) {
        throw std::runtime_error(
            "GPU pre-FFT pipeline requires copy_metadata_to_host=true when auxiliary subtraction is enabled.");
    }

    time_cpu_stage(batch.timings_ms.cpu_aux_selection_policy, [&]() {
        batch.selected_inverse.assign(batch.B, -1);
        for (size_t i = 0; i < batch.B; ++i) {
            if (std::abs(batch.Q_host[i]) > config.q_abs_threshold && batch.A0_host[i] > config.a0_min) {
                batch.selected_inverse[i] = static_cast<int>(batch.selected_columns.size());
                batch.selected_columns.push_back(static_cast<int>(i));
            }
        }
        batch.B_selected = batch.selected_columns.size();
    });

    if (batch.B_selected == 0) {
        batch.sigma_host.assign(batch.B, 0.0);
        batch.alpha_host.assign(batch.B, 0.0);
        finalize_total();
        return batch;
    }

    batch.ld_selected = batch.K;
    const size_t total_selected = batch.K * batch.B_selected;
    const int blocks_sel = static_cast<int>((total_selected + threads - 1) / threads);
    time_gpu_stage(batch.timings_ms.subset_compaction_gather, [&]() {
        allocDevice(batch.selected_columns_device, batch.B_selected, "cudaMalloc(selected_columns_device)");
        allocDevice(batch.selected_inverse_device, batch.B, "cudaMalloc(selected_inverse_device)");
        allocDevice(batch.rho_selected_device, batch.ld_selected * batch.B_selected, "cudaMalloc(rho_selected_device)");
        allocDevice(batch.sigma_selected_device, batch.B_selected, "cudaMalloc(sigma_selected_device)");
        allocDevice(batch.alpha_selected_device, batch.B_selected, "cudaMalloc(alpha_selected_device)");

        checkCuda(cudaMemcpyAsync(batch.selected_columns_device, batch.selected_columns.data(), sizeof(int) * batch.B_selected, cudaMemcpyHostToDevice, stream), "copy selected columns");
        checkCuda(cudaMemcpyAsync(batch.selected_inverse_device, batch.selected_inverse.data(), sizeof(int) * batch.B, cudaMemcpyHostToDevice, stream), "copy selected inverse");

        gather_selected_columns_kernel<<<blocks_sel, threads, 0, stream>>>(
            batch.rho_device,
            static_cast<int>(batch.K),
            static_cast<int>(batch.ld),
            batch.selected_columns_device,
            static_cast<int>(batch.B_selected),
            static_cast<int>(batch.ld_selected),
            batch.rho_selected_device);
        checkCuda(cudaGetLastError(), "gather_selected_columns_kernel");
    });

    // Match CPU sigma bounds semantics from CoulombSolver.
    const auto& lattice = ctx.lattice();
    std::vector<double> uvec{
        lattice[0][0], lattice[1][0], lattice[2][0]
    };
    double length = sqrt(uvec[0] * uvec[0] + uvec[1] * uvec[1] + uvec[2] * uvec[2]);
    double min_length = length;
    double discret_length = length / static_cast<double>(ctx.dimx());

    uvec = std::vector<double>{lattice[0][1], lattice[1][1], lattice[2][1]};
    length = sqrt(uvec[0] * uvec[0] + uvec[1] * uvec[1] + uvec[2] * uvec[2]);
    min_length = std::min(min_length, length);
    discret_length = std::min(discret_length, length / static_cast<double>(ctx.dimy()));

    uvec = std::vector<double>{lattice[0][2], lattice[1][2], lattice[2][2]};
    length = sqrt(uvec[0] * uvec[0] + uvec[1] * uvec[1] + uvec[2] * uvec[2]);
    min_length = std::min(min_length, length);
    discret_length = std::min(discret_length, length / static_cast<double>(ctx.dimz()));

    const double sigma_min = config.points_per_std * discret_length;
    const double sigma_max = min_length / config.std_per_cell;
    if (sigma_max <= sigma_min) {
        throw std::runtime_error("Invalid sigma bounds for auxiliary Gaussian construction.");
    }

    const int blocks_sel_cols = static_cast<int>((batch.B_selected + threads - 1) / threads);
    time_gpu_stage(batch.timings_ms.sigma_alpha, [&]() {
        sigma_alpha_selected_kernel<<<blocks_sel_cols, threads, 0, stream>>>(
            batch.selected_columns_device,
            static_cast<int>(batch.B_selected),
            batch.Q_device,
            batch.r0x_device,
            batch.r0y_device,
            batch.r0z_device,
            batch.Mx_device,
            batch.My_device,
            batch.Mz_device,
            batch.M2_device,
            sigma_min,
            sigma_max,
            batch.sigma_device,
            batch.alpha_device,
            batch.sigma_selected_device,
            batch.alpha_selected_device);
        checkCuda(cudaGetLastError(), "sigma_alpha_selected_kernel");
    });

    ScopedDeviceBuffer<double> aux_selected;
    time_gpu_stage(batch.timings_ms.auxiliary_build, [&]() {
        allocDevice(aux_selected.ptr, batch.ld_selected * batch.B_selected, "cudaMalloc(aux_selected)");
        build_auxiliary_kernel<<<blocks_sel, threads, 0, stream>>>(
            aux_selected.ptr,
            static_cast<int>(batch.K),
            static_cast<int>(batch.ld_selected),
            batch.selected_columns_device,
            static_cast<int>(batch.B_selected),
            batch.Q_device,
            batch.r0x_device,
            batch.r0y_device,
            batch.r0z_device,
            batch.alpha_device,
            ctx.d_x(),
            ctx.d_y(),
            ctx.d_z(),
            config.wrap_aux,
            ctx.d_shifts27());
        checkCuda(cudaGetLastError(), "build_auxiliary_kernel");
    });

    time_gpu_stage(batch.timings_ms.auxiliary_subtraction, [&]() {
        subtract_auxiliary_kernel<<<blocks_sel, threads, 0, stream>>>(
            batch.rho_selected_device,
            aux_selected.ptr,
            static_cast<int>(batch.K),
            static_cast<int>(batch.B_selected),
            static_cast<int>(batch.ld_selected));
        checkCuda(cudaGetLastError(), "subtract_auxiliary_kernel");
    });

    time_gpu_stage(batch.timings_ms.scatter_back, [&]() {
        scatter_selected_columns_kernel<<<blocks_sel, threads, 0, stream>>>(
            batch.rho_selected_device,
            static_cast<int>(batch.K),
            static_cast<int>(batch.ld_selected),
            batch.selected_columns_device,
            static_cast<int>(batch.B_selected),
            static_cast<int>(batch.ld),
            batch.rho_device);
        checkCuda(cudaGetLastError(), "scatter_selected_columns_kernel");
    });

    if (config.copy_metadata_to_host) {
        time_cpu_stage(batch.timings_ms.metadata_copy_gpu_to_cpu, [&]() {
            copyDeviceVectorToHost(batch.sigma_host, batch.sigma_device, batch.B);
            copyDeviceVectorToHost(batch.alpha_host, batch.alpha_device, batch.B);
        });
    }

    finalize_total();

    return batch;
}
