#include "backend/transition_dipole.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

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

void checkCublas(cublasStatus_t status, const char* what)
{
    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    }
    std::ostringstream msg;
    msg << what << " failed with code " << static_cast<int>(status);
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

template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() = default;
    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    ~DeviceBuffer()
    {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    void allocate(size_t n, const char* what)
    {
        if (ptr_ != nullptr) {
            checkCuda(cudaFree(ptr_), "cudaFree(reallocate)");
            ptr_ = nullptr;
            size_ = 0;
        }
        if (n == 0) {
            return;
        }
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T)), what);
        size_ = n;
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

template <typename T>
class PinnedBuffer
{
public:
    PinnedBuffer() = default;
    PinnedBuffer(PinnedBuffer const&) = delete;
    PinnedBuffer& operator=(PinnedBuffer const&) = delete;

    ~PinnedBuffer()
    {
        if (ptr_ != nullptr) {
            cudaFreeHost(ptr_);
        }
    }

    void allocate(size_t n, const char* what)
    {
        if (ptr_ != nullptr) {
            checkCuda(cudaFreeHost(ptr_), "cudaFreeHost(reallocate)");
            ptr_ = nullptr;
            size_ = 0;
        }
        if (n == 0) {
            return;
        }
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&ptr_), n * sizeof(T)), what);
        size_ = n;
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

struct ColumnKey
{
    int v = -1;
    int sx = 0;
    int sy = 0;
    int sz = 0;

    bool operator==(const ColumnKey& other) const
    {
        return v == other.v && sx == other.sx && sy == other.sy && sz == other.sz;
    }
};

struct ColumnKeyHash
{
    size_t operator()(const ColumnKey& key) const
    {
        // FNV-1a style mix over small integer tuple.
        size_t h = 1469598103934665603ull;
        auto mix = [&h](uint64_t x) {
            h ^= static_cast<size_t>(x);
            h *= 1099511628211ull;
        };
        mix(static_cast<uint64_t>(static_cast<uint32_t>(key.v)));
        mix(static_cast<uint64_t>(static_cast<uint32_t>(key.sx)));
        mix(static_cast<uint64_t>(static_cast<uint32_t>(key.sy)));
        mix(static_cast<uint64_t>(static_cast<uint32_t>(key.sz)));
        return h;
    }
};

struct ColumnSpec
{
    int v_index = -1;
    int offx = 0;
    int offy = 0;
    int offz = 0;
    int valid = 1;
    double shift_vec[3]{0.0, 0.0, 0.0};
    std::vector<int> task_indices{};
};

__global__ void expand_a_big_kernel(
    const double* d_x,
    const double* d_y,
    const double* d_z,
    int K,
    int M,
    double* d_a_big)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(K) * static_cast<size_t>(M);
    if (idx >= total) {
        return;
    }

    const int i = static_cast<int>(idx % static_cast<size_t>(K));
    const int c = static_cast<int>(idx / static_cast<size_t>(K));

    const size_t base = static_cast<size_t>(c) * static_cast<size_t>(K) + static_cast<size_t>(i);
    const double value = d_a_big[base];

    d_a_big[static_cast<size_t>(M + c) * static_cast<size_t>(K) + static_cast<size_t>(i)] = d_x[i] * value;
    d_a_big[static_cast<size_t>(2 * M + c) * static_cast<size_t>(K) + static_cast<size_t>(i)] = d_y[i] * value;
    d_a_big[static_cast<size_t>(3 * M + c) * static_cast<size_t>(K) + static_cast<size_t>(i)] = d_z[i] * value;
}

__global__ void pack_shifted_valence_kernel(
    const double* d_v_base,
    const int* d_col_v,
    const int* d_offx,
    const int* d_offy,
    const int* d_offz,
    const int* d_valid,
    int dimx,
    int dimy,
    int dimz,
    int K,
    int nb,
    double* d_b)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(K) * static_cast<size_t>(nb);
    if (idx >= total) {
        return;
    }

    const int i = static_cast<int>(idx % static_cast<size_t>(K));
    const int col = static_cast<int>(idx / static_cast<size_t>(K));

    double value = 0.0;
    if (d_valid[col] != 0) {
        const int offx = d_offx[col];
        const int offy = d_offy[col];
        const int offz = d_offz[col];

        const int ix = i % dimx;
        const int yz = i / dimx;
        const int iy = yz % dimy;
        const int iz = yz / dimy;

        const int ix2 = ix + offx;
        const int iy2 = iy + offy;
        const int iz2 = iz + offz;

        if (ix2 >= 0 && ix2 < dimx && iy2 >= 0 && iy2 < dimy && iz2 >= 0 && iz2 < dimz) {
            const int i2 = ix2 + dimx * (iy2 + dimy * iz2);
            const int v = d_col_v[col];
            value = d_v_base[static_cast<size_t>(v) * static_cast<size_t>(K) + static_cast<size_t>(i2)];
        }
    }

    d_b[static_cast<size_t>(col) * static_cast<size_t>(K) + static_cast<size_t>(i)] = value;
}

__global__ void apply_transition_correction_kernel(
    double* d_c,
    int M,
    int ldc,
    int nb,
    const int* d_col_v,
    const double* d_shift_vec,
    const double* d_center_c,
    const double* d_center_v)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(M) * static_cast<size_t>(nb);
    if (idx >= total) {
        return;
    }

    const int c = static_cast<int>(idx % static_cast<size_t>(M));
    const int col = static_cast<int>(idx / static_cast<size_t>(M));

    const int v = d_col_v[col];
    const size_t col_offset = static_cast<size_t>(col) * static_cast<size_t>(ldc);

    const double q = d_c[col_offset + static_cast<size_t>(c)];
    const double factor = 0.5 * q;

    const double cx = d_center_c[3 * c + 0];
    const double cy = d_center_c[3 * c + 1];
    const double cz = d_center_c[3 * c + 2];

    const double vx = d_center_v[3 * v + 0];
    const double vy = d_center_v[3 * v + 1];
    const double vz = d_center_v[3 * v + 2];

    const double sx = d_shift_vec[3 * col + 0];
    const double sy = d_shift_vec[3 * col + 1];
    const double sz = d_shift_vec[3 * col + 2];

    d_c[col_offset + static_cast<size_t>(M + c)] -= factor * (sx + vx + cx);
    d_c[col_offset + static_cast<size_t>(2 * M + c)] -= factor * (sy + vy + cy);
    d_c[col_offset + static_cast<size_t>(3 * M + c)] -= factor * (sz + vz + cz);
}

int round_to_int(double value)
{
    return static_cast<int>(std::llround(value));
}

void ensure_non_null(const double* ptr, const char* name)
{
    if (ptr == nullptr) {
        std::ostringstream msg;
        msg << name << " is null.";
        throw std::runtime_error(msg.str());
    }
}

int choose_batch_columns(
    size_t num_columns,
    int K,
    int M,
    size_t bytes_fixed)
{
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    checkCuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

    const size_t bytes_per_col_per_slot = sizeof(double) * (static_cast<size_t>(K) + static_cast<size_t>(4 * M));
    if (bytes_per_col_per_slot == 0) {
        return 1;
    }

    // Keep headroom to avoid OOM in mixed workloads.
    const size_t reserve = static_cast<size_t>(0.2 * static_cast<double>(free_bytes));
    size_t usable = 0;
    if (free_bytes > reserve + bytes_fixed) {
        usable = free_bytes - reserve - bytes_fixed;
    }

    size_t guess = usable / (2 * bytes_per_col_per_slot);  // double-buffering
    if (guess == 0) {
        guess = 1;
    }

    guess = std::min<size_t>(guess, num_columns);
    guess = std::min<size_t>(guess, 8192);

    return static_cast<int>(std::max<size_t>(guess, 1));
}

}  // namespace

bool transition_dipole_gpu_available()
{
    return true;
}

std::vector<TransitionDipoleValue> compute_transition_dipoles_gpu(
    std::vector<const double*> const& conduction_values,
    std::vector<const double*> const& valence_values,
    std::vector<int> const& dim,
    std::vector<double> const& supercell_cond,
    std::vector<double> const& supercell_val,
    double dV,
    const double* XX,
    const double* YY,
    const double* ZZ,
    std::vector<std::vector<double>> const& center_c,
    std::vector<std::vector<double>> const& center_v,
    std::vector<std::vector<double>> const& unitcell_t,
    std::vector<TransitionDipoleTask> const& tasks,
    bool apply_correction)
{
    if (tasks.empty()) {
        return {};
    }
    if (dim.size() != 3) {
        throw std::runtime_error("Transition GPU path expects dim size == 3.");
    }
    if (supercell_cond.size() != 3 || supercell_val.size() != 3) {
        throw std::runtime_error("Transition GPU path expects supercell vectors of size 3.");
    }
    if (unitcell_t.size() != 3 || unitcell_t[0].size() != 3 || unitcell_t[1].size() != 3 || unitcell_t[2].size() != 3) {
        throw std::runtime_error("Transition GPU path expects a 3x3 transposed unit cell matrix.");
    }

    const int M = static_cast<int>(conduction_values.size());
    const int Nv = static_cast<int>(valence_values.size());
    if (M <= 0 || Nv <= 0) {
        throw std::runtime_error("Transition GPU path requires at least one conduction and one valence WF.");
    }

    if (static_cast<int>(center_c.size()) != M || static_cast<int>(center_v.size()) != Nv) {
        throw std::runtime_error("Center arrays are incompatible with WF counts in transition GPU path.");
    }

    const int dimx = dim[0];
    const int dimy = dim[1];
    const int dimz = dim[2];
    const int64_t K64 = static_cast<int64_t>(dimx) * static_cast<int64_t>(dimy) * static_cast<int64_t>(dimz);
    if (K64 <= 0 || K64 > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Transition GPU path supports 0 < K <= INT_MAX.");
    }
    const int K = static_cast<int>(K64);

    ensure_non_null(XX, "XX");
    ensure_non_null(YY, "YY");
    ensure_non_null(ZZ, "ZZ");

    for (int c = 0; c < M; ++c) {
        ensure_non_null(conduction_values[c], "conduction_values[c]");
        if (center_c[c].size() != 3) {
            throw std::runtime_error("Each conduction center must have exactly 3 components.");
        }
    }
    for (int v = 0; v < Nv; ++v) {
        ensure_non_null(valence_values[v], "valence_values[v]");
        if (center_v[v].size() != 3) {
            throw std::runtime_error("Each valence center must have exactly 3 components.");
        }
    }

    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA devices available for transition dipoles.");
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int local_rank = get_local_rank();
    const int device_index = (local_rank >= 0 ? local_rank : rank) % device_count;
    checkCuda(cudaSetDevice(device_index), "cudaSetDevice");

    std::vector<TransitionDipoleValue> out(tasks.size());

    std::unordered_map<ColumnKey, int, ColumnKeyHash> column_map;
    column_map.reserve(tasks.size());

    std::vector<ColumnSpec> columns;
    columns.reserve(tasks.size());

    const int scale_x = round_to_int(static_cast<double>(dimx) / supercell_val[0]);
    const int scale_y = round_to_int(static_cast<double>(dimy) / supercell_val[1]);
    const int scale_z = round_to_int(static_cast<double>(dimz) / supercell_val[2]);

    const int supercell_cond_x = round_to_int(supercell_cond[0]);
    const int supercell_cond_y = round_to_int(supercell_cond[1]);
    const int supercell_cond_z = round_to_int(supercell_cond[2]);

    for (size_t t = 0; t < tasks.size(); ++t) {
        const TransitionDipoleTask& task = tasks[t];

        if (task.c_index < 0 || task.c_index >= M || task.v_index < 0 || task.v_index >= Nv) {
            throw std::runtime_error("Transition task has out-of-range c_index or v_index.");
        }

        const ColumnKey key{task.v_index, task.shift[0], task.shift[1], task.shift[2]};
        auto it = column_map.find(key);

        int col_idx = -1;
        if (it == column_map.end()) {
            ColumnSpec spec;
            spec.v_index = task.v_index;

            const int neg_sx = -task.shift[0];
            const int neg_sy = -task.shift[1];
            const int neg_sz = -task.shift[2];

            if (supercell_cond_x <= std::abs(neg_sx) ||
                supercell_cond_y <= std::abs(neg_sy) ||
                supercell_cond_z <= std::abs(neg_sz)) {
                spec.valid = 0;
            }

            // indexOffset = -R * round(dim/supercell), with R = -S
            spec.offx = -neg_sx * scale_x;
            spec.offy = -neg_sy * scale_y;
            spec.offz = -neg_sz * scale_z;

            spec.shift_vec[0] = unitcell_t[0][0] * neg_sx + unitcell_t[0][1] * neg_sy + unitcell_t[0][2] * neg_sz;
            spec.shift_vec[1] = unitcell_t[1][0] * neg_sx + unitcell_t[1][1] * neg_sy + unitcell_t[1][2] * neg_sz;
            spec.shift_vec[2] = unitcell_t[2][0] * neg_sx + unitcell_t[2][1] * neg_sy + unitcell_t[2][2] * neg_sz;

            col_idx = static_cast<int>(columns.size());
            columns.push_back(spec);
            column_map.insert({key, col_idx});
        } else {
            col_idx = it->second;
        }

        columns[col_idx].task_indices.push_back(static_cast<int>(t));
    }

    const int num_cols = static_cast<int>(columns.size());
    if (num_cols <= 0) {
        return out;
    }

    DeviceBuffer<double> d_a_big{};
    DeviceBuffer<double> d_v_base{};
    DeviceBuffer<double> d_x{};
    DeviceBuffer<double> d_y{};
    DeviceBuffer<double> d_z{};
    DeviceBuffer<double> d_center_c{};
    DeviceBuffer<double> d_center_v{};

    DeviceBuffer<double> d_b[2]{};
    DeviceBuffer<double> d_c[2]{};

    DeviceBuffer<int> d_col_v[2]{};
    DeviceBuffer<int> d_offx[2]{};
    DeviceBuffer<int> d_offy[2]{};
    DeviceBuffer<int> d_offz[2]{};
    DeviceBuffer<int> d_valid[2]{};
    DeviceBuffer<double> d_shift_vec[2]{};

    PinnedBuffer<double> h_c[2]{};

    cudaStream_t streams[2]{nullptr, nullptr};
    cudaEvent_t done[2]{nullptr, nullptr};
    cublasHandle_t cublas_handle = nullptr;

    auto cleanup = [&]() {
        for (auto& event : done) {
            if (event != nullptr) {
                cudaEventDestroy(event);
                event = nullptr;
            }
        }
        for (auto& stream : streams) {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
                stream = nullptr;
            }
        }
        if (cublas_handle != nullptr) {
            cublasDestroy(cublas_handle);
            cublas_handle = nullptr;
        }
    };

    try {
        checkCuda(cudaStreamCreate(&streams[0]), "cudaStreamCreate(stream0)");
        checkCuda(cudaStreamCreate(&streams[1]), "cudaStreamCreate(stream1)");
        checkCuda(cudaEventCreateWithFlags(&done[0], cudaEventDisableTiming), "cudaEventCreate(done0)");
        checkCuda(cudaEventCreateWithFlags(&done[1], cudaEventDisableTiming), "cudaEventCreate(done1)");
        checkCublas(cublasCreate(&cublas_handle), "cublasCreate");

        d_a_big.allocate(static_cast<size_t>(K) * static_cast<size_t>(4 * M), "cudaMalloc(d_a_big)");
        d_v_base.allocate(static_cast<size_t>(K) * static_cast<size_t>(Nv), "cudaMalloc(d_v_base)");
        d_x.allocate(static_cast<size_t>(K), "cudaMalloc(d_x)");
        d_y.allocate(static_cast<size_t>(K), "cudaMalloc(d_y)");
        d_z.allocate(static_cast<size_t>(K), "cudaMalloc(d_z)");

        d_center_c.allocate(static_cast<size_t>(3 * M), "cudaMalloc(d_center_c)");
        d_center_v.allocate(static_cast<size_t>(3 * Nv), "cudaMalloc(d_center_v)");

        // Upload coordinates and WF matrices.
        checkCuda(cudaMemcpyAsync(d_x.get(), XX, sizeof(double) * static_cast<size_t>(K), cudaMemcpyHostToDevice, streams[0]),
            "cudaMemcpyAsync(d_x)");
        checkCuda(cudaMemcpyAsync(d_y.get(), YY, sizeof(double) * static_cast<size_t>(K), cudaMemcpyHostToDevice, streams[0]),
            "cudaMemcpyAsync(d_y)");
        checkCuda(cudaMemcpyAsync(d_z.get(), ZZ, sizeof(double) * static_cast<size_t>(K), cudaMemcpyHostToDevice, streams[0]),
            "cudaMemcpyAsync(d_z)");

        for (int c = 0; c < M; ++c) {
            checkCuda(cudaMemcpyAsync(
                d_a_big.get() + static_cast<size_t>(c) * static_cast<size_t>(K),
                conduction_values[c],
                sizeof(double) * static_cast<size_t>(K),
                cudaMemcpyHostToDevice,
                streams[0]),
                "cudaMemcpyAsync(upload conduction)");
        }

        for (int v = 0; v < Nv; ++v) {
            checkCuda(cudaMemcpyAsync(
                d_v_base.get() + static_cast<size_t>(v) * static_cast<size_t>(K),
                valence_values[v],
                sizeof(double) * static_cast<size_t>(K),
                cudaMemcpyHostToDevice,
                streams[0]),
                "cudaMemcpyAsync(upload valence)");
        }

        std::vector<double> h_center_c(3 * static_cast<size_t>(M), 0.0);
        std::vector<double> h_center_v(3 * static_cast<size_t>(Nv), 0.0);
        for (int c = 0; c < M; ++c) {
            h_center_c[3 * c + 0] = center_c[c][0];
            h_center_c[3 * c + 1] = center_c[c][1];
            h_center_c[3 * c + 2] = center_c[c][2];
        }
        for (int v = 0; v < Nv; ++v) {
            h_center_v[3 * v + 0] = center_v[v][0];
            h_center_v[3 * v + 1] = center_v[v][1];
            h_center_v[3 * v + 2] = center_v[v][2];
        }

        checkCuda(cudaMemcpyAsync(
            d_center_c.get(), h_center_c.data(), sizeof(double) * h_center_c.size(),
            cudaMemcpyHostToDevice, streams[0]),
            "cudaMemcpyAsync(d_center_c)");

        checkCuda(cudaMemcpyAsync(
            d_center_v.get(), h_center_v.data(), sizeof(double) * h_center_v.size(),
            cudaMemcpyHostToDevice, streams[0]),
            "cudaMemcpyAsync(d_center_v)");

        const int threads = 256;
        const size_t total_a = static_cast<size_t>(K) * static_cast<size_t>(M);
        const int blocks_a = static_cast<int>((total_a + threads - 1) / threads);
        expand_a_big_kernel<<<blocks_a, threads, 0, streams[0]>>>(
            d_x.get(), d_y.get(), d_z.get(), K, M, d_a_big.get());
        checkCuda(cudaGetLastError(), "expand_a_big_kernel");

        checkCuda(cudaStreamSynchronize(streams[0]), "cudaStreamSynchronize(setup)");

        const size_t bytes_fixed = sizeof(double) * (
            static_cast<size_t>(K) * static_cast<size_t>(4 * M) +
            static_cast<size_t>(K) * static_cast<size_t>(Nv) +
            static_cast<size_t>(3 * K) +
            static_cast<size_t>(3 * (M + Nv)));

        int batch_cols = choose_batch_columns(static_cast<size_t>(num_cols), K, M, bytes_fixed);

        while (true) {
            try {
                for (int slot = 0; slot < 2; ++slot) {
                    d_b[slot].allocate(static_cast<size_t>(K) * static_cast<size_t>(batch_cols), "cudaMalloc(d_b)");
                    d_c[slot].allocate(static_cast<size_t>(4 * M) * static_cast<size_t>(batch_cols), "cudaMalloc(d_c)");
                    d_col_v[slot].allocate(static_cast<size_t>(batch_cols), "cudaMalloc(d_col_v)");
                    d_offx[slot].allocate(static_cast<size_t>(batch_cols), "cudaMalloc(d_offx)");
                    d_offy[slot].allocate(static_cast<size_t>(batch_cols), "cudaMalloc(d_offy)");
                    d_offz[slot].allocate(static_cast<size_t>(batch_cols), "cudaMalloc(d_offz)");
                    d_valid[slot].allocate(static_cast<size_t>(batch_cols), "cudaMalloc(d_valid)");
                    d_shift_vec[slot].allocate(static_cast<size_t>(3 * batch_cols), "cudaMalloc(d_shift_vec)");
                    h_c[slot].allocate(static_cast<size_t>(4 * M) * static_cast<size_t>(batch_cols), "cudaMallocHost(h_c)");
                }
                break;
            } catch (const std::runtime_error&) {
                if (batch_cols <= 1) {
                    throw;
                }
                batch_cols = std::max(1, batch_cols / 2);
            }
        }

        std::vector<int> h_col_v(static_cast<size_t>(batch_cols), 0);
        std::vector<int> h_offx(static_cast<size_t>(batch_cols), 0);
        std::vector<int> h_offy(static_cast<size_t>(batch_cols), 0);
        std::vector<int> h_offz(static_cast<size_t>(batch_cols), 0);
        std::vector<int> h_valid(static_cast<size_t>(batch_cols), 0);
        std::vector<double> h_shift_vec(static_cast<size_t>(3 * batch_cols), 0.0);

        const int ldc = 4 * M;
        const double alpha = dV;
        const double beta = 0.0;

        int slot_batch_start[2]{0, 0};
        int slot_batch_size[2]{0, 0};
        bool slot_busy[2]{false, false};

        auto consume_slot = [&](int slot) {
            if (!slot_busy[slot]) {
                return;
            }

            checkCuda(cudaEventSynchronize(done[slot]), "cudaEventSynchronize(done)");

            const int start = slot_batch_start[slot];
            const int nb = slot_batch_size[slot];
            const double* c_host = h_c[slot].get();

            for (int local_col = 0; local_col < nb; ++local_col) {
                const int global_col = start + local_col;
                const ColumnSpec& col = columns[global_col];

                for (int task_idx : col.task_indices) {
                    const int c_idx = tasks[task_idx].c_index;
                    const size_t col_offset = static_cast<size_t>(local_col) * static_cast<size_t>(ldc);
                    out[task_idx].q = c_host[col_offset + static_cast<size_t>(c_idx)];
                    out[task_idx].dx = c_host[col_offset + static_cast<size_t>(M + c_idx)];
                    out[task_idx].dy = c_host[col_offset + static_cast<size_t>(2 * M + c_idx)];
                    out[task_idx].dz = c_host[col_offset + static_cast<size_t>(3 * M + c_idx)];
                }
            }

            slot_busy[slot] = false;
        };

        for (int batch_start = 0, batch_id = 0; batch_start < num_cols; batch_start += batch_cols, ++batch_id) {
            const int slot = batch_id % 2;
            consume_slot(slot);

            const int nb = std::min(batch_cols, num_cols - batch_start);
            for (int j = 0; j < nb; ++j) {
                const ColumnSpec& col = columns[batch_start + j];
                h_col_v[j] = col.v_index;
                h_offx[j] = col.offx;
                h_offy[j] = col.offy;
                h_offz[j] = col.offz;
                h_valid[j] = col.valid;
                h_shift_vec[3 * j + 0] = col.shift_vec[0];
                h_shift_vec[3 * j + 1] = col.shift_vec[1];
                h_shift_vec[3 * j + 2] = col.shift_vec[2];
            }

            checkCuda(cudaMemcpyAsync(d_col_v[slot].get(), h_col_v.data(), sizeof(int) * static_cast<size_t>(nb), cudaMemcpyHostToDevice, streams[slot]),
                "cudaMemcpyAsync(d_col_v)");
            checkCuda(cudaMemcpyAsync(d_offx[slot].get(), h_offx.data(), sizeof(int) * static_cast<size_t>(nb), cudaMemcpyHostToDevice, streams[slot]),
                "cudaMemcpyAsync(d_offx)");
            checkCuda(cudaMemcpyAsync(d_offy[slot].get(), h_offy.data(), sizeof(int) * static_cast<size_t>(nb), cudaMemcpyHostToDevice, streams[slot]),
                "cudaMemcpyAsync(d_offy)");
            checkCuda(cudaMemcpyAsync(d_offz[slot].get(), h_offz.data(), sizeof(int) * static_cast<size_t>(nb), cudaMemcpyHostToDevice, streams[slot]),
                "cudaMemcpyAsync(d_offz)");
            checkCuda(cudaMemcpyAsync(d_valid[slot].get(), h_valid.data(), sizeof(int) * static_cast<size_t>(nb), cudaMemcpyHostToDevice, streams[slot]),
                "cudaMemcpyAsync(d_valid)");
            checkCuda(cudaMemcpyAsync(d_shift_vec[slot].get(), h_shift_vec.data(), sizeof(double) * static_cast<size_t>(3 * nb), cudaMemcpyHostToDevice, streams[slot]),
                "cudaMemcpyAsync(d_shift_vec)");

            const size_t total_pack = static_cast<size_t>(K) * static_cast<size_t>(nb);
            const int blocks_pack = static_cast<int>((total_pack + threads - 1) / threads);
            pack_shifted_valence_kernel<<<blocks_pack, threads, 0, streams[slot]>>>(
                d_v_base.get(),
                d_col_v[slot].get(),
                d_offx[slot].get(),
                d_offy[slot].get(),
                d_offz[slot].get(),
                d_valid[slot].get(),
                dimx,
                dimy,
                dimz,
                K,
                nb,
                d_b[slot].get());
            checkCuda(cudaGetLastError(), "pack_shifted_valence_kernel");

            checkCublas(cublasSetStream(cublas_handle, streams[slot]), "cublasSetStream");
            checkCublas(cublasDgemm(
                cublas_handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                ldc,
                nb,
                K,
                &alpha,
                d_a_big.get(),
                K,
                d_b[slot].get(),
                K,
                &beta,
                d_c[slot].get(),
                ldc),
                "cublasDgemm(transition)");

            if (apply_correction) {
                const size_t total_corr = static_cast<size_t>(M) * static_cast<size_t>(nb);
                const int blocks_corr = static_cast<int>((total_corr + threads - 1) / threads);
                apply_transition_correction_kernel<<<blocks_corr, threads, 0, streams[slot]>>>(
                    d_c[slot].get(),
                    M,
                    ldc,
                    nb,
                    d_col_v[slot].get(),
                    d_shift_vec[slot].get(),
                    d_center_c.get(),
                    d_center_v.get());
                checkCuda(cudaGetLastError(), "apply_transition_correction_kernel");
            }

            checkCuda(cudaMemcpyAsync(
                h_c[slot].get(),
                d_c[slot].get(),
                sizeof(double) * static_cast<size_t>(ldc) * static_cast<size_t>(nb),
                cudaMemcpyDeviceToHost,
                streams[slot]),
                "cudaMemcpyAsync(C->host)");

            checkCuda(cudaEventRecord(done[slot], streams[slot]), "cudaEventRecord(done)");
            slot_batch_start[slot] = batch_start;
            slot_batch_size[slot] = nb;
            slot_busy[slot] = true;
        }

        consume_slot(0);
        consume_slot(1);

        cleanup();
        return out;

    } catch (...) {
        cleanup();
        throw;
    }
}
