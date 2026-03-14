#include "backend/gpu/solver_gpu_factory.h"

#include "backend/pre_fft_density.h"
#include "timing.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <list>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
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

void checkCufft(cufftResult status, const char* what)
{
    if (status == CUFFT_SUCCESS) {
        return;
    }
    std::ostringstream msg;
    msg << what << " failed with cuFFT code " << static_cast<int>(status);
    throw std::runtime_error(msg.str());
}

template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() = default;
    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    ~DeviceBuffer() { release(); }

    void allocate(size_t n, const char* what)
    {
        if (n == size_ && ptr_ != nullptr) {
            return;
        }
        release();
        if (n == 0) {
            return;
        }
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T) * n), what);
        size_ = n;
    }

    void release()
    {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

struct FftDensityCacheEntry
{
    cufftDoubleComplex* fft = nullptr;
    double charge = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double alpha = 0.0;
};

void freeDensityCache(std::map<Density_descr, FftDensityCacheEntry>& cache)
{
    for (auto& kv : cache) {
        if (kv.second.fft != nullptr) {
            cudaFree(kv.second.fft);
            kv.second.fft = nullptr;
        }
    }
    cache.clear();
}

struct SameBandFftCacheEntry
{
    cufftDoubleComplex* fft_aux = nullptr;  // auxiliary-corrected spectrum (for Coulomb)
    cufftDoubleComplex* fft_raw = nullptr;  // raw spectrum (for Yukawa)
    double charge = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double alpha = 0.0;
};

struct SolverStageTimingsMs
{
    double density_build = 0.0;
    double density_materialization = 0.0;
    double auxiliary_build = 0.0;
    double auxiliary_subtraction = 0.0;
    double fft = 0.0;
    double contraction = 0.0;
    double host_device_copy = 0.0;
};

struct SameBandCacheBuildConfig
{
    bool need_aux_fft = true;
    bool need_raw_fft = false;
    bool keep_dual_spectral = true;
    bool wrap_aux = true;
    bool enable_timing = false;
};

size_t parse_bytes_with_suffix(const std::string& text)
{
    std::string s(text);
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c) != 0; }), s.end());
    if (s.empty()) {
        throw std::runtime_error("Empty cache size string.");
    }

    auto lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    double scale = 1.0;
    auto strip_suffix = [&](const char* suffix, double factor) -> bool {
        const std::string sf(suffix);
        if (lower.size() >= sf.size() && lower.compare(lower.size() - sf.size(), sf.size(), sf) == 0) {
            lower.erase(lower.size() - sf.size());
            scale = factor;
            return true;
        }
        return false;
    };

    if (!strip_suffix("gib", 1024.0 * 1024.0 * 1024.0) &&
        !strip_suffix("mib", 1024.0 * 1024.0) &&
        !strip_suffix("kib", 1024.0) &&
        !strip_suffix("gb", 1000.0 * 1000.0 * 1000.0) &&
        !strip_suffix("mb", 1000.0 * 1000.0) &&
        !strip_suffix("kb", 1000.0) &&
        !strip_suffix("g", 1024.0 * 1024.0 * 1024.0) &&
        !strip_suffix("m", 1024.0 * 1024.0) &&
        !strip_suffix("k", 1024.0)) {
        scale = 1.0;
    }

    if (lower.empty()) {
        throw std::runtime_error("Invalid cache size string.");
    }
    const double value = std::stod(lower);
    if (!(value > 0.0)) {
        throw std::runtime_error("Cache size must be positive.");
    }
    return static_cast<size_t>(value * scale);
}

size_t same_band_cache_default_bytes()
{
    // Per-band cache default. Coulomb/Yukawa each maintain conduction + valence stores.
    return static_cast<size_t>(3) * 1024ull * 1024ull * 1024ull;
}

size_t same_band_cache_limit_bytes_from_env()
{
    const char* env = std::getenv("WO_GPU_DENSITY_CACHE_BYTES");
    if (env == nullptr || *env == '\0') {
        return same_band_cache_default_bytes();
    }
    try {
        return parse_bytes_with_suffix(env);
    } catch (const std::exception&) {
        return same_band_cache_default_bytes();
    }
}

void release_same_band_entry(SameBandFftCacheEntry& entry)
{
    if (entry.fft_aux != nullptr) {
        cudaFree(entry.fft_aux);
        entry.fft_aux = nullptr;
    }
    if (entry.fft_raw != nullptr) {
        cudaFree(entry.fft_raw);
        entry.fft_raw = nullptr;
    }
}

size_t same_band_entry_device_bytes(const SameBandFftCacheEntry& entry, int N)
{
    const size_t n = static_cast<size_t>(std::max(0, N));
    size_t bytes = 0;
    if (entry.fft_aux != nullptr) {
        bytes += sizeof(cufftDoubleComplex) * n;
    }
    if (entry.fft_raw != nullptr) {
        bytes += sizeof(cufftDoubleComplex) * n;
    }
    return bytes;
}

struct SameBandPersistentCacheNode
{
    SameBandFftCacheEntry entry{};
    size_t bytes = 0;
    size_t last_used_iteration = 0;
    std::list<Density_descr>::iterator lru_it{};
};

class SameBandPersistentCache
{
public:
    explicit SameBandPersistentCache(size_t capacity_bytes)
        : capacity_bytes_(std::max<size_t>(1, capacity_bytes))
    {
    }

    ~SameBandPersistentCache()
    {
        clear();
    }

    SameBandPersistentCache(const SameBandPersistentCache&) = delete;
    SameBandPersistentCache& operator=(const SameBandPersistentCache&) = delete;

    bool contains(const Density_descr& key) const
    {
        return nodes_.find(key) != nodes_.end();
    }

    void touch(const Density_descr& key)
    {
        auto it = nodes_.find(key);
        if (it == nodes_.end()) {
            return;
        }
        touch(it);
    }

    const SameBandFftCacheEntry& at(const Density_descr& key)
    {
        auto it = nodes_.find(key);
        if (it == nodes_.end()) {
            throw std::runtime_error("Density FFT cache miss.");
        }
        touch(it);
        return it->second.entry;
    }

    void insert(
        const Density_descr& key,
        SameBandFftCacheEntry&& entry,
        size_t entry_bytes,
        const std::set<Density_descr>& protected_keys)
    {
        if (entry_bytes > capacity_bytes_) {
            release_same_band_entry(entry);
            throw std::runtime_error("Density FFT cache entry exceeds configured memory cap.");
        }

        auto existing = nodes_.find(key);
        if (existing != nodes_.end()) {
            total_bytes_ -= existing->second.bytes;
            release_same_band_entry(existing->second.entry);
            lru_.erase(existing->second.lru_it);
            nodes_.erase(existing);
        }

        evict_until_fit(entry_bytes, protected_keys);

        SameBandPersistentCacheNode node{};
        node.entry = std::move(entry);
        node.bytes = entry_bytes;
        node.last_used_iteration = ++iteration_counter_;
        lru_.push_front(key);
        node.lru_it = lru_.begin();

        total_bytes_ += node.bytes;
        nodes_.insert({key, std::move(node)});
    }

    void clear()
    {
        for (auto& kv : nodes_) {
            release_same_band_entry(kv.second.entry);
        }
        nodes_.clear();
        lru_.clear();
        total_bytes_ = 0;
        iteration_counter_ = 0;
    }

private:
    void touch(std::map<Density_descr, SameBandPersistentCacheNode>::iterator it)
    {
        it->second.last_used_iteration = ++iteration_counter_;
        if (it->second.lru_it != lru_.begin()) {
            lru_.splice(lru_.begin(), lru_, it->second.lru_it);
        }
    }

    void evict_until_fit(size_t bytes_needed, const std::set<Density_descr>& protected_keys)
    {
        while (total_bytes_ + bytes_needed > capacity_bytes_) {
            bool evicted = false;
            for (auto rit = lru_.rbegin(); rit != lru_.rend(); ++rit) {
                if (protected_keys.find(*rit) != protected_keys.end()) {
                    continue;
                }

                auto map_it = nodes_.find(*rit);
                if (map_it == nodes_.end()) {
                    continue;
                }

                total_bytes_ -= map_it->second.bytes;
                release_same_band_entry(map_it->second.entry);
                lru_.erase(map_it->second.lru_it);
                nodes_.erase(map_it);
                evicted = true;
                break;
            }

            if (!evicted) {
                throw std::runtime_error(
                    "Density FFT cache memory cap is too small for the active descriptor working set.");
            }
        }
    }

    size_t capacity_bytes_ = 1;
    size_t total_bytes_ = 0;
    size_t iteration_counter_ = 0;
    std::map<Density_descr, SameBandPersistentCacheNode> nodes_{};
    std::list<Density_descr> lru_{};
};

size_t recommend_density_build_batch(size_t K)
{
    size_t free_mem = 0;
    size_t total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) {
        return 32;
    }

    const size_t bytes_per_density = K * (sizeof(double) + sizeof(cufftDoubleComplex));
    if (bytes_per_density == 0) {
        return 32;
    }

    // Conservative: reserve most memory for existing caches/workspaces.
    const size_t budget = free_mem / 8;
    size_t batch = budget / bytes_per_density;
    batch = std::max<size_t>(1, batch);
    batch = std::min<size_t>(batch, 256);
    return batch;
}

struct CufftPlanCache3D
{
    explicit CufftPlanCache3D(const std::vector<int>& dims)
    {
        if (dims.size() != 3) {
            throw std::runtime_error("CufftPlanCache3D expects dims with size=3.");
        }
        n_[0] = dims[2];
        n_[1] = dims[1];
        n_[2] = dims[0];
        idist_ = dims[0] * dims[1] * dims[2];
    }

    ~CufftPlanCache3D()
    {
        for (auto& kv : plans_) {
            cufftDestroy(kv.second);
        }
    }

    CufftPlanCache3D(CufftPlanCache3D const&) = delete;
    CufftPlanCache3D& operator=(CufftPlanCache3D const&) = delete;

    cufftHandle getPlan(int batch)
    {
        auto it = plans_.find(batch);
        if (it != plans_.end()) {
            return it->second;
        }

        cufftHandle plan = 0;
        checkCufft(
            cufftPlanMany(
                &plan,
                3,
                n_,
                n_,
                1,
                idist_,
                n_,
                1,
                idist_,
                CUFFT_Z2Z,
                batch),
            "cufftPlanMany(Z2Z)");

        plans_.insert({batch, plan});
        return plan;
    }

    void execForward(cufftDoubleComplex* data, int batch, cudaStream_t stream)
    {
        cufftHandle plan = getPlan(batch);
        checkCufft(cufftSetStream(plan, stream), "cufftSetStream");
        checkCufft(cufftExecZ2Z(plan, data, data, CUFFT_FORWARD), "cufftExecZ2Z forward");
    }

private:
    int n_[3]{};
    int idist_ = 0;
    std::map<int, cufftHandle> plans_{};
};

__device__ inline cufftDoubleComplex cadd(cufftDoubleComplex a, cufftDoubleComplex b)
{
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

__device__ inline cufftDoubleComplex cmul(cufftDoubleComplex a, cufftDoubleComplex b)
{
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ inline cufftDoubleComplex cconj(cufftDoubleComplex a)
{
    return make_cuDoubleComplex(a.x, -a.y);
}

__device__ inline cufftDoubleComplex cscale(cufftDoubleComplex a, double s)
{
    return make_cuDoubleComplex(a.x * s, a.y * s);
}

__global__ void real_to_complex_columns_kernel(
    const double* rho,
    int ld_rho,
    int K,
    int B,
    cufftDoubleComplex* out)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(K) * static_cast<size_t>(B);
    if (idx >= total) {
        return;
    }

    const int i = static_cast<int>(idx % static_cast<size_t>(K));
    const int col = static_cast<int>(idx / static_cast<size_t>(K));
    const double v = rho[static_cast<size_t>(col) * static_cast<size_t>(ld_rho) + static_cast<size_t>(i)];
    out[static_cast<size_t>(col) * static_cast<size_t>(K) + static_cast<size_t>(i)] = make_cuDoubleComplex(v, 0.0);
}

__global__ void gather_weighted_modes_kernel(
    cufftDoubleComplex const* const* fft_ptrs,
    int n_rows,
    const int* q_indices,
    const double* sqrt_w,
    int n_q,
    cufftDoubleComplex* out)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n_rows) * static_cast<size_t>(n_q);
    if (idx >= total) {
        return;
    }

    const int row = static_cast<int>(idx % static_cast<size_t>(n_rows));
    const int q = static_cast<int>(idx / static_cast<size_t>(n_rows));

    const int l = q_indices[q];
    const cufftDoubleComplex val = fft_ptrs[row][l];
    out[static_cast<size_t>(q) * static_cast<size_t>(n_rows) + static_cast<size_t>(row)] = cscale(val, sqrt_w[q]);
}

__global__ void gather_weighted_modes_dual_offset_kernel(
    cufftDoubleComplex const* const* fft_ptrs,
    int n_rows_batch,
    int row_offset,
    int n_rows_total,
    const int* q_indices,
    const int* q_mirror_indices,
    const double* sqrt_w,
    int n_q,
    cufftDoubleComplex* out_plus,
    cufftDoubleComplex* out_minus)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n_rows_batch) * static_cast<size_t>(n_q);
    if (idx >= total) {
        return;
    }

    const int row_local = static_cast<int>(idx % static_cast<size_t>(n_rows_batch));
    const int q = static_cast<int>(idx / static_cast<size_t>(n_rows_batch));
    const int row = row_offset + row_local;

    const int l_plus = q_indices[q];
    const int l_minus = q_mirror_indices[q];
    const double w = sqrt_w[q];

    const cufftDoubleComplex val_plus = fft_ptrs[row_local][l_plus];
    const cufftDoubleComplex val_minus = fft_ptrs[row_local][l_minus];

    out_plus[static_cast<size_t>(q) * static_cast<size_t>(n_rows_total) + static_cast<size_t>(row)] = cscale(val_plus, w);
    out_minus[static_cast<size_t>(q) * static_cast<size_t>(n_rows_total) + static_cast<size_t>(row)] = cscale(val_minus, w);
}

__global__ void extract_gram_real_kernel(
    const cufftDoubleComplex* gram,
    int n_rows,
    const int* row_idx,
    const int* col_idx,
    int n_tasks,
    double* out)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tasks) {
        return;
    }

    const int i = row_idx[t];
    const int j = col_idx[t];
    out[t] = gram[static_cast<size_t>(j) * static_cast<size_t>(n_rows) + static_cast<size_t>(i)].x;
}

__device__ inline cufftDoubleComplex rec_gaussian(
    double qx,
    double qy,
    double qz,
    double x0,
    double y0,
    double z0,
    double alpha,
    double charge,
    double ox,
    double oy,
    double oz,
    bool minus_q)
{
    if (!(alpha > 0.0) || fabs(charge) < 1e-18) {
        return make_cuDoubleComplex(0.0, 0.0);
    }

    const double q2 = qx * qx + qy * qy + qz * qz;
    const double pref = charge * exp(-q2 / (4.0 * alpha));
    const double phase = qx * (x0 - ox) + qy * (y0 - oy) + qz * (z0 - oz);
    const double c = cos(phase);
    const double s = sin(phase);

    if (minus_q) {
        return make_cuDoubleComplex(pref * c, pref * s);
    }
    return make_cuDoubleComplex(pref * c, -pref * s);
}

__global__ void build_coulomb_spectrum_kernel(
    const cufftDoubleComplex* f1,
    const cufftDoubleComplex* f2,
    const double* qx,
    const double* qy,
    const double* qz,
    const double* vq,
    int n_q,
    double dV,
    double invN,
    double charge1,
    double x1,
    double y1,
    double z1,
    double alpha1,
    double charge2,
    double x2,
    double y2,
    double z2,
    double alpha2,
    double ox,
    double oy,
    double oz,
    cufftDoubleComplex* out)
{
    const int iq = blockIdx.x * blockDim.x + threadIdx.x;
    if (iq >= n_q) {
        return;
    }

    const int k = iq + 1;  // q=0 is skipped by construction

    const double qxv = qx[iq];
    const double qyv = qy[iq];
    const double qzv = qz[iq];

    const cufftDoubleComplex ff1 = f1[k];
    const cufftDoubleComplex ff2_minus_q = cconj(f2[k]);

    const cufftDoubleComplex g1 = rec_gaussian(
        qxv, qyv, qzv,
        x1, y1, z1,
        alpha1, charge1,
        ox, oy, oz,
        false);

    const cufftDoubleComplex g2_minus_q = rec_gaussian(
        qxv, qyv, qzv,
        x2, y2, z2,
        alpha2, charge2,
        ox, oy, oz,
        true);

    const cufftDoubleComplex term1 = cscale(cmul(ff1, ff2_minus_q), dV);
    const cufftDoubleComplex term2 = cmul(ff1, g2_minus_q);
    const cufftDoubleComplex term3 = cmul(g1, ff2_minus_q);

    const cufftDoubleComplex sum = cadd(cadd(term1, term2), term3);
    out[iq] = cscale(sum, vq[iq] * invN);
}

__global__ void build_yukawa_spectrum_kernel(
    const cufftDoubleComplex* f1,
    const cufftDoubleComplex* f2,
    const double* qx,
    const double* qy,
    const double* qz,
    int n_q,
    double alpha,
    double dV,
    double invN,
    cufftDoubleComplex* out)
{
    const int iq = blockIdx.x * blockDim.x + threadIdx.x;
    if (iq >= n_q) {
        return;
    }

    const cufftDoubleComplex ff1 = f1[iq];
    const cufftDoubleComplex ff2_minus_q = cconj(f2[iq]);

    const double q2 = qx[iq] * qx[iq] + qy[iq] * qy[iq] + qz[iq] * qz[iq];
    const double denom = q2 + alpha * alpha;
    if (denom <= 1e-16) {
        out[iq] = make_cuDoubleComplex(0.0, 0.0);
        return;
    }

    const double vq = (4.0 * M_PI / denom) * ARB_TO_EV;
    out[iq] = cscale(cmul(ff1, ff2_minus_q), vq * dV * invN);
}

__global__ void build_phase_matrix_kernel(
    const double* qx,
    const double* qy,
    const double* qz,
    const double* shifts,
    int n_q,
    int n_shift,
    cufftDoubleComplex* out)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n_q) * static_cast<size_t>(n_shift);
    if (idx >= total) {
        return;
    }

    const int iq = static_cast<int>(idx % static_cast<size_t>(n_q));
    const int s = static_cast<int>(idx / static_cast<size_t>(n_q));

    const double sx = shifts[3 * s + 0];
    const double sy = shifts[3 * s + 1];
    const double sz = shifts[3 * s + 2];

    const double angle = qx[iq] * sx + qy[iq] * sy + qz[iq] * sz;
    out[static_cast<size_t>(s) * static_cast<size_t>(n_q) + static_cast<size_t>(iq)] =
        make_cuDoubleComplex(cos(angle), sin(angle));
}

__global__ void fused_coulomb_shell_dot_kernel(
    const cufftDoubleComplex* spectrum,
    const double* qx,
    const double* qy,
    const double* qz,
    const double* shifts,
    int n_q,
    int n_shift,
    cufftDoubleComplex* out)
{
    const int s = blockIdx.x;
    if (s >= n_shift) {
        return;
    }

    const int tid = threadIdx.x;
    const double sx = shifts[3 * s + 0];
    const double sy = shifts[3 * s + 1];
    const double sz = shifts[3 * s + 2];

    double acc_r = 0.0;
    double acc_i = 0.0;

    for (int iq = tid; iq < n_q; iq += blockDim.x) {
        const double angle = qx[iq] * sx + qy[iq] * sy + qz[iq] * sz;
        const double c = cos(angle);
        const double si = sin(angle);
        const cufftDoubleComplex ph = make_cuDoubleComplex(c, si);
        const cufftDoubleComplex v = cmul(spectrum[iq], ph);
        acc_r += v.x;
        acc_i += v.y;
    }

    extern __shared__ double sdata[];
    double* s_r = sdata;
    double* s_i = sdata + blockDim.x;
    s_r[tid] = acc_r;
    s_i[tid] = acc_i;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_r[tid] += s_r[tid + stride];
            s_i[tid] += s_i[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[s] = make_cuDoubleComplex(s_r[0], s_i[0]);
    }
}

double elapsed_ms(const std::chrono::steady_clock::time_point& t0, const std::chrono::steady_clock::time_point& t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

bool dual_spectral_enabled_from_env(bool default_value = true)
{
    const char* raw = std::getenv("WO_GPU_DUAL_SPECTRAL");
    if (raw == nullptr) {
        return default_value;
    }
    std::string v(raw);
    for (char& c : v) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    return default_value;
}

void forward_fft_real_columns(
    const double* rho_device,
    int ld,
    int K,
    int B,
    CufftPlanCache3D& plan_cache,
    cudaStream_t stream,
    DeviceBuffer<cufftDoubleComplex>& out_fft)
{
    if (rho_device == nullptr || B == 0 || K == 0) {
        out_fft.release();
        return;
    }

    out_fft.allocate(static_cast<size_t>(K) * static_cast<size_t>(B), "cudaMalloc(batch FFT)");

    const int threads = 256;
    const size_t total = static_cast<size_t>(K) * static_cast<size_t>(B);
    const int blocks = static_cast<int>((total + threads - 1) / threads);

    real_to_complex_columns_kernel<<<blocks, threads, 0, stream>>>(
        rho_device,
        ld,
        K,
        B,
        out_fft.get());
    checkCuda(cudaGetLastError(), "real_to_complex_columns_kernel");

    plan_cache.execForward(out_fft.get(), B, stream);
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(fft)");
}

void forward_fft_real_batch(
    const GpuDensityBatch& batch,
    CufftPlanCache3D& plan_cache,
    cudaStream_t stream,
    DeviceBuffer<cufftDoubleComplex>& out_fft)
{
    forward_fft_real_columns(
        batch.rho_device,
        static_cast<int>(batch.ld),
        static_cast<int>(batch.K),
        static_cast<int>(batch.B),
        plan_cache,
        stream,
        out_fft);
}

std::array<double, 3> rd_to_cart_shift(
    const std::vector<std::vector<double>>& unitcell_t,
    const std::array<int, 3>& rd)
{
    const std::vector<double> tmp = matVecMul3x3(unitcell_t, std::vector<int>{rd[0], rd[1], rd[2]});
    return {tmp[0], tmp[1], tmp[2]};
}

double analytic_i4(
    const SameBandFftCacheEntry& c,
    const SameBandFftCacheEntry& v,
    const std::array<double, 3>& shift)
{
    if (!(c.alpha > 0.0) || !(v.alpha > 0.0)) {
        return 0.0;
    }

    const double beta = std::sqrt((c.alpha * v.alpha) / (c.alpha + v.alpha));
    const double dx = c.x - v.x - shift[0];
    const double dy = c.y - v.y - shift[1];
    const double dz = c.z - v.z - shift[2];
    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);

    const double pref = ARB_TO_EV * c.charge * v.charge;
    if (r < 1e-12) {
        return pref * 2.0 * beta / std::sqrt(M_PI);
    }
    return pref * std::erf(beta * r) / r;
}

void ensure_same_band_cache_entries(
    const std::vector<Density_descr>& required,
    SameBandPersistentCache& cache,
    const std::map<int, WannierFunction>& wann_map,
    CufftPlanCache3D& fft_plans,
    cudaStream_t stream,
    int N,
    const SameBandCacheBuildConfig& cfg,
    SolverStageTimingsMs* timings)
{
    if (required.empty()) {
        return;
    }

    std::set<Density_descr> required_set(required.begin(), required.end());
    std::vector<Density_descr> missing{};
    missing.reserve(required.size());
    for (const auto& d : required) {
        if (cache.contains(d)) {
            cache.touch(d);
        } else {
            missing.push_back(d);
        }
    }

    if (missing.empty()) {
        return;
    }

    const size_t max_batch = recommend_density_build_batch(static_cast<size_t>(N));

    for (size_t start = 0; start < missing.size(); start += max_batch) {
        const size_t end = std::min(start + max_batch, missing.size());
        const size_t B = end - start;

        std::vector<PreFftDensitySpec> specs{};
        specs.reserve(B);
        for (size_t i = start; i < end; ++i) {
            PreFftDensitySpec s{};
            s.idx1 = missing[i].id1;
            s.idx2 = missing[i].id2;
            s.R = std::array<int, 3>{missing[i].R[0], missing[i].R[1], missing[i].R[2]};
            s.user_tag = static_cast<int>(i - start);
            specs.push_back(s);
        }

        if (cfg.need_aux_fft && cfg.need_raw_fft && !cfg.keep_dual_spectral) {
            throw std::runtime_error(
                "Invalid same-band cache config: need_raw_fft + need_aux_fft requires keep_dual_spectral=true.");
        }

        PreFftAuxConfig pre_cfg{};
        pre_cfg.apply_auxiliary_subtraction = cfg.need_aux_fft;
        pre_cfg.copy_metadata_to_host = cfg.need_aux_fft;
        pre_cfg.wrap_aux = cfg.wrap_aux;
        pre_cfg.q_abs_threshold = 0.0;
        pre_cfg.a0_min = 1e-16;
        pre_cfg.points_per_std = 2.0;
        pre_cfg.std_per_cell = 11.0;
        pre_cfg.enable_timing = cfg.enable_timing;
        pre_cfg.keep_raw_density_copy = cfg.need_aux_fft && cfg.need_raw_fft && cfg.keep_dual_spectral;

        const auto t_density_start = std::chrono::steady_clock::now();
        GpuDensityBatch batch = build_pre_fft_density_batch_gpu(
            PreFftDensityKind::SameBand,
            wann_map,
            wann_map,
            specs,
            pre_cfg);
        const auto t_density_stop = std::chrono::steady_clock::now();

        if (timings != nullptr) {
            if (cfg.enable_timing) {
                timings->density_build += batch.timings_ms.total;
                timings->density_materialization += batch.timings_ms.density_materialization;
                timings->auxiliary_build += batch.timings_ms.auxiliary_build;
                timings->auxiliary_subtraction += batch.timings_ms.auxiliary_subtraction;
                timings->host_device_copy += batch.timings_ms.metadata_copy_gpu_to_cpu;
            } else {
                timings->density_build += elapsed_ms(t_density_start, t_density_stop);
            }
        }

        DeviceBuffer<cufftDoubleComplex> d_fft_aux{};
        DeviceBuffer<cufftDoubleComplex> d_fft_raw{};

        if (cfg.need_aux_fft) {
            const auto t_fft_start = std::chrono::steady_clock::now();
            forward_fft_real_columns(
                batch.rho_device,
                static_cast<int>(batch.ld),
                static_cast<int>(batch.K),
                static_cast<int>(batch.B),
                fft_plans,
                stream,
                d_fft_aux);
            if (timings != nullptr) {
                const auto t_fft_stop = std::chrono::steady_clock::now();
                timings->fft += elapsed_ms(t_fft_start, t_fft_stop);
            }
        }

        if (cfg.need_raw_fft) {
            const double* raw_src = nullptr;
            if (cfg.need_aux_fft) {
                raw_src = batch.rho_raw_device;
            } else {
                raw_src = batch.rho_device;
            }
            if (raw_src == nullptr) {
                throw std::runtime_error("Raw density source missing while building raw same-band FFT cache.");
            }

            const auto t_fft_start = std::chrono::steady_clock::now();
            forward_fft_real_columns(
                raw_src,
                static_cast<int>(batch.ld),
                static_cast<int>(batch.K),
                static_cast<int>(batch.B),
                fft_plans,
                stream,
                d_fft_raw);
            if (timings != nullptr) {
                const auto t_fft_stop = std::chrono::steady_clock::now();
                timings->fft += elapsed_ms(t_fft_start, t_fft_stop);
            }
        }

        for (size_t local = 0; local < B; ++local) {
            SameBandFftCacheEntry entry{};

            if (cfg.need_aux_fft) {
                checkCuda(
                    cudaMalloc(
                        reinterpret_cast<void**>(&entry.fft_aux),
                        sizeof(cufftDoubleComplex) * static_cast<size_t>(N)),
                    "cudaMalloc(cache fft_aux)");
                checkCuda(
                    cudaMemcpyAsync(
                        entry.fft_aux,
                        d_fft_aux.get() + local * static_cast<size_t>(N),
                        sizeof(cufftDoubleComplex) * static_cast<size_t>(N),
                        cudaMemcpyDeviceToDevice,
                        stream),
                    "copy cache fft_aux");

                if (batch.Q_host.size() == batch.B) {
                    entry.charge = batch.Q_host[local];
                }
                if (batch.r0x_host.size() == batch.B) {
                    entry.x = batch.r0x_host[local];
                    entry.y = batch.r0y_host[local];
                    entry.z = batch.r0z_host[local];
                }
                if (batch.alpha_host.size() == batch.B) {
                    entry.alpha = batch.alpha_host[local];
                }
            }

            if (cfg.need_raw_fft) {
                checkCuda(
                    cudaMalloc(
                        reinterpret_cast<void**>(&entry.fft_raw),
                        sizeof(cufftDoubleComplex) * static_cast<size_t>(N)),
                    "cudaMalloc(cache fft_raw)");
                checkCuda(
                    cudaMemcpyAsync(
                        entry.fft_raw,
                        d_fft_raw.get() + local * static_cast<size_t>(N),
                        sizeof(cufftDoubleComplex) * static_cast<size_t>(N),
                        cudaMemcpyDeviceToDevice,
                        stream),
                    "copy cache fft_raw");
            }
            const size_t entry_bytes = same_band_entry_device_bytes(entry, N);
            cache.insert(missing[start + local], std::move(entry), entry_bytes, required_set);
        }
        checkCuda(cudaStreamSynchronize(stream), "sync same-band cache build");
    }
}

class LocalFieldEffectsGpuSolver final : public Solver
{
public:
    using Solver::calculate;

    LocalFieldEffectsGpuSolver(
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap)
        : Solver("LocalFieldEffectsGPU", vWannMap, cWannMap),
          rec_mesh_(vWannMap.begin()->second.getMeshgrid()),
          fft_plans_(rec_mesh_.getDim())
    {
        const std::vector<double> supercell_d = vWannMap.begin()->second.getLatticeInUnitcellBasis();
        for (int i = 0; i < 3; ++i) {
            const int rounded = static_cast<int>(std::llround(supercell_d[i]));
            if (std::abs(supercell_d[i] - static_cast<double>(rounded)) > 1e-5) {
                throw std::runtime_error("Supercell is not compatible with unitcell for LFE GPU solver.");
            }
            supercell_[i] = rounded;
        }

        const RealMeshgrid* real_mesh = vWannMap.begin()->second.getMeshgrid();
        dV_ = real_mesh->getdV();
        V_unitcell_ = vWannMap.begin()->second.getVunitcell();
        N_ = real_mesh->getNumDataPoints();

        checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate(LFE solver)");
        checkCublas(cublasCreate(&cublas_), "cublasCreate(LFE solver)");
        checkCublas(cublasSetStream(cublas_, stream_), "cublasSetStream(LFE solver)");

        const std::vector<int> dims = rec_mesh_.getDim();
        std::vector<int> q_indices{};
        std::vector<double> sqrt_w{};

        for (int i = 0; i < dims[0]; i += supercell_[0]) {
            for (int j = 0; j < dims[1]; j += supercell_[1]) {
                for (int k = 0; k < dims[2]; k += supercell_[2]) {
                    if (i == 0 && j == 0 && k == 0) {
                        continue;
                    }
                    double qx = 0.0;
                    double qy = 0.0;
                    double qz = 0.0;
                    rec_mesh_.xyz(i, j, k, qx, qy, qz);
                    const double w = potential_.fourierCart(qx, qy, qz) * dV_ * dV_ /
                        std::pow(2.0 * M_PI, 3.0) /
                        V_unitcell_;
                    const int l = rec_mesh_.getGlobId(i, j, k);
                    q_indices.push_back(l);
                    sqrt_w.push_back(std::sqrt(std::max(0.0, w)));
                }
            }
        }

        Q_ = static_cast<int>(q_indices.size());
        d_q_indices_.allocate(q_indices.size(), "cudaMalloc(LFE q indices)");
        d_sqrt_w_.allocate(sqrt_w.size(), "cudaMalloc(LFE sqrt weights)");
        if (!q_indices.empty()) {
            checkCuda(
                cudaMemcpyAsync(
                    d_q_indices_.get(),
                    q_indices.data(),
                    sizeof(int) * q_indices.size(),
                    cudaMemcpyHostToDevice,
                    stream_),
                "copy LFE q indices");
            checkCuda(
                cudaMemcpyAsync(
                    d_sqrt_w_.get(),
                    sqrt_w.data(),
                    sizeof(double) * sqrt_w.size(),
                    cudaMemcpyHostToDevice,
                    stream_),
                "copy LFE sqrt weights");
            checkCuda(cudaStreamSynchronize(stream_), "sync LFE precompute");
        }
    }

    ~LocalFieldEffectsGpuSolver() override
    {
        freeDensityCache(cache_);
        release_pre_fft_density_gpu_workspace();
        if (cublas_ != nullptr) {
            cublasDestroy(cublas_);
            cublas_ = nullptr;
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    void calculate(
        std::vector<Integral>& integrals,
        const bool /*verbose*/ = true,
        const unsigned int /*numOuterThreads*/ = 1,
        const unsigned int numInnerThreads = 1) override
    {
        if (integrals.empty()) {
            return;
        }

        omp_set_num_threads(static_cast<int>(numInnerThreads));

        struct TaskData {
            int integral_index = -1;
            Density_descr d1{};
            Density_descr d2{};
        };

        std::vector<TaskData> tasks{};
        tasks.reserve(integrals.size());

        for (size_t i = 0; i < integrals.size(); ++i) {
            if (integrals[i].isEmpty()) {
                continue;
            }

            const std::array<int, 3> RD{integrals[i].indexes[4], integrals[i].indexes[5], integrals[i].indexes[6]};
            if (RD[0] != 0 || RD[1] != 0 || RD[2] != 0) {
                integrals[i].setFailed("RD has to be zero when calculating local field effects (GPU).");
                continue;
            }

            const std::vector<int> R1{
                -integrals[i].indexes[7],
                -integrals[i].indexes[8],
                -integrals[i].indexes[9]};
            const std::vector<int> R2{
                -integrals[i].indexes[10],
                -integrals[i].indexes[11],
                -integrals[i].indexes[12]};

            TaskData t{};
            t.integral_index = static_cast<int>(i);
            t.d1 = Density_descr(integrals[i].indexes[0], integrals[i].indexes[2], R1);
            t.d2 = Density_descr(integrals[i].indexes[1], integrals[i].indexes[3], R2);
            tasks.push_back(t);
        }

        if (tasks.empty()) {
            return;
        }

        std::set<Density_descr> required_set{};
        for (const auto& t : tasks) {
            required_set.insert(t.d1);
            required_set.insert(t.d2);
        }
        std::vector<Density_descr> required(required_set.begin(), required_set.end());

        ensure_cache(required);

        std::map<Density_descr, int> tuple_to_idx{};
        tuple_to_idx.clear();
        for (size_t i = 0; i < required.size(); ++i) {
            tuple_to_idx.insert({required[i], static_cast<int>(i)});
        }

        const int n_u = static_cast<int>(required.size());
        if (n_u == 0) {
            return;
        }

        if (Q_ == 0) {
            for (const auto& t : tasks) {
                integrals[t.integral_index].value = 0.0;
            }
            freeDensityCache(cache_);
            return;
        }

        std::vector<cufftDoubleComplex*> h_fft_ptrs(required.size(), nullptr);
        for (size_t i = 0; i < required.size(); ++i) {
            h_fft_ptrs[i] = cache_.at(required[i]).fft;
        }

        DeviceBuffer<cufftDoubleComplex*> d_fft_ptrs{};
        d_fft_ptrs.allocate(h_fft_ptrs.size(), "cudaMalloc(LFE fft ptrs)");
        checkCuda(
            cudaMemcpyAsync(
                d_fft_ptrs.get(),
                h_fft_ptrs.data(),
                sizeof(cufftDoubleComplex*) * h_fft_ptrs.size(),
                cudaMemcpyHostToDevice,
                stream_),
            "copy LFE fft ptrs");

        DeviceBuffer<cufftDoubleComplex> d_modes{};
        d_modes.allocate(static_cast<size_t>(n_u) * static_cast<size_t>(Q_), "cudaMalloc(LFE modes)");

        const int threads = 256;
        const size_t total_modes = static_cast<size_t>(n_u) * static_cast<size_t>(Q_);
        const int blocks_modes = static_cast<int>((total_modes + threads - 1) / threads);
        gather_weighted_modes_kernel<<<blocks_modes, threads, 0, stream_>>>(
            d_fft_ptrs.get(),
            n_u,
            d_q_indices_.get(),
            d_sqrt_w_.get(),
            Q_,
            d_modes.get());
        checkCuda(cudaGetLastError(), "gather_weighted_modes_kernel");

        DeviceBuffer<cufftDoubleComplex> d_gram{};
        d_gram.allocate(static_cast<size_t>(n_u) * static_cast<size_t>(n_u), "cudaMalloc(LFE gram)");

        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

        checkCublas(
            cublasZgemm(
                cublas_,
                CUBLAS_OP_N,
                CUBLAS_OP_C,
                n_u,
                n_u,
                Q_,
                &alpha,
                reinterpret_cast<const cuDoubleComplex*>(d_modes.get()),
                n_u,
                reinterpret_cast<const cuDoubleComplex*>(d_modes.get()),
                n_u,
                &beta,
                reinterpret_cast<cuDoubleComplex*>(d_gram.get()),
                n_u),
            "cublasZgemm(LFE gram)");

        std::vector<int> h_row(tasks.size(), 0);
        std::vector<int> h_col(tasks.size(), 0);
        for (size_t i = 0; i < tasks.size(); ++i) {
            h_row[i] = tuple_to_idx.at(tasks[i].d1);
            h_col[i] = tuple_to_idx.at(tasks[i].d2);
        }

        DeviceBuffer<int> d_row{};
        DeviceBuffer<int> d_col{};
        d_row.allocate(h_row.size(), "cudaMalloc(LFE row)");
        d_col.allocate(h_col.size(), "cudaMalloc(LFE col)");

        checkCuda(
            cudaMemcpyAsync(
                d_row.get(),
                h_row.data(),
                sizeof(int) * h_row.size(),
                cudaMemcpyHostToDevice,
                stream_),
            "copy LFE rows");
        checkCuda(
            cudaMemcpyAsync(
                d_col.get(),
                h_col.data(),
                sizeof(int) * h_col.size(),
                cudaMemcpyHostToDevice,
                stream_),
            "copy LFE cols");

        DeviceBuffer<double> d_out{};
        d_out.allocate(tasks.size(), "cudaMalloc(LFE out)");
        const int blocks_tasks = static_cast<int>((tasks.size() + threads - 1) / threads);
        extract_gram_real_kernel<<<blocks_tasks, threads, 0, stream_>>>(
            d_gram.get(),
            n_u,
            d_row.get(),
            d_col.get(),
            static_cast<int>(tasks.size()),
            d_out.get());
        checkCuda(cudaGetLastError(), "extract_gram_real_kernel");

        std::vector<double> h_out(tasks.size(), 0.0);
        checkCuda(
            cudaMemcpyAsync(
                h_out.data(),
                d_out.get(),
                sizeof(double) * h_out.size(),
                cudaMemcpyDeviceToHost,
                stream_),
            "copy LFE out");
        checkCuda(cudaStreamSynchronize(stream_), "sync LFE results");

        for (size_t i = 0; i < tasks.size(); ++i) {
            integrals[tasks[i].integral_index].value = h_out[i];
        }

        // Keep GPU memory bounded on small cards by limiting cache lifetime
        // to one scheduler batch.
        freeDensityCache(cache_);
    }

private:
    void ensure_cache(const std::vector<Density_descr>& required)
    {
        std::vector<Density_descr> missing{};
        missing.reserve(required.size());
        for (const auto& d : required) {
            if (cache_.find(d) == cache_.end()) {
                missing.push_back(d);
            }
        }

        if (missing.empty()) {
            return;
        }

        if (cache_.size() + missing.size() > max_cache_entries_) {
            freeDensityCache(cache_);
            missing = required;
        }

        const size_t max_batch = recommend_density_build_batch(static_cast<size_t>(N_));

        for (size_t start = 0; start < missing.size(); start += max_batch) {
            const size_t end = std::min(start + max_batch, missing.size());
            const size_t B = end - start;

            std::vector<PreFftDensitySpec> specs{};
            specs.reserve(B);
            for (size_t i = start; i < end; ++i) {
                PreFftDensitySpec s{};
                s.idx1 = missing[i].id1;
                s.idx2 = missing[i].id2;
                s.R = std::array<int, 3>{missing[i].R[0], missing[i].R[1], missing[i].R[2]};
                s.user_tag = static_cast<int>(i - start);
                specs.push_back(s);
            }

            PreFftAuxConfig cfg{};
            cfg.apply_auxiliary_subtraction = false;
            cfg.copy_metadata_to_host = false;
            cfg.enable_timing = false;

            GpuDensityBatch batch = build_pre_fft_density_batch_gpu(
                PreFftDensityKind::TransitionCv,
                cWannMap,
                vWannMap,
                specs,
                cfg);

            DeviceBuffer<cufftDoubleComplex> d_fft{};
            forward_fft_real_batch(batch, fft_plans_, stream_, d_fft);

            for (size_t local = 0; local < B; ++local) {
                FftDensityCacheEntry entry{};
                checkCuda(
                    cudaMalloc(
                        reinterpret_cast<void**>(&entry.fft),
                        sizeof(cufftDoubleComplex) * static_cast<size_t>(N_)),
                    "cudaMalloc(LFE cached fft)");
                checkCuda(
                    cudaMemcpyAsync(
                        entry.fft,
                        d_fft.get() + local * static_cast<size_t>(N_),
                        sizeof(cufftDoubleComplex) * static_cast<size_t>(N_),
                        cudaMemcpyDeviceToDevice,
                        stream_),
                    "copy LFE cached fft");

                cache_.insert({missing[start + local], entry});
            }
            checkCuda(cudaStreamSynchronize(stream_), "sync LFE cache build");
        }
    }

    ReciprocalMeshgrid rec_mesh_;
    CoulombPotential potential_{};
    int supercell_[3]{1, 1, 1};
    double dV_ = 0.0;
    double V_unitcell_ = 1.0;
    int N_ = 0;
    int Q_ = 0;

    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;

    CufftPlanCache3D fft_plans_;

    DeviceBuffer<int> d_q_indices_{};
    DeviceBuffer<double> d_sqrt_w_{};

    std::map<Density_descr, FftDensityCacheEntry> cache_{};
    const size_t max_cache_entries_ = 4096;
};

double bytes_to_gib(size_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

std::vector<Density_descr> build_lfe_survivor_tuples(
    std::map<Density_descr, Density_indicator> const& lfe_indicators,
    double abscharge_threshold)
{
    std::vector<Density_descr> survivors{};
    survivors.reserve(lfe_indicators.size());
    for (const auto& kv : lfe_indicators) {
        if (kv.second.absCharge < abscharge_threshold) {
            continue;
        }
        Density_descr converted(kv.first);
        converted.R[0] = -converted.R[0];
        converted.R[1] = -converted.R[1];
        converted.R[2] = -converted.R[2];
        survivors.push_back(converted);
    }
    return survivors;
}

size_t estimate_cufft_workspace_bytes(const std::vector<int>& dims, int batch)
{
    if (dims.size() != 3) {
        throw std::runtime_error("estimate_cufft_workspace_bytes expects 3D dims.");
    }
    if (batch <= 0) {
        return 0;
    }

    cufftHandle plan = 0;
    checkCufft(cufftCreate(&plan), "cufftCreate(workspace estimate)");

    int n[3]{dims[2], dims[1], dims[0]};
    const int idist = dims[0] * dims[1] * dims[2];
    size_t work_size = 0;
    checkCufft(
        cufftMakePlanMany(
            plan,
            3,
            n,
            n,
            1,
            idist,
            n,
            1,
            idist,
            CUFFT_Z2Z,
            batch,
            &work_size),
        "cufftMakePlanMany(workspace estimate)");

    cufftDestroy(plan);
    return work_size;
}

struct LfeDensePlan
{
    bool valid = false;
    bool use_full_gemm = false;
    size_t density_batch = 1;
    size_t tile_size = 0;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    size_t safety_bytes = 0;
    size_t budget_bytes = 0;
    size_t spectral_bytes = 0;
    size_t qmeta_bytes = 0;
    size_t cufft_workspace_bytes = 0;
    size_t build_peak_bytes = 0;
    size_t contraction_full_bytes = 0;
    size_t contraction_tile_bytes = 0;
};

LfeDensePlan plan_lfe_dense(
    size_t n_tuple,
    size_t n_q,
    size_t n_grid,
    const std::vector<int>& dims)
{
    LfeDensePlan plan{};
    plan.valid = false;

    if (n_tuple == 0 || n_q == 0 || n_grid == 0) {
        plan.valid = true;
        plan.use_full_gemm = true;
        plan.density_batch = 1;
        plan.tile_size = 0;
        return plan;
    }

    checkCuda(cudaMemGetInfo(&plan.free_bytes, &plan.total_bytes), "cudaMemGetInfo(LFE dense plan)");

    const size_t safety_floor = 512ull * 1024ull * 1024ull;
    const size_t safety_cap = 4ull * 1024ull * 1024ull * 1024ull;
    const size_t safety_fraction = static_cast<size_t>(0.20 * static_cast<double>(plan.free_bytes));
    plan.safety_bytes = std::min(safety_cap, std::max(safety_floor, safety_fraction));
    if (plan.free_bytes <= plan.safety_bytes) {
        return plan;
    }
    plan.budget_bytes = plan.free_bytes - plan.safety_bytes;

    const size_t cbytes = sizeof(cufftDoubleComplex);
    plan.spectral_bytes = 2ull * n_tuple * n_q * cbytes; // A(q) and B(-q)
    plan.qmeta_bytes = n_q * (2ull * sizeof(int) + sizeof(double));
    plan.contraction_full_bytes = n_tuple * n_tuple * cbytes;

    const size_t persistent_bytes = plan.spectral_bytes + plan.qmeta_bytes;
    if (persistent_bytes >= plan.budget_bytes) {
        return plan;
    }

    const size_t per_density_bytes = n_grid * (sizeof(double) + sizeof(cufftDoubleComplex));
    const size_t build_budget = plan.budget_bytes - persistent_bytes;
    size_t candidate_batch = std::min(n_tuple, recommend_density_build_batch(n_grid));
    candidate_batch = std::max<size_t>(1, std::min<size_t>(candidate_batch, 256));

    size_t selected_batch = 0;
    size_t selected_ws = 0;
    while (candidate_batch >= 1) {
        const size_t ws = estimate_cufft_workspace_bytes(dims, static_cast<int>(candidate_batch));
        const size_t batch_bytes = candidate_batch * per_density_bytes + ws;
        if (batch_bytes <= build_budget) {
            selected_batch = candidate_batch;
            selected_ws = ws;
            break;
        }
        candidate_batch /= 2;
    }

    if (selected_batch == 0) {
        return plan;
    }

    plan.density_batch = selected_batch;
    plan.cufft_workspace_bytes = selected_ws;
    plan.build_peak_bytes = persistent_bytes + plan.density_batch * per_density_bytes + plan.cufft_workspace_bytes;
    if (plan.build_peak_bytes > plan.budget_bytes) {
        return plan;
    }

    const size_t full_peak = persistent_bytes + plan.contraction_full_bytes;
    if (full_peak <= plan.budget_bytes) {
        plan.use_full_gemm = true;
        plan.tile_size = std::min<size_t>(n_tuple, 512);
        plan.contraction_tile_bytes = plan.contraction_full_bytes;
        plan.valid = true;
        return plan;
    }

    const size_t available_for_tile = plan.budget_bytes - persistent_bytes;
    size_t tile = static_cast<size_t>(std::floor(std::sqrt(static_cast<double>(available_for_tile) / static_cast<double>(cbytes))));
    tile = std::min(tile, n_tuple);
    if (tile == 0) {
        return plan;
    }

    plan.use_full_gemm = false;
    plan.tile_size = tile;
    plan.contraction_tile_bytes = tile * tile * cbytes;
    plan.valid = true;
    return plan;
}

class LocalFieldEffectsGpuDenseRunner
{
public:
    LocalFieldEffectsGpuDenseRunner(
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap)
        : vWannMap_(vWannMap),
          cWannMap_(cWannMap),
          rec_mesh_(vWannMap.begin()->second.getMeshgrid()),
          fft_plans_(rec_mesh_.getDim())
    {
        const std::vector<double> supercell_d = vWannMap.begin()->second.getLatticeInUnitcellBasis();
        for (int i = 0; i < 3; ++i) {
            const int rounded = static_cast<int>(std::llround(supercell_d[i]));
            if (std::abs(supercell_d[i] - static_cast<double>(rounded)) > 1e-5) {
                throw std::runtime_error("Supercell is not compatible with unitcell for dense LFE GPU solver.");
            }
            supercell_[i] = rounded;
        }

        const RealMeshgrid* real_mesh = vWannMap.begin()->second.getMeshgrid();
        dV_ = real_mesh->getdV();
        V_unitcell_ = vWannMap.begin()->second.getVunitcell();
        N_ = real_mesh->getNumDataPoints();
        dims_ = rec_mesh_.getDim();

        checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate(LFE dense)");
        checkCublas(cublasCreate(&cublas_), "cublasCreate(LFE dense)");
        checkCublas(cublasSetStream(cublas_, stream_), "cublasSetStream(LFE dense)");

        std::vector<int> q_indices{};
        std::vector<int> q_mirror_indices{};
        std::vector<double> sqrt_w{};

        for (int i = 0; i < dims_[0]; i += supercell_[0]) {
            for (int j = 0; j < dims_[1]; j += supercell_[1]) {
                for (int k = 0; k < dims_[2]; k += supercell_[2]) {
                    if (i == 0 && j == 0 && k == 0) {
                        continue;
                    }

                    double qx = 0.0;
                    double qy = 0.0;
                    double qz = 0.0;
                    rec_mesh_.xyz(i, j, k, qx, qy, qz);

                    const double w = potential_.fourierCart(qx, qy, qz) * dV_ * dV_ /
                        std::pow(2.0 * M_PI, 3.0) /
                        V_unitcell_;

                    const int l = rec_mesh_.getGlobId(i, j, k);
                    const int mi = (dims_[0] - i) % dims_[0];
                    const int mj = (dims_[1] - j) % dims_[1];
                    const int mk = (dims_[2] - k) % dims_[2];
                    const int l_mirror = rec_mesh_.getGlobId(mi, mj, mk);

                    q_indices.push_back(l);
                    q_mirror_indices.push_back(l_mirror);
                    sqrt_w.push_back(std::sqrt(std::max(0.0, w)));
                }
            }
        }

        Q_ = static_cast<int>(q_indices.size());
        d_q_indices_.allocate(q_indices.size(), "cudaMalloc(LFE dense q indices)");
        d_q_mirror_indices_.allocate(q_mirror_indices.size(), "cudaMalloc(LFE dense q mirror indices)");
        d_sqrt_w_.allocate(sqrt_w.size(), "cudaMalloc(LFE dense sqrt weights)");
        if (!q_indices.empty()) {
            checkCuda(
                cudaMemcpyAsync(
                    d_q_indices_.get(),
                    q_indices.data(),
                    sizeof(int) * q_indices.size(),
                    cudaMemcpyHostToDevice,
                    stream_),
                "copy LFE dense q indices");
            checkCuda(
                cudaMemcpyAsync(
                    d_q_mirror_indices_.get(),
                    q_mirror_indices.data(),
                    sizeof(int) * q_mirror_indices.size(),
                    cudaMemcpyHostToDevice,
                    stream_),
                "copy LFE dense q mirror indices");
            checkCuda(
                cudaMemcpyAsync(
                    d_sqrt_w_.get(),
                    sqrt_w.data(),
                    sizeof(double) * sqrt_w.size(),
                    cudaMemcpyHostToDevice,
                    stream_),
                "copy LFE dense sqrt weights");
            checkCuda(cudaStreamSynchronize(stream_), "sync LFE dense q precompute");
        }
    }

    ~LocalFieldEffectsGpuDenseRunner()
    {
        release_pre_fft_density_gpu_workspace();
        if (cublas_ != nullptr) {
            cublasDestroy(cublas_);
            cublas_ = nullptr;
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    bool run(
        const std::vector<Density_descr>& survivors_s,
        double energy_threshold,
        std::list<Integral>& out_integrals)
    {
        out_integrals.clear();
        const size_t Nt = survivors_s.size();
        if (Nt == 0) {
            std::cout << "[LFE GPU dense] No surviving tuples after ABSCHARGE screening.\n";
            return true;
        }
        if (Q_ == 0) {
            std::cout << "[LFE GPU dense] Q=0 after excluding G=0. Output is empty.\n";
            return true;
        }

        const LfeDensePlan plan = plan_lfe_dense(
            Nt,
            static_cast<size_t>(Q_),
            static_cast<size_t>(N_),
            dims_);

        std::cout
            << "[LFE GPU dense] Planner: Nt=" << Nt
            << ", Q=" << Q_
            << ", N=" << N_
            << ", free=" << bytes_to_gib(plan.free_bytes) << " GiB"
            << ", safety=" << bytes_to_gib(plan.safety_bytes) << " GiB"
            << ", budget=" << bytes_to_gib(plan.budget_bytes) << " GiB"
            << ", spectral=" << bytes_to_gib(plan.spectral_bytes) << " GiB"
            << ", qmeta=" << bytes_to_gib(plan.qmeta_bytes) << " GiB"
            << ", cufft_ws=" << bytes_to_gib(plan.cufft_workspace_bytes) << " GiB"
            << ", density_batch=" << plan.density_batch
            << "\n";

        if (!plan.valid) {
            std::cout << "[LFE GPU dense] Planner could not find a safe memory plan. Fallback required.\n";
            return false;
        }

        if (plan.use_full_gemm) {
            std::cout << "[LFE GPU dense] Contraction mode: FULL GEMM\n";
        } else {
            std::cout << "[LFE GPU dense] Contraction mode: TILED GEMM (tile=" << plan.tile_size << ")\n";
        }

        DeviceBuffer<cufftDoubleComplex> d_modes_plus{};
        DeviceBuffer<cufftDoubleComplex> d_modes_minus{};
        d_modes_plus.allocate(Nt * static_cast<size_t>(Q_), "cudaMalloc(LFE dense modes plus)");
        d_modes_minus.allocate(Nt * static_cast<size_t>(Q_), "cudaMalloc(LFE dense modes minus)");
        build_global_modes(survivors_s, plan.density_batch, d_modes_plus, d_modes_minus);

        if (plan.use_full_gemm) {
            contract_full_and_emit(survivors_s, energy_threshold, plan.tile_size, d_modes_plus, d_modes_minus, out_integrals);
        } else {
            contract_tiled_and_emit(survivors_s, energy_threshold, plan.tile_size, d_modes_plus, d_modes_minus, out_integrals);
        }

        return true;
    }

private:
    void build_global_modes(
        const std::vector<Density_descr>& survivors_s,
        size_t density_batch,
        DeviceBuffer<cufftDoubleComplex>& d_modes_plus,
        DeviceBuffer<cufftDoubleComplex>& d_modes_minus)
    {
        const int threads = 256;
        const int n_tuple = static_cast<int>(survivors_s.size());

        DeviceBuffer<cufftDoubleComplex*> d_fft_ptrs{};
        DeviceBuffer<cufftDoubleComplex> d_fft{};

        for (size_t start = 0; start < survivors_s.size(); start += density_batch) {
            const size_t end = std::min(start + density_batch, survivors_s.size());
            const size_t B = end - start;

            std::vector<PreFftDensitySpec> specs{};
            specs.reserve(B);
            for (size_t i = start; i < end; ++i) {
                PreFftDensitySpec s{};
                s.idx1 = survivors_s[i].id1;
                s.idx2 = survivors_s[i].id2;
                s.R = std::array<int, 3>{
                    -survivors_s[i].R[0],
                    -survivors_s[i].R[1],
                    -survivors_s[i].R[2]};
                s.user_tag = static_cast<int>(i - start);
                specs.push_back(s);
            }

            PreFftAuxConfig cfg{};
            cfg.apply_auxiliary_subtraction = false;
            cfg.copy_metadata_to_host = false;
            cfg.enable_timing = false;

            GpuDensityBatch batch = build_pre_fft_density_batch_gpu(
                PreFftDensityKind::TransitionCv,
                cWannMap_,
                vWannMap_,
                specs,
                cfg);

            forward_fft_real_batch(batch, fft_plans_, stream_, d_fft);

            std::vector<cufftDoubleComplex*> h_fft_ptrs(B, nullptr);
            for (size_t i = 0; i < B; ++i) {
                h_fft_ptrs[i] = d_fft.get() + i * static_cast<size_t>(N_);
            }
            d_fft_ptrs.allocate(B, "cudaMalloc(LFE dense fft ptrs)");
            checkCuda(
                cudaMemcpyAsync(
                    d_fft_ptrs.get(),
                    h_fft_ptrs.data(),
                    sizeof(cufftDoubleComplex*) * B,
                    cudaMemcpyHostToDevice,
                    stream_),
                "copy LFE dense fft ptrs");

            const size_t total = B * static_cast<size_t>(Q_);
            const int blocks = static_cast<int>((total + threads - 1) / threads);
            gather_weighted_modes_dual_offset_kernel<<<blocks, threads, 0, stream_>>>(
                d_fft_ptrs.get(),
                static_cast<int>(B),
                static_cast<int>(start),
                n_tuple,
                d_q_indices_.get(),
                d_q_mirror_indices_.get(),
                d_sqrt_w_.get(),
                Q_,
                d_modes_plus.get(),
                d_modes_minus.get());
            checkCuda(cudaGetLastError(), "gather_weighted_modes_dual_offset_kernel");
            checkCuda(cudaStreamSynchronize(stream_), "sync LFE dense gather modes");
        }
    }

    void append_entries_from_host_tile(
        const std::vector<Density_descr>& survivors_s,
        size_t row0,
        size_t col0,
        size_t rows,
        size_t cols,
        const std::vector<cufftDoubleComplex>& tile,
        double energy_threshold,
        std::list<Integral>& out_integrals) const
    {
        for (size_t cj = 0; cj < cols; ++cj) {
            const size_t j = col0 + cj;
            for (size_t ri = 0; ri < rows; ++ri) {
                const size_t i = row0 + ri;
                if (i < j) {
                    continue;
                }

                const cufftDoubleComplex v = tile[cj * rows + ri];
                if (std::abs(v.y) > 1e-8) {
                    std::ostringstream msg;
                    msg << "Dense LFE GPU value has non-zero imaginary part at (" << i << "," << j << "): " << v.y;
                    throw std::runtime_error(msg.str());
                }

                if (std::abs(v.x) < energy_threshold) {
                    continue;
                }

                Integral a(
                    survivors_s[i].id1,
                    survivors_s[j].id1,
                    survivors_s[i].id2,
                    survivors_s[j].id2,
                    std::vector<int>{0, 0, 0},
                    survivors_s[i].R,
                    survivors_s[j].R);
                a.value = v.x;
                out_integrals.push_back(a);

                if (i != j) {
                    Integral b(
                        survivors_s[j].id1,
                        survivors_s[i].id1,
                        survivors_s[j].id2,
                        survivors_s[i].id2,
                        std::vector<int>{0, 0, 0},
                        survivors_s[j].R,
                        survivors_s[i].R);
                    b.value = v.x;
                    out_integrals.push_back(b);
                }
            }
        }
    }

    void contract_full_and_emit(
        const std::vector<Density_descr>& survivors_s,
        double energy_threshold,
        size_t emit_tile_hint,
        DeviceBuffer<cufftDoubleComplex>& d_modes_plus,
        DeviceBuffer<cufftDoubleComplex>& d_modes_minus,
        std::list<Integral>& out_integrals)
    {
        const int n_tuple = static_cast<int>(survivors_s.size());
        DeviceBuffer<cufftDoubleComplex> d_gram{};
        d_gram.allocate(static_cast<size_t>(n_tuple) * static_cast<size_t>(n_tuple), "cudaMalloc(LFE dense full gram)");

        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        checkCublas(
            cublasZgemm(
                cublas_,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                n_tuple,
                n_tuple,
                Q_,
                &alpha,
                reinterpret_cast<const cuDoubleComplex*>(d_modes_plus.get()),
                n_tuple,
                reinterpret_cast<const cuDoubleComplex*>(d_modes_minus.get()),
                n_tuple,
                &beta,
                reinterpret_cast<cuDoubleComplex*>(d_gram.get()),
                n_tuple),
            "cublasZgemm(LFE dense full)");
        checkCuda(cudaStreamSynchronize(stream_), "sync LFE dense full contraction");

        const size_t emit_tile = std::max<size_t>(1, std::min<size_t>(emit_tile_hint, survivors_s.size()));
        for (size_t row0 = 0; row0 < survivors_s.size(); row0 += emit_tile) {
            const size_t rows = std::min(emit_tile, survivors_s.size() - row0);
            for (size_t col0 = 0; col0 <= row0; col0 += emit_tile) {
                const size_t cols = std::min(emit_tile, survivors_s.size() - col0);
                std::vector<cufftDoubleComplex> host_tile(rows * cols);
                checkCublas(
                    cublasGetMatrix(
                        static_cast<int>(rows),
                        static_cast<int>(cols),
                        sizeof(cufftDoubleComplex),
                        d_gram.get() + col0 * survivors_s.size() + row0,
                        n_tuple,
                        host_tile.data(),
                        static_cast<int>(rows)),
                    "cublasGetMatrix(LFE dense full tile)");
                append_entries_from_host_tile(
                    survivors_s,
                    row0,
                    col0,
                    rows,
                    cols,
                    host_tile,
                    energy_threshold,
                    out_integrals);
            }
        }
    }

    void contract_tiled_and_emit(
        const std::vector<Density_descr>& survivors_s,
        double energy_threshold,
        size_t tile_size,
        DeviceBuffer<cufftDoubleComplex>& d_modes_plus,
        DeviceBuffer<cufftDoubleComplex>& d_modes_minus,
        std::list<Integral>& out_integrals)
    {
        const int n_tuple = static_cast<int>(survivors_s.size());
        const size_t tile = std::max<size_t>(1, std::min(tile_size, survivors_s.size()));
        DeviceBuffer<cufftDoubleComplex> d_tile{};
        d_tile.allocate(tile * tile, "cudaMalloc(LFE dense tile)");

        const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

        for (size_t row0 = 0; row0 < survivors_s.size(); row0 += tile) {
            const int rows = static_cast<int>(std::min(tile, survivors_s.size() - row0));
            for (size_t col0 = 0; col0 <= row0; col0 += tile) {
                const int cols = static_cast<int>(std::min(tile, survivors_s.size() - col0));

                checkCublas(
                    cublasZgemm(
                        cublas_,
                        CUBLAS_OP_N,
                        CUBLAS_OP_T,
                        rows,
                        cols,
                        Q_,
                        &alpha,
                        reinterpret_cast<const cuDoubleComplex*>(d_modes_plus.get() + row0),
                        n_tuple,
                        reinterpret_cast<const cuDoubleComplex*>(d_modes_minus.get() + col0),
                        n_tuple,
                        &beta,
                        reinterpret_cast<cuDoubleComplex*>(d_tile.get()),
                        rows),
                    "cublasZgemm(LFE dense tiled)");
                checkCuda(cudaStreamSynchronize(stream_), "sync LFE dense tiled contraction");

                std::vector<cufftDoubleComplex> host_tile(static_cast<size_t>(rows) * static_cast<size_t>(cols));
                checkCuda(
                    cudaMemcpy(
                        host_tile.data(),
                        d_tile.get(),
                        sizeof(cufftDoubleComplex) * host_tile.size(),
                        cudaMemcpyDeviceToHost),
                    "copy LFE dense tile");

                append_entries_from_host_tile(
                    survivors_s,
                    row0,
                    col0,
                    static_cast<size_t>(rows),
                    static_cast<size_t>(cols),
                    host_tile,
                    energy_threshold,
                    out_integrals);
            }
        }
    }

    std::map<int, WannierFunction> const& vWannMap_;
    std::map<int, WannierFunction> const& cWannMap_;
    ReciprocalMeshgrid rec_mesh_;
    CoulombPotential potential_{};
    std::vector<int> dims_{};
    int supercell_[3]{1, 1, 1};
    double dV_ = 0.0;
    double V_unitcell_ = 1.0;
    int N_ = 0;
    int Q_ = 0;

    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;

    CufftPlanCache3D fft_plans_;

    DeviceBuffer<int> d_q_indices_{};
    DeviceBuffer<int> d_q_mirror_indices_{};
    DeviceBuffer<double> d_sqrt_w_{};
};

struct DensityPairLess
{
    bool operator()(const std::pair<Density_descr, Density_descr>& a, const std::pair<Density_descr, Density_descr>& b) const
    {
        if (a.first < b.first) {
            return true;
        }
        if (b.first < a.first) {
            return false;
        }
        return a.second < b.second;
    }
};

class CoulombGpuSolver final : public Solver
{
public:
    using Solver::calculate;

    CoulombGpuSolver(
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap,
        bool wrap_aux)
        : Solver("FourierGaussGPU", vWannMap, cWannMap),
          wrap_aux_(wrap_aux),
          real_mesh_(vWannMap.begin()->second.getMeshgrid()),
          rec_mesh_(vWannMap.begin()->second.getMeshgrid()),
          fft_plans_(rec_mesh_.getDim()),
          c_cache_(same_band_cache_limit_bytes_from_env()),
          v_cache_(same_band_cache_limit_bytes_from_env())
    {
        dV_ = real_mesh_->getdV();
        N_ = real_mesh_->getNumDataPoints();
        Q_ = std::max(0, N_ - 1);

        origin_ = real_mesh_->getOrigin();
        supercell_ = vWannMap.begin()->second.getLatticeInUnitcellBasis();

        // Determine sigma bounds (same logic as CPU solver).
        std::vector<double> uvec = matVecMul3x3(real_mesh_->getLattice(), std::vector<int>{1, 0, 0});
        double length = std::sqrt(uvec[0] * uvec[0] + uvec[1] * uvec[1] + uvec[2] * uvec[2]);
        double min_length = length;
        double discret_length = length / real_mesh_->getDim()[0];

        uvec = matVecMul3x3(real_mesh_->getLattice(), std::vector<int>{0, 1, 0});
        length = std::sqrt(uvec[0] * uvec[0] + uvec[1] * uvec[1] + uvec[2] * uvec[2]);
        discret_length = std::min(discret_length, length / real_mesh_->getDim()[1]);
        min_length = std::min(min_length, length);

        uvec = matVecMul3x3(real_mesh_->getLattice(), std::vector<int>{0, 0, 1});
        length = std::sqrt(uvec[0] * uvec[0] + uvec[1] * uvec[1] + uvec[2] * uvec[2]);
        discret_length = std::min(discret_length, length / real_mesh_->getDim()[2]);
        min_length = std::min(min_length, length);

        std_min_ = points_per_std_ * discret_length;
        std_max_ = min_length / std_per_cell_;

        if (std_max_ <= std_min_) {
            throw std::runtime_error("std_min > std_max in CoulombGpuSolver.");
        }

        checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate(Coulomb solver)");
        checkCublas(cublasCreate(&cublas_), "cublasCreate(Coulomb solver)");
        checkCublas(cublasSetStream(cublas_, stream_), "cublasSetStream(Coulomb solver)");
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        timing_enabled_ = (rank_ == 0) && wo_timing::timingEnabledFromEnv(false);

        if (Q_ > 0) {
            std::vector<double> h_qx(Q_, 0.0);
            std::vector<double> h_qy(Q_, 0.0);
            std::vector<double> h_qz(Q_, 0.0);
            std::vector<double> h_vq(Q_, 0.0);
            for (int iq = 0; iq < Q_; ++iq) {
                double qx = 0.0;
                double qy = 0.0;
                double qz = 0.0;
                rec_mesh_.xyz(iq + 1, qx, qy, qz);
                h_qx[iq] = qx;
                h_qy[iq] = qy;
                h_qz[iq] = qz;
                h_vq[iq] = potential_.fourierCart(qx, qy, qz);
            }

            d_qx_.allocate(Q_, "cudaMalloc(Coulomb qx)");
            d_qy_.allocate(Q_, "cudaMalloc(Coulomb qy)");
            d_qz_.allocate(Q_, "cudaMalloc(Coulomb qz)");
            d_vq_.allocate(Q_, "cudaMalloc(Coulomb vq)");

            checkCuda(cudaMemcpyAsync(d_qx_.get(), h_qx.data(), sizeof(double) * Q_, cudaMemcpyHostToDevice, stream_), "copy Coulomb qx");
            checkCuda(cudaMemcpyAsync(d_qy_.get(), h_qy.data(), sizeof(double) * Q_, cudaMemcpyHostToDevice, stream_), "copy Coulomb qy");
            checkCuda(cudaMemcpyAsync(d_qz_.get(), h_qz.data(), sizeof(double) * Q_, cudaMemcpyHostToDevice, stream_), "copy Coulomb qz");
            checkCuda(cudaMemcpyAsync(d_vq_.get(), h_vq.data(), sizeof(double) * Q_, cudaMemcpyHostToDevice, stream_), "copy Coulomb vq");
            checkCuda(cudaStreamSynchronize(stream_), "sync Coulomb precompute");
        }
    }

    ~CoulombGpuSolver() override
    {
        c_cache_.clear();
        v_cache_.clear();
        release_pre_fft_density_gpu_workspace();
        if (cublas_ != nullptr) {
            cublasDestroy(cublas_);
            cublas_ = nullptr;
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    void calculate(
        std::vector<Integral>& integrals,
        const bool /*verbose*/ = false,
        const unsigned int /*numOuterThreads*/ = 1,
        const unsigned int numInnerThreads = 1) override
    {
        if (integrals.empty()) {
            return;
        }

        omp_set_num_threads(static_cast<int>(numInnerThreads));
        SolverStageTimingsMs timings{};
        auto time_gpu_stage = [&](double& out_ms, auto&& fn, const char* sync_label) {
            if (!timing_enabled_) {
                fn();
                return;
            }
            const auto t0 = std::chrono::steady_clock::now();
            fn();
            checkCuda(cudaStreamSynchronize(stream_), sync_label);
            const auto t1 = std::chrono::steady_clock::now();
            out_ms += elapsed_ms(t0, t1);
        };
        auto time_copy_stage = [&](double& out_ms, auto&& fn) {
            if (!timing_enabled_) {
                fn();
                return;
            }
            const auto t0 = std::chrono::steady_clock::now();
            fn();
            const auto t1 = std::chrono::steady_clock::now();
            out_ms += elapsed_ms(t0, t1);
        };

        struct TaskData {
            int integral_index = -1;
            Density_descr c_dens{};
            Density_descr v_dens{};
            std::array<int, 3> RD{0, 0, 0};
        };

        std::vector<TaskData> tasks{};
        tasks.reserve(integrals.size());

        for (size_t i = 0; i < integrals.size(); ++i) {
            if (integrals[i].isEmpty()) {
                continue;
            }

            const std::array<int, 3> RD{integrals[i].indexes[4], integrals[i].indexes[5], integrals[i].indexes[6]};
            if (!check_aliasing(RD)) {
                integrals[i].setFailed("The supercell is not large enough to protect against aliasing (GPU Coulomb).");
                continue;
            }

            const std::vector<int> Rc{integrals[i].indexes[7], integrals[i].indexes[8], integrals[i].indexes[9]};
            const std::vector<int> Rv{integrals[i].indexes[10], integrals[i].indexes[11], integrals[i].indexes[12]};

            TaskData t{};
            t.integral_index = static_cast<int>(i);
            t.c_dens = Density_descr(integrals[i].indexes[0], integrals[i].indexes[1], Rc);
            t.v_dens = Density_descr(integrals[i].indexes[2], integrals[i].indexes[3], Rv);
            t.RD = RD;
            tasks.push_back(t);
        }

        if (tasks.empty()) {
            return;
        }

        std::set<Density_descr> c_required_set{};
        std::set<Density_descr> v_required_set{};
        for (const auto& t : tasks) {
            c_required_set.insert(t.c_dens);
            v_required_set.insert(t.v_dens);
        }

        const std::vector<Density_descr> c_required(c_required_set.begin(), c_required_set.end());
        const std::vector<Density_descr> v_required(v_required_set.begin(), v_required_set.end());

        ensure_same_band_cache(c_required, c_cache_, cWannMap, &timings);
        ensure_same_band_cache(v_required, v_cache_, vWannMap, &timings);

        std::map<std::pair<Density_descr, Density_descr>, std::vector<int>, DensityPairLess> grouped{};
        for (size_t i = 0; i < tasks.size(); ++i) {
            grouped[{tasks[i].c_dens, tasks[i].v_dens}].push_back(static_cast<int>(i));
        }

        if (Q_ == 0) {
            for (const auto& t : tasks) {
                integrals[t.integral_index].value = analytic_i4(
                    c_cache_.at(t.c_dens),
                    v_cache_.at(t.v_dens),
                    rd_to_cart_shift(unitcell_T, t.RD));
            }
            return;
        }

        const int threads = 256;
        const double invN = 1.0 / static_cast<double>(N_);

        for (const auto& kv : grouped) {
            const auto& c_entry = c_cache_.at(kv.first.first);
            const auto& v_entry = v_cache_.at(kv.first.second);
            const std::vector<int>& idx = kv.second;
            const int ns = static_cast<int>(idx.size());

            std::vector<double> h_shifts(static_cast<size_t>(3 * ns), 0.0);
            std::vector<std::array<double, 3>> shift_cache(static_cast<size_t>(ns));
            for (int s = 0; s < ns; ++s) {
                const std::array<double, 3> shift = rd_to_cart_shift(unitcell_T, tasks[idx[s]].RD);
                shift_cache[s] = shift;
                h_shifts[3 * s + 0] = shift[0];
                h_shifts[3 * s + 1] = shift[1];
                h_shifts[3 * s + 2] = shift[2];
            }

            DeviceBuffer<double> d_shifts{};
            d_shifts.allocate(h_shifts.size(), "cudaMalloc(Coulomb shifts)");
            time_copy_stage(timings.host_device_copy, [&]() {
                checkCuda(
                    cudaMemcpyAsync(
                        d_shifts.get(),
                        h_shifts.data(),
                        sizeof(double) * h_shifts.size(),
                        cudaMemcpyHostToDevice,
                        stream_),
                    "copy Coulomb shifts");
            });

            DeviceBuffer<cufftDoubleComplex> d_spectrum{};
            d_spectrum.allocate(Q_, "cudaMalloc(Coulomb spectrum)");
            const int blocks_q = static_cast<int>((Q_ + threads - 1) / threads);
            time_gpu_stage(timings.contraction, [&]() {
                build_coulomb_spectrum_kernel<<<blocks_q, threads, 0, stream_>>>(
                        c_entry.fft_aux,
                        v_entry.fft_aux,
                    d_qx_.get(),
                    d_qy_.get(),
                    d_qz_.get(),
                    d_vq_.get(),
                    Q_,
                    dV_,
                    invN,
                    c_entry.charge,
                    c_entry.x,
                    c_entry.y,
                    c_entry.z,
                    c_entry.alpha,
                    v_entry.charge,
                    v_entry.x,
                    v_entry.y,
                    v_entry.z,
                    v_entry.alpha,
                    origin_[0],
                    origin_[1],
                    origin_[2],
                    d_spectrum.get());
                checkCuda(cudaGetLastError(), "build_coulomb_spectrum_kernel");
            }, "sync Coulomb spectrum");

            DeviceBuffer<cufftDoubleComplex> d_shell_values{};
            d_shell_values.allocate(ns, "cudaMalloc(Coulomb shell values)");

            const bool use_gemm = (ns >= 16);
            time_gpu_stage(timings.contraction, [&]() {
                if (use_gemm) {
                    DeviceBuffer<cufftDoubleComplex> d_phase{};
                    d_phase.allocate(static_cast<size_t>(Q_) * static_cast<size_t>(ns), "cudaMalloc(Coulomb phase)");
                    const size_t total_phase = static_cast<size_t>(Q_) * static_cast<size_t>(ns);
                    const int blocks_phase = static_cast<int>((total_phase + threads - 1) / threads);
                    build_phase_matrix_kernel<<<blocks_phase, threads, 0, stream_>>>(
                        d_qx_.get(),
                        d_qy_.get(),
                        d_qz_.get(),
                        d_shifts.get(),
                        Q_,
                        ns,
                        d_phase.get());
                    checkCuda(cudaGetLastError(), "build_phase_matrix_kernel");

                    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
                    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

                    checkCublas(
                        cublasZgemm(
                            cublas_,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            1,
                            ns,
                            Q_,
                            &alpha,
                            reinterpret_cast<const cuDoubleComplex*>(d_spectrum.get()),
                            1,
                            reinterpret_cast<const cuDoubleComplex*>(d_phase.get()),
                            Q_,
                            &beta,
                            reinterpret_cast<cuDoubleComplex*>(d_shell_values.get()),
                            1),
                        "cublasZgemm(Coulomb shell)");
                } else {
                    fused_coulomb_shell_dot_kernel<<<ns, threads, sizeof(double) * threads * 2, stream_>>>(
                        d_spectrum.get(),
                        d_qx_.get(),
                        d_qy_.get(),
                        d_qz_.get(),
                        d_shifts.get(),
                        Q_,
                        ns,
                        d_shell_values.get());
                    checkCuda(cudaGetLastError(), "fused_coulomb_shell_dot_kernel");
                }
            }, "sync Coulomb shell contraction");

            std::vector<cufftDoubleComplex> h_shell_values(static_cast<size_t>(ns));
            time_copy_stage(timings.host_device_copy, [&]() {
                checkCuda(
                    cudaMemcpyAsync(
                        h_shell_values.data(),
                        d_shell_values.get(),
                        sizeof(cufftDoubleComplex) * static_cast<size_t>(ns),
                        cudaMemcpyDeviceToHost,
                        stream_),
                    "copy Coulomb shell values");
                checkCuda(cudaStreamSynchronize(stream_), "sync Coulomb shell values");
            });

            for (int s = 0; s < ns; ++s) {
                const double analytic = analytic_i4(c_entry, v_entry, shift_cache[s]);
                const double imag = h_shell_values[s].y;
                if (std::abs(imag) > 1e-7) {
                    integrals[tasks[idx[s]].integral_index].setFailed("Coulomb GPU integral has non-zero imaginary part.");
                    continue;
                }
                integrals[tasks[idx[s]].integral_index].value = h_shell_values[s].x + analytic;
            }
        }

        maybe_print_timings(timings, tasks.size());
    }

private:
    bool check_aliasing(const std::array<int, 3>& R) const
    {
        for (int i = 0; i < 3; ++i) {
            if (supercell_[i] < std::abs(static_cast<double>(R[i])) / 2.0) {
                return false;
            }
        }
        return true;
    }

    void ensure_same_band_cache(
        const std::vector<Density_descr>& required,
        SameBandPersistentCache& cache,
        const std::map<int, WannierFunction>& wann_map,
        SolverStageTimingsMs* timings)
    {
        SameBandCacheBuildConfig cfg{};
        cfg.need_aux_fft = true;
        cfg.need_raw_fft = keep_dual_spectral_;
        cfg.keep_dual_spectral = keep_dual_spectral_;
        cfg.wrap_aux = wrap_aux_;
        cfg.enable_timing = timing_enabled_;
        ensure_same_band_cache_entries(required, cache, wann_map, fft_plans_, stream_, N_, cfg, timings);
    }

    void maybe_print_timings(const SolverStageTimingsMs& t, size_t n_tasks) const
    {
        if (!timing_enabled_) {
            return;
        }
        const double total = t.density_build + t.fft + t.contraction + t.host_device_copy;
        std::cout
            << "\n[Timing][GPU Coulomb] tasks=" << n_tasks
            << " density_build_ms=" << t.density_build
            << " density_materialization_ms=" << t.density_materialization
            << " auxiliary_build_ms=" << t.auxiliary_build
            << " auxiliary_subtraction_ms=" << t.auxiliary_subtraction
            << " fft_ms=" << t.fft
            << " contraction_ms=" << t.contraction
            << " host_device_copy_ms=" << t.host_device_copy
            << " total_ms=" << total
            << std::endl;
    }

    const double points_per_std_ = 2.0;
    const double std_per_cell_ = 11.0;

    const bool wrap_aux_ = true;
    const RealMeshgrid* real_mesh_ = nullptr;
    ReciprocalMeshgrid rec_mesh_;
    CoulombPotential potential_{};

    double dV_ = 0.0;
    int N_ = 0;
    int Q_ = 0;
    std::vector<double> origin_{};
    std::vector<double> supercell_{};

    double std_min_ = 0.0;
    double std_max_ = 0.0;

    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;

    CufftPlanCache3D fft_plans_;

    DeviceBuffer<double> d_qx_{};
    DeviceBuffer<double> d_qy_{};
    DeviceBuffer<double> d_qz_{};
    DeviceBuffer<double> d_vq_{};

    SameBandPersistentCache c_cache_;
    SameBandPersistentCache v_cache_;

    int rank_ = 0;
    bool timing_enabled_ = false;
    const bool keep_dual_spectral_ = dual_spectral_enabled_from_env(true);
};

class YukawaGpuSolver final : public Solver
{
public:
    using Solver::calculate;

    YukawaGpuSolver(
        std::map<int, WannierFunction> const& vWannMap,
        std::map<int, WannierFunction> const& cWannMap,
        std::map<int, double> const& vMeanDensity,
        std::map<int, double> const& cMeanDensity,
        double relativePermittivity,
        double screeningAlpha)
        : Solver("FourierYukawaGPU", vWannMap, cWannMap),
          rec_mesh_(vWannMap.begin()->second.getMeshgrid()),
          fft_plans_(rec_mesh_.getDim()),
          vMeanDensity_(vMeanDensity),
          cMeanDensity_(cMeanDensity),
          relativePermittivity_(relativePermittivity),
          screeningAlpha_(screeningAlpha),
          c_cache_(same_band_cache_limit_bytes_from_env()),
          v_cache_(same_band_cache_limit_bytes_from_env())
    {
        if (!check_maps(vWannMap, vMeanDensity_) || !check_maps(cWannMap, cMeanDensity_)) {
            throw std::runtime_error("MeanDensity maps are not compatible with Wannier maps (GPU Yukawa).");
        }

        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        timing_enabled_ = (rank_ == 0) && wo_timing::timingEnabledFromEnv(false);

        dV_ = vWannMap.begin()->second.getMeshgrid()->getdV();
        N_ = vWannMap.begin()->second.getMeshgrid()->getNumDataPoints();
        Q_ = N_;  // Yukawa has no q=0 divergence.
        supercell_ = vWannMap.begin()->second.getLatticeInUnitcellBasis();

        checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate(Yukawa solver)");
        checkCublas(cublasCreate(&cublas_), "cublasCreate(Yukawa solver)");
        checkCublas(cublasSetStream(cublas_, stream_), "cublasSetStream(Yukawa solver)");

        if (Q_ > 0) {
            std::vector<double> h_qx(Q_, 0.0);
            std::vector<double> h_qy(Q_, 0.0);
            std::vector<double> h_qz(Q_, 0.0);
            for (int iq = 0; iq < Q_; ++iq) {
                double qx = 0.0;
                double qy = 0.0;
                double qz = 0.0;
                rec_mesh_.xyz(iq, qx, qy, qz);
                h_qx[iq] = qx;
                h_qy[iq] = qy;
                h_qz[iq] = qz;
            }

            d_qx_.allocate(Q_, "cudaMalloc(Yukawa qx)");
            d_qy_.allocate(Q_, "cudaMalloc(Yukawa qy)");
            d_qz_.allocate(Q_, "cudaMalloc(Yukawa qz)");

            checkCuda(cudaMemcpyAsync(d_qx_.get(), h_qx.data(), sizeof(double) * Q_, cudaMemcpyHostToDevice, stream_), "copy Yukawa qx");
            checkCuda(cudaMemcpyAsync(d_qy_.get(), h_qy.data(), sizeof(double) * Q_, cudaMemcpyHostToDevice, stream_), "copy Yukawa qy");
            checkCuda(cudaMemcpyAsync(d_qz_.get(), h_qz.data(), sizeof(double) * Q_, cudaMemcpyHostToDevice, stream_), "copy Yukawa qz");
            checkCuda(cudaStreamSynchronize(stream_), "sync Yukawa precompute");
        }
    }

    ~YukawaGpuSolver() override
    {
        c_cache_.clear();
        v_cache_.clear();
        release_pre_fft_density_gpu_workspace();
        if (cublas_ != nullptr) {
            cublasDestroy(cublas_);
            cublas_ = nullptr;
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    void calculate(
        std::vector<Integral>& integrals,
        const bool /*verbose*/ = true,
        const unsigned int /*numOuterThreads*/ = 1,
        const unsigned int numInnerThreads = 1) override
    {
        if (integrals.empty()) {
            return;
        }

        omp_set_num_threads(static_cast<int>(numInnerThreads));

        struct TaskData {
            int integral_index = -1;
            Density_descr c_dens{};
            Density_descr v_dens{};
            std::array<int, 3> RD{0, 0, 0};
            double alpha = 0.0;
        };

        std::vector<TaskData> tasks{};
        tasks.reserve(integrals.size());

        for (size_t i = 0; i < integrals.size(); ++i) {
            if (integrals[i].isEmpty()) {
                continue;
            }

            const std::array<int, 3> RD{integrals[i].indexes[4], integrals[i].indexes[5], integrals[i].indexes[6]};
            if (!check_aliasing(RD)) {
                integrals[i].setFailed("The supercell is not large enough to protect against aliasing (GPU Yukawa).");
                continue;
            }

            const std::vector<int> Rc{integrals[i].indexes[7], integrals[i].indexes[8], integrals[i].indexes[9]};
            const std::vector<int> Rv{integrals[i].indexes[10], integrals[i].indexes[11], integrals[i].indexes[12]};

            TaskData t{};
            t.integral_index = static_cast<int>(i);
            t.c_dens = Density_descr(integrals[i].indexes[0], integrals[i].indexes[1], Rc);
            t.v_dens = Density_descr(integrals[i].indexes[2], integrals[i].indexes[3], Rv);
            t.RD = RD;
            t.alpha = yukawa_alpha(t.c_dens, t.v_dens);
            tasks.push_back(t);
        }

        if (tasks.empty()) {
            return;
        }

        std::set<Density_descr> c_required_set{};
        std::set<Density_descr> v_required_set{};
        for (const auto& t : tasks) {
            c_required_set.insert(t.c_dens);
            v_required_set.insert(t.v_dens);
        }
        const std::vector<Density_descr> c_required(c_required_set.begin(), c_required_set.end());
        const std::vector<Density_descr> v_required(v_required_set.begin(), v_required_set.end());

        SolverStageTimingsMs timings{};
        ensure_same_band_cache(c_required, c_cache_, cWannMap, &timings);
        ensure_same_band_cache(v_required, v_cache_, vWannMap, &timings);

        std::map<std::pair<Density_descr, Density_descr>, std::vector<int>, DensityPairLess> grouped{};
        for (size_t i = 0; i < tasks.size(); ++i) {
            grouped[{tasks[i].c_dens, tasks[i].v_dens}].push_back(static_cast<int>(i));
        }

        if (Q_ == 0) {
            for (const auto& t : tasks) {
                integrals[t.integral_index].value = 0.0;
            }
            maybe_print_timings(timings, tasks.size());
            return;
        }

        auto time_gpu_stage = [&](double& out_ms, auto&& fn, const char* sync_label) {
            const auto t0 = std::chrono::steady_clock::now();
            fn();
            checkCuda(cudaStreamSynchronize(stream_), sync_label);
            const auto t1 = std::chrono::steady_clock::now();
            out_ms += elapsed_ms(t0, t1);
        };
        auto time_copy_stage = [&](double& out_ms, auto&& fn) {
            const auto t0 = std::chrono::steady_clock::now();
            fn();
            const auto t1 = std::chrono::steady_clock::now();
            out_ms += elapsed_ms(t0, t1);
        };

        const int threads = 256;
        const double invN = 1.0 / static_cast<double>(N_);

        for (const auto& kv : grouped) {
            const auto& c_entry = c_cache_.at(kv.first.first);
            const auto& v_entry = v_cache_.at(kv.first.second);
            if (c_entry.fft_raw == nullptr || v_entry.fft_raw == nullptr) {
                throw std::runtime_error("GPU Yukawa requires raw FFT spectra in same-band cache.");
            }

            const std::vector<int>& idx = kv.second;
            const int ns = static_cast<int>(idx.size());
            const double alpha = tasks[idx.front()].alpha;

            std::vector<double> h_shifts(static_cast<size_t>(3 * ns), 0.0);
            for (int s = 0; s < ns; ++s) {
                const std::array<double, 3> shift = rd_to_cart_shift(unitcell_T, tasks[idx[s]].RD);
                h_shifts[3 * s + 0] = shift[0];
                h_shifts[3 * s + 1] = shift[1];
                h_shifts[3 * s + 2] = shift[2];
            }

            DeviceBuffer<double> d_shifts{};
            d_shifts.allocate(h_shifts.size(), "cudaMalloc(Yukawa shifts)");
            time_copy_stage(timings.host_device_copy, [&]() {
                checkCuda(
                    cudaMemcpyAsync(
                        d_shifts.get(),
                        h_shifts.data(),
                        sizeof(double) * h_shifts.size(),
                        cudaMemcpyHostToDevice,
                        stream_),
                    "copy Yukawa shifts");
                checkCuda(cudaStreamSynchronize(stream_), "sync Yukawa shifts copy");
            });

            DeviceBuffer<cufftDoubleComplex> d_spectrum{};
            d_spectrum.allocate(Q_, "cudaMalloc(Yukawa spectrum)");
            const int blocks_q = static_cast<int>((Q_ + threads - 1) / threads);
            time_gpu_stage(timings.contraction, [&]() {
                build_yukawa_spectrum_kernel<<<blocks_q, threads, 0, stream_>>>(
                    c_entry.fft_raw,
                    v_entry.fft_raw,
                    d_qx_.get(),
                    d_qy_.get(),
                    d_qz_.get(),
                    Q_,
                    alpha,
                    dV_,
                    invN,
                    d_spectrum.get());
                checkCuda(cudaGetLastError(), "build_yukawa_spectrum_kernel");
            }, "sync Yukawa spectrum");

            DeviceBuffer<cufftDoubleComplex> d_shell_values{};
            d_shell_values.allocate(ns, "cudaMalloc(Yukawa shell values)");
            const bool use_gemm = (ns >= 16);
            time_gpu_stage(timings.contraction, [&]() {
                if (use_gemm) {
                    DeviceBuffer<cufftDoubleComplex> d_phase{};
                    d_phase.allocate(static_cast<size_t>(Q_) * static_cast<size_t>(ns), "cudaMalloc(Yukawa phase)");
                    const size_t total_phase = static_cast<size_t>(Q_) * static_cast<size_t>(ns);
                    const int blocks_phase = static_cast<int>((total_phase + threads - 1) / threads);
                    build_phase_matrix_kernel<<<blocks_phase, threads, 0, stream_>>>(
                        d_qx_.get(),
                        d_qy_.get(),
                        d_qz_.get(),
                        d_shifts.get(),
                        Q_,
                        ns,
                        d_phase.get());
                    checkCuda(cudaGetLastError(), "build_phase_matrix_kernel(Yukawa)");

                    const cuDoubleComplex alpha_mul = make_cuDoubleComplex(1.0, 0.0);
                    const cuDoubleComplex beta_mul = make_cuDoubleComplex(0.0, 0.0);
                    checkCublas(
                        cublasZgemm(
                            cublas_,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            1,
                            ns,
                            Q_,
                            &alpha_mul,
                            reinterpret_cast<const cuDoubleComplex*>(d_spectrum.get()),
                            1,
                            reinterpret_cast<const cuDoubleComplex*>(d_phase.get()),
                            Q_,
                            &beta_mul,
                            reinterpret_cast<cuDoubleComplex*>(d_shell_values.get()),
                            1),
                        "cublasZgemm(Yukawa shell)");
                } else {
                    fused_coulomb_shell_dot_kernel<<<ns, threads, sizeof(double) * threads * 2, stream_>>>(
                        d_spectrum.get(),
                        d_qx_.get(),
                        d_qy_.get(),
                        d_qz_.get(),
                        d_shifts.get(),
                        Q_,
                        ns,
                        d_shell_values.get());
                    checkCuda(cudaGetLastError(), "fused_coulomb_shell_dot_kernel(Yukawa)");
                }
            }, "sync Yukawa shell contraction");

            std::vector<cufftDoubleComplex> h_shell_values(static_cast<size_t>(ns));
            time_copy_stage(timings.host_device_copy, [&]() {
                checkCuda(
                    cudaMemcpyAsync(
                        h_shell_values.data(),
                        d_shell_values.get(),
                        sizeof(cufftDoubleComplex) * static_cast<size_t>(ns),
                        cudaMemcpyDeviceToHost,
                        stream_),
                    "copy Yukawa shell values");
                checkCuda(cudaStreamSynchronize(stream_), "sync Yukawa shell copy");
            });

            for (int s = 0; s < ns; ++s) {
                if (std::abs(h_shell_values[s].y) > 1e-7) {
                    integrals[tasks[idx[s]].integral_index].setFailed("Yukawa GPU integral has non-zero imaginary part.");
                    continue;
                }
                integrals[tasks[idx[s]].integral_index].value = h_shell_values[s].x;
            }
        }

        maybe_print_timings(timings, tasks.size());
    }

private:
    bool check_maps(
        const std::map<int, WannierFunction>& wann_map,
        const std::map<int, double>& mean_density) const
    {
        for (const auto& [key, _] : wann_map) {
            if (mean_density.find(key) == mean_density.end()) {
                return false;
            }
        }
        return true;
    }

    bool check_aliasing(const std::array<int, 3>& R) const
    {
        for (int i = 0; i < 3; ++i) {
            if (supercell_[i] < std::abs(static_cast<double>(R[i])) / 2.0) {
                return false;
            }
        }
        return true;
    }

    double yukawa_alpha(const Density_descr& c_dens, const Density_descr& v_dens) const
    {
        const double nv1 = vMeanDensity_.at(v_dens.id1);
        const double nv2 = vMeanDensity_.at(v_dens.id2);
        const double nc1 = cMeanDensity_.at(c_dens.id1);
        const double nc2 = cMeanDensity_.at(c_dens.id2);
        const double mean_density = std::sqrt(std::sqrt(nv1 * nv2) * std::sqrt(nc1 * nc2));
        return YukawaPotential::calc_yukawa_screening_factor(
            mean_density,
            relativePermittivity_,
            screeningAlpha_);
    }

    void ensure_same_band_cache(
        const std::vector<Density_descr>& required,
        SameBandPersistentCache& cache,
        const std::map<int, WannierFunction>& wann_map,
        SolverStageTimingsMs* timings)
    {
        SameBandCacheBuildConfig cfg{};
        cfg.need_aux_fft = keep_dual_spectral_;
        cfg.need_raw_fft = true;
        cfg.keep_dual_spectral = keep_dual_spectral_;
        cfg.wrap_aux = true;
        cfg.enable_timing = timing_enabled_;
        ensure_same_band_cache_entries(required, cache, wann_map, fft_plans_, stream_, N_, cfg, timings);
    }

    void maybe_print_timings(const SolverStageTimingsMs& t, size_t n_tasks) const
    {
        if (!timing_enabled_) {
            return;
        }
        const double total = t.density_build + t.fft + t.contraction + t.host_device_copy;
        std::cout
            << "\n[Timing][GPU Yukawa] tasks=" << n_tasks
            << " density_build_ms=" << t.density_build
            << " fft_ms=" << t.fft
            << " contraction_ms=" << t.contraction
            << " host_device_copy_ms=" << t.host_device_copy
            << " total_ms=" << total
            << std::endl;
    }

    ReciprocalMeshgrid rec_mesh_;
    CufftPlanCache3D fft_plans_;

    std::map<int, double> vMeanDensity_;
    std::map<int, double> cMeanDensity_;
    double relativePermittivity_ = 1.0;
    double screeningAlpha_ = 1.0;

    double dV_ = 0.0;
    int N_ = 0;
    int Q_ = 0;
    std::vector<double> supercell_{};

    int rank_ = 0;
    bool timing_enabled_ = false;
    const bool keep_dual_spectral_ = dual_spectral_enabled_from_env(true);

    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;

    DeviceBuffer<double> d_qx_{};
    DeviceBuffer<double> d_qy_{};
    DeviceBuffer<double> d_qz_{};

    SameBandPersistentCache c_cache_;
    SameBandPersistentCache v_cache_;
};

}  // namespace

bool run_local_field_effects_gpu_dense_single_rank(
    std::map<int, WannierFunction> const& vWannMap,
    std::map<int, WannierFunction> const& cWannMap,
    std::map<Density_descr, Density_indicator> const& lfe_indicators,
    double abscharge_threshold,
    double energy_threshold,
    std::list<Integral>& out_integrals)
{
    out_integrals.clear();

    int rank = 0;
    int num_worker = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_worker);
    if (num_worker != 1 || rank != 0) {
        return false;
    }

    try {
        std::vector<Density_descr> survivors = build_lfe_survivor_tuples(
            lfe_indicators,
            abscharge_threshold);

        std::cout << "[LFE GPU dense] Surviving tuples: " << survivors.size() << std::endl;

        LocalFieldEffectsGpuDenseRunner runner(vWannMap, cWannMap);
        return runner.run(survivors, energy_threshold, out_integrals);
    } catch (const std::exception& e) {
        std::cout << "[WARN] run_local_field_effects_gpu_dense_single_rank failed: " << e.what() << std::endl;
        out_integrals.clear();
        return false;
    }
}

std::unique_ptr<Solver> make_coulomb_solver_gpu(
    std::map<int, WannierFunction> const& vWannMap,
    std::map<int, WannierFunction> const& cWannMap,
    bool wrap_aux)
{
    return std::make_unique<CoulombGpuSolver>(vWannMap, cWannMap, wrap_aux);
}

std::unique_ptr<Solver> make_local_field_effects_solver_gpu(
    std::map<int, WannierFunction> const& vWannMap,
    std::map<int, WannierFunction> const& cWannMap)
{
    return std::make_unique<LocalFieldEffectsGpuSolver>(vWannMap, cWannMap);
}

std::unique_ptr<Solver> make_yukawa_solver_gpu(
    std::map<int, WannierFunction> const& vWannMap,
    std::map<int, WannierFunction> const& cWannMap,
    std::map<int, double> const& vMeanDensity,
    std::map<int, double> const& cMeanDensity,
    double relativePermittivity,
    double screeningAlpha)
{
    return std::make_unique<YukawaGpuSolver>(
        vWannMap,
        cWannMap,
        vMeanDensity,
        cMeanDensity,
        relativePermittivity,
        screeningAlpha);
}
