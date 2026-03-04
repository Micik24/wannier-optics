#include "backend/density_metrics.h"

#include "density_metrics_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

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

struct DeviceWannier
{
    double* values = nullptr;
    const double* host_values = nullptr;
};

class GpuDensityMetricsContext
{
public:
    GpuDensityMetricsContext()
    {
        checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    ~GpuDensityMetricsContext()
    {
        for (auto& [id, wf] : c_cache_) {
            if (wf.values) {
                cudaFree(wf.values);
            }
        }
        for (auto& [id, wf] : v_cache_) {
            if (wf.values) {
                cudaFree(wf.values);
            }
        }

        if (d_w1_ptrs_) cudaFree(d_w1_ptrs_);
        if (d_w2_ptrs_) cudaFree(d_w2_ptrs_);
        if (d_offsets_xyz_) cudaFree(d_offsets_xyz_);
        if (d_valid_) cudaFree(d_valid_);
        if (d_abs_charge_) cudaFree(d_abs_charge_);
        if (h_abs_charge_) cudaFreeHost(h_abs_charge_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    std::vector<double> compute(
        DensityMetricKind kind,
        std::map<int, WannierFunction> const& cWannMap,
        std::map<int, WannierFunction> const& vWannMap,
        std::vector<DensityMetricSpec> const& specs)
    {
        if (kind != DensityMetricKind::TransitionCv) {
            throw std::runtime_error("Unsupported density metric kind.");
        }
        if (specs.empty()) {
            return {};
        }

        initMesh(cWannMap, vWannMap);
        ensureCapacity(specs.size());

        std::vector<const double*> h_w1_ptrs(specs.size(), nullptr);
        std::vector<const double*> h_w2_ptrs(specs.size(), nullptr);
        std::vector<int> h_offsets_xyz(specs.size() * 3, 0);
        std::vector<int> h_valid(specs.size(), 1);

        for (size_t i = 0; i < specs.size(); ++i) {
            auto c_it = cWannMap.find(specs[i].idx1);
            auto v_it = vWannMap.find(specs[i].idx2);
            if (c_it == cWannMap.end() || v_it == vWannMap.end()) {
                throw std::runtime_error("Density metric spec references unknown Wannier index.");
            }

            WannierFunction const& c_wann = c_it->second;
            WannierFunction const& v_wann = v_it->second;

            std::vector<int> Rvec{specs[i].R[0], specs[i].R[1], specs[i].R[2]};

            const std::vector<double> supercell = c_wann.getLatticeInUnitcellBasis();
            if ((std::round(supercell[0]) <= std::abs(Rvec[0])) ||
                (std::round(supercell[1]) <= std::abs(Rvec[1])) ||
                (std::round(supercell[2]) <= std::abs(Rvec[2]))) {
                h_valid[i] = 0;
                continue;
            }

            h_w1_ptrs[i] = ensureUploaded(c_wann, c_cache_, "conduction");
            h_w2_ptrs[i] = ensureUploaded(v_wann, v_cache_, "valence");

            const std::vector<int> offset =
                v_wann.getMeshgrid()->getIndexOffset(Rvec, v_wann.getLatticeInUnitcellBasis());

            h_offsets_xyz[3 * i + 0] = offset[0];
            h_offsets_xyz[3 * i + 1] = offset[1];
            h_offsets_xyz[3 * i + 2] = offset[2];
        }

        checkCuda(cudaMemcpyAsync(d_w1_ptrs_, h_w1_ptrs.data(), sizeof(double*) * specs.size(),
                      cudaMemcpyHostToDevice, stream_),
            "cudaMemcpyAsync(d_w1_ptrs)");
        checkCuda(cudaMemcpyAsync(d_w2_ptrs_, h_w2_ptrs.data(), sizeof(double*) * specs.size(),
                      cudaMemcpyHostToDevice, stream_),
            "cudaMemcpyAsync(d_w2_ptrs)");
        checkCuda(cudaMemcpyAsync(d_offsets_xyz_, h_offsets_xyz.data(), sizeof(int) * specs.size() * 3,
                      cudaMemcpyHostToDevice, stream_),
            "cudaMemcpyAsync(d_offsets_xyz)");
        checkCuda(cudaMemcpyAsync(d_valid_, h_valid.data(), sizeof(int) * specs.size(),
                      cudaMemcpyHostToDevice, stream_),
            "cudaMemcpyAsync(d_valid)");
        checkCuda(cudaMemsetAsync(d_abs_charge_, 0, sizeof(double) * specs.size(), stream_),
            "cudaMemsetAsync(d_abs_charge)");

        launch_abs_charge_kernel(
            d_w1_ptrs_, d_w2_ptrs_, d_offsets_xyz_, d_valid_,
            static_cast<int>(specs.size()), dimx_, dimy_, dimz_, dV_, d_abs_charge_, stream_);
        checkCuda(cudaGetLastError(), "launch_abs_charge_kernel");

        checkCuda(cudaMemcpyAsync(h_abs_charge_, d_abs_charge_, sizeof(double) * specs.size(),
                      cudaMemcpyDeviceToHost, stream_),
            "cudaMemcpyAsync(h_abs_charge)");
        checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(abs_charge)");

        return std::vector<double>(h_abs_charge_, h_abs_charge_ + specs.size());
    }

private:
    void initMesh(
        std::map<int, WannierFunction> const& cWannMap,
        std::map<int, WannierFunction> const& vWannMap)
    {
        if (cWannMap.empty() || vWannMap.empty()) {
            throw std::runtime_error("Wannier maps must not be empty for GPU density metrics.");
        }

        const RealMeshgrid* c_mesh = cWannMap.begin()->second.getMeshgrid();
        const RealMeshgrid* v_mesh = vWannMap.begin()->second.getMeshgrid();

        const std::vector<int> cdim = c_mesh->getDim();
        const std::vector<int> vdim = v_mesh->getDim();
        if (cdim != vdim) {
            throw std::runtime_error("Conduction and valence meshes are not compatible.");
        }

        if (!initialized_) {
            dimx_ = cdim[0];
            dimy_ = cdim[1];
            dimz_ = cdim[2];
            N_ = static_cast<size_t>(dimx_) * dimy_ * dimz_;
            dV_ = c_mesh->getdV();
            initialized_ = true;
            return;
        }

        if (dimx_ != cdim[0] || dimy_ != cdim[1] || dimz_ != cdim[2]) {
            throw std::runtime_error("Mesh dimensions changed during GPU density metric computation.");
        }
    }

    const double* ensureUploaded(
        WannierFunction const& wf,
        std::unordered_map<int, DeviceWannier>& cache,
        const char* label)
    {
        auto it = cache.find(wf.getId());
        const double* host_values = wf.getValue();

        if (it == cache.end()) {
            DeviceWannier device_wf{};
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_wf.values), sizeof(double) * N_),
                "cudaMalloc(wannier values)");
            checkCuda(cudaMemcpyAsync(device_wf.values, host_values, sizeof(double) * N_,
                          cudaMemcpyHostToDevice, stream_),
                "cudaMemcpyAsync(upload wannier)");
            device_wf.host_values = host_values;
            auto [inserted_it, ok] = cache.insert({wf.getId(), device_wf});
            if (!ok) {
                throw std::runtime_error("Failed to insert uploaded Wannier function into GPU cache.");
            }
            return inserted_it->second.values;
        }

        if (it->second.host_values != host_values) {
            checkCuda(cudaMemcpyAsync(it->second.values, host_values, sizeof(double) * N_,
                          cudaMemcpyHostToDevice, stream_),
                "cudaMemcpyAsync(refresh wannier)");
            it->second.host_values = host_values;
        }

        (void)label;
        return it->second.values;
    }

    void ensureCapacity(size_t batch_size)
    {
        if (batch_size <= capacity_) {
            return;
        }

        if (d_w1_ptrs_) cudaFree(d_w1_ptrs_);
        if (d_w2_ptrs_) cudaFree(d_w2_ptrs_);
        if (d_offsets_xyz_) cudaFree(d_offsets_xyz_);
        if (d_valid_) cudaFree(d_valid_);
        if (d_abs_charge_) cudaFree(d_abs_charge_);
        if (h_abs_charge_) cudaFreeHost(h_abs_charge_);

        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_w1_ptrs_), sizeof(double*) * batch_size),
            "cudaMalloc(d_w1_ptrs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_w2_ptrs_), sizeof(double*) * batch_size),
            "cudaMalloc(d_w2_ptrs)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_offsets_xyz_), sizeof(int) * batch_size * 3),
            "cudaMalloc(d_offsets_xyz)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_valid_), sizeof(int) * batch_size),
            "cudaMalloc(d_valid)");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_abs_charge_), sizeof(double) * batch_size),
            "cudaMalloc(d_abs_charge)");
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&h_abs_charge_), sizeof(double) * batch_size),
            "cudaMallocHost(h_abs_charge)");

        capacity_ = batch_size;
    }

    bool initialized_ = false;
    int dimx_ = 0;
    int dimy_ = 0;
    int dimz_ = 0;
    size_t N_ = 0;
    double dV_ = 0.0;

    std::unordered_map<int, DeviceWannier> c_cache_;
    std::unordered_map<int, DeviceWannier> v_cache_;

    cudaStream_t stream_ = nullptr;

    size_t capacity_ = 0;
    const double** d_w1_ptrs_ = nullptr;
    const double** d_w2_ptrs_ = nullptr;
    int* d_offsets_xyz_ = nullptr;
    int* d_valid_ = nullptr;
    double* d_abs_charge_ = nullptr;
    double* h_abs_charge_ = nullptr;
};

}  // namespace

bool density_metrics_gpu_available()
{
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    return status == cudaSuccess && device_count > 0;
}

size_t density_metrics_recommended_max_specs()
{
    if (!density_metrics_gpu_available()) {
        return 32768;
    }

    size_t free_mem = 0;
    size_t total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) {
        return 32768;
    }

    // Per-spec buffers we allocate in the current implementation:
    // d_w1_ptr, d_w2_ptr, d_offset(3 ints), d_valid, d_abs_charge, h_abs_charge(pinned).
    const size_t bytes_per_spec =
        sizeof(double*) * 2 + sizeof(int) * 3 + sizeof(int) + sizeof(double) + sizeof(double);

    // Keep this lightweight and safe: use at most 5% of currently free memory.
    const size_t mem_budget = free_mem / 20;
    size_t max_specs = (bytes_per_spec > 0) ? (mem_budget / bytes_per_spec) : 0;

    // 2D kernel launch uses grid.y = batch_size (spec count), which must stay below device limits.
    // Use a conservative cap for portability.
    constexpr size_t launch_cap = 60000;
    max_specs = std::min(max_specs, launch_cap);

    // Avoid tiny batches.
    max_specs = std::max<size_t>(4096, max_specs);
    return max_specs;
}

std::vector<double> compute_abs_charge_batch_gpu(
    DensityMetricKind kind,
    std::map<int, WannierFunction> const& cWannMap,
    std::map<int, WannierFunction> const& vWannMap,
    std::vector<DensityMetricSpec> const& specs)
{
    if (!density_metrics_gpu_available()) {
        throw std::runtime_error("No CUDA device available for GPU density metrics.");
    }

    static GpuDensityMetricsContext context{};
    return context.compute(kind, cWannMap, vWannMap, specs);
}
