#include "backend/backend.h"

#if !defined(WO_BACKEND_GPU) && !defined(WO_BACKEND_CPU)
#error "Define WO_BACKEND_CPU or WO_BACKEND_GPU"
#endif

#if defined(WO_BACKEND_GPU)
std::unique_ptr<Backend> make_gpu_backend();
#else
std::unique_ptr<Backend> make_cpu_backend();
#endif

std::unique_ptr<Backend> makeBackend()
{
#if defined(WO_BACKEND_GPU)
    return make_gpu_backend();
#else
    return make_cpu_backend();
#endif
}
