#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s failed: %s\n", #call,                 \
                         cudaGetErrorString(err));                            \
            return EXIT_FAILURE;                                             \
        }                                                                    \
    } while (0)

__global__ void compute_load(float* output, unsigned long long iterations) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float x = 1.000001f + static_cast<float>(i) * 1e-7f;
    float y = 0.999999f;

    for (unsigned long long j = 0; j < iterations; ++j) {
        x = fmaf(x, y, 0.000001f);
        y = fmaf(y, x, -0.000001f);
    }

    // Keep the compiler from removing the workload as dead code.
    output[i] = x + y;
}

int main(int argc, char** argv) {
    int device = 0;
    double duration_seconds = 300.0;

    if (argc >= 2) {
        duration_seconds = std::atof(argv[1]);
    }
    if (argc >= 3) {
        device = std::atoi(argv[2]);
    }

    if (duration_seconds <= 0.0) {
        std::fprintf(stderr, "Duration must be positive\n");
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const int threads = 256;
    const int blocks = prop.multiProcessorCount * 8;
    const size_t element_count =
        static_cast<size_t>(blocks) * static_cast<size_t>(threads);

    float* output = nullptr;
    CUDA_CHECK(cudaMalloc(&output, element_count * sizeof(float)));

    constexpr unsigned long long iterations = 1000000ULL;

    std::printf("GPU: %s\n", prop.name);
    std::printf("Device: %d\n", device);
    std::printf("Duration: %.1f seconds\n", duration_seconds);
    std::printf("Launch: %d blocks x %d threads\n", blocks, threads);
    std::fflush(stdout);

    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    const auto deadline =
        start + std::chrono::duration<double>(duration_seconds);

    unsigned long long launches = 0;

    while (clock::now() < deadline) {
        compute_load<<<blocks, threads>>>(output, iterations);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        ++launches;
    }

    const double elapsed =
        std::chrono::duration<double>(clock::now() - start).count();

    CUDA_CHECK(cudaFree(output));

    std::printf("Completed %llu kernel launches in %.3f seconds\n",
                launches, elapsed);

    return EXIT_SUCCESS;
}
