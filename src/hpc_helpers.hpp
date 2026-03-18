#ifndef HPC_HELPERS_HPP
#define HPC_HELPERS_HPP

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

#ifndef __CUDACC__
    #include <chrono>
#endif

// ---------- Timing macros (from SPM course) ----------

#ifndef __CUDACC__
#define TIMERSTART(label)                                                      \
    auto a##label = std::chrono::steady_clock::now();                          \
    auto b##label = a##label;                                                  \
    double elapsed_##label = 0.0;

#define TIMERSTOP(label)                                                       \
    b##label = std::chrono::steady_clock::now();                               \
    elapsed_##label = std::chrono::duration<double>(b##label - a##label).count(); \
    std::cout << "# elapsed time (" << #label << "): "                         \
              << elapsed_##label << " s\n";

#define TIMERSUM(label1, label2)                                               \
    std::cout << "# elapsed time (" << #label1 << "+" << #label2 << "): "      \
              << (elapsed_##label1 + elapsed_##label2) << " s\n";

#else
    #define TIMERSTART(label)                                                  \
        cudaEvent_t start##label, stop##label;                                 \
        float time##label;                                                     \
        cudaEventCreate(&start##label);                                        \
        cudaEventCreate(&stop##label);                                         \
        cudaEventRecord(start##label, 0);

    #define TIMERSTOP(label)                                                   \
            cudaEventRecord(stop##label, 0);                                   \
            cudaEventSynchronize(stop##label);                                 \
            cudaEventElapsedTime(&time##label, start##label, stop##label);     \
            std::cout << "TIMING: " << time##label << " ms (" << #label << ")" \
                      << std::endl;

    #define TIMERSUM(label1, label2)                                           \
        std::cout << "(" << #label1 << "+" << #label2 <<"): "                  \
                  << (time##label1 + time##label2) << "ms" << std::endl;
#endif

#ifdef __CUDACC__
    #define CUERR {                                                            \
        cudaError_t err;                                                       \
        if ((err = cudaGetLastError()) != cudaSuccess) {                       \
            std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "    \
                      << __FILE__ << ", line " << __LINE__ << std::endl;       \
            exit(1);                                                           \
        }                                                                      \
    }
    #define H2D (cudaMemcpyHostToDevice)
    #define D2H (cudaMemcpyDeviceToHost)
#endif

#define SDIV(x,y) (((x)+(y)-1)/(y))

// ---------- Benchmarking statistics ----------

struct BenchStats {
    double median_s;
    double stddev_s;
    double throughput_Mkeys_s;
};

inline BenchStats compute_stats(std::vector<double>& times, size_t N) {
    std::sort(times.begin(), times.end());
    BenchStats s;
    size_t n = times.size();
    s.median_s = (n % 2 == 0)
        ? (times[n/2 - 1] + times[n/2]) / 2.0
        : times[n/2];
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / n;
    double sq_sum = 0;
    for (auto t : times) sq_sum += (t - mean) * (t - mean);
    s.stddev_s = std::sqrt(sq_sum / n);
    s.throughput_Mkeys_s = (double)N / s.median_s / 1e6;
    return s;
}

#endif
