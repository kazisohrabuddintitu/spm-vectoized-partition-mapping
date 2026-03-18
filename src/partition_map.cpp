//
// Plain C++ partition mapping kernel.
// Compiled into two binaries from this same source:
//   - partition_map_baseline  (auto-vectorization OFF)
//   - partition_map_autovec   (auto-vectorization ON, with GCC report)
//
// Usage: ./partition_map_baseline N P [seed] [num_runs]
//        ./partition_map_autovec  N P [seed] [num_runs]
//

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h>

#include "keygen.hpp"
#include "verify.hpp"
#include "hpc_helpers.hpp"

// --- Hash strategy selected at compile time ---
// -DHASH64: full 64-bit multiply-shift (Knuth golden ratio)
// default:  32-bit multiply-shift on upper bits (lighter, vectorizes easily)

#ifdef HASH64
static constexpr uint64_t HASH_CONST64 = 0x9E3779B97F4A7C15ULL;

__attribute__((noinline))
void partition_map(const uint64_t* __restrict__ keys,
                   uint32_t* __restrict__ part_id,
                   size_t N,
                   uint32_t shift) {
    for (size_t i = 0; i < N; ++i) {
        part_id[i] = (uint32_t)((keys[i] * HASH_CONST64) >> shift);
    }
}
#else
// 32-bit multiply-shift: extract upper 32 bits, multiply by Knuth's 32-bit
// constant, shift right. Lighter compute, GCC vectorizes easily with AVX2.
static constexpr uint32_t HASH_CONST32 = 0x9E3779B9U;  // golden ratio * 2^32

__attribute__((noinline))
void partition_map(const uint64_t* __restrict__ keys,
                   uint32_t* __restrict__ part_id,
                   size_t N,
                   uint32_t shift) {
    for (size_t i = 0; i < N; ++i) {
        uint32_t hi = (uint32_t)(keys[i] >> 32);
        part_id[i] = (hi * HASH_CONST32) >> shift;
    }
}
#endif

static void print_usage(const char* prog) {
    std::printf("Usage: %s N P [seed] [num_runs]\n", prog);
    std::printf("  N        : number of keys\n");
    std::printf("  P        : number of partitions (must be power of 2)\n");
    std::printf("  seed     : PRNG seed (default: 42)\n");
    std::printf("  num_runs : benchmark repetitions (default: 11)\n");
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const size_t   N        = std::stoul(argv[1]);
    const uint32_t P        = std::stoul(argv[2]);
    const uint64_t seed     = (argc > 3) ? std::stoull(argv[3]) : 42;
    const int      num_runs = (argc > 4) ? std::stoi(argv[4])   : 11;

    // Validate P is power of two
    if (P == 0 || (P & (P - 1)) != 0) {
        std::cerr << "Error: P must be a power of 2\n";
        return 1;
    }

    const uint32_t log2P = __builtin_ctz(P);
#ifdef HASH64
    const uint32_t shift = 64 - log2P;
#else
    const uint32_t shift = 32 - log2P;
#endif

    std::cout << "=== Partition Map (Plain C++) ===\n"
              << "  N=" << N << " P=" << P
              << " seed=" << seed << " runs=" << num_runs << "\n";

    // Allocate aligned memory
    uint64_t* keys    = (uint64_t*)aligned_alloc(32, N * sizeof(uint64_t));
    uint32_t* part_id = (uint32_t*)aligned_alloc(32, N * sizeof(uint32_t));
    if (!keys || !part_id) {
        std::cerr << "Error: memory allocation failed\n";
        return 1;
    }

    // Generate keys
    generate_keys(keys, N, seed);

    // Warmup
    partition_map(keys, part_id, N, shift);

    // Benchmark
    std::vector<double> times(num_runs);
    for (int r = 0; r < num_runs; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        partition_map(keys, part_id, N, shift);
        auto t1 = std::chrono::steady_clock::now();
        times[r] = std::chrono::duration<double>(t1 - t0).count();
    }

    auto stats = compute_stats(times, N);
    std::printf("  median: %.6f s  stddev: %.6f s  throughput: %.2f Mkeys/s\n",
                stats.median_s, stats.stddev_s, stats.throughput_Mkeys_s);

    // Verification: print checksum
    uint64_t cs = compute_checksum(part_id, N);
    std::printf("  checksum: 0x%016lx\n", cs);

    // For small N, print first few values
    if (N <= 20) {
        std::cout << "  part_id: ";
        for (size_t i = 0; i < N; ++i) std::cout << part_id[i] << " ";
        std::cout << "\n";
    }

    free(keys);
    free(part_id);
    return 0;
}
