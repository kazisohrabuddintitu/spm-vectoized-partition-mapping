//
// AVX2 intrinsics partition mapping kernel.
// Uses emulated 64x64 multiply (AVX2 lacks native _mm256_mullo_epi64).
//
// Usage: ./partition_map_avx2 N P [seed] [num_runs]
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
// -DHASH64: full 64-bit multiply-shift
// default:  32-bit multiply-shift on upper bits

#ifdef HASH64
static constexpr uint64_t HASH_CONST64 = 0x9E3779B97F4A7C15ULL;
#else
static constexpr uint32_t HASH_CONST32 = 0x9E3779B9U;
#endif

// Scalar reference (for verification)
__attribute__((noinline))
void partition_map_scalar(const uint64_t* __restrict__ keys,
                          uint32_t* __restrict__ part_id,
                          size_t N,
                          uint32_t shift) {
    for (size_t i = 0; i < N; ++i) {
#ifdef HASH64
        part_id[i] = (uint32_t)((keys[i] * HASH_CONST64) >> shift);
#else
        uint32_t hi = (uint32_t)(keys[i] >> 32);
        part_id[i] = (hi * HASH_CONST32) >> shift;
#endif
    }
}

#ifdef HASH64
// AVX2 64-bit hash: emulated 64x64 multiply (3 x mul_epu32)
__attribute__((noinline))
void partition_map_avx2(const uint64_t* __restrict__ keys,
                        uint32_t* __restrict__ part_id,
                        size_t N,
                        uint32_t shift) {
    const __m256i vconst    = _mm256_set1_epi64x(HASH_CONST64);
    const __m256i vconst_hi = _mm256_srli_epi64(vconst, 32);
    const __m256i pack_mask = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    const __m256i vshift    = _mm256_set1_epi64x(shift);

    size_t i = 0;
    for (; i + 4 <= N; i += 4) {
        __m256i k = _mm256_loadu_si256((const __m256i*)(keys + i));

        __m256i lo_lo = _mm256_mul_epu32(k, vconst);
        __m256i k_hi  = _mm256_srli_epi64(k, 32);
        __m256i hi_lo = _mm256_mul_epu32(k_hi, vconst);
        __m256i lo_hi = _mm256_mul_epu32(k, vconst_hi);
        __m256i cross = _mm256_slli_epi64(_mm256_add_epi64(hi_lo, lo_hi), 32);
        __m256i hash  = _mm256_add_epi64(lo_lo, cross);
        __m256i pid64 = _mm256_srlv_epi64(hash, vshift);

        __m256i packed = _mm256_permutevar8x32_epi32(pid64, pack_mask);
        _mm_storeu_si128((__m128i*)(part_id + i),
                         _mm256_castsi256_si128(packed));
    }
    for (; i < N; ++i)
        part_id[i] = (uint32_t)((keys[i] * HASH_CONST64) >> shift);
}

#else
// AVX2 32-bit hash: processes 8 x uint32_t per iteration.
// Extract upper 32 bits of each key, multiply by 32-bit constant, shift right.
__attribute__((noinline))
void partition_map_avx2(const uint64_t* __restrict__ keys,
                        uint32_t* __restrict__ part_id,
                        size_t N,
                        uint32_t shift) {
    const __m256i vconst32  = _mm256_set1_epi32(HASH_CONST32);
    // Shuffle mask: from 8 x 64-bit loaded as 16 x 32-bit, pick the high 32 bits
    // In each 64-bit lane, the high 32 bits are at odd positions: 1,3,5,7,9,11,13,15
    // But we load 4 keys at a time (256 bits), so positions 1,3,5,7
    const __m256i hi32_mask = _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 4, 6);

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        // Load 8 keys = 2 x 256-bit loads
        __m256i k0 = _mm256_loadu_si256((const __m256i*)(keys + i));
        __m256i k1 = _mm256_loadu_si256((const __m256i*)(keys + i + 4));

        // Extract upper 32 bits from each 64-bit key
        // k0 has 4 keys: [k0_lo, k0_hi, k1_lo, k1_hi, k2_lo, k2_hi, k3_lo, k3_hi]
        // We want [k0_hi, k1_hi, k2_hi, k3_hi] packed
        __m256i hi0 = _mm256_permutevar8x32_epi32(k0, hi32_mask); // hi in low 128
        __m256i hi1 = _mm256_permutevar8x32_epi32(k1, hi32_mask); // hi in low 128

        // Combine: 4 from hi0 (low 128) + 4 from hi1 (low 128) -> 8 x uint32_t
        __m256i hi_all = _mm256_permute2x128_si256(hi0, hi1, 0x20);

        // 32-bit multiply: _mm256_mullo_epi32(hi_all, vconst32)
        __m256i product = _mm256_mullo_epi32(hi_all, vconst32);

        // Shift right
        __m256i pid = _mm256_srli_epi32(product, shift);

        // Store 8 x uint32_t
        _mm256_storeu_si256((__m256i*)(part_id + i), pid);
    }
    // Scalar tail
    for (; i < N; ++i) {
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

    if (P == 0 || (P & (P - 1)) != 0) {
        std::cerr << "Error: P must be a power of 2\n";
        return 1;
    }

    const uint32_t log2P = __builtin_ctz(P);
    const uint32_t shift = 64 - log2P;

    std::cout << "=== Partition Map (AVX2 Intrinsics) ===\n"
              << "  N=" << N << " P=" << P
              << " seed=" << seed << " runs=" << num_runs << "\n";

    // Allocate aligned memory
    uint64_t* keys        = (uint64_t*)aligned_alloc(32, N * sizeof(uint64_t));
    uint32_t* part_id     = (uint32_t*)aligned_alloc(32, N * sizeof(uint32_t));
    uint32_t* ref_part_id = (uint32_t*)aligned_alloc(32, N * sizeof(uint32_t));
    if (!keys || !part_id || !ref_part_id) {
        std::cerr << "Error: memory allocation failed\n";
        return 1;
    }

    // Generate keys
    generate_keys(keys, N, seed);

    // Compute scalar reference for verification
    partition_map_scalar(keys, ref_part_id, N, shift);

    // Warmup AVX2
    partition_map_avx2(keys, part_id, N, shift);

    // Verify AVX2 against scalar reference
    std::cout << "Correctness check:\n";
    bool ok = verify(ref_part_id, part_id, N, "avx2_vs_scalar");
    if (!ok) {
        std::cerr << "ERROR: AVX2 output does not match scalar reference!\n";
        free(keys); free(part_id); free(ref_part_id);
        return 1;
    }

    // Benchmark AVX2
    std::vector<double> times(num_runs);
    for (int r = 0; r < num_runs; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        partition_map_avx2(keys, part_id, N, shift);
        auto t1 = std::chrono::steady_clock::now();
        times[r] = std::chrono::duration<double>(t1 - t0).count();
    }

    auto stats = compute_stats(times, N);
    std::printf("  median: %.6f s  stddev: %.6f s  throughput: %.2f Mkeys/s\n",
                stats.median_s, stats.stddev_s, stats.throughput_Mkeys_s);

    uint64_t cs = compute_checksum(part_id, N);
    std::printf("  checksum: 0x%016lx\n", cs);

    if (N <= 20) {
        std::cout << "  part_id: ";
        for (size_t i = 0; i < N; ++i) std::cout << part_id[i] << " ";
        std::cout << "\n";
    }

    free(keys);
    free(part_id);
    free(ref_part_id);
    return 0;
}
