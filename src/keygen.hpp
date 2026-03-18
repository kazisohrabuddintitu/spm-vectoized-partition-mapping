#ifndef KEYGEN_HPP
#define KEYGEN_HPP

#include <cstdint>
#include <cstddef>

// Deterministic key generation using splitmix64.
// Same seed always produces the same sequence.
inline void generate_keys(uint64_t* keys, size_t N, uint64_t seed) {
    uint64_t state = seed;
    for (size_t i = 0; i < N; ++i) {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        keys[i] = z ^ (z >> 31);
    }
}

#endif
