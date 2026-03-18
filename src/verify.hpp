#ifndef VERIFY_HPP
#define VERIFY_HPP

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <iomanip>

// Position-sensitive checksum over partition IDs.
inline uint64_t compute_checksum(const uint32_t* part_id, size_t N) {
    uint64_t cs = 0;
    for (size_t i = 0; i < N; ++i)
        cs ^= ((uint64_t)part_id[i] * 2654435761ULL) ^ (i * 0x9E3779B97F4A7C15ULL);
    return cs;
}

// Element-by-element comparison, prints first mismatches.
inline bool verify_elementwise(const uint32_t* ref, const uint32_t* test,
                               size_t N, const char* label) {
    int errors = 0;
    for (size_t i = 0; i < N; ++i) {
        if (ref[i] != test[i]) {
            if (errors < 20)
                std::cerr << "[" << label << "] mismatch at i=" << i
                          << ": ref=" << ref[i] << " got=" << test[i] << "\n";
            ++errors;
        }
    }
    if (errors > 20)
        std::cerr << "[" << label << "] ... " << (errors - 20) << " more errors\n";
    return errors == 0;
}

// Full verification: checksum + element-wise for small N.
inline bool verify(const uint32_t* ref, const uint32_t* test,
                   size_t N, const char* label, size_t elem_limit = 10000) {
    uint64_t cs_ref  = compute_checksum(ref, N);
    uint64_t cs_test = compute_checksum(test, N);
    bool ok = (cs_ref == cs_test);
    std::cout << "  checksum " << label << ": "
              << (ok ? "PASS" : "FAIL")
              << " (ref=0x" << std::hex << cs_ref
              << " test=0x" << cs_test << std::dec << ")\n";
    if (N <= elem_limit)
        ok = verify_elementwise(ref, test, N, label) && ok;
    return ok;
}

#endif
